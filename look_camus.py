"""Look at the CAMUS data: check the mask labels, estimate a rough EF from the masks, save a picture.

Install once:  python -m pip install h5py matplotlib numpy
Set DATA_PATH below, then run from the project root:  python look_camus.py
"""
import re

import h5py
import matplotlib

matplotlib.use("Agg")  # draw to a file, no window needed
import matplotlib.pyplot as plt
import numpy as np

DATA_PATH = r"c:\Users\gabit\Downloads\subject_based_dataset.hdf5\subject_based_dataset.hdf5"


def tokens(name):
    return set(re.split(r"[^a-z0-9]+", name.lower()))


with h5py.File(DATA_PATH, "r") as f:
    names = []
    f.visititems(lambda n, o: names.append(n) if isinstance(o, h5py.Dataset) else None)
    print("Datasets in the file:")
    for n in names:
        print("  ", n, f[n].shape)

    def get(*parts):
        hits = [n for n in names if set(parts) <= tokens(n)]
        if len(hits) != 1:
            raise SystemExit(f"Expected exactly one dataset matching {parts}, found {hits}")
        return f[hits[0]]

    ed_img, ed_msk = get("train", "4ch", "ed", "frames"), get("train", "4ch", "ed", "masks")
    es_img, es_msk = get("train", "4ch", "es", "frames"), get("train", "4ch", "es", "masks")

    # 1. Which label values do the masks use? In CAMUS, 1 is usually the left ventricle cavity,
    #    2 the heart muscle around it and 3 the left atrium, but check the picture to be sure.
    ed_all = ed_msk[:, :, :, 0]
    es_all = es_msk[:, :, :, 0]
    print("\nMask label values:", np.unique(ed_all[:50]))

    # 2. Rough EF from the ventricle cavity (label 1). Area ratio ** 1.5 approximates the volume ratio.
    #    This is only a sanity check; a proper volume method comes later.
    a_ed = (ed_all == 1).sum(axis=(1, 2)).astype(float)
    a_es = (es_all == 1).sum(axis=(1, 2)).astype(float)
    ok = (a_ed > 0) & (a_es > 0)
    ef = 1 - (a_es[ok] / a_ed[ok]) ** 1.5
    print(f"\nPatients with a ventricle mask in both frames: {ok.sum()} of {len(ok)}")
    print(f"Rough EF  min {ef.min():.0%}  median {np.median(ef):.0%}  max {ef.max():.0%}")
    print("(Real EF values mostly fall between about 20% and 80%.)")

    # 3. Save a picture of patient 0: the two frames with their masks drawn on top.
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    for ax, img, msk, title in [(axes[0], ed_img, ed_msk, "End of filling (ED)"),
                                (axes[1], es_img, es_msk, "End of squeezing (ES)")]:
        ax.imshow(img[0, :, :, 0], cmap="gray")
        m = msk[0, :, :, 0]
        ax.imshow(np.ma.masked_where(m == 0, m), alpha=0.4, cmap="viridis")
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("camus_preview.png", dpi=120)
    print("\nSaved camus_preview.png in the current folder. Open it to check the masks line up.")

    # 4. Look at the lowest rough-EF patients, since extreme values are the most likely to be errors.
    idx = np.where(ok)[0]
    rough = {i: 1 - (a_es[i] / a_ed[i]) ** 1.5 for i in idx}
    lowest = sorted(idx, key=lambda i: rough[i])[:8]
    print("\nLowest rough EF (patient index: EF):")
    for i in lowest:
        print(f"  patient {i}: {rough[i]:.0%}")

    w = lowest[0]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    for ax, img, msk, title in [(axes[0], ed_img, ed_msk, f"Patient {w} ED"),
                                (axes[1], es_img, es_msk, f"Patient {w} ES")]:
        ax.imshow(img[w, :, :, 0], cmap="gray")
        m = msk[w, :, :, 0]
        ax.imshow(np.ma.masked_where(m == 0, m), alpha=0.4, cmap="viridis")
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("camus_lowest_ef.png", dpi=120)
    print("Saved camus_lowest_ef.png, the patient with the lowest rough EF.")