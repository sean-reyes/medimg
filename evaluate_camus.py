"""Check the trained CAMUS outliner on the test patients.

1. Draws the model's outline next to the experts' for six patients, from best to worst.
2. Computes ejection fraction (EF) with the method of disks, from the experts' masks and from the model's.

Run from the project root after train_camus.py:  python evaluate_camus.py
"""
import re

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy import ndimage  # python -m pip install scipy
import torch.nn.functional as F

DATA_PATH = r"c:\Users\gabit\Downloads\subject_based_dataset.hdf5\subject_based_dataset.hdf5"
device = "cuda" if torch.cuda.is_available() else "cpu"
RES = 192  # image size: 128 is faster, 192 keeps more detail
K = 384 // RES
MODEL_PATH = "models/camus_unet.pt" if RES == 128 else f"models/camus_unet_{RES}.pt"


def tokens(name):
    return set(re.split(r"[^a-z0-9]+", name.lower()))


# ---------- load the data exactly as train_camus.py did ----------
with h5py.File(DATA_PATH, "r") as f:
    names = []
    f.visititems(lambda n, o: names.append(n) if isinstance(o, h5py.Dataset) else None)

    def get(*parts):
        hits = [n for n in names if set(parts) <= tokens(n)]
        if len(hits) != 1:
            raise SystemExit(f"Expected one dataset matching {parts}, found {hits}")
        return f[hits[0]][:, :, :, 0]

    data = {}
    for phase in ("ed", "es"):
        img, msk = get("train", "4ch", phase, "frames"), get("train", "4ch", phase, "masks")
        n = img.shape[0]
        img = img.reshape(n, RES, K, RES, K).mean(axis=(2, 4)).astype(np.float32)
        data[phase] = (img, msk[:, K // 2::K, K // 2::K].astype(np.int64))

ed_img, ed_msk = data["ed"]
es_img, es_msk = data["es"]
split = np.load("models/camus_split.npz")
train_p, test_p = split["train"], split["test"]


def norm(x):
    m = x.mean(axis=(1, 2), keepdims=True)
    s = x.std(axis=(1, 2), keepdims=True) + 1e-6
    return ((x - m) / s).astype(np.float32)


# ---------- the same U-Net as in train_camus.py ----------
def block(i, o):
    return nn.Sequential(
        nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
        nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, c=16, classes=4):
        super().__init__()
        self.e1, self.e2, self.e3 = block(1, c), block(c, 2 * c), block(2 * c, 4 * c)
        self.mid = block(4 * c, 8 * c)
        self.u3, self.d3 = nn.ConvTranspose2d(8 * c, 4 * c, 2, stride=2), block(8 * c, 4 * c)
        self.u2, self.d2 = nn.ConvTranspose2d(4 * c, 2 * c, 2, stride=2), block(4 * c, 2 * c)
        self.u1, self.d1 = nn.ConvTranspose2d(2 * c, c, 2, stride=2), block(2 * c, c)
        self.out = nn.Conv2d(c, classes, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(F.max_pool2d(e1, 2))
        e3 = self.e3(F.max_pool2d(e2, 2))
        m = self.mid(F.max_pool2d(e3, 2))
        d3 = self.d3(torch.cat([self.u3(m), e3], 1))
        d2 = self.d2(torch.cat([self.u2(d3), e2], 1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
        return self.out(d1)


def predict(model, X):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 32):
            out.append(model(torch.from_numpy(X[i:i + 32, None]).to(device)).argmax(1).cpu().numpy())
    return np.concatenate(out)


# ---------- EF by the method of disks ----------
def disk_volume(mask, n_slices=20):
    """Slice the ventricle along its long axis and add up thin discs (pixel units; the unit cancels in EF)."""
    ys, xs = np.nonzero(mask == 1)
    if len(ys) < 50:
        return np.nan
    pts = np.stack([xs, ys], 1).astype(float)
    centre = pts.mean(0)
    axis = np.linalg.svd(pts - centre, full_matrices=False)[2][0]  # direction of the long axis
    t = (pts - centre) @ axis
    h = (t.max() - t.min()) / n_slices
    idx = np.minimum(((t - t.min()) / h).astype(int), n_slices - 1)
    width = np.bincount(idx, minlength=n_slices) / h  # average width of each slice
    return float(np.sum(np.pi / 4 * width ** 2 * h))


def ef(ed_mask, es_mask):
    return 1 - disk_volume(es_mask) / disk_volume(ed_mask)


model = UNet().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
pred_ed = predict(model, norm(ed_img[test_p]))
pred_es = predict(model, norm(es_img[test_p]))

ef_true = np.array([ef(a, b) for a, b in zip(ed_msk[test_p], es_msk[test_p])])
ef_pred = np.array([ef(a, b) for a, b in zip(pred_ed, pred_es)])
ef_train = np.array([ef(a, b) for a, b in zip(ed_msk[train_p], es_msk[train_p])])

ok = ~np.isnan(ef_true) & ~np.isnan(ef_pred)
err = np.abs(ef_true - ef_pred)
base = np.abs(ef_true - np.nanmedian(ef_train))
print(f"Test patients used: {ok.sum()} of {len(ok)}")
print(f"Model EF error:    mean {err[ok].mean() * 100:.1f}  median {np.median(err[ok]) * 100:.1f} percentage points")
print(f"Baseline (typical EF for everyone): mean {base[ok].mean() * 100:.1f}  median {np.median(base[ok]) * 100:.1f}")
print(f"Within 10 points of the experts' EF: model {np.mean(err[ok] < 0.10):.0%}, baseline {np.mean(base[ok] < 0.10):.0%}")
print(f"Correlation between model EF and expert EF: {np.corrcoef(ef_true[ok], ef_pred[ok])[0, 1]:.2f}")


# ---------- does cleaning up stray pixels help? ----------
def keep_largest(masks):
    """Keep only the biggest connected ventricle region in each outline and drop stray specks."""
    out = masks.copy()
    for i, m in enumerate(masks):
        lab, n = ndimage.label(m == 1)
        if n > 1:
            sizes = ndimage.sum(m == 1, lab, range(1, n + 1))
            out[i][(m == 1) & (lab != 1 + int(np.argmax(sizes)))] = 0
    return out


ef_clean = np.array([ef(a, b) for a, b in zip(keep_largest(pred_ed), keep_largest(pred_es))])
ok2 = ok & ~np.isnan(ef_clean)
err2 = np.abs(ef_true - ef_clean)
print(f"\nAfter keeping only the largest ventricle blob: mean {err2[ok2].mean() * 100:.1f}  "
      f"median {np.median(err2[ok2]) * 100:.1f}  within 10 points {np.mean(err2[ok2] < 0.10):.0%}  "
      f"correlation {np.corrcoef(ef_true[ok2], ef_clean[ok2])[0, 1]:.2f}")

# ---------- picture: six patients from smallest to largest EF error ----------
order = np.argsort(np.where(ok, err, np.inf))[: ok.sum()]
picks = [order[int(q * (len(order) - 1))] for q in (0, 0.2, 0.4, 0.6, 0.8, 1.0)]
fig, axes = plt.subplots(3, 4, figsize=(14, 10))
for k, p in enumerate(picks):
    for j, (img, true_m, pred_m, name) in enumerate([(ed_img, ed_msk, pred_ed, "ED"), (es_img, es_msk, pred_es, "ES")]):
        ax = axes[k // 2, (k % 2) * 2 + j]
        ax.imshow(img[test_p[p]], cmap="gray")
        ax.contour(true_m[test_p[p]] == 1, levels=[0.5], colors="white", linewidths=1.2)
        ax.contour(pred_m[p] == 1, levels=[0.5], colors="red", linewidths=1.2)
        ax.set_title(f"Test patient {p} {name}\nEF experts {ef_true[p]:.0%}, model {ef_pred[p]:.0%}", fontsize=9)
        ax.axis("off")
fig.suptitle("White = experts' ventricle outline, red = model's. Left to right, top to bottom: best to worst EF error.")
plt.tight_layout()
plt.savefig("camus_eval.png", dpi=110)
print("Saved camus_eval.png")