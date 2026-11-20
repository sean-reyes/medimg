"""Save a few held-out test patients as PNG files, to try in the app (step 8: test it like a skeptic).

Run from the project root after train_camus.py:  python export_test_frames.py
"""
import re
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

DATA_PATH = r"c:\Users\gabit\Downloads\subject_based_dataset.hdf5\subject_based_dataset.hdf5"
HOW_MANY = 5


def tokens(name):
    return set(re.split(r"[^a-z0-9]+", name.lower()))


test_p = np.load("models/camus_split.npz")["test"][:HOW_MANY]
Path("test_frames").mkdir(exist_ok=True)
with h5py.File(DATA_PATH, "r") as f:
    names = []
    f.visititems(lambda n, o: names.append(n) if isinstance(o, h5py.Dataset) else None)
    for phase in ("ed", "es"):
        hits = [n for n in names if {"train", "4ch", phase, "frames"} <= tokens(n)]
        frames = f[hits[0]]
        for p in test_p:
            x = frames[int(p), :, :, 0].astype(np.float32)
            x = (x - x.min()) / (x.max() - x.min() + 1e-6) * 255
            Image.fromarray(x.astype(np.uint8)).save(f"test_frames/patient{p:03d}_{phase.upper()}.png")
print("Saved PNG files in test_frames/. Upload a matching ED and ES pair in the app.")