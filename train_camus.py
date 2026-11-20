"""Train a small U-Net to outline the heart on four-chamber CAMUS frames (built to run on a normal PC).

Install once:  python -m pip install torch h5py numpy
Set DATA_PATH, then run from the project root:  python train_camus.py
Tip: set EPOCHS = 3 for a first run to check everything works, then raise it.
"""
import re
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DATA_PATH = r"c:\Users\gabit\Downloads\subject_based_dataset.hdf5\subject_based_dataset.hdf5"
RES = 192  # image size: 128 is faster, 192 keeps more detail
K = 384 // RES
MODEL_PATH = "models/camus_unet.pt" if RES == 128 else f"models/camus_unet_{RES}.pt"
EPOCHS = 10
BATCH = 16
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Training on:", device)
Path("models").mkdir(exist_ok=True)


def tokens(name):
    return set(re.split(r"[^a-z0-9]+", name.lower()))


# ---------- load and shrink the data ----------
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
        img = get("train", "4ch", phase, "frames")
        msk = get("train", "4ch", phase, "masks")
        n = img.shape[0]
        # 384 -> RES pixels: average KxK blocks for the images, keep every Kth pixel for the masks
        img = img.reshape(n, RES, K, RES, K).mean(axis=(2, 4)).astype(np.float32)
        msk = msk[:, K // 2::K, K // 2::K].astype(np.int64)
        data[phase] = (img, msk)

ed_img, ed_msk = data["ed"]
es_img, es_msk = data["es"]
n_patients = ed_img.shape[0]


def norm(x):
    """Standardize each image so overall brightness differences don't matter."""
    m = x.mean(axis=(1, 2), keepdims=True)
    s = x.std(axis=(1, 2), keepdims=True) + 1e-6
    return ((x - m) / s).astype(np.float32)


# ---------- split by patient, so one person never appears in two sets ----------
perm = np.random.default_rng(SEED).permutation(n_patients)
n_val = n_test = int(0.15 * n_patients)
test_p, val_p, train_p = perm[:n_test], perm[n_test:n_test + n_val], perm[n_test + n_val:]
np.savez("models/camus_split.npz", train=train_p, val=val_p, test=test_p)
print(f"Patients: {len(train_p)} train, {len(val_p)} validation, {len(test_p)} test")


def stack(p):
    """ED frames followed by ES frames for the given patients."""
    x = norm(np.concatenate([ed_img[p], es_img[p]]))
    y = np.concatenate([ed_msk[p], es_msk[p]])
    return x, y


Xtr, Ytr = stack(train_p)
Xva, Yva = stack(val_p)


# ---------- the model: a small U-Net ----------
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
            xb = torch.from_numpy(X[i:i + 32, None]).to(device)
            out.append(model(xb).argmax(1).cpu().numpy())
    return np.concatenate(out)


def dice_lv(pred, true):
    """Overlap between predicted and expert ventricle (label 1): 1.0 is perfect, 0 is none."""
    p, t = pred == 1, true == 1
    inter = (p & t).sum(axis=(1, 2))
    denom = p.sum(axis=(1, 2)) + t.sum(axis=(1, 2))
    return float(np.mean(2 * inter / np.maximum(denom, 1)))


def rough_ef(ed_masks, es_masks):
    """Same rough EF as look_camus.py: 1 - (ES area / ED area) ** 1.5. NaN if there is no ventricle."""
    a_ed = (ed_masks == 1).sum(axis=(1, 2)).astype(float)
    a_es = (es_masks == 1).sum(axis=(1, 2)).astype(float)
    return np.where(a_ed > 0, 1 - (a_es / np.maximum(a_ed, 1)) ** 1.5, np.nan)


# ---------- train ----------
model = UNet().to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
best, t0 = 0.0, time.time()
for epoch in range(1, EPOCHS + 1):
    model.train()
    order = np.random.permutation(len(Xtr))
    total = 0.0
    for i in range(0, len(order), BATCH):
        b = order[i:i + BATCH]
        xb = torch.from_numpy(Xtr[b, None]).to(device)
        yb = torch.from_numpy(Ytr[b]).to(device)
        # light augmentation: random contrast and brightness (no flips, they change the heart's orientation)
        xb = xb * (1 + 0.2 * (torch.rand(len(b), 1, 1, 1, device=device) - 0.5)) \
            + 0.2 * (torch.rand(len(b), 1, 1, 1, device=device) - 0.5)
        loss = F.cross_entropy(model(xb), yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total += loss.item() * len(b)
    d = dice_lv(predict(model, Xva), Yva)
    print(f"epoch {epoch}/{EPOCHS}  loss {total / len(Xtr):.3f}  validation Dice {d:.3f}  [{(time.time() - t0) / 60:.1f} min]")
    if d > best:
        best = d
        torch.save(model.state_dict(), MODEL_PATH)

# ---------- final check on patients the model never saw ----------
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
pred_ed = predict(model, norm(ed_img[test_p]))
pred_es = predict(model, norm(es_img[test_p]))
dice = dice_lv(np.concatenate([pred_ed, pred_es]), np.concatenate([ed_msk[test_p], es_msk[test_p]]))
print(f"\nTest Dice (ventricle outline): {dice:.3f}")

ef_true = rough_ef(ed_msk[test_p], es_msk[test_p])
ef_pred = rough_ef(pred_ed, pred_es)
ok = ~np.isnan(ef_true) & ~np.isnan(ef_pred)
mae = np.abs(ef_true[ok] - ef_pred[ok]).mean() * 100
baseline = np.abs(ef_true[ok] - np.nanmedian(rough_ef(ed_msk[train_p], es_msk[train_p]))).mean() * 100
print(f"Rough EF error vs the expert masks: {mae:.1f} percentage points on {ok.sum()} test patients")
print(f"Baseline (give everyone the typical EF): {baseline:.1f} percentage points")
print(f"Saved {MODEL_PATH} and models/camus_split.npz")