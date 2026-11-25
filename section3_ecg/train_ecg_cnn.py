"""A small 1D convolutional network for the five PTB-XL categories, trained on the raw 100 Hz signals.

Run from the project root:  python section3_ecg/train_ecg_cnn.py   (or python train_ecg_cnn.py)
Needs ptbxl_data.py in the same folder. Tip: set EPOCHS = 2 first to check it runs and to time one epoch.
"""
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score

from ptbxl_data import CLASSES, load

EPOCHS = 15
BATCH = 64
SEED = 42
MODEL_PATH = "models/ecg_cnn.pt"
BASELINE_MACRO_AUROC = 0.809  # the random forest from train_ecg.py, for comparison

torch.manual_seed(SEED)
np.random.seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Training on:", device)

x, y, fold, _ = load()  # x: (n, 1000, 12) float32
train_idx, val_idx, test_idx = (np.where(m)[0] for m in (fold <= 8, fold == 9, fold == 10))
print(f"Records: {len(train_idx)} train, {len(val_idx)} validation, {len(test_idx)} test")

# Standardize each lead using the training records only.
mu = torch.tensor(x[train_idx].mean(axis=(0, 1)), dtype=torch.float32).view(1, 12, 1).to(device)
sd = torch.tensor(x[train_idx].std(axis=(0, 1)), dtype=torch.float32).view(1, 12, 1).to(device)


def prep(ids):
    """A batch of records as a (batch, 12 leads, 1000 samples) tensor, standardized."""
    xb = torch.from_numpy(x[ids]).permute(0, 2, 1).contiguous().to(device)
    return (xb - mu) / sd


class Block(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.c1 = nn.Conv1d(cin, cout, 7, stride=stride, padding=3, bias=False)
        self.b1 = nn.BatchNorm1d(cout)
        self.c2 = nn.Conv1d(cout, cout, 7, padding=3, bias=False)
        self.b2 = nn.BatchNorm1d(cout)
        self.skip = (nn.Sequential(nn.Conv1d(cin, cout, 1, stride=stride, bias=False), nn.BatchNorm1d(cout))
                     if cin != cout or stride != 1 else nn.Identity())

    def forward(self, t):
        out = F.relu(self.b1(self.c1(t)))
        out = self.b2(self.c2(out))
        return F.relu(out + self.skip(t))


class ECGNet(nn.Module):
    def __init__(self, classes=5):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(12, 32, 15, stride=2, padding=7, bias=False), nn.BatchNorm1d(32), nn.ReLU())
        self.blocks = nn.Sequential(Block(32, 32, 1), Block(32, 64, 2), Block(64, 128, 2), Block(128, 128, 2))
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(128, classes)

    def forward(self, t):
        t = self.blocks(self.stem(t))
        return self.fc(self.drop(t.mean(dim=2)))  # average over time, then one score per category


def predict(model, ids):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(ids), 256):
            out.append(torch.sigmoid(model(prep(ids[i:i + 256]))).cpu().numpy())
    return np.concatenate(out)


def macro_auroc(truth, p):
    return float(np.mean([roc_auc_score(truth[:, i], p[:, i]) for i in range(truth.shape[1])]))


model = ECGNet().to(device)
print(f"Model size: {sum(p.numel() for p in model.parameters()):,} parameters")
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

best, t0 = 0.0, time.time()
Path("models").mkdir(exist_ok=True)
for epoch in range(1, EPOCHS + 1):
    model.train()
    order = np.random.permutation(train_idx)
    total = 0.0
    for i in range(0, len(order), BATCH):
        ids = np.sort(order[i:i + BATCH])
        xb = prep(ids)
        # light augmentation: random amplitude change and a small shift in time
        xb = xb * (0.9 + 0.2 * torch.rand(len(ids), 1, 1, device=device))
        xb = torch.roll(xb, shifts=int(np.random.randint(-50, 51)), dims=2)
        yb = torch.from_numpy(y[ids]).float().to(device)
        loss = F.binary_cross_entropy_with_logits(model(xb), yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total += loss.item() * len(ids)
    sched.step()
    val_auc = macro_auroc(y[val_idx], predict(model, val_idx))
    print(f"epoch {epoch}/{EPOCHS}  loss {total / len(order):.3f}  validation macro AUROC {val_auc:.3f}  "
          f"[{(time.time() - t0) / 60:.1f} min]")
    if val_auc > best:
        best = val_auc
        torch.save(model.state_dict(), MODEL_PATH)

# ---------- final scores: thresholds from the validation fold, then the test fold, looked at once ----------
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
p_val, p_test = predict(model, val_idx), predict(model, test_idx)
grid = np.arange(0.05, 0.96, 0.05)
thresholds, aucs = {}, []
print(f"\n{'category':9}{'val AUROC':>11}{'test AUROC':>12}{'threshold':>11}{'test F1':>9}")
for i, c in enumerate(CLASSES):
    f1s = [f1_score(y[val_idx][:, i], p_val[:, i] >= t, zero_division=0) for t in grid]
    thresholds[c] = float(grid[int(np.argmax(f1s))])
    auc_val = roc_auc_score(y[val_idx][:, i], p_val[:, i])
    auc_test = roc_auc_score(y[test_idx][:, i], p_test[:, i])
    aucs.append(auc_test)
    f1_test = f1_score(y[test_idx][:, i], p_test[:, i] >= thresholds[c], zero_division=0)
    print(f"{c:9}{auc_val:>11.3f}{auc_test:>12.3f}{thresholds[c]:>11.2f}{f1_test:>9.3f}")
print(f"\nMacro test AUROC: {np.mean(aucs):.3f}   (the random forest baseline scored {BASELINE_MACRO_AUROC:.3f})")

torch.save({"state_dict": model.state_dict(), "classes": CLASSES, "thresholds": thresholds,
            "mu": mu.cpu().view(-1).tolist(), "sd": sd.cpu().view(-1).tolist()}, MODEL_PATH)
print(f"Saved {MODEL_PATH}")