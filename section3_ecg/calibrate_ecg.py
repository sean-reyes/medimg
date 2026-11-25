"""Calibrate the ECG network's scores on the validation fold, so a score of 0.30 means that roughly 30% of
records with that score really have the pattern. The test fold is used only to check that it worked.

Run after train_ecg_cnn.py, from the project root:  python section3_ecg/calibrate_ecg.py
It adds the calibration to models/ecg_cnn.pt.
"""
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from ecg_model import ECGNet, MODEL_PATH, _logit, _sigmoid
from ptbxl_data import CLASSES, load

x, y, fold, _ = load()
val_idx, test_idx = np.where(fold == 9)[0], np.where(fold == 10)[0]

ckpt = torch.load(MODEL_PATH, map_location="cpu")
net = ECGNet()
net.load_state_dict(ckpt["state_dict"])
net.eval()
mu, sd = torch.tensor(ckpt["mu"]).view(1, 12, 1), torch.tensor(ckpt["sd"]).view(1, 12, 1)


def predict(ids):
    out = []
    with torch.no_grad():
        for i in range(0, len(ids), 256):
            xb = torch.from_numpy(x[ids[i:i + 256]]).permute(0, 2, 1)
            out.append(torch.sigmoid(net((xb - mu) / sd)).numpy())
    return np.concatenate(out)


def ece(p, truth, bins=10):
    """Expected calibration error: the average gap between the score and the real share of positives."""
    edges, total = np.linspace(0, 1, bins + 1), 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
        if m.any():
            total += m.mean() * abs(p[m].mean() - truth[m].mean())
    return total


p_val, p_test = predict(val_idx), predict(test_idx)
calibration, cal_thresholds = {}, {}
print(f"{'category':9}{'Brier raw':>11}{'Brier cal.':>12}{'ECE raw':>9}{'ECE cal.':>10}{'AUROC raw':>11}{'AUROC cal.':>12}")
for i, c in enumerate(CLASSES):
    # Platt scaling: a straight-line correction on the log-odds, fitted on the validation fold.
    lr = LogisticRegression(C=1e6).fit(_logit(p_val[:, i]).reshape(-1, 1), y[val_idx][:, i])
    a, b = float(lr.coef_[0, 0]), float(lr.intercept_[0])
    calibration[c] = [a, b]
    cal_thresholds[c] = float(_sigmoid(a * _logit(ckpt["thresholds"][c]) + b))  # same decisions as before
    p_cal = _sigmoid(a * _logit(p_test[:, i]) + b)
    t = y[test_idx][:, i]
    print(f"{c:9}{np.mean((p_test[:, i] - t) ** 2):>11.4f}{np.mean((p_cal - t) ** 2):>12.4f}"
          f"{ece(p_test[:, i], t):>9.3f}{ece(p_cal, t):>10.3f}{roc_auc_score(t, p_test[:, i]):>11.3f}{roc_auc_score(t, p_cal):>12.3f}")

ckpt["calibration"], ckpt["cal_thresholds"] = calibration, cal_thresholds
torch.save(ckpt, MODEL_PATH)
print(f"\nSaved the calibration into {MODEL_PATH}. Lower Brier and ECE are better, and AUROC should not change.")