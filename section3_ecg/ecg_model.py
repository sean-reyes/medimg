"""ECG pattern model for the app: loads the trained CNN and returns scores for the five PTB-XL categories.

Needs models/ecg_cnn.pt from train_ecg_cnn.py (run calibrate_ecg.py afterwards so the scores are calibrated).
Input: a (1000, 12) array, 10 seconds at 100 Hz, 12 leads in millivolts, in the PTB-XL lead order.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_PATH = "models/ecg_cnn.pt"


class Block(nn.Module):  # identical to train_ecg_cnn.py
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


class ECGNet(nn.Module):  # identical to train_ecg_cnn.py
    def __init__(self, classes=5):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(12, 32, 15, stride=2, padding=7, bias=False), nn.BatchNorm1d(32), nn.ReLU())
        self.blocks = nn.Sequential(Block(32, 32, 1), Block(32, 64, 2), Block(64, 128, 2), Block(128, 128, 2))
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(128, classes)

    def forward(self, t):
        t = self.blocks(self.stem(t))
        return self.fc(self.drop(t.mean(dim=2)))


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _sigmoid(z):
    return 1 / (1 + np.exp(-z))


def load_model():
    ckpt = torch.load(MODEL_PATH, map_location="cpu")
    net = ECGNet()
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    return {"net": net, "ckpt": ckpt,
            "mu": torch.tensor(ckpt["mu"]).view(1, 12, 1), "sd": torch.tensor(ckpt["sd"]).view(1, 12, 1)}


def analyse(model, signal):
    """Return {'status': 'ok' or 'reject', 'messages': [...], 'scores': {class: {'score', 'flagged'}}, 'calibrated': bool}."""
    sig = np.asarray(signal, dtype=np.float32)
    problems = []
    if sig.shape != (1000, 12):
        problems.append(f"Expected 1000 rows (10 seconds at 100 Hz) by 12 lead columns, but got {sig.shape[0]} by "
                        f"{sig.shape[1] if sig.ndim > 1 else 1}.")
    elif not np.isfinite(sig).all():
        problems.append("The file contains missing or non-numeric values.")
    elif np.abs(sig).max() > 50:
        problems.append("The values are too large to be millivolts, so the units or the file format may be wrong.")
    elif (sig.std(axis=0) < 1e-3).sum() >= 3:
        problems.append("Several leads are flat, which suggests a disconnected lead or a wrong file.")
    if problems:
        return {"status": "reject", "messages": problems, "scores": {}, "calibrated": False}

    ckpt = model["ckpt"]
    x = (torch.from_numpy(sig).T.unsqueeze(0) - model["mu"]) / model["sd"]
    with torch.no_grad():
        raw = torch.sigmoid(model["net"](x))[0].numpy()

    cal, cal_thr = ckpt.get("calibration"), ckpt.get("cal_thresholds")
    scores = {}
    for c, p in zip(ckpt["classes"], raw):
        if cal and cal_thr:
            a, b = cal[c]
            score, thr = float(_sigmoid(a * _logit(p) + b)), cal_thr[c]
        else:
            score, thr = float(p), ckpt["thresholds"][c]
        scores[c] = {"score": score, "flagged": bool(score >= thr)}
    return {"status": "ok", "messages": [], "scores": scores, "calibrated": bool(cal and cal_thr)}