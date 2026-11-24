"""Baseline ECG pattern classifier on PTB-XL: simple summary features and a random forest.

1. Download PTB-XL from https://physionet.org/content/ptb-xl/1.0.3/ and set PTBXL_DIR in ptbxl_data.py.
2. python -m pip install wfdb scikit-learn joblib
3. Run from the project root:  python section3_ecg/train_ecg.py   (or python train_ecg.py if it is in the root)

Folds 1-8 train the model, fold 9 picks each category's decision threshold, and fold 10 is the test set, looked at once.
"""
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score

from ptbxl_data import CLASSES, load

x, y, fold, _ = load()
print(f"Loaded {len(x)} ECGs with at least one diagnostic label")


def features(sig):
    """Simple per-lead summaries (48 numbers). The app must use this exact function. A neural network on the raw signal is the next step up."""
    return np.concatenate([sig.mean(1), sig.std(1), sig.min(1), sig.max(1)], axis=1)


X = features(x)
train, val, test = fold <= 8, fold == 9, fold == 10
print(f"Records: {train.sum()} train, {val.sum()} validation, {test.sum()} test")

model = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=42)
model.fit(X[train], y[train])  # one yes/no output per category, since a record can have several labels


def probs(m, feats):
    return np.column_stack([p[:, 1] for p in m.predict_proba(feats)])


p_val, p_test = probs(model, X[val]), probs(model, X[test])

print(f"\n{'category':9}{'val AUROC':>11}{'test AUROC':>12}{'threshold':>11}{'test F1':>9}")
thresholds, aucs = {}, []
grid = np.arange(0.05, 0.96, 0.05)
for i, c in enumerate(CLASSES):
    # The cut-off between "yes" and "no" is chosen on the validation fold, never on the test fold.
    f1s = [f1_score(y[val][:, i], p_val[:, i] >= t, zero_division=0) for t in grid]
    thresholds[c] = float(grid[int(np.argmax(f1s))])
    auc_val = roc_auc_score(y[val][:, i], p_val[:, i])
    auc_test = roc_auc_score(y[test][:, i], p_test[:, i])
    f1_test = f1_score(y[test][:, i], p_test[:, i] >= thresholds[c], zero_division=0)
    aucs.append(auc_test)
    print(f"{c:9}{auc_val:>11.3f}{auc_test:>12.3f}{thresholds[c]:>11.2f}{f1_test:>9.3f}")
print(f"\nMacro test AUROC: {np.mean(aucs):.3f}   (0.5 is guessing, 1.0 is perfect)")

Path("models").mkdir(exist_ok=True)
joblib.dump({"model": model, "classes": CLASSES, "thresholds": thresholds}, "models/ecg_model.joblib")
np.savetxt("sample_ecg.csv", x[test][0], delimiter=",")  # a test ECG to try in the app
print("Saved models/ecg_model.joblib and sample_ecg.csv")