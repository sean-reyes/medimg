"""Load PTB-XL at 100 Hz into numpy arrays.

The first run reads about 21,400 files, which takes a few minutes, and saves a cache so later runs start in seconds.
Set PTBXL_DIR to the folder where you saved PTB-XL (it can be anywhere). The cache is about 1 GB: add cache/ to .gitignore.
"""
import ast
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

PTBXL_DIR = r"c:\Users\gabit\Downloads\ptbxl"
CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
CACHE = Path("cache/ptbxl_100hz.npz")


def load():
    """Return signals (n, 1000, 12), labels (n, 5) of 0/1, folds (n,) and ECG ids (n,).

    Records with no diagnostic label (only form or rhythm statements) are left out.
    """
    if CACHE.exists():
        d = np.load(CACHE)
        return d["x"], d["y"], d["fold"], d["ecg_id"]

    db = next(Path(PTBXL_DIR).rglob("ptbxl_database.csv"), None)
    if db is None:
        raise SystemExit(f"Could not find ptbxl_database.csv anywhere under {PTBXL_DIR}. Check PTBXL_DIR in ptbxl_data.py.")
    base = str(db.parent) + "/"

    meta = pd.read_csv(base + "ptbxl_database.csv", index_col="ecg_id")
    meta["scp_codes"] = meta["scp_codes"].apply(ast.literal_eval)
    stmts = pd.read_csv(base + "scp_statements.csv", index_col=0)
    stmts = stmts[stmts["diagnostic"] == 1]
    meta["labels"] = meta["scp_codes"].apply(
        lambda codes: sorted({stmts.loc[c, "diagnostic_class"] for c in codes if c in stmts.index})
    )
    meta = meta[meta["labels"].map(len) > 0]

    x = np.zeros((len(meta), 1000, 12), dtype=np.float32)
    for i, f in enumerate(meta["filename_lr"]):
        x[i] = wfdb.rdsamp(base + f)[0]
        if (i + 1) % 2000 == 0:
            print(f"  read {i + 1} of {len(meta)} ECGs")
    x = np.nan_to_num(x)
    y = np.array([[c in labs for c in CLASSES] for labs in meta["labels"]], dtype=np.int8)
    fold = meta["strat_fold"].to_numpy()
    ids = meta.index.to_numpy()

    CACHE.parent.mkdir(exist_ok=True)
    np.savez(CACHE, x=x, y=y, fold=fold, ecg_id=ids)
    return x, y, fold, ids