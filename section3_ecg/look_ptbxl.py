"""Look at PTB-XL: load the labels, count the five categories, and save a picture of one ECG.

Install once:  python -m pip install wfdb pandas matplotlib numpy
Run from the project root:  python look_ptbxl.py
"""
import ast
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to a file, no window needed
import matplotlib.pyplot as plt
import pandas as pd
import wfdb

CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]

# Change this to the folder where you saved PTB-XL. It can be anywhere on your computer.
PTBXL_DIR = r"c:\Users\gabit\Downloads\ptbxl"

# The unzipped download is often nested in an extra folder, so search for the label file.
db = next(Path(PTBXL_DIR).rglob("ptbxl_database.csv"), None)
if db is None:
    raise SystemExit(f"Could not find ptbxl_database.csv anywhere under {PTBXL_DIR}. Check the path.")
PATH = str(db.parent) + "/"
print("Using the dataset in:", PATH)

meta = pd.read_csv(PATH + "ptbxl_database.csv", index_col="ecg_id")
meta["scp_codes"] = meta["scp_codes"].apply(ast.literal_eval)

# Roll the detailed statements up into the five diagnostic categories.
stmts = pd.read_csv(PATH + "scp_statements.csv", index_col=0)
stmts = stmts[stmts["diagnostic"] == 1]


def to_classes(codes):
    return sorted({stmts.loc[c, "diagnostic_class"] for c in codes if c in stmts.index})


meta["labels"] = meta["scp_codes"].apply(to_classes)

print(f"\nRecords: {len(meta)}   patients: {meta['patient_id'].nunique()}")
print(f"Records with no diagnostic label: {(meta['labels'].map(len) == 0).sum()}")
print("\nRecords per category (a record can have several):")
for c in CLASSES:
    print(f"   {c}: {meta['labels'].map(lambda labs: c in labs).sum()}")
print("\nHow many labels each record has:", meta["labels"].map(len).value_counts().sort_index().to_dict())
print("Records per fold:", meta["strat_fold"].value_counts().sort_index().to_dict())
print("(Folds 1-8 are for training, 9 for validation and 10 for testing, as the dataset's authors recommend.)")

# Draw all 12 leads of the first labeled record, using the 100 Hz version.
row = meta[meta["labels"].map(len) > 0].iloc[0]
signal, fields = wfdb.rdsamp(PATH + row["filename_lr"])
print(f"\nExample ECG {row.name}: signal shape {signal.shape}, sampling rate {fields['fs']} Hz, labels {row['labels']}")
print("Leads:", fields["sig_name"])

fig, axes = plt.subplots(12, 1, figsize=(10, 14), sharex=True)
for i, ax in enumerate(axes):
    ax.plot(signal[:, i], linewidth=0.8)
    ax.set_ylabel(fields["sig_name"][i], rotation=0, labelpad=18)
    ax.set_yticks([])
axes[-1].set_xlabel("Sample (100 per second, so 1000 samples = 10 seconds)")
fig.suptitle(f"ECG {row.name}: {', '.join(row['labels'])}")
plt.tight_layout()
plt.savefig("ptbxl_example.png", dpi=100)
print("Saved ptbxl_example.png in the current folder.")