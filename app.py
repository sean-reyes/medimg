# import streamlit as st
# import pandas as pd
# import joblib

# model = joblib.load("models/model.joblib")

# st.title("Heart Disease Prediction")
# # st.caption("Not a medical tool. For demonstration only.")

# age = st.number_input("Age", 20, 90, 55)
# sex = st.selectbox("Sex", ["Male", "Female"])
# cp = st.selectbox("Chest pain type", ["typical angina", "atypical angina", "non-anginal", "asymptomatic"])
# bp1, bp2 = st.columns(2)
# trestbps = bp1.number_input("Systolic Blood Pressure", 80, 220, 130)
# diastolic = bp2.number_input("Diastolic Blood Pressure", 40, 130, 80)
# ldl = st.number_input("LDL", 0, 400, 110)
# hdl = st.number_input("HDL", 0, 150, 50)
# tg = st.number_input("Triglycerides", 0, 1000, 150)
# chol = ldl + hdl + tg / 5   # the model was trained on total cholesterol only
# if tg >= 400:
#     st.warning("Triglycerides of 400 mg/dl or more make the total cholesterol calculation unreliable.")
# glucose = st.number_input("Fasting blood sugar (mg/dl)", 40, 400, 95)
# fbs = 1 if glucose > 120 else 0
# restecg = st.selectbox("Resting ECG", ["normal", "lv hypertrophy", "st-t abnormality"])
# thalch = st.number_input("Max heart rate", 60, 210, 140)
# target_hr = 0.85 * (220 - age)
# if thalch < target_hr:
#     st.warning(
#         f"Max heart rate is below 85% of the age-predicted maximum ({target_hr:.0f} bpm). "
#         "A real exercise test this submaximal can be considered inconclusive, and the "
#         "exercise-related inputs (ST depression, slope, exercise chest pain) may be less reliable. "
#         "The model does not account for this."
#     )
# exang = st.selectbox("Experiencing exercise-induced angina?", ["NO", "YES"])
# oldpeak = st.number_input("ST depression in mm (oldpeak)", -3.0, 7.0, 1.0)
# slope = st.selectbox("Slope", ["upsloping", "flat", "downsloping", "unknown"])

# if st.button("Predict"):
#     row = pd.DataFrame([{
#         "age": age, "sex": sex, "dataset": "Cleveland", "cp": cp,
#         "trestbps": trestbps, "chol": chol, "fbs": fbs, "restecg": restecg,
#         "thalch": thalch, "exang": exang, "oldpeak": oldpeak, "slope": slope,
#         "fbs": fbs,
#         "exang": 1 if exang == "YES" else 0,
#     }])
#     prob = model.predict_proba(row)[0][1]
#     st.write(f"Estimated probability of heart disease: {prob:.0%}")

"""Heart tools: a heart disease risk estimator, with an echo ejection fraction prototype below it.

Run from the project root:
    python -m streamlit run app.py
Needs models/model.joblib, saved by train_model.py.
"""
import joblib
import pandas as pd
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Heart tools", layout="wide")


@st.cache_resource
def load_model():
    return joblib.load("models/model.joblib")


model = load_model()
prep = model.named_steps["prep"]

# Read columns and category spellings from the trained pipeline so they always match training.
used, cats = set(), {}
for name, trans, cols in prep.transformers_:
    if name == "remainder":
        continue
    used.update(cols)
    if name == "cat":
        for col, c in zip(cols, trans.named_steps["onehot"].categories_):
            cats[col] = [str(x) for x in c]

# ---------- Section 1: heart disease risk estimator ----------
st.title("Heart tools")
st.header("Heart disease risk estimator")
st.warning("Learning project, not a medical tool. Do not use it to make decisions about anyone's health.")

row = {}
c = st.columns(4)
if "age" in used:
    row["age"] = c[0].number_input("Age (years)", 20, 90, 55)
if "sex" in cats:
    row["sex"] = c[1].selectbox("Sex", cats["sex"])
if "dataset" in cats:
    row["dataset"] = c[2].selectbox("Hospital the data came from", cats["dataset"])
if "cp" in cats:
    row["cp"] = c[3].selectbox("Chest pain type", cats["cp"])

c = st.columns(4)
if "trestbps" in used:
    row["trestbps"] = c[0].number_input("Resting blood pressure, top number (mm Hg)", 80, 220, 130)
if "thalch" in used:
    row["thalch"] = c[1].number_input("Maximum heart rate (bpm)", 60, 210, 140)
if "oldpeak" in used:
    row["oldpeak"] = c[2].number_input("ST depression during exercise", -3.0, 7.0, 1.0, step=0.1)
if "slope" in cats:
    row["slope"] = c[3].selectbox("Slope of the peak exercise ST segment", cats["slope"])

c = st.columns(4)
if "restecg" in cats:
    row["restecg"] = c[0].selectbox("Resting ECG result", cats["restecg"])
if "fbs" in used:
    glucose = c[1].number_input("Fasting blood sugar (mg/dl)", 40, 400, 95)
    row["fbs"] = int(glucose > 120)
if "exang" in used:
    row["exang"] = int(c[2].selectbox("Chest pain during exercise", ["No", "Yes"]) == "Yes")

if "chol" in used:
    c = st.columns(4)
    ldl = c[0].number_input("LDL cholesterol (mg/dl)", 0, 400, 110)
    hdl = c[1].number_input("HDL cholesterol (mg/dl)", 0, 150, 50)
    tg = c[2].number_input("Triglycerides (mg/dl)", 0, 1000, 150)
    row["chol"] = ldl + hdl + tg / 5  # the model was trained on total cholesterol only
    c[3].caption(f"Total cholesterol used by the model: {row['chol']:.0f} mg/dl")
    if tg >= 400:
        st.warning("Triglycerides of 400 mg/dl or more make the total cholesterol calculation unreliable.")

if "thalch" in row and "age" in row:
    target_hr = 0.85 * (220 - row["age"])
    if row["thalch"] < target_hr:
        st.warning(
            f"Max heart rate is below 85% of the age-predicted maximum ({target_hr:.0f} bpm). "
            "A real exercise test this submaximal can be considered inconclusive. The model does not account for this."
        )

if st.button("Estimate risk", type="primary"):
    prob = model.predict_proba(pd.DataFrame([row]))[0][1]
    left, right = st.columns([1, 3])
    left.metric("Estimated probability", f"{prob:.0%}")
    right.progress(float(prob))
    right.caption("A rough guide from a small dataset. It can be wrong in both directions.")

# ---------- Section 2: echo ejection fraction prototype ----------
st.divider()
st.header("Echo ejection fraction (research prototype)")
st.warning("Research prototype. Not validated for clinical use. No trained model is connected yet.")

a, b = st.columns(2)
ed = a.file_uploader("End of filling frame (end-diastole)", type=["png", "jpg", "jpeg"])
es = b.file_uploader("End of squeezing frame (end-systole)", type=["png", "jpg", "jpeg"])
if ed:
    a.image(Image.open(ed), use_container_width=True)
if es:
    b.image(Image.open(es), use_container_width=True)


@st.cache_resource
def load_echo():
    import echo_model
    return echo_model, echo_model.load_model()


if st.button("Analyse frames", disabled=not (ed and es)):
    try:
        echo_model, echo_net = load_echo()
    except FileNotFoundError:
        st.error("Echo model file not found. Run train_camus.py first (it saves models/camus_unet_192.pt).")
    else:
        res = echo_model.analyse(echo_net, Image.open(ed), Image.open(es))
        c1, c2 = st.columns(2)
        c1.image(res["ed_overlay"], caption="End of filling, ventricle outlined in red", use_container_width=True)
        c2.image(res["es_overlay"], caption="End of squeezing, ventricle outlined in red", use_container_width=True)
        if res["status"] == "reject":
            st.error("Scan rejected. " + " ".join(res["messages"]) + " Please upload a new pair of frames.")
        elif res["status"] == "unsure":
            st.info("Unsure, so no number is shown. " + " ".join(res["messages"]))
        else:
            st.metric("Estimated ejection fraction", f"{res['ef']:.0%}")
            st.caption(echo_model.ERROR_NOTE)
        st.caption("Check the outlines against the images before relying on any result.")


# ---------- Section 3: ECG pattern classifier ----------
from pathlib import Path

import numpy as np

st.divider()
st.header("ECG pattern classifier (research prototype)")
st.warning("Research prototype. Not validated for clinical use. It estimates ECG patterns, not diagnoses.")

ECG_LABELS = {
    "NORM": "Normal ECG",
    "MI": "Pattern consistent with myocardial infarction",
    "STTC": "ST/T change",
    "CD": "Conduction disturbance",
    "HYP": "Hypertrophy",
}

ecg_file = st.file_uploader(
    "ECG as a CSV: 1000 rows (10 seconds at 100 Hz) by 12 lead columns, no header", type=["csv"]
)
if ecg_file:
    sig = pd.read_csv(ecg_file, header=None).to_numpy(dtype=float)
    if sig.shape != (1000, 12):
        st.error(f"Expected 1000 rows by 12 columns, got {sig.shape[0]} by {sig.shape[1]}.")
    else:
        st.line_chart(pd.DataFrame({"Lead I": sig[:, 0]}))
        if not Path("models/ecg_model.joblib").exists():
            st.info("Unsure: no ECG model found yet. Run train_ecg.py first.")
        else:
            bundle = joblib.load("models/ecg_model.joblib")
            x = sig[None]  # (1, 1000, 12); must match features() in train_ecg.py
            feats = np.concatenate([x.mean(1), x.std(1), x.min(1), x.max(1)], axis=1)
            probs = [float(p[0, 1]) for p in bundle["model"].predict_proba(feats)]
            for cls, p in zip(bundle["classes"], probs):
                st.write(f"{ECG_LABELS.get(cls, cls)}: {p:.0%}")
                st.progress(p)
            if max(probs) < 0.5:
                st.info("Unsure: no category reached 50%, so treat this result as inconclusive.")
            st.caption("Each category is estimated separately, so a record can match more than one.")


# ---------- Sections 4 and 5: more scan types (prototype stubs) ----------
def scan_section(number, title, upload_label, file_types, classes, predict, note, multiple=False):
    """One upload box, one button, one probability bar per class, and an 'unsure' fallback."""
    st.divider()
    st.header(f"{title}")
    st.warning("Research prototype. Not validated for clinical use. It estimates patterns, not diagnoses.")
    files = st.file_uploader(upload_label, type=file_types, accept_multiple_files=multiple, key=title)
    if files and st.button("Analyse", key=title + "_go"):
        probs = predict(files)
        if probs is None:
            st.info("Unsure: no model is connected yet, so no result is shown.")
        else:
            for cls, p in zip(classes, probs):
                st.write(f"{cls}: {p:.0%}")
                st.progress(float(p))
            if max(probs) < 0.5:
                st.info("Unsure: no category reached 50%, so treat this result as inconclusive.")
    st.caption(note)


def predict_mri(file):
    """Plug in: segment the ventricles and heart muscle, compute volumes, ejection fraction and
    muscle mass, then classify those measurements. Return five probabilities, or None."""
    return None


def predict_valve(files):
    """Plug in: classify each echo image by view, keep the aortic valve views, combine them into
    one study-level result. Return three probabilities, or None."""
    return None


scan_section(
    4, "Cardiac MRI: cardiomyopathy patterns (ACDC)",
    "Cine MRI file (NIfTI, .nii or .nii.gz)", ["gz", "nii"],
    ["Normal", "Dilated cardiomyopathy", "Hypertrophic cardiomyopathy",
     "Previous myocardial infarction", "Right ventricular abnormality"],
    predict_mri,
    "Plan: segment, measure volumes and ejection fraction, then classify the measurements.",
)

scan_section(
    5, "Echo images: aortic stenosis (TMED-2)",
    "Echo images from one study (PNG or JPG)", ["png", "jpg", "jpeg"],
    ["No aortic stenosis", "Early aortic stenosis", "Significant aortic stenosis"],
    predict_valve,
    "Covers the aortic valve only, not valve disease in general.",
    multiple=True,
)