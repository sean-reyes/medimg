import streamlit as st
import pandas as pd
import joblib

model = joblib.load("models/model.joblib")

st.title("Heart Disease Prediction")
# st.caption("Not a medical tool. For demonstration only.")

age = st.number_input("Age", 20, 90, 55)
sex = st.selectbox("Sex", ["Male", "Female"])
cp = st.selectbox("Chest pain type", ["typical angina", "atypical angina", "non-anginal", "asymptomatic"])
bp1, bp2 = st.columns(2)
trestbps = bp1.number_input("Systolic Blood Pressure", 80, 220, 130)
diastolic = bp2.number_input("Diastolic Blood Pressure", 40, 130, 80)
ldl = st.number_input("LDL", 0, 400, 110)
hdl = st.number_input("HDL", 0, 150, 50)
tg = st.number_input("Triglycerides", 0, 1000, 150)
chol = ldl + hdl + tg / 5   # the model was trained on total cholesterol only
if tg >= 400:
    st.warning("Triglycerides of 400 mg/dl or more make the total cholesterol calculation unreliable.")
glucose = st.number_input("Fasting blood sugar (mg/dl)", 40, 400, 95)
fbs = 1 if glucose > 120 else 0
restecg = st.selectbox("Resting ECG", ["normal", "lv hypertrophy", "st-t abnormality"])
thalch = st.number_input("Max heart rate", 60, 210, 140)
exang = st.selectbox("Experiencing exercise-induced angina?", ["NO", "YES"])
oldpeak = st.number_input("ST depression in mm (oldpeak)", -3.0, 7.0, 1.0)
slope = st.selectbox("Slope", ["upsloping", "flat", "downsloping", "unknown"])

if st.button("Predict"):
    row = pd.DataFrame([{
        "age": age, "sex": sex, "dataset": "Cleveland", "cp": cp,
        "trestbps": trestbps, "chol": chol, "fbs": fbs, "restecg": restecg,
        "thalch": thalch, "exang": exang, "oldpeak": oldpeak, "slope": slope,
        "fbs": fbs,
        "exang": 1 if exang == "YES" else 0,
    }])
    prob = model.predict_proba(row)[0][1]
    st.write(f"Estimated probability of heart disease: {prob:.0%}")