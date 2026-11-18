import streamlit as st
import pandas as pd
import joblib

model = joblib.load("models/model.joblib")

st.title("Heart Disease Prediction (learning project)")
st.caption("Not a medical tool. For demonstration only.")

age = st.number_input("Age", 20, 90, 55)
sex = st.selectbox("Sex", ["Male", "Female"])
cp = st.selectbox("Chest pain type", ["typical angina", "atypical angina", "non-anginal", "asymptomatic"])
trestbps = st.number_input("Resting blood pressure", 80, 220, 130)
chol = st.number_input("Cholesterol", 100, 600, 220)
fbs = st.selectbox("Fasting blood sugar > 120mg/dl?", ["NO", "YES"])
restecg = st.selectbox("Resting ECG", ["normal", "lv hypertrophy", "st-t abnormality"])
thalch = st.number_input("Max heart rate", 60, 210, 140)
exang = st.selectbox("Experiencing exercise-induced angina?", ["NO", "YES"])
oldpeak = st.number_input("ST depression (oldpeak)", -3.0, 7.0, 1.0)
slope = st.selectbox("Slope", ["upsloping", "flat", "downsloping", "unknown"])

if st.button("Predict"):
    row = pd.DataFrame([{
        "age": age, "sex": sex, "dataset": "Cleveland", "cp": cp,
        "trestbps": trestbps, "chol": chol, "fbs": fbs, "restecg": restecg,
        "thalch": thalch, "exang": exang, "oldpeak": oldpeak, "slope": slope,
        "fbs": 1 if fbs == "YES" else 0,
        "exang": 1 if exang == "YES" else 0,
    }])
    prob = model.predict_proba(row)[0][1]
    st.write(f"Estimated probability of heart disease: {prob:.0%}")