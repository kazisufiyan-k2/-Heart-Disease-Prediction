import streamlit as st
import pandas as pd
import joblib

# Page config
st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="centered")

# Custom CSS
st.markdown("""
<style>

body {
    background-color: #0e1117;
}

.main-title {
    text-align:center;
    font-size:40px;
    color:#ff4b4b;
    font-weight:bold;
}

.subtitle{
    text-align:center;
    color:gray;
    margin-bottom:30px;
}

.stButton>button {
    width:100%;
    background-color:#ff4b4b;
    color:white;
    font-size:18px;
    border-radius:10px;
    height:50px;
}

.stButton>button:hover{
    background-color:#ff2a2a;
}

.result-box{
    padding:20px;
    border-radius:10px;
    text-align:center;
    font-size:20px;
}

</style>
""", unsafe_allow_html=True)

# Load model files
model = joblib.load("knn_heart_model.pkl")
scaler = joblib.load("heart_scaler.pkl")
expected_columns = joblib.load("heart_columns.pkl")

# Title
st.markdown('<p class="main-title">❤️ Heart Disease Prediction</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Enter medical details to check heart disease risk</p>', unsafe_allow_html=True)

# Layout columns
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 100, 40)
    sex = st.selectbox("Sex", ["M", "F"])
    chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
    resting_bp = st.number_input("Resting Blood Pressure", 80, 200, 120)
    cholesterol = st.number_input("Cholesterol", 100, 600, 200)

with col2:
    fasting_bs = st.selectbox("Fasting Blood Sugar >120", [0,1])
    resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
    max_hr = st.slider("Max Heart Rate", 60, 220, 150)
    exercise_angina = st.selectbox("Exercise Angina", ["Y","N"])
    oldpeak = st.slider("Oldpeak", 0.0, 6.0, 1.0)
    st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

# Prediction button
if st.button("Predict Heart Risk"):

    raw_input = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,
        'Sex_' + sex: 1,
        'ChestPainType_' + chest_pain: 1,
        'RestingECG_' + resting_ecg: 1,
        'ExerciseAngina_' + exercise_angina: 1,
        'ST_Slope_' + st_slope: 1
    }

    input_df = pd.DataFrame([raw_input])

    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_columns]

    scaled_input = scaler.transform(input_df)

    prediction = model.predict(scaled_input)[0]

    if prediction == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")