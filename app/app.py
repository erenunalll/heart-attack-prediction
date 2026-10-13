import streamlit as st
import pandas as pd
import joblib

st.title("Heart Attack Prediction")
st.write("Machine Learning Demo")

patient_data = []

# features
age = st.number_input(
    "Enter your age",
    min_value=0,
    max_value=120
)
gender = st.selectbox(
    "Select your gender",
    ["Male", "Female"]
)
heart_rate = st.number_input(
    "Enter your heart rate"
)
sys_blood_pressure = st.number_input(
    "Enter your systolic blood pressure",
    min_value=40,
    max_value=300
)
dias_blood_pressure = st.number_input(
    "Enter your diastolic blood pressure"
)
blood_sugar = st.number_input(
    "Enter your blood sugar"
)
ckmb = st.number_input(
    "Enter your CK-MB"
)
troponin = st.number_input(
    "Enter your troponin"
)

# submit
if st.button("Submit"):
    patient_data = [
        age,
        1 if gender == "Male" else 0,
        heart_rate,
        sys_blood_pressure,
        dias_blood_pressure,
        blood_sugar,
        ckmb,
        troponin
    ]

    # prediction
    model = joblib.load("models/decision_tree_model.joblib")

    input_data = pd.DataFrame(
        [patient_data],
        columns=[
            "Age",
            "Gender",
            "Heart rate",
            "Systolic blood pressure",
            "Diastolic blood pressure",
            "Blood sugar",
            "CK-MB",
            "Troponin"
        ]
    )

    prediction = model.predict(input_data)
    st.write("Prediction:", prediction[0])