import streamlit as st
import numpy as np
import pickle
import pandas as pd

# Load saved model and scaler
with open("diabetes_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Page config
st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🏥",
    layout="centered"
)

# Title
st.title("🏥 Diabetes Prediction System")
st.markdown("Enter the patient's medical details below to predict diabetes risk.")
st.divider()

# Input form
st.subheader("Patient Medical Details")

col1, col2 = st.columns(2)

with col1:
    pregnancies    = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
    glucose        = st.number_input("Glucose Level", min_value=0, max_value=300, value=120)
    blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=150, value=70)
    skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)

with col2:
    insulin = st.number_input("Insulin Level", min_value=0, max_value=900, value=80)
    bmi     = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
    dpf     = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
    age     = st.number_input("Age", min_value=1, max_value=120, value=25)

st.divider()

# Predict button
if st.button("🔍 Predict Diabetes Risk", use_container_width=True):

    # Prepare input data
    input_data = np.array([[pregnancies, glucose, blood_pressure,
                            skin_thickness, insulin, bmi, dpf, age]])

    # Scale the input
    input_scaled = scaler.transform(input_data)

    # Make prediction
    prediction  = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]

    st.divider()
    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ High Risk — This patient is likely DIABETIC")
        st.metric("Diabetes Probability", f"{probability[1]*100:.1f}%")
        st.warning("Please consult a doctor immediately for proper diagnosis.")
    else:
        st.success("✅ Low Risk — This patient is likely NOT DIABETIC")
        st.metric("Healthy Probability", f"{probability[0]*100:.1f}%")
        st.info("Maintain a healthy lifestyle to stay diabetes-free!")

    # Show input summary table
    st.divider()
    st.subheader("Input Summary")
    summary = pd.DataFrame({
        "Feature": ["Pregnancies", "Glucose", "Blood Pressure",
                    "Skin Thickness", "Insulin", "BMI", "DPF", "Age"],
        "Value":   [pregnancies, glucose, blood_pressure,
                    skin_thickness, insulin, bmi, dpf, age]
    })
    st.dataframe(summary, use_container_width=True)