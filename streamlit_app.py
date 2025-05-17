import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and scaler
try:
    model = joblib.load('random_forest_model.pkl')
    scaler = joblib.load('scaler.pkl')
except FileNotFoundError:
    st.error("Model or scaler file not found. Please ensure 'random_forest_model.pkl' and 'scaler.pkl' are in the repository.")
    st.stop()

# Expected feature names
expected_features = ['Pregnancies', 'Glucose', 'BloodPressure', 'Insulin', 'BMI', 
                    'DiabetesPedigreeFunction', 'Age', 'Glucose_BMI_Ratio']

st.title('Diabetes Prediction App')
st.write('Enter patient details to predict diabetes risk')

# Input fields
pregnancies = st.number_input('Pregnancies', 0, 20, 1)
glucose = st.number_input('Glucose', 0.0, 200.0, 120.0)
blood_pressure = st.number_input('Blood Pressure', 0.0, 150.0, 70.0)
insulin = st.number_input('Insulin', 0.0, 1000.0, 100.0)
bmi = st.number_input('BMI', 0.0, 70.0, 30.0)
dpf = st.number_input('Diabetes Pedigree Function', 0.0, 3.0, 0.5)
age = st.number_input('Age', 0, 100, 30)

# Predict
if st.button('Predict'):
    if bmi == 0:
        st.error("BMI cannot be zero. Please enter a valid BMI.")
    else:
        # Create input data with explicit column order
        input_data = pd.DataFrame({
            'Pregnancies': [pregnancies],
            'Glucose': [glucose],
            'BloodPressure': [blood_pressure],
            'Insulin': [insulin],
            'BMI': [bmi],
            'DiabetesPedigreeFunction': [dpf],
            'Age': [age],
            'Glucose_BMI_Ratio': [glucose / bmi]
        }, columns=expected_features)  # Ensure correct column order
        
        try:
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)
            st.write('Prediction:', 'Diabetic' if prediction[0] == 1 else 'Non-Diabetic')
        except ValueError as e:
            st.error(f"Error in prediction: {str(e)}. Expected features: {expected_features}")
