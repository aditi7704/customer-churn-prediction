import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load(
    r'C:\Users\Its Adii\Desktop\customer-churn-prediction\models\churn_model.pkl'
)

st.title("Customer Churn Prediction")

tenure = st.slider("Tenure", 0, 72)

monthly_charges = st.number_input(
    "Monthly Charges"
)

if st.button("Predict"):

    input_data = np.array([
        [tenure, monthly_charges]
    ])

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("Customer Likely to Churn")
    else:
        st.success("Customer Likely to Stay")