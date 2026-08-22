import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)


# ============================================================
# MODEL PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"


# ============================================================
# LOAD RANDOM FOREST MODEL
# ============================================================

@st.cache_resource
def load_artifacts():

    model = joblib.load(
        MODEL_DIR / "churn_model.pkl"
    )

    feature_columns = joblib.load(
        MODEL_DIR / "feature_columns.pkl"
    )

    return model, feature_columns


model, feature_columns = load_artifacts()


# ============================================================
# TITLE
# ============================================================

st.title("📊 Customer Churn Prediction")

st.write(
    "Enter customer details to predict whether "
    "the customer is likely to churn."
)


# ============================================================
# INPUTS
# ============================================================

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)


SeniorCitizen = st.selectbox(
    "Senior Citizen",
    ["Yes", "No"]
)


Partner = st.selectbox(
    "Partner",
    ["Yes", "No"]
)


Dependents = st.selectbox(
    "Dependents",
    ["Yes", "No"]
)


tenure = st.number_input(
    "Tenure (Months)",
    min_value=0,
    max_value=72,
    value=12,
    step=1
)


PhoneService = st.selectbox(
    "Phone Service",
    ["Yes", "No"]
)


MultipleLines = st.selectbox(
    "Multiple Lines",
    [
        "No",
        "Yes",
        "No phone service"
    ]
)


InternetService = st.selectbox(
    "Internet Service",
    [
        "DSL",
        "Fiber optic",
        "No"
    ]
)


OnlineSecurity = st.selectbox(
    "Online Security",
    [
        "Yes",
        "No",
        "No internet service"
    ]
)


OnlineBackup = st.selectbox(
    "Online Backup",
    [
        "Yes",
        "No",
        "No internet service"
    ]
)


DeviceProtection = st.selectbox(
    "Device Protection",
    [
        "Yes",
        "No",
        "No internet service"
    ]
)


TechSupport = st.selectbox(
    "Tech Support",
    [
        "Yes",
        "No",
        "No internet service"
    ]
)


StreamingTV = st.selectbox(
    "Streaming TV",
    [
        "Yes",
        "No",
        "No internet service"
    ]
)


StreamingMovies = st.selectbox(
    "Streaming Movies",
    [
        "Yes",
        "No",
        "No internet service"
    ]
)


Contract = st.selectbox(
    "Contract",
    [
        "Month-to-month",
        "One year",
        "Two year"
    ]
)


PaperlessBilling = st.selectbox(
    "Paperless Billing",
    [
        "Yes",
        "No"
    ]
)


PaymentMethod = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)


MonthlyCharges = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    max_value=200.0,
    value=70.0,
    step=1.0
)


TotalCharges = st.number_input(
    "Total Charges",
    min_value=0.0,
    max_value=10000.0,
    value=1000.0,
    step=10.0
)


# ============================================================
# PREDICTION
# ============================================================

if st.button("Predict"):

    # --------------------------------------------------------
    # Convert Senior Citizen
    # --------------------------------------------------------

    senior_citizen_value = (
        1 if SeniorCitizen == "Yes" else 0
    )


    # --------------------------------------------------------
    # Create input dataframe
    # --------------------------------------------------------

    input_data = pd.DataFrame([{

        "gender": gender,

        "SeniorCitizen": senior_citizen_value,

        "Partner": Partner,

        "Dependents": Dependents,

        "tenure": tenure,

        "PhoneService": PhoneService,

        "MultipleLines": MultipleLines,

        "InternetService": InternetService,

        "OnlineSecurity": OnlineSecurity,

        "OnlineBackup": OnlineBackup,

        "DeviceProtection": DeviceProtection,

        "TechSupport": TechSupport,

        "StreamingTV": StreamingTV,

        "StreamingMovies": StreamingMovies,

        "Contract": Contract,

        "PaperlessBilling": PaperlessBilling,

        "PaymentMethod": PaymentMethod,

        "MonthlyCharges": MonthlyCharges,

        "TotalCharges": TotalCharges

    }])


    # --------------------------------------------------------
    # Create dataframe using training features
    # --------------------------------------------------------

    input_encoded = pd.DataFrame(
        0,
        index=[0],
        columns=feature_columns
    )


    # --------------------------------------------------------
    # Numeric columns
    # --------------------------------------------------------

    numeric_columns = [
        "SeniorCitizen",
        "tenure",
        "MonthlyCharges",
        "TotalCharges"
    ]


    for column in numeric_columns:

        if column in input_encoded.columns:

            input_encoded.loc[
                0,
                column
            ] = input_data.loc[
                0,
                column
            ]


    # --------------------------------------------------------
    # Categorical columns
    # --------------------------------------------------------

    categorical_columns = [

        "gender",
        "Partner",
        "Dependents",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod"

    ]


    # --------------------------------------------------------
    # One-hot encoding
    # --------------------------------------------------------

    for column in categorical_columns:

        value = input_data.loc[
            0,
            column
        ]

        dummy_column = f"{column}_{value}"

        if dummy_column in input_encoded.columns:

            input_encoded.loc[
                0,
                dummy_column
            ] = 1


    # --------------------------------------------------------
    # Convert to numeric
    # --------------------------------------------------------

    input_encoded = input_encoded.astype(float)


    # --------------------------------------------------------
    # RANDOM FOREST PREDICTION
    # --------------------------------------------------------

    prediction = model.predict(
        input_encoded
    )[0]


    probability = model.predict_proba(
        input_encoded
    )[0][1]


    # --------------------------------------------------------
    # DISPLAY RESULT
    # --------------------------------------------------------

    st.subheader("Prediction Result")


    if prediction == 1:

        st.error(
            "⚠️ Customer is likely to Churn"
        )

        st.write(
            f"**Churn Probability: {probability:.2%}**"
        )

    else:

        st.success(
            "✅ Customer is likely to Stay"
        )

        st.write(
            f"**Stay Probability: {(1 - probability):.2%}**"
        )


    # --------------------------------------------------------
    # Probability bar
    # --------------------------------------------------------

    st.progress(
        int(probability * 100)
    )

    st.caption(
        f"Churn Probability: {probability:.2%}"
    )