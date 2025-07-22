import pickle

import pandas as pd
import streamlit as st
from keras.models import load_model  # type: ignore
from sklearn.pipeline import Pipeline  # type: ignore

st.set_page_config(page_title="Customer Churn Predictor", layout="wide")


# Load Preprocessing Pipeline & Model
@st.cache_resource
def load_pipeline() -> Pipeline:
    with open("models/preprocessing_pipeline.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_churn_model():
    return load_model("models/churn_model.keras")


pipeline = load_pipeline()
model = load_churn_model()

# Streamlit UI Enhancements
st.markdown(
    "<h1 style='text-align: center; color: #2E86C1;'>Customer Churn Prediction App 🚀</h1>",
    unsafe_allow_html=True,
)
st.write("Enter customer details below to predict the likelihood of churn.")

# Layout with columns for a structured UI
with st.form("churn_form"):
    col1, col2 = st.columns(2, gap='medium')

    with col1:
        credit_score = st.number_input(
            "Credit Score",
            min_value=300,
            max_value=850,
            step=1,
            help="Higher score reduces churn risk.",
        )
        geography = st.selectbox(
            "Geography", pipeline.named_steps["preprocessor"].transformers_[1][1].categories_[0]
        )
        gender = st.selectbox(
            "Gender", pipeline.named_steps["preprocessor"].transformers_[1][1].categories_[1]
        )
        age = st.slider(
            "Age", 18, 92, help="Older customers may have different churn behavior."
        )
        tenure = st.slider(
            "Tenure (years)", 0, 10, help="Years customer has been with the company."
        )

    with col2:
        balance = st.number_input(
            "Balance", min_value=0.0, format="%.2f", help="Customer's account balance."
        )
        num_of_products = st.slider(
            "Number of Products", 1, 4, help="More products often mean lower churn."
        )
        has_cr_card = st.radio("Has Credit Card?", [0, 1], help="1 = Yes, 0 = No")
        is_active_member = st.radio("Is Active Member?", [0, 1], help="1 = Yes, 0 = No")
        estimated_salary = st.number_input(
            "Estimated Salary",
            min_value=0.0,
            format="%.2f",
            help="Higher salaries may impact churn.",
        )

    # Submit Button
    submitted = st.form_submit_button("Predict Churn 🔍")

# Process input & prediction
if submitted:
    input_data = pd.DataFrame(
        [
            {
                "CreditScore": credit_score,
                "Geography": geography,
                "Gender": gender,
                "Age": age,
                "Tenure": tenure,
                "Balance": balance,
                "NumOfProducts": num_of_products,
                "HasCrCard": has_cr_card,
                "IsActiveMember": is_active_member,
                "EstimatedSalary": estimated_salary,
            }
        ]
    )

    # Preprocess input
    X_new = pipeline.transform(input_data)

    # Prediction
    if model is None:
        st.error("Churn prediction model could not be loaded. Please check the model file.")
    else:
        pred_proba = float(model.predict(X_new)[0][0])
        prediction = "⚠️ Likely to Churn!" if pred_proba > 0.5 else "✅ Not Likely to Churn!"

        # Display Results
        st.markdown(
            "<h3 style='text-align: center;'>Prediction Result</h3>", unsafe_allow_html=True
        )
        st.markdown(
            f"<h2 style='text-align: center; color: {'red' if pred_proba > 0.5 else 'green'};'>{prediction}</h2>",
            unsafe_allow_html=True,
        )

        # Progress Bar
        st.progress(pred_proba)

        st.write(f"**Churn Probability:** {pred_proba:.2%}")
