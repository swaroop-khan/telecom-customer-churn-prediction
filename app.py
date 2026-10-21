import streamlit as st
import pandas as pd
import joblib

# Load the saved model and preprocessor
model = joblib.load('notebooks/model.pkl')
preprocessor = joblib.load('notebooks/preprocessor.pkl')

st.set_page_config(page_title="Telecom Customer Churn Predictor", page_icon="📊", layout="centered")

st.title("📊 Telecom Customer Churn Prediction App")
st.write("Enter customer details below to predict whether they are likely to churn.")

# Create input form for user features
with st.form("churn_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependent", ["Yes", "No"])
        tenure = st.slider("Tenure (Months)", 0, 72, 1)
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])

    with col2:
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check", 
            "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        monthly_charges = st.number_input("Monthly Charges (₹)", min_value=0.0, max_value=15000.0, value=1500.0)
        total_charges = st.number_input("Total Charges (₹)", min_value=0.0, max_value=200000.0, value=25000.0)

    submitted = st.form_submit_button("Predict Churn")

if submitted:
    # Bundle inputs into a dataframe matching the training schema
    input_data = pd.DataFrame({
        'gender': [gender],
        'SeniorCitizen': [senior_citizen],
        'Partner': [partner],
        'Dependents': [dependents],
        'tenure': [tenure],
        'PhoneService': [phone_service],
        'MultipleLines': [multiple_lines],
        'InternetService': [internet_service],
        'OnlineSecurity': [online_security],
        'OnlineBackup': [online_backup],
        'DeviceProtection': [device_protection],
        'TechSupport': [tech_support],
        'StreamingTV': [streaming_tv],
        'StreamingMovies': [streaming_movies],
        'Contract': [contract],
        'PaperlessBilling': [paperless_billing],
        'PaymentMethod': [payment_method],
        'MonthlyCharges': [monthly_charges],
        'TotalCharges': [total_charges]
    })
    
    # Preprocess input data
    processed_input = preprocessor.transform(input_data)
    
    # Predict
    prediction = model.predict(processed_input)
    prediction_proba = model.predict_proba(processed_input)
    
    churn_probability = prediction_proba[0][1] * 100

    # Display results
    st.subheader("Prediction Result:")
    if prediction[0] == 1:
        st.error(f"⚠️ **This customer is likely to churn!** (Probability: {churn_probability:.2f}%)")
    else:
        st.success(f"✅ **This customer is likely to stay.** (Churn Probability: {churn_probability:.2f}%)")