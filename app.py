import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle

# Page config
st.set_page_config(page_title="Customer Churn Predictor", layout="centered")

# Load model and preprocessing objects
@st.cache_resource
def load_assets():
    model = load_model('churn_model.h5')
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('onehot_encoder_geography.pkl', 'rb') as f:
        onehot_encoder_geography = pickle.load(f)
    with open('label_encoder_gender.pkl', 'rb') as f:
        label_encoder_gender = pickle.load(f)
    return model, scaler, onehot_encoder_geography, label_encoder_gender

model, scaler, onehot_encoder_geography, label_encoder_gender = load_assets()

# Title and description
st.title("🔍 Customer Churn Prediction App")
st.markdown("""
This application predicts whether a customer is likely to churn based on their personal and financial information.  
Fill out the details below and click **Predict**.
""")

# Form layout
with st.form("churn_form"):
    st.subheader("📋 Customer Information")
    col1, col2 = st.columns(2)

    with col1:
        geography = st.selectbox('🌍 Geography', onehot_encoder_geography.categories_[0])
        gender = st.selectbox('👤 Gender', label_encoder_gender.classes_)
        age = st.slider('🎂 Age', 18, 92)
        credit_score = st.number_input('💳 Credit Score', min_value=300, max_value=1000, value=650)
        tenure = st.slider('📅 Tenure (Years)', 0, 10)
    with col2:
        balance = st.number_input('💰 Account Balance')
        estimated_salary = st.number_input('💼 Estimated Salary')
        num_of_products = st.slider('📦 Number of Products', 1, 4)
        has_cr_card = st.selectbox('💳 Has Credit Card?', [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        is_active_member = st.selectbox('✅ Active Member?', [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

    # Predict button
    submit = st.form_submit_button("🔮 Predict")

# Run prediction
if submit:
    try:
        # Prepare input
        input_data = pd.DataFrame({
            'Geography': [geography],
            'CreditScore': [credit_score],
            'Gender': [label_encoder_gender.transform([gender])[0]],
            'Age': [age],
            'Tenure': [tenure],
            'Balance': [balance],
            'NumOfProducts': [num_of_products],
            'HasCrCard': [has_cr_card],
            'IsActiveMember': [is_active_member],
            'EstimatedSalary': [estimated_salary]
        })

        # One-hot encode geography
        geo_encoded = onehot_encoder_geography.transform([[geography]]).toarray()
        geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geography.get_feature_names_out(['Geography']))
        input_data = pd.concat([input_data.drop("Geography", axis=1), geo_encoded_df], axis=1)

        # Scale
        input_scaled = scaler.transform(input_data)

        # Predict
        prob = model.predict(input_scaled)[0][0]

        # Display
        st.subheader("📊 Prediction Result")
        if prob > 0.5:
            st.error(f"⚠️ The customer is **likely to churn** with a probability of {prob * 100:.2f}%")
        else:
            st.success(f"✅ The customer is **unlikely to churn** with a probability of {(1 - prob) * 100:.2f}%")

    except Exception as e:
        st.error(f"❌ An error occurred during prediction:\n\n{e}")
