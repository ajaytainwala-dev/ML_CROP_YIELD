import streamlit as st
import pandas as pd
import joblib

# Load saved model and label encoder
best_model = joblib.load("models/best_model_RandomForest.pkl")  # or XGBoost
le = joblib.load("models/label_encoder.pkl")

numeric_features = ['N','P','K','temperature','humidity','ph','rainfall']

st.title("🌱 Crop Yield Prediction for Sustainable Farming")
st.write("Enter soil nutrients and weather conditions to get the most suitable crop:")

# Form input
with st.form("crop_form"):
    inputs = {}
    for feature in numeric_features:
        inputs[feature] = st.number_input(f"{feature}", min_value=0.0, step=0.1, value=50.0)
    
    submitted = st.form_submit_button("Predict Crop")

if submitted:
    # Prepare DataFrame
    sample = pd.DataFrame([inputs], columns=numeric_features)
    
    # Predict
    pred_enc = best_model.predict(sample)[0]
    pred_crop = le.inverse_transform([pred_enc])[0]
    
    st.success(f"✅ Recommended Crop: **{pred_crop}**")
    
    # Top 3 probabilities (if model supports predict_proba)
    if hasattr(best_model.named_steps['model'], 'predict_proba'):
        probs = best_model.predict_proba(sample)[0]
        top_idx = probs.argsort()[::-1][:3]
        st.write("### Top 3 crop predictions:")
        for i in top_idx:
            st.write(f"{le.inverse_transform([i])[0]} : {probs[i]:.2f}")
