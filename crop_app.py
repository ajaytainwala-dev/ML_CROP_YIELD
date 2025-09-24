import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px

from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class OutlierClipper(BaseEstimator, TransformerMixin):
    """Clip features by lower/upper quantiles per-column (works on numpy arrays)."""
    def __init__(self, lower_quantile=0.01, upper_quantile=0.99):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.lower_bounds_ = None
        self.upper_bounds_ = None
    
    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        # compute per-column quantiles
        self.lower_bounds_ = np.quantile(X, self.lower_quantile, axis=0)
        self.upper_bounds_ = np.quantile(X, self.upper_quantile, axis=0)
        return self
    
    def transform(self, X):
        X = np.asarray(X, dtype=float).copy()
        X = np.clip(X, self.lower_bounds_, self.upper_bounds_)
        return X


# Load saved model and label encoder
best_model = joblib.load("models/best_model_RandomForest.pkl")  # or your model
le = joblib.load("models/label_encoder.pkl")

features = ['N','P','K','temperature','humidity','ph','rainfall']

st.set_page_config(page_title="🌱 Crop Prediction App", layout="wide")
st.title("🌱 Crop Yield Prediction for Sustainable Farming")
st.write("Made for Farmers by Ajay Tainwala")
st.write("Enter soil nutrients and weather conditions to get the most suitable crop:")


# --- Input Form ---
with st.form("crop_input_form", clear_on_submit=False):
    col1, col2, col3 = st.columns(3)
    inputs = {}
    with col1:
        inputs['N'] = st.number_input("Nitrogen (N)", min_value=0.0, max_value=200.0, step=1.0, value=50.0)
        inputs['P'] = st.number_input("Phosphorus (P)", min_value=0.0, max_value=150.0, step=1.0, value=50.0)
        inputs['K'] = st.number_input("Potassium (K)", min_value=0.0, max_value=200.0, step=1.0, value=50.0)
    with col2:
        inputs['temperature'] = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, step=0.5, value=25.0)
        inputs['humidity'] = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, step=1.0, value=50.0)
        inputs['ph'] = st.number_input("Soil pH", min_value=0.0, max_value=14.0, step=0.1, value=6.5)
    with col3:
        inputs['rainfall'] = st.number_input("Rainfall (mm)", min_value=0.0, max_value=5000.0, step=10.0, value=200.0)
    
    submitted = st.form_submit_button("Predict Crop")

# --- Prediction & Top 3 ---
if submitted:
    input_df = pd.DataFrame([inputs], columns=features)
    
    # Predict
    pred_enc = best_model.predict(input_df)[0]
    pred_crop = le.inverse_transform([pred_enc])[0]

    # Predict probabilities for top 3
    if hasattr(best_model.named_steps['model'], 'predict_proba'):
        probs = best_model.predict_proba(input_df)[0]
        top_idx = probs.argsort()[::-1][:3]
        top_crops = [(le.inverse_transform([i])[0], probs[i]) for i in top_idx]
    else:
        top_crops = [(pred_crop, 1.0)]
    
    # Display results
    st.success(f"✅ **Recommended Crop:** {pred_crop.capitalize()}")
    st.subheader("🌿 Top 3 Crop Recommendations")
    for crop, prob in top_crops:
        st.write(f"{crop.capitalize()} - Probability: {prob:.2f}")

    # --- Browser TTS ---
    tts_text = f"The recommended crop is {pred_crop}"
    tts_html = f"""
    <audio autoplay>
        <source src="https://translate.google.com/translate_tts?ie=UTF-8&q={tts_text}&tl=en&client=tw-ob" type="audio/mpeg">
    </audio>
    """
    st.components.v1.html(tts_html)

# --- Feature Importance Chart ---
st.subheader("📊 Feature Importance")
model = best_model.named_steps['model']
importances = model.feature_importances_
feat_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance', ascending=True)

fig = px.bar(feat_df, x='Importance', y='Feature', orientation='h', color='Importance',
             color_continuous_scale='Viridis', title='Feature Importance for Crop Prediction')
st.plotly_chart(fig, use_container_width=True)
