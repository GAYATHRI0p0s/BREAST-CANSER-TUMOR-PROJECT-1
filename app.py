import streamlit as st
import pandas as pd
import joblib

# 1. Load the Model and Scaler
# This is how you "open" the .pkl files
model = joblib.load('best_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("Breast Cancer Diagnostic Assistant")

st.write("""
### Enter clinical features for prediction:
""")

# 2. Define Input Sliders
# Based on the features used in your notebook (radius_mean, texture_mean, etc.)
def get_user_input():
    radius_mean = st.sidebar.slider('Radius Mean', 6.0, 30.0, 14.0)
    texture_mean = st.sidebar.slider('Texture Mean', 9.0, 40.0, 19.0)
    perimeter_mean = st.sidebar.slider('Perimeter Mean', 43.0, 190.0, 92.0)
    area_mean = st.sidebar.slider('Area Mean', 143.0, 2500.0, 650.0)
    smoothness_mean = st.sidebar.slider('Smoothness Mean', 0.05, 0.25, 0.1)
    
    # IMPORTANT: You must include all 30 features in the exact order 
    # used during training in your notebook.
    data = {
        'radius_mean': radius_mean,
        'texture_mean': texture_mean,
        'perimeter_mean': perimeter_mean,
        'area_mean': area_mean,
        'smoothness_mean': smoothness_mean,
        # ... add all other features here
    }
    return pd.DataFrame(data, index=[0])

user_data = get_user_input()

# 3. Preprocess and Predict
if st.button('Predict'):
    # We MUST scale the data using the saved scaler from the notebook
    scaled_data = scaler.transform(user_data)
    
    prediction = model.predict(scaled_data)
    prediction_proba = model.predict_proba(scaled_data)
    
    if prediction[0] == 1:
        st.error("Prediction: Malignant")
    else:
        st.success("Prediction: Benign")
        
    st.write(f"Confidence: {max(prediction_proba[0])*100:.2f}%")