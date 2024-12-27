import streamlit as st
import pandas as pd
import joblib
import numpy as np

model = joblib.load('artifacts/energy_demand_model.joblib')
scaler = joblib.load('artifacts/scaler.joblib')

st.title('Energy Demand Prediction')

st.sidebar.header('Enter Input Parameters')

features = {
    'generation fossil gas': st.sidebar.number_input('Generation Fossil Gas', value=4000),
    'generation nuclear': st.sidebar.number_input('Generation Nuclear', value=6000),
    'generation wind onshore': st.sidebar.number_input('Generation Wind Onshore', value=7000),
    'generation solar': st.sidebar.number_input('Generation Solar', value=1000),
    'temp': st.sidebar.number_input('Temperature', value=280),
    'pressure': st.sidebar.number_input('Pressure', value=1015),
    'humidity': st.sidebar.number_input('Humidity', value=70),
    'wind_speed': st.sidebar.number_input('Wind Speed', value=4),
    'clouds_all': st.sidebar.number_input('Cloud Coverage', value=20),
    'total_generation': st.sidebar.number_input('Total Generation', value=30000)
}

if st.sidebar.button('Predict Energy Demand'):
    input_data = pd.DataFrame([features])
    
    input_scaled = scaler.transform(input_data)
    
    prediction = model.predict(input_scaled)[0]
    
    st.success(f'Predicted Energy Demand: {prediction:.2f} MW')
    
    st.subheader('Feature Importance')
    importance_df = pd.DataFrame({
        'Feature': list(features.keys()),
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    st.bar_chart(importance_df.set_index('Feature'))