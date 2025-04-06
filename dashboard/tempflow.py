# tempflow.py
import requests
import pandas as pd
import numpy as np
import joblib
from datetime import timedelta
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

def load_model_and_scaler(model_path, scaler_path):
    """Loads the pre-trained LSTM model and associated scaler"""
    try:
        model = tf.keras.models.load_model(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    except Exception as e:
        raise RuntimeError(f"Model loading failed: {str(e)}")

def get_past_temperatures(lat, lon, user_date):
    """Fetches available historical temperature data and handles missing future data."""
    try:
        end_date = min(pd.to_datetime(user_date) - timedelta(days=1), pd.to_datetime("today") - timedelta(days=1))
        start_date = end_date - timedelta(days=30)  # Ensure at least 30 past days

        url = (f"https://archive-api.open-meteo.com/v1/archive?"
               f"latitude={lat}&longitude={lon}&start_date={start_date.date()}&end_date={end_date.date()}&"
               f"daily=temperature_2m_max&timezone=auto")
        
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        if 'daily' not in data:
            raise ValueError("Invalid API response format - missing 'daily' key")
        
        return data['daily']['temperature_2m_max'], data['daily']['time']
    
    except Exception as e:
        raise RuntimeError(f"API request failed: {str(e)}")



def prepare_input_sequence(past_temperatures, dates, scaler, lat, lon):
    """Prepares input sequence matching Kaggle implementation"""
    df = pd.DataFrame({
        'Date': pd.to_datetime(dates),
        'Latitude': lat,
        'Longitude': lon,
        'temperature': past_temperatures
    })
    
    # Handle missing values
    df['temperature'] = df['temperature'].replace(0, np.nan)
    df['temperature'] = df['temperature'].interpolate(method='linear')
    df = df.dropna(subset=['temperature'])
    
    # Feature engineering
    df['year'] = df['Date'].dt.year
    df['month'] = df['Date'].dt.month
    df['dayofyear'] = df['Date'].dt.dayofyear
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['dayofyear_sin'] = np.sin(2 * np.pi * df['dayofyear'] / 365.25)
    df['dayofyear_cos'] = np.cos(2 * np.pi * df['dayofyear'] / 365.25)
    
    # Ensure minimum sequence length
    if len(df) < 30:
        raise ValueError(f"Need at least 30 days of data, got {len(df)}")
    
    # Prepare features
    feature_columns = ['Latitude', 'Longitude', 'year', 
                      'month_sin', 'month_cos', 
                      'dayofyear_sin', 'dayofyear_cos']
    
    # Scale features
    try:
        scaled_features = scaler.transform(df[feature_columns])
    except ValueError as e:
        raise RuntimeError(f"Feature scaling failed: {str(e)}")
    
    return scaled_features[-30:]

def predict_future(model, sequence, start_date, end_date, scaler, lat, lon):
    """Prediction function matching Kaggle implementation"""
    predictions = []
    current_seq = sequence.copy()
    current_date = start_date

    while current_date <= end_date:
        # Predict next day's temperature
        pred = model.predict(current_seq[np.newaxis, :, :], verbose=0)[0][0]
        predictions.append((current_date, pred))
        
        # Prepare new features
        new_features = {
            'Latitude': lat,
            'Longitude': lon,
            'year': current_date.year,
            'month_sin': np.sin(2 * np.pi * current_date.month / 12),
            'month_cos': np.cos(2 * np.pi * current_date.month / 12),
            'dayofyear_sin': np.sin(2 * np.pi * current_date.timetuple().tm_yday / 365.25),
            'dayofyear_cos': np.cos(2 * np.pi * current_date.timetuple().tm_yday / 365.25)
        }
        
        # Update sequence
        new_row = pd.DataFrame([new_features])
        scaled_row = scaler.transform(new_row)
        current_seq = np.vstack([current_seq[1:], scaled_row])
        
        current_date += timedelta(days=1)
    
    return predictions
