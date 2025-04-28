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
    
    return 
    

# temp_forecaster.py

# import ee
# import pandas as pd
# import numpy as np
# import joblib
# import tensorflow as tf
# from sklearn.preprocessing import StandardScaler
# from datetime import datetime, timedelta

# # --- 1) GEE Initialization ---
# def initialize_gee():
#     """Initialize Google Earth Engine with service account credentials"""
#     try:
#         credentials = ee.ServiceAccountCredentials('', "ee-ansersohaibstudy-46d7846227ad.json")
#         ee.Initialize(credentials, project="ee-ansersohaibstudy")
#     except Exception as e:
#         raise RuntimeError(f"GEE init failed: {str(e)}")

# # --- 2) Fetch 0.02‑scaled MODIS LST and return a 2‑col DataFrame [date, temperature] ---
# def fetch_recent_lst(latitude, longitude, days):
#     """Fetch MODIS LST data with Open-Meteo-like reliability"""
#     # Convert inputs to scalars immediately
#     lat = float(latitude)
#     lon = float(longitude)
    
#     # Date range calculation (UTC dates only)
#     end_date = datetime.utcnow().date()
#     start_date = end_date - timedelta(days=days)
    
#     # GEE query with explicit error boundaries
#     try:
#         point = ee.Geometry.Point(lon, lat)
#         coll = (ee.ImageCollection('MODIS/061/MOD11A1')
#                 .filterDate(start_date.isoformat(), end_date.isoformat())
#                 .select('LST_Day_1km'))
        
#         # Extract values with null safety
#         def extractor(img):
#             date = ee.Date(img.get('system:time_start')).format('YYYY-MM-dd')
#             lst = img.reduceRegion(ee.Reducer.first(), point, 1000).get('LST_Day_1km')
#             return ee.Feature(None, {'date': date, 'lst': lst})
        
#         features = coll.map(extractor).getInfo().get('features', [])
        
#     except Exception as e:
#         raise RuntimeError(f"GEE query failed: {str(e)}")

#     # Process features with Open-Meteo-like data cleaning
#     clean_data = []
#     for f in features:
#         props = f.get('properties', {})
#         lst = props.get('lst')
        
#         # Type normalization (matches Open-Meteo's simple numbers)
#         if isinstance(lst, (list, np.ndarray)) and len(lst) > 0:
#             lst = float(lst[0])
#         elif isinstance(lst, (int, float)):
#             lst = float(lst)
#         else:
#             continue  # Skip nulls/undefined
            
#         # Convert to Celsius (same formula as Open-Meteo's 2m temp)
#         temp_c = (lst * 0.02) - 273.15
#         clean_data.append({
#             'date': pd.to_datetime(props['date']).date(),
#             'temperature': round(float(temp_c), 2)
#         })
    
#     # Create DataFrame with guaranteed daily coverage
#     df = pd.DataFrame(clean_data)
    
#     # Create full date index (like Open-Meteo's complete series)
#     full_dates = pd.date_range(start=start_date, end=end_date, freq='D')
#     df = df.set_index('date').reindex(full_dates).rename_axis('date')
    
#     # Open-Meteo-style interpolation
#     df['temperature'] = df['temperature'].interpolate(method='time').ffill().bfill()
    
#     return df.reset_index()


# # --- 3) Slice out the 30 days before a given user_date by pure position (no KeyErrors) ---
# def get_past_temperatures(lat, lon, user_date):
#     """Mirror Open-Meteo's API response structure"""
#     # Date window calculation (identical to Open-Meteo version)
#     ud = pd.to_datetime(user_date).date()
#     endd = ud - timedelta(days=1)
#     startd = endd - timedelta(days=35)
    
#     # Fetch data (now returns Open-Meteo-like format)
#     df = fetch_recent_lst(lat, lon, 35)
    
#     # Convert df['date'] to date objects for consistent comparison
#     df['date'] = df['date'].dt.date
    
#     # Filter using pandas datetime operations
#     mask = (df['date'] >= startd) & (df['date'] <= endd)
#     sub = df.loc[mask]
    
#     if len(sub) < 30:
#         raise ValueError(f"Insufficient data: {len(sub)}/30 days")
    
#     # Return numpy arrays like Open-Meteo
#     return (
#         sub['temperature'].values[-30:], 
#         pd.to_datetime(sub['date']).dt.strftime('%Y-%m-%d').values[-30:]
#     )

# # --- 4) Prepare the model input sequence (no boolean array tests) ---
# def prepare_input_sequence(past_temps, past_dates, scaler, lat, lon):
#     past = np.array(past_temps, dtype=float).flatten()
#     if past.shape[0] != len(past_dates):
#         raise ValueError(f"Length mismatch: {past.shape[0]} vs {len(past_dates)}")

#     df = pd.DataFrame({
#         'Date':        pd.to_datetime(past_dates),
#         'Latitude':    float(lat),
#         'Longitude':   float(lon),
#         'temperature': past
#     })

#     # replace zeros, interpolate, drop any remaining NaNs
#     df['temperature'] = df['temperature'].replace(0, np.nan)
#     df['temperature'] = df['temperature'].interpolate(method='linear')
#     df = df.dropna(subset=['temperature'])

#     if df.shape[0] < 30:
#         raise ValueError(f"Need ≥30 days, got {df.shape[0]}")

#     # cyclical features
#     df['year']        = df['Date'].dt.year
#     df['month']       = df['Date'].dt.month
#     df['dayofyear']   = df['Date'].dt.dayofyear
#     df['month_sin']   = np.sin(2 * np.pi * df['month'] / 12)
#     df['month_cos']   = np.cos(2 * np.pi * df['month'] / 12)
#     df['doy_sin']     = np.sin(2 * np.pi * df['dayofyear'] / 365.25)
#     df['doy_cos']     = np.cos(2 * np.pi * df['dayofyear'] / 365.25)

#     features = [
#         'Latitude', 'Longitude', 'year',
#         'month_sin', 'month_cos',
#         'doy_sin', 'doy_cos'
#     ]
#     X = scaler.transform(df[features])
#     return X[-30:]


# # --- 5) Load your LSTM model & scaler ---
# def load_model_and_scaler(model_path, scaler_path):
#     model  = tf.keras.models.load_model(model_path)
#     scaler = joblib.load(scaler_path)
#     return model, scaler


# # --- 6) Roll‐forward prediction (no ambiguous array tests) ---
# def predict_future(model, seq, start_date, end_date, scaler, lat, lon):
#     preds = []
#     curr = seq.copy()
#     d0   = pd.to_datetime(start_date).date()
#     d1   = pd.to_datetime(end_date).date()

#     while d0 <= d1:
#         p = model.predict(curr[np.newaxis, ...], verbose=0)[0,0]
#         preds.append((d0, p))

#         # build next‐day features
#         f = {
#             'Latitude':  float(lat),
#             'Longitude': float(lon),
#             'year':      d0.year,
#             'month_sin': np.sin(2*np.pi*d0.month/12),
#             'month_cos': np.cos(2*np.pi*d0.month/12),
#             'doy_sin':   np.sin(2*np.pi*d0.timetuple().tm_yday/365.25),
#             'doy_cos':   np.cos(2*np.pi*d0.timetuple().tm_yday/365.25)
#         }
#         row = pd.DataFrame([f])
#         nxt = scaler.transform(row)
#         curr = np.vstack([curr[1:], nxt])

#         d0 += timedelta(days=1)
#     return preds


# # --- 7) Example usage ---
# # if __name__ == "__main__":
# #     initialize_gee(SERVICE_ACCOUNT_KEY_PATH, GEE_PROJECT_ID)

# #     # 30‑day look‑back prior to this date
# #     USER_DATE = "2025-04-24"
# #     LAT, LON   = 36.39861, 74.4912

# #     # Fetch & prep
# #     temps, dates = get_past_temperatures(LAT, LON, USER_DATE)
# #     model, scaler = load_model_and_scaler("path/to/model.h5", "path/to/scaler.pkl")
# #     seq = prepare_input_sequence(temps, dates, scaler, LAT, LON)

# #     # Predict next week (example)
# #     start_pred = pd.to_datetime(USER_DATE).date()
# #     end_pred   = start_pred + timedelta(days=7)
# #     forecast   = predict_future(model, seq, start_pred, end_pred, scaler, LAT, LON)

# #     print("Next 7 days forecast:")
# #     for dt, val in forecast:
# #         print(f"  {dt}: {val:.2f} °C")
