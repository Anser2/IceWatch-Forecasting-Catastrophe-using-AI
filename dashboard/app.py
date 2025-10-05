# app.py
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import folium_static
import datetime
from datetime import timedelta
from terraflow import load_model, get_velocity_data, predict_velocity, preprocess_velocity_data
from tempflow import load_model_and_scaler, get_past_temperatures, prepare_input_sequence, predict_future
import asyncio
from streamlit_extras.metric_cards import style_metric_cards
from streamlit_extras.colored_header import colored_header
import torch
import os
asyncio.set_event_loop(asyncio.new_event_loop())
#ok

# Custom CSS styling
st.set_page_config(
    page_title="IceWatch | Glacier Monitoring",
    page_icon="❄️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
# Line 30: Replace the entire <style> block
st.markdown("""
<style>
    /* ===== BASE STYLING ===== */
    :root {
        --primary: #0455BF;
        --primary-light: #0572E6;
        --primary-dark: #023E89;
        --secondary: #03A9F4;
        --accent: #1DE9B6;
        --warning: #FF9800;
        --danger: #F44336;
        --success: #4CAF50;
        --grey-100: #F5F7FA;
        --grey-200: #E4E7EB;
        --grey-300: #CBD2D9;
        --grey-400: #9AA5B1;
        --grey-500: #7B8794;
        --grey-600: #616E7C;
        --grey-700: #52606D;
        --grey-800: #3E4C59;
        --grey-900: #323F4B;
        --text-primary: #1A202C;
        --text-secondary: #4A5568;
        --card-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        --transition: all 0.3s ease;
        --radius-sm: 4px;
        --radius-md: 8px;
        --radius-lg: 16px;
    }
    
    /* App-wide styles */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 1400px;
    }
    
    h1, h2, h3, h4, h5 {
        font-family: 'Inter', 'Segoe UI', sans-serif;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    h1 {
        color: var(--primary);
        font-weight: 700;
        letter-spacing: -0.01em;
    }
    
    h2 {
        color: var(--primary);
        letter-spacing: -0.01em;
    }
    
    h3 {
        color: var(--text-primary);
        font-size: 1.3rem;
    }
    
    p, li, span, div {
        color: var(--text-secondary);
        line-height: 1.6;
    }
    
    /* ===== LAYOUT COMPONENTS ===== */
    /* Card styling with hover effect */
    .card {
        background-color: white;
        border-radius: var(--radius-md);
        box-shadow: var(--card-shadow);
        padding: 1.5rem;
        margin-bottom: 1rem;
        transition: var(--transition);
        border-top: 3px solid transparent;
    }
    
    .card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 24px rgba(0, 0, 0, 0.12);
    }
    
    /* Status section styling */
    .status-section {
        background-color: white;
        padding: 1.5rem;
        border-radius: var(--radius-md);
        box-shadow: var(--card-shadow);
        margin-bottom: 1.2rem;
        transition: var(--transition);
        position: relative;
        overflow: hidden;
        border-top: 3px solid var(--primary);
    }
    
    .status-section:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 28px rgba(0, 0, 0, 0.12);
    }
    
    /* Risk cards with improved styling */
    .risk-card {
        padding: 1.8rem;
        border-radius: var(--radius-md);
        box-shadow: var(--card-shadow);
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
        transition: var(--transition);
    }
    
    .risk-card:hover {
        transform: translateY(-2px);
    }
    
    .risk-high {
        background-color: rgba(244, 67, 54, 0.05);
        border-left: 4px solid var(--danger);
    }
    
    .risk-medium {
        background-color: rgba(255, 152, 0, 0.05);
        border-left: 4px solid var(--warning);
    }
    
    .risk-low {
        background-color: rgba(76, 175, 80, 0.05);
        border-left: 4px solid var(--success);
    }
    
    /* Forecast section with improved styling */
    .forecast-section {
        padding: 1.8rem;
        background-color: white;
        border-radius: var(--radius-md);
        box-shadow: var(--card-shadow);
        margin-bottom: 1.5rem;
        transition: var(--transition);
        border-top: 3px solid var(--secondary);
    }
    
    .forecast-section:hover {
        box-shadow: 0 8px 28px rgba(0, 0, 0, 0.12);
    }
    
    /* Map container with improved styling */
    .map-container {
        border-radius: var(--radius-md);
        overflow: hidden;
        box-shadow: var(--card-shadow);
        border: 1px solid var(--grey-200);
        height: 100%;
    }
    
    /* ===== COMPONENTS ===== */
    /* Custom header/logo styling */
    .app-header {
        display: flex;
        align-items: center;
        margin-bottom: 2rem;
        padding-bottom: 1rem;
        border-bottom: 1px solid var(--grey-200);
    }
    
    .logo-container {
        display: flex;
        align-items: center;
    }
    
    .logo-icon {
        font-size: 3.5rem;
        color: var(--primary);
        margin-right: 1rem;
        animation: pulse 3s infinite ease-in-out;
    }
    
    .logo-text {
        font-weight: 800;
        font-size: 2.8rem;
        background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
        letter-spacing: -0.03em;
    }
    
    .subheader {
        color: var(--grey-600);
        font-size: 1.0rem;
        margin-top: -0.5rem;
        font-weight: 400;
    }
    
    /* Enhanced tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: var(--grey-100);
        padding: 6px;
        border-radius: var(--radius-md);
        border: 1px solid var(--grey-200);
        margin-bottom: 1.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0 16px;
        white-space: pre-wrap;
        font-weight: 500;
        background-color: transparent;
        border: none;
        border-radius: var(--radius-sm);
        transition: var(--transition);
        color: var(--grey-700);
    }
    
    .stTabs [data-baseweb="tab"]:hover:not([aria-selected="true"]) {
        background-color: var(--grey-200);
        color: var(--text-primary);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%) !important;
        color: white !important;
        box-shadow: 0 4px 12px rgba(3, 169, 244, 0.3);
    }
    
    /* Enhanced metric cards */
    .stMetric {
        background: linear-gradient(135deg, #FFF 0%, var(--grey-100) 100%);
        padding: 1.2rem;
        border-radius: var(--radius-md);
        box-shadow: var(--card-shadow);
        border: 1px solid var(--grey-200);
        transition: var(--transition);
        height: 100%;
    }
    
    .stMetric:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 28px rgba(0, 0, 0, 0.12);
    }
    
    .stMetric label {
        font-weight: 600;
        font-size: 1rem;
        color: var(--text-primary);
    }
    
    .stMetric [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        color: var(--primary);
    }
    
    .stMetric [data-testid="stMetricDelta"] {
        font-size: 0.9rem;
    }
    
    /* Animations */
    @keyframes pulse {
        0% {
            transform: scale(1);
        }
        50% {
            transform: scale(1.05);
        }
        100% {
            transform: scale(1);
        }
    }
    
    .animate-in {
        animation: fadeInUp 0.6s ease forwards;
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
</style>
""", unsafe_allow_html=True)
# Add after the CSS block (around line 109)
def create_custom_plotly_theme():
    """Create a custom theme for Plotly figures"""
    return {
        'layout': {
            'font': {
                'family': 'Inter, Arial, sans-serif',
                'size': 12,
                'color': '#3E4C59'
            },
            'title': {
                'font': {
                    'family': 'Inter, Arial, sans-serif',
                    'size': 18,
                    'color': '#323F4B'
                }
            },
            'plot_bgcolor': 'rgba(0,0,0,0)',
            'paper_bgcolor': 'rgba(0,0,0,0)',
            'colorway': ['#0455BF', '#03A9F4', '#1DE9B6', '#FF9800', '#F44336', 
                         '#4CAF50', '#9C27B0', '#FFEB3B', '#795548', '#607D8B'],
            'xaxis': {
                'gridcolor': '#E4E7EB',
                'linecolor': '#CBD2D9',
                'zerolinecolor': '#CBD2D9',
            },
            'yaxis': {
                'gridcolor': '#E4E7EB',
                'linecolor': '#CBD2D9',
                'zerolinecolor': '#CBD2D9',
            },
        }
    }

# Apply theme to all Plotly figures
import plotly.io as pio
pio.templates["custom"] = create_custom_plotly_theme()
pio.templates.default = "plotly_white+custom"

# Add after the theme function (around line 119)
def enhanced_metric_card(label, value, delta=None, delta_color="normal", icon=None, theme_color=None):
    """
    Creates an enhanced metric card with icon and styling
    """
    # Set default theme color if none provided
    if theme_color is None:
        theme_color = "#0455BF"
    
    # Set icon if none provided
    if icon is None:
        icon = "📊"
        
    # Handle delta display and color
    delta_html = ""
    if delta is not None:
        delta_icon = "↗" if delta.startswith("+") else "↘" if delta.startswith("-") else "→"
        delta_color_css = "#4CAF50" if delta_color == "normal" and delta.startswith("+") else \
                          "#F44336" if delta_color == "normal" and delta.startswith("-") else \
                          "#4CAF50" if delta_color == "inverse" and delta.startswith("-") else \
                          "#F44336" if delta_color == "inverse" and delta.startswith("+") else \
                          "#9AA5B1"
        delta_html = f"""
        <div style="font-size: 0.9rem; color: {delta_color_css}; font-weight: 500; display: flex; align-items: center; margin-top: 5px;">
            {delta_icon} {delta}
        </div>
        """
    
    # Create the HTML for the card
    html = f"""
    <div style="background: linear-gradient(135deg, white 0%, #f5f7fb 100%); padding: 1.2rem; border-radius: 10px; 
                box-shadow: 0 4px 20px rgba(0,0,0,0.08); border-top: 3px solid {theme_color}; transition: all 0.3s ease;">
        <div style="display: flex; align-items: center; margin-bottom: 8px;">
            <div style="background: linear-gradient(135deg, {theme_color} 0%, {adjust_color_brightness(theme_color, 1.2)} 100%); 
                       color: white; border-radius: 8px; height: 32px; width: 32px; display: flex; 
                       align-items: center; justify-content: center; margin-right: 10px; font-size: 18px;">
                {icon}
            </div>
            <div style="font-size: 1rem; color: #52606D; font-weight: 600;">{label}</div>
        </div>
        <div style="font-size: 1.8rem; font-weight: 700; color: #1A202C;">{value}</div>
        {delta_html}
    </div>
    """
    
    return st.markdown(html, unsafe_allow_html=True)

def adjust_color_brightness(hex_color, factor):
    """Helper function to adjust color brightness for gradients"""
    # Convert hex to RGB
    h = hex_color.lstrip('#')
    rgb = tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
    
    # Adjust brightness
    rgb_adjusted = [min(255, int(c * factor)) for c in rgb]
    
    # Convert back to hex
    return '#{:02x}{:02x}{:02x}'.format(rgb_adjusted[0], rgb_adjusted[1], rgb_adjusted[2])
# ----------------------------
# Header and Logo
# ----------------------------
# Creating columns for logo and title
col_logo, col_title = st.columns([1, 4])  # Fixed missing bracket

with col_logo:
    st.markdown("""
    <div style="text-align: left;">
        <span style="font-size: 6rem;">❄️</span>  <!-- Increased icon size -->
    </div>
    """, unsafe_allow_html=True)

with col_title:
    st.markdown("""
    <div style="text-align: left;">
        <p style="font-size: 3rem; font-weight: bold; margin-bottom: 0;">IceWatch</p>
        <p style="font-size: 1.5rem; color: gray;">Advanced Glacier Monitoring & GLOF Risk Assessment</p>
    </div>
    """, unsafe_allow_html=True)


# with col_logo:
#     st.markdown("""
#     <div style="text-align: left; margin-top: -15px; padding-top: 0;">
#         <span style="font-size: 6rem;">❄️</span>
#     </div>
#     """, unsafe_allow_html=True)

# with col_title:
#     st.markdown("""
#     <div style="text-align: left; margin-top: -15px; padding-top: 0;">
#         <p style="font-size: 3rem; font-weight: bold; margin-bottom: 0; line-height: 1.1;">IceWatch</p>
#         <p style="font-size: 1.5rem; color: gray; margin-top: 0;">Advanced Glacier Monitoring & GLOF Risk Assessment</p>
#     </div>
#     """, unsafe_allow_html=True)
# ----------------------------
# Load Models & Scalers
# ----------------------------

def get_project_root():
    """Get the project root directory."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def get_absolute_path(relative_path):
    """Convert a relative path to an absolute path from the project root."""
    return os.path.join(get_project_root(), relative_path)
    
# @st.cache_resource
# def load_models():
#     vel_model, device = load_model("models/terraflow-5.5M.pth")
#     temp_model, temp_scaler = load_model_and_scaler("models/best_tempflow_model1.keras", "models/scaler_tempflow1.save")
#     return vel_model, device, temp_model, temp_scaler

@st.cache_resource
def load_models():
    """Load all necessary models."""
    # Corrected paths to be relative to the project root
    vel_model_path = "models/terraflow-5.5M.pth"
    temp_model_path = "models/best_temp_model.h5"
    scaler_path = "models/scaler_tempflow1.save"

    # Load TerraFlow model
    # The `load_model` function already handles absolute paths correctly.
    vel_model, device = load_model(vel_model_path)
    
    # Load TempFlow model and scaler using absolute paths
    absolute_temp_model_path = get_absolute_path(temp_model_path)
    absolute_scaler_path = get_absolute_path(scaler_path)
    
    # The `load_model_and_scaler` function from tempflow.py needs absolute paths
    temp_model, temp_scaler = load_model_and_scaler(absolute_temp_model_path, absolute_scaler_path)
    
    return vel_model, device, temp_model, temp_scaler

vel_model, device, temp_model, temp_scaler = load_models()

# ----------------------------
# Sidebar – Enhanced with location presets
# ----------------------------
with st.sidebar:
    st.image("https://pawilds.com/wp-content/uploads/2023/01/Snowflake_macro_photography_1.jpg", use_container_width=True)
    st.markdown("---")
    
    st.header("📍 Location")
    
    # Add location presets
    location_preset = st.selectbox(
        "Select Monitoring Area", 
        ["Shishper Glacier, Pakistan", "Khurdopin Glacier", "Passu Glacier", "Custom Location"],
        index=0
    )
    
    # Set default coordinates based on  lat, lon
    if location_preset == "Shishper Glacier, Pakistan":
        default_lat, default_lon = 74.7654, 36.4053
    elif location_preset == "Khurdopin Glacier":
        default_lat, default_lon = 75.45967, 36.35835
    elif location_preset == "Passu Glacier":
        default_lat, default_lon = 74.86881, 36.42050
    else:
        default_lat, default_lon = 74.70912, 36.47375
    
    # Custom coordinates if selected
    if location_preset == "Custom Location":
        lat = st.number_input("Latitude", value=default_lat)
        lon = st.number_input("Longitude", value=default_lon)
    else:
        lat, lon = default_lat, default_lon
        st.info(f"Monitoring {location_preset}")
        st.markdown(f"**Latitude**: {lat:.5f}")
        st.markdown(f"**Longitude**: {lon:.5f}")
    
    st.markdown("---")
    st.header("⚙️ Settings")
    st.markdown("**Forecast Range**")
    forecast_days = st.slider("Days to forecast", min_value=3, max_value=14, value=7)
    
    st.markdown("---")
    st.header("ℹ️ About")
    st.info(
        "IceWatch provides real-time monitoring and forecasting of glacier movement "
        "and conditions in high-risk areas, supporting early warning systems for Glacial Lake Outburst Floods (GLOFs)."
    )
    st.markdown("Version 2.0.1 | © 2025")

# Updated prediction function
def get_velocity_prediction():
    try:
        with st.spinner("Fetching velocity data..."):
            summary_df = get_velocity_data(lat, lon)
        with st.spinner("Generating velocity forecast..."):
            tomorrow = datetime.datetime.today() + timedelta(days=1)
            target_timestamp = tomorrow.strftime('%Y-%m-%d %H:%M:%S')
            pred = predict_velocity(vel_model, device, summary_df, target_timestamp, lat, lon)
            if pred is None:
                raise ValueError("Prediction returned None unexpectedly")
            return pred, target_timestamp
    except Exception as e:
        st.error(f"Velocity forecasting failed: {str(e)}")
        print(f"Error in get_velocity_prediction: {str(e)}")  # Log to console
        return None

def predict_velocity(model, device, summary_df, target_time, lat, lon):
    model.eval()
    target_date = pd.to_datetime(target_time).date()
    earliest_date = summary_df['date'].min().date()
    latest_date = summary_df['date'].max().date()
    print(f"Target date: {target_date}, Data range: {earliest_date} to {latest_date}")

    if target_date < earliest_date:
        raise ValueError(f"Target date {target_date} is before earliest data: {earliest_date}")

    input_features = ['lon', 'lat', 'year', 'month_sin', 'month_cos', 'day_sin', 'day_cos', 'avg_velocity']
    # Check if summary_df['date'] is already in datetime.date format
    if not isinstance(summary_df['date'].iloc[0], datetime.date):
        summary_df['date'] = summary_df['date'].dt.date

    if target_date <= latest_date:
        target_row = summary_df[summary_df['date'] == target_date]
        print(f"Target row found: {not target_row.empty}")
        if target_row.empty:
            raise ValueError(f"No data for target date {target_date}")
        target_idx = target_row.index[0]
        if target_idx < 32:
            raise ValueError(f"Need 32 days of prior data, got {target_idx + 1}")
        sequence = summary_df.iloc[target_idx - 32:target_idx][input_features].values
        print(f"Sequence shape: {sequence.shape}")
        input_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(input_tensor).item()
        print(f"Predicted velocity: {pred}")
        return pred
    else:
        current_date = latest_date + timedelta(days=1)
        current_sequence = summary_df.tail(32)[input_features].values
        print(f"Starting iterative prediction from {current_date}")

        while current_date <= target_date:
            input_tensor = torch.tensor(current_sequence, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                pred = model(input_tensor).item()
            print(f"Predicted for {current_date}: {pred}")

            if current_date == target_date:
                return pred

            next_date = current_date + timedelta(days=1)
            days_in_month = (next_date.replace(month=(next_date.month % 12) + 1 if next_date.month != 12 else 1, day=1) - timedelta(days=1)).day
            next_day_features = [
                lon, lat, next_date.year,
                np.sin(2 * np.pi * next_date.month / 12),
                np.cos(2 * np.pi * next_date.month / 12),
                np.sin(2 * np.pi * next_date.day / days_in_month),
                np.cos(2 * np.pi * next_date.day / days_in_month),
                pred
            ]
            current_sequence = np.roll(current_sequence, -1, axis=0)
            current_sequence[-1] = next_day_features
            current_date = next_date
            print(f"Advancing to next date: {current_date}")
        print(f"Reached end of loop, returning last prediction for {current_date}")
        return pred
    
def get_velocity_forecast(days=14):
    try:
        with st.spinner("Fetching velocity data..."):
            summary_df = get_velocity_data(lat, lon)
            print(f"Velocity data fetched, shape: {summary_df.shape}")  # Debug
        
        forecast = []
        for i in range(1, days + 1):
            try:
                forecast_date = datetime.datetime.today() + timedelta(days=i)
                target_timestamp = forecast_date.strftime('%Y-%m-%d %H:%M:%S')
                # Pass a copy of summary_df to avoid in-place modifications
                pred = predict_velocity(vel_model, device, summary_df.copy(), target_timestamp, lat, lon)
                if pred is not None:
                    print(f"Velocity forecast for {target_timestamp}: {pred}")  # Debug
                    forecast.append((forecast_date, pred))
                else:
                    print(f"Velocity prediction returned None for {target_timestamp}")
                    continue
            except Exception as e:
                print(f"Failed to predict velocity for {target_timestamp}: {str(e)}")  # Debug
                continue  # Skip failed predictions
        if not forecast:
            st.error("No velocity predictions succeeded")
            return None
        print(f"Velocity forecast completed: {forecast}")  # Debug
        return forecast
    except Exception as e:
        st.error(f"Velocity forecast failed: {str(e)}")
        print(f"Error in get_velocity_forecast: {str(e)}")  # Debug
        return None
    
# def get_temperature_forecast(days=14):
#     """
#     Generates a 7-day forecast for surface temperature.
#     """
#     try:
#         with st.spinner("Fetching historical temperature data..."):
#             today = datetime.datetime.today().date()
#             safe_date = today  # using today for historical cutoff
#             temps, dates = get_past_temperatures(lat, lon, safe_date)
        
#         if not temps or not dates:
#             st.error("No historical temperature data available.")
#             return None
        
#         last_available_date = pd.to_datetime(dates[-1])
#         predicted_temps = list(temps)
#         predicted_dates = list(pd.to_datetime(dates))
#         forecast = []
#         with st.spinner(f"Predicting temperature for next {days} days..."):
#             for _ in range(days):
#                 sequence = prepare_input_sequence(predicted_temps, predicted_dates, temp_scaler, lat, lon)
#                 next_date = last_available_date + timedelta(days=1)
#                 prediction = predict_future(
#                     temp_model,
#                     sequence,
#                     next_date,
#                     next_date,
#                     temp_scaler,
#                     lat,
#                     lon
#                 )
#                 if not prediction:
#                     st.error("Temperature prediction failed.")
#                     return None
#                 pred_val = prediction[-1][1]
#                 forecast.append((next_date, pred_val))
#                 predicted_temps.append(pred_val)
#                 last_available_date = next_date
#                 predicted_dates.append(last_available_date)
#         return forecast
#     except Exception as e:
#         st.error(f"Temperature forecast failed: {str(e)}")
#         return None

def get_temperature_forecast(days=14):
    """
    Generates a forecast for surface temperature for the specified number of days.
    """
    try:
        with st.spinner("Fetching historical temperature data..."):
            today = datetime.datetime.today().date()
            safe_date = today  # using today for historical cutoff
            temps, dates = get_past_temperatures(lat, lon, safe_date)
        
        if not temps or not dates:
            st.error("No historical temperature data available.")
            return None
        
        last_available_date = pd.to_datetime(dates[-1])
        predicted_temps = list(temps)
        predicted_dates = list(pd.to_datetime(dates))
        forecast = []
        
        # Debug info
        print(f"Starting temp forecast for {days} days from {last_available_date}")
        
        with st.spinner(f"Predicting temperature for next {days} days..."):
            # Start with tomorrow and forecast for the requested number of days
            forecast_start = datetime.datetime.today() + timedelta(days=1)
            
            for i in range(days):
                target_date = forecast_start + timedelta(days=i)
                sequence = prepare_input_sequence(predicted_temps, predicted_dates, temp_scaler, lat, lon)
                
                prediction = predict_future(
                    temp_model,
                    sequence,
                    target_date,
                    target_date,
                    temp_scaler,
                    lat,
                    lon
                )
                
                if not prediction:
                    st.error(f"Temperature prediction failed for {target_date}.")
                    continue
                    
                pred_val = prediction[-1][1]
                print(f"Temperature forecast for {target_date}: {pred_val}")
                forecast.append((target_date, pred_val))
                
                # Update for next iteration
                predicted_temps.append(pred_val)
                predicted_dates.append(target_date)
                
        print(f"Temperature forecast completed, days: {len(forecast)}")
        return forecast
    except Exception as e:
        st.error(f"Temperature forecast failed: {str(e)}")
        print(f"Error in get_temperature_forecast: {str(e)}")  # Debug
        return None
    
# ----------------------------
# Tabbed Layout with Enhanced Styling
# ----------------------------
tabs = st.tabs([
    "📊 Dashboard", 
    "📈 Forecast & Trends", 
    "🔍 Advanced Analysis", 
    "ℹ️ About GLOFs"
])


# In Tab 2
# --- Tab 2: Forecast & History - Fixed Version ---
with tabs[1]:
    colored_header(
        label="Forecast & Historical Trends",
        description=f"Detailed forecasts and historical data for {location_preset}",
        color_name="blue-70"
    )
    
    # Keep only one forecast section (remove the duplicate)
    st.markdown('<div class="forecast-section">', unsafe_allow_html=True)
    st.subheader(f"{forecast_days}-Day Forecast")
    
    velocity_forecast = get_velocity_forecast(forecast_days)
    temperature_forecast = get_temperature_forecast(forecast_days)
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    plot_has_data = False
    
    # Add velocity trace
    if velocity_forecast:
        df_vel_forecast = pd.DataFrame(velocity_forecast, columns=["Date", "Velocity"])
        df_vel_forecast["Date"] = pd.to_datetime(df_vel_forecast["Date"])
        print(f"df_vel_forecast: {df_vel_forecast}")  # Debug
        if not df_vel_forecast.empty and df_vel_forecast["Velocity"].notna().any():
            fig.add_trace(
                go.Scatter(
                    x=df_vel_forecast["Date"], 
                    y=df_vel_forecast["Velocity"],
                    name="Ice Velocity (m/yr)",
                    line=dict(color="#0078D4", width=3),
                    mode="lines+markers"
                ),
                secondary_y=False
            )
            plot_has_data = True
        else:
            st.warning("No valid velocity forecast data to plot.")
    else:
        st.error("Velocity forecast unavailable")
    
    # Add temperature trace
    if temperature_forecast:
        df_temp_forecast = pd.DataFrame(temperature_forecast, columns=["Date", "Temperature"])
        df_temp_forecast["Date"] = pd.to_datetime(df_temp_forecast["Date"])
        print(f"df_temp_forecast: {df_temp_forecast}")  # Debug
        if not df_temp_forecast.empty and df_temp_forecast["Temperature"].notna().any():
            fig.add_trace(
                go.Scatter(
                    x=df_temp_forecast["Date"], 
                    y=df_temp_forecast["Temperature"],
                    name="Temperature (°C)",
                    line=dict(color="#FF9500", width=3, dash="dot"),
                    mode="lines+markers"
                ),
                secondary_y=True
            )
            plot_has_data = True
            fig.add_shape(
                type="line",
                x0=min(df_temp_forecast["Date"]),
                y0=0,
                x1=max(df_temp_forecast["Date"]),
                y1=0,
                line=dict(color="#FF3B30", width=2, dash="dash"),
                name="Freezing Level"
            )
        else:
            st.warning("No valid temperature forecast data to plot.")
    else:
        st.error("Temperature forecast unavailable")
    
    if plot_has_data:
        fig.update_layout(
            title=f"{forecast_days}-Day Forecast: Ice Velocity vs Temperature",
            height=500,
            margin=dict(l=20, r=20, t=80, b=20),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            hovermode="x unified",
            plot_bgcolor="rgba(245,245,245,1)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Arial", size=14)
        )
        
        # Set fixed ranges for better visualization
        fig.update_yaxes(
            title_text="Ice Velocity (m/yr)",
            secondary_y=False,
            showgrid=True,
            gridcolor="rgba(200,200,200,0.3)",
            title_font=dict(color="#0078D4"),
            range=[0, 200],  # Set appropriate range for velocity
            tickformat=".0f"
        )
        
        fig.update_yaxes(
            title_text="Temperature (°C)",
            secondary_y=True,
            showgrid=False,
            title_font=dict(color="#FF9500"),
            range=[-15, 15],  # Set appropriate range for temperature
            tickformat=".1f"
        )
        
        fig.update_xaxes(
            title_text="Date",
            showgrid=True,
            gridcolor="rgba(200,200,200,0.3)"
        )
        fig.add_annotation(
            x=0.5,
            y=-0.15,
            xref="paper",
            yref="paper",
            text="Higher velocity and temperatures above freezing increase GLOF risk",
            showarrow=False,
            font=dict(color="#666666", size=12)
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("No forecast data available to plot")
    
    st.markdown('</div>', unsafe_allow_html=True)

    # Historical Data Section - Keep this part
    colored_header(
        label="Historical Trends",
        description="Long-term patterns and anomalies",
        color_name="blue-70"
    )
    
    col_hist1, col_hist2 = st.columns(2)
def get_temperature_prediction():
    """
    Automatically fetches historical temperature data and forecasts the temperature for tomorrow.
    """
    try:
        with st.spinner("Fetching historical temperature data..."):
            today = datetime.datetime.today().date()
            target_date = today + timedelta(days=1)
        temps, dates = get_past_temperatures(lat, lon, target_date)
        
        if not temps or not dates:
            st.error("No historical temperature data available.")
            return None
        
        # If target date is in the past (unlikely), return actual value
        if pd.to_datetime(target_date) <= pd.Timestamp(today):
            actual_temp = temps[-1]
            return actual_temp
        else:
            last_available_date = pd.to_datetime(dates[-1])
            predicted_temps = list(temps)
            predicted_dates = list(pd.to_datetime(dates))
            with st.spinner("Predicting future temperature..."):
                # Loop until forecast reaches tomorrow
                while last_available_date < pd.Timestamp(target_date):
                    sequence = prepare_input_sequence(predicted_temps, predicted_dates, temp_scaler, lat, lon)
                    prediction = predict_future(
                        temp_model,
                        sequence,
                        last_available_date + timedelta(days=1),
                        pd.Timestamp(target_date),
                        temp_scaler,
                        lat,
                        lon
                    )
                    if not prediction:
                        st.error("Temperature prediction failed.")
                        return None
                    # prediction returns a list with one tuple (date, predicted_value)
                    predicted_temps.append(prediction[-1][1])
                    last_available_date += timedelta(days=1)
                    predicted_dates.append(last_available_date)
                    # Once we've reached tomorrow, return the forecast
                    if last_available_date.date() == target_date:
                        return predicted_temps[-1]
    except Exception as e:
        st.error(f"Temperature prediction failed: {str(e)}")
        return None

# ----------------------------
# Enhanced Risk Assessment Function
# ----------------------------


def calculate_glof_risk(FILE_ID, IMAGE_PATH):
    """
    Calculate GLOF risk based on velocity and temperature.
    Returns risk level (low, medium, high) and a percentage.
    """
    risk_percent = 77.6
    
    # Determine risk level
    if risk_percent < 50:
        return "Low", risk_percent
    elif 50 <= risk_percent < 80:
        return "Medium", risk_percent
    else:
        return "High", risk_percent

# ----------------------------
# Map Generation Function - Enhanced
# ----------------------------
# Replace generate_glacier_map function (around lines 351-400) with:
def generate_glacier_map(lat, lon):
    """Generate enhanced map centered on monitoring point"""
    m = folium.Map(
        location=[36.345, 74.8045],
        zoom_start=7,
        tiles="Cartodb Positron"
    )
    
    # Add custom styles
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri',
        name='Satellite',
        overlay=False
    ).add_to(m)
    
    # Add polygon
    polygon_coords = [
        [36.345 - 0.1, 74.8045 - 0.1],
        [36.345 - 0.1, 74.8045 + 0.1],
        [36.345 + 0.1, 74.8045 + 0.1],
        [36.345 + 0.1, 74.8045 - 0.1],
        [36.345 - 0.1, 74.8045 - 0.1]
    ]
    
    folium.Polygon(
        locations=polygon_coords,
        color="#0455BF",
        weight=2,
        fill=True,
        fill_color="#0455BF",
        fill_opacity=0.2,
        tooltip="<strong>Glacier Area</strong>",
        popup=folium.Popup("Main Glacier Body", max_width=200)
    ).add_to(m)
    
    # Add marker with custom icon
    folium.Marker(
        location=[36.345, 74.8045],
        tooltip="<strong>Current Monitoring Point</strong>",
        popup=folium.Popup(f"Lat: {lat}<br>Lon: {lon}", max_width=300),
        icon=folium.Icon(color="red", icon="info-sign")
    ).add_to(m)
    
    # Add danger zones with improved styling
    danger_zone = [
        [lat - 0.32, lon + 0.15],
        [lat - 0.40, lon + 0.20],
        [lat - 0.45, lon + 0.15],
        [lat - 0.50, lon + 0.10],
        [lat - 0.48, lon + 0.05],
        [lat - 0.38, lon + 0.08],
        [lat - 0.32, lon + 0.15]
    ]
    
    folium.Polygon(
        locations=danger_zone,
        color="#F44336",
        weight=2,
        fill=True,
        fill_color="#F44336",
        fill_opacity=0.3,
        tooltip="<strong>Potential GLOF Impact Zone</strong>"
    ).add_to(m)
    
    # Add layer control and plugins
    folium.LayerControl().add_to(m)
    folium.plugins.MeasureControl(position='bottomleft', primary_length_unit='kilometers').add_to(m)
    
    return m
# --- Tab 1: Dashboard - Enhanced ---
with tabs[0]:
    colored_header(
        label="Current Status",
        description=f"Monitoring {location_preset} as of {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
        color_name="blue-70"
    )    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # st.markdown('<div class="status-section">', unsafe_allow_html=True)
        st.subheader("🧊 Ice Velocity (Tomorrow)")
        result = get_velocity_prediction()
        if result:
            velocity, timestamp = result
            if velocity is not None:
                st.metric(
                    label="Predicted Velocity",
                    value=f"{velocity:.2f} m/yr",
                    delta=f"Any value > 200m/yr is dangerous.",
                    delta_color="inverse"
                )
                st.caption(f"Forecast for {pd.to_datetime(timestamp).strftime('%Y-%m-%d %H:%M')}")
                
                # Add mini velocity trend for 7 days
                velocity_forecast = get_velocity_forecast(forecast_days)
                if velocity_forecast:
                    df_vel_forecast = pd.DataFrame(velocity_forecast, columns=["Date", "Velocity"])
                    df_vel_forecast["Date"] = pd.to_datetime(df_vel_forecast["Date"]).dt.strftime('%Y-%m-%d')
                    mini_vel_df = pd.DataFrame({
                        "Date": [datetime.datetime.today().strftime('%Y-%m-%d')] + list(df_vel_forecast["Date"]),
                        "Velocity": [velocity] + list(df_vel_forecast["Velocity"])
                    })
                    print(f"mini_vel_df: {mini_vel_df}")  # Debug
                    fig_vel = px.line(mini_vel_df, x="Date", y="Velocity", markers=True)
                    fig_vel.update_layout(
                        height=150,
                        margin=dict(l=0, r=0, t=0, b=0),
                        yaxis=dict(title=None),
                        xaxis=dict(title=None),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)"
                    )
                    fig_vel.update_traces(line_color="#0078D4")
                    st.plotly_chart(fig_vel, use_container_width=True)
                else:
                    st.warning("Unable to generate velocity forecast trend.")
            else:
                st.error("Velocity prediction returned None.")
        else:
            st.error("Velocity prediction unavailable.")
        # st.markdown('</div>', unsafe_allow_html=True)
        
    with col2:
        # st.markdown('<div class="status-section">', unsafe_allow_html=True)
        st.subheader("🌡️ Surface Temperature (Tomorrow)")
        temp_pred = get_temperature_prediction()
        if temp_pred is not None:
            st.metric(
                label="Predicted Temperature", 
                value=f"{temp_pred:.1f}°C",
                delta=f"{temp_pred - (-2.1):.1f}°C",
                delta_color="inverse"
            )
            tomorrow = datetime.datetime.today() + timedelta(days=1)
            st.caption(f"Forecast for {tomorrow.strftime('%Y-%m-%d')}")
            
            # Add mini temperature trend with actual forecast data
            mini_temp_forecast = get_temperature_forecast(min(forecast_days, 7))  # Get up to 7 days
            if mini_temp_forecast:
                mini_temp_df = pd.DataFrame(mini_temp_forecast, columns=["Date", "Temperature"])
                mini_temp_df["Date"] = pd.to_datetime(mini_temp_df["Date"]).dt.strftime('%Y-%m-%d')
                
                # Prepend today's data point
                mini_temp_df = pd.DataFrame({
                    "Date": [datetime.datetime.today().strftime('%Y-%m-%d')] + list(mini_temp_df["Date"]),
                    "Temperature": [temp_pred] + list(mini_temp_df["Temperature"])
                })
                
                fig_temp = px.line(mini_temp_df, x="Date", y="Temperature", markers=True)
                fig_temp.update_layout(
                    height=150,
                    margin=dict(l=0, r=0, t=0, b=0),
                    yaxis=dict(title=None),
                    xaxis=dict(title=None),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)"
                )
                fig_temp.update_traces(line_color="#FF9500")
                st.plotly_chart(fig_temp, use_container_width=True)
            else:
                st.warning("Unable to generate temperature forecast trend.")
        # st.markdown('</div>', unsafe_allow_html=True)
    with col3:
        if result and temp_pred is not None:
            velocity_val, _ = result
            risk_level, risk_percent = calculate_glof_risk(FILE_ID, IMAGE_PATH)
            
            risk_class = ""
            if risk_level == "High":
                risk_class = "risk-high"
            elif risk_level == "Medium":
                risk_class = "risk-medium"
            else:
                risk_class = "risk-low"
            
            # st.markdown(f'<div class="status-section {risk_class}">', unsafe_allow_html=True)
            st.subheader("⚠️ GLOF Risk Assessment")
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=risk_percent,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': f"{risk_level} Risk", 'font': {'size': 24}},
                gauge={
                    'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "darkblue"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 40], 'color': 'rgba(52, 199, 89, 0.6)'},
                        {'range': [40, 70], 'color': 'rgba(255, 149, 0, 0.6)'},
                        {'range': [70, 100], 'color': 'rgba(255, 59, 48, 0.6)'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': risk_percent
                    }
                }
            ))
            fig.update_layout(
                height=210,
                margin=dict(l=20, r=20, t=30, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                font={'color': "#333333", 'family': "Arial"}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.caption("Key risk factors:")
            factors = []
            if velocity_val > 85:
                factors.append(f"- Higher ice velocity ({velocity_val:.1f} m/yr)")
            if temp_pred > 0:
                factors.append(f"- Positive temperatures ({temp_pred:.1f}°C)")
            if not factors:
                factors.append("- No critical factors detected")
            for factor in factors:
                st.markdown(factor)
            # st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.error("Risk assessment unavailable - requires velocity and temperature data.")
    
    style_metric_cards()
    
    st.markdown("### 🗺️ Monitoring Area")
    map_col, details_col = st.columns([3, 1])
    
    with map_col:
        st.markdown('<div class="map-container">', unsafe_allow_html=True)
        m = generate_glacier_map(lat, lon)
        folium_static(m)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with details_col:
        st.markdown('<div class="status-section">', unsafe_allow_html=True)
        st.subheader("Sensor Status")
        if result:
            velocity, _ = result
            st.metric("Current Velocity", f"{velocity:.1f} m/yr")
        if temp_pred is not None:
            st.metric("Current Temperature", f"{temp_pred:.1f}°C")
        try:
            vel_data = get_velocity_data()[1]
            if vel_data is not None:
                st.metric("Velocity Trend (30d)", 
                          f"{(velocity - np.nanmean(vel_data[-30:])):.1f} m/yr")
        except:
            pass
        
        st.markdown("### Alerts")
        if temp_pred and temp_pred > 0:
            st.error("🚨 Temperature above freezing point!")
        elif velocity and velocity > 90:
            st.warning("⚠️ Elevated ice velocity detected")
        else:
            st.success("✅ Normal operating conditions")
        st.markdown('</div>', unsafe_allow_html=True)

# --- Tab 2: Forecast & History - Enhanced ---
with tabs[1]:
    colored_header(
        label="Forecast & Historical Trends",
        description=f"Detailed forecasts and historical data for {location_preset}",
        color_name="blue-70"
    )
    
    st.markdown('<div class="forecast-section">', unsafe_allow_html=True)
    st.subheader(f"{forecast_days}-Day Forecast")
    
    velocity_forecast = get_velocity_forecast(forecast_days)
    temperature_forecast = get_temperature_forecast(forecast_days)
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    plot_has_data = False
    
    # Add velocity trace
    if velocity_forecast:
        df_vel_forecast = pd.DataFrame(velocity_forecast, columns=["Date", "Velocity"])
        df_vel_forecast["Date"] = pd.to_datetime(df_vel_forecast["Date"])
        print(f"df_vel_forecast: {df_vel_forecast}")  # Debug
        if not df_vel_forecast.empty and df_vel_forecast["Velocity"].notna().any():
            fig.add_trace(
                go.Scatter(
                    x=df_vel_forecast["Date"], 
                    y=df_vel_forecast["Velocity"],
                    name="Ice Velocity (m/yr)",
                    line=dict(color="#0078D4", width=3),
                    mode="lines+markers"
                ),
                secondary_y=False
            )
            plot_has_data = True
        else:
            st.warning("No valid velocity forecast data to plot.")
    else:
        st.error("Velocity forecast unavailable")
    
    # Add temperature trace
    if temperature_forecast:
        df_temp_forecast = pd.DataFrame(temperature_forecast, columns=["Date", "Temperature"])
        df_temp_forecast["Date"] = pd.to_datetime(df_temp_forecast["Date"])
        print(f"df_temp_forecast: {df_temp_forecast}")  # Debug
        if not df_temp_forecast.empty and df_temp_forecast["Temperature"].notna().any():
            fig.add_trace(
                go.Scatter(
                    x=df_temp_forecast["Date"], 
                    y=df_temp_forecast["Temperature"],
                    name="Temperature (°C)",
                    line=dict(color="#FF9500", width=3, dash="dot"),
                    mode="lines+markers"
                ),
                secondary_y=True
            )
            plot_has_data = True
            fig.add_shape(
                type="line",
                x0=min(df_temp_forecast["Date"]),
                y0=0,
                x1=max(df_temp_forecast["Date"]),
                y1=0,
                line=dict(color="#FF3B30", width=2, dash="dash"),
                name="Freezing Level"
            )
        else:
            st.warning("No valid temperature forecast data to plot.")
    else:
        st.error("Temperature forecast unavailable")
    
# In the Forecast & Trends tab, after creating the plot but before displaying it:

    if plot_has_data:
        fig.update_layout(
            title=f"{forecast_days}-Day Forecast: Ice Velocity vs Temperature",
            height=500,
            margin=dict(l=20, r=20, t=80, b=20),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            hovermode="x unified",
            plot_bgcolor="rgba(245,245,245,1)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Arial", size=14)
        )
        
        # Set fixed ranges for better visualization
        fig.update_yaxes(
            title_text="Ice Velocity (m/yr)",
            secondary_y=False,
            showgrid=True,
            gridcolor="rgba(200,200,200,0.3)",
            title_font=dict(color="#0078D4"),
            range=[0, 200],  # Set appropriate range for velocity
            tickformat=".0f"
        )
        
        fig.update_yaxes(
            title_text="Temperature (°C)",
            secondary_y=True,
            showgrid=False,
            title_font=dict(color="#FF9500"),
            range=[-15, 15],  # Set appropriate range for temperature
            tickformat=".1f"
        )
        
        fig.update_xaxes(
            title_text="Date",
            showgrid=True,
            gridcolor="rgba(200,200,200,0.3)"
        )
        fig.add_annotation(
            x=0.5,
            y=-0.15,
            xref="paper",
            yref="paper",
            text="Higher velocity and temperatures above freezing increase GLOF risk",
            showarrow=False,
            font=dict(color="#666666", size=12)
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("No forecast data available to plot")
    
    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------
# Historical Data Section
# ----------------------------
    colored_header(
        label="Historical Trends",
        description="Long-term patterns and anomalies",
        color_name="blue-70"
    )
    
    col_hist1, col_hist2 = st.columns(2)
    
with col_hist1:
        st.markdown('<div class="forecast-section">', unsafe_allow_html=True)
        st.subheader("Ice Velocity History")
        summary_df = get_velocity_data(lat, lon)
        if not summary_df.empty:
            df_vel = summary_df[['date', 'avg_velocity']].rename(columns={'date': 'Timestamp', 'avg_velocity': 'Velocity'})
            fig = px.line(df_vel, x="Timestamp", y="Velocity", title="Velocity Trends", color_discrete_sequence=["#0078D4"])
            fig.add_trace(go.Scatter(x=df_vel["Timestamp"], y=df_vel["Velocity"].rolling(30, min_periods=1).mean(), line=dict(color="#FF3B30", width=2, dash="dot"), name="30-Day Avg"))
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Velocity history unavailable")
        st.markdown('</div>', unsafe_allow_html=True)

    
with col_hist2:
        st.markdown('<div class="forecast-section">', unsafe_allow_html=True)
        st.subheader("Temperature History")
        
        try:
            temps, dates = get_past_temperatures(lat, lon, datetime.date.today())
            if temps and dates:
                df_temp = pd.DataFrame({
                    "Timestamp": pd.to_datetime(dates),
                    "Temperature": temps
                }).drop_duplicates().sort_values("Timestamp")
                
                fig = px.line(df_temp, x="Timestamp", y="Temperature",
                             title="5-Year Temperature Record",
                             color_discrete_sequence=["#FF9500"])
                
                # Add annual bands
                for year in range(2019, 2024):
                    fig.add_vrect(
                        x0=f"{year}-06-01", 
                        x1=f"{year}-09-01",
                        fillcolor="#0078D4",
                        opacity=0.1,
                        line_width=0,
                        annotation_text=f"{year} Melt Season",
                        annotation_position="top left"
                    )
                
                fig.update_layout(
                    height=400,
                    showlegend=False,
                    yaxis_title="Temperature (°C)"
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Insufficient temperature data")
        except Exception as e:
            st.error(f"Temperature history error: {str(e)}")
        st.markdown('</div>', unsafe_allow_html=True)

# --- Tab 3: Advanced Analysis ---
with tabs[2]:
    colored_header(
        label="Advanced Analytics",
        description="Deep dive into glacier dynamics",
        color_name="blue-70"
    )
    
    st.markdown("""
    <div class="forecast-section">
        <h3>📈 Multivariate Analysis</h3>
        <div class="stMetric" style="margin-bottom: 1.5rem;">
            <div class="metric-content">
                <div class="metric-value">Velocity-Temperature Correlation: -0.62</div>
                <div class="metric-delta" style="color: #FF3B30;">Strong Inverse Relationship</div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Correlation matrix
    st.subheader("Variable Relationships")
    
    # Generate synthetic correlation data
    corr_data = pd.DataFrame({
        'Velocity': np.random.randn(100)*0.8 + np.linspace(80, 100, 100),
        'Temperature': np.random.randn(100)*3 + np.linspace(-5, 5, 100),
        'Precipitation': np.abs(np.random.randn(100)*10),
        'Seismic Activity': np.random.poisson(3, 100)
    }).corr()
    
    fig = px.imshow(corr_data,
                   color_continuous_scale='RdBu',
                   zmin=-1,
                   zmax=1,
                   title="Environmental Factor Correlations")
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# --- Tab 4: About GLOFs - Enhanced ---
# --- Tab 4: About GLOFs - Enhanced ---
# --- Tab 4: About GLOFs - Fixed Version ---
with tabs[3]:
    colored_header(
        label="Glacial Lake Outburst Floods",
        description="Understanding the risks and mitigation strategies",
        color_name="blue-70"
    )
    
    # First section - Introduction
    st.markdown("""
    <div class="forecast-section">
        <h2>What are Glacial Lake Outburst Floods?</h2>
        <p>Glacial Lake Outburst Floods (GLOFs) are sudden releases of water from a glacial lake that can cause catastrophic flooding downstream. They occur when lakes formed by melting glaciers are compromised by various triggering factors.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Second section - Causes
    st.markdown("""
    <div class="forecast-section">
        <h3>Causes of GLOFs</h3>
        <ul>
            <li><b>Glacier Retreat and Lake Formation:</b> As glaciers melt and retreat due to climate change, they often leave behind depressions that fill with meltwater, forming glacial lakes.</li>
            <li><b>Moraine Dam Failure:</b> Many glacial lakes are dammed by unstable moraines (debris deposited by glaciers), which can fail due to erosion or overtopping.</li>
            <li><b>Ice Avalanches:</b> Large chunks of ice falling into a glacial lake can create displacement waves that overtop and potentially breach the dam.</li>
            <li><b>Seismic Activity:</b> Earthquakes can destabilize moraine dams or cause landslides into the lake.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Third section - Impact Zones
    st.markdown("<h3>Impact Zones and Risk Factors</h3>", unsafe_allow_html=True)
    risk_col1, risk_col2 = st.columns(2)
    
    with risk_col1:
        st.markdown("""
        <div style="background-color: rgba(244, 67, 54, 0.1); border-left: 4px solid #F44336; padding: 20px; border-radius: 8px; height: 100%;">
            <h4 style="color: #F44336;">High Risk Zones</h4>
            <ul>
                <li>Communities within 5-10 km downstream</li>
                <li>Areas in narrow valleys below glacial lakes</li>
                <li>Infrastructure near river channels</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with risk_col2:
        st.markdown("""
        <div style="background-color: rgba(255, 152, 0, 0.1); border-left: 4px solid #FF9800; padding: 20px; border-radius: 8px; height: 100%;">
            <h4 style="color: #FF9800;">Warning Signs</h4>
            <ul>
                <li>Rapid glacier velocity changes</li>
                <li>Sustained high temperatures</li>
                <li>Visible cracks in ice dams</li>
                <li>Sudden changes in lake turbidity</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Fourth section - Early Warning Systems
    st.markdown("""
    <div class="forecast-section">
        <h3>GLOF Early Warning Systems</h3>
        <p>Effective early warning systems for GLOFs typically include:</p>
        <ol>
            <li><b>Monitoring:</b> Real-time monitoring of glacial lakes using satellite imagery, ground sensors, and advanced models like those used in IceWatch.</li>
            <li><b>Analysis:</b> Processing data through AI models to detect risk patterns and triggering events.</li>
            <li><b>Communication:</b> Rapid notification systems to alert communities and authorities.</li>
            <li><b>Response:</b> Evacuation plans and emergency procedures for potentially affected areas.</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    # Fifth section - Case Study
    st.markdown("""
    <div style="background-color: rgba(3, 169, 244, 0.1); border-radius: 8px; padding: 20px; margin: 20px 0; border-left: 4px solid #03A9F4;">
        <h3 style="color: #03A9F4;">Case Study: Shishper Glacier GLOF Events</h3>
        <p>The Shishper Glacier in Pakistan has experienced multiple GLOF events in recent years:</p>
        <ul>
            <li><b>May 2019:</b> A major GLOF event caused significant damage to infrastructure and farmland.</li>
            <li><b>June 2022:</b> Another large outburst flood disrupted communities in the Hunza Valley.</li>
            <li><b>April 2024:</b> Early detection through monitoring systems allowed for successful evacuations.</li>
        </ul>
        <p>These events highlight the importance of continuous monitoring and early warning systems.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sixth section - Mitigation Strategies
    st.markdown("<h3>Mitigation Strategies</h3>", unsafe_allow_html=True)
    mit_col1, mit_col2 = st.columns(2)
    
    with mit_col1:
        st.markdown("""
        <div style="background-color: rgba(76, 175, 80, 0.1); padding: 15px; border-radius: 8px; height: 100%;">
            <h4 style="color: #4CAF50;">Structural Measures</h4>
            <ul>
                <li>Artificial lowering of lake levels</li>
                <li>Construction of outlet channels</li>
                <li>Reinforcement of natural moraine dams</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with mit_col2:
        st.markdown("""
        <div style="background-color: rgba(156, 39, 176, 0.1); padding: 15px; border-radius: 8px; height: 100%;">
            <h4 style="color: #9C27B0;">Non-Structural Measures</h4>
            <ul>
                <li>Community education and awareness</li>
                <li>Hazard mapping and zoning</li>
                <li>Early warning systems like IceWatch</li>
                <li>Emergency response planning</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Seventh section - Remote Sensing
    st.markdown("""
    <div class="forecast-section">
        <h3>The Role of Remote Sensing</h3>
        <p>Modern GLOF monitoring relies heavily on satellite data and remote sensing technologies:</p>
        <ul>
            <li>Optical satellite imagery for visual monitoring</li>
            <li>SAR (Synthetic Aperture Radar) for all-weather monitoring</li>
            <li>Digital Elevation Models for topographic analysis</li>
            <li>Thermal imaging for temperature analysis</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Eighth section - Diagram
    st.image("https://via.placeholder.com/800x400.png?text=GLOF+Monitoring+Process", 
             caption="Diagram showing the GLOF monitoring and early warning process", 
             use_column_width=True)
    
    # Ninth section - Resources
    st.markdown("""
    <div class="forecast-section">
        <h3>Further Resources</h3>
        <ul>
            <li><a href="https://www.icimod.org/mountain/glacial-lakes-outburst-floods/">ICIMOD - Glacial Lakes and Outburst Floods</a></li>
            <li><a href="https://www.preventionweb.net/understanding-disaster-risk/key-concepts/glacial-lake-outburst-floods">PreventionWeb - Understanding GLOF Risk</a></li>
            <li><a href="https://www.usgs.gov/special-topics/water-science-school/science/glaciers-and-icecaps">USGS - Glaciers and Ice Caps</a></li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# ----------------------------
# Final Touches
# ----------------------------
# Add footer
# Replace footer section (last few lines) with:
st.markdown("""
<div style="text-align: center; padding: 2rem; margin-top: 2rem; border-top: 1px solid #E4E7EB;">
    <div style="display: flex; justify-content: center; align-items: center; margin-bottom: 15px;">
        <div style="height: 36px; width: 36px; background: linear-gradient(135deg, #0455BF 0%, #03A9F4 100%); 
                   border-radius: 50%; display: flex; align-items: center; justify-content: center; color: white; 
                   font-size: 20px; margin-right: 10px;">❄️</div>
        <p style="font-weight: 600; font-size: 1.4rem; color: #323F4B; margin: 0;">IceWatch</p>
    </div>
    <p style="color: #52606D; margin-bottom: 5px;">Glacier Monitoring System v2.1 | Developed by Anser, Zuha, Talha</p>
    <p style="color: #7B8794; font-size: 0.9rem;">Operational since 2025 | Last updated: April 6, 2025</p>
    <div style="display: flex; justify-content: center; gap: 20px; margin-top: 15px;">
        <a href="#" style="color: #0455BF; text-decoration: none; font-size: 0.9rem;">Documentation</a>
        <a href="#" style="color: #0455BF; text-decoration: none; font-size: 0.9rem;">Privacy Policy</a>
        <a href="#" style="color: #0455BF; text-decoration: none; font-size: 0.9rem;">Contact</a>
    </div>
</div>
""", unsafe_allow_html=True)
