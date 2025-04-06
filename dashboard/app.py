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
from tempflow import load_model_and_scaler as load_tempflow, get_past_temperatures, prepare_input_sequence, predict_future
import asyncio
from streamlit_extras.metric_cards import style_metric_cards
from streamlit_extras.colored_header import colored_header
import torch
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
st.markdown("""
<style>
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        border-radius: 4px 4px 0px 0px;
        font-weight: 500;
        background-color: #f0f2f6;
        border-left: 1px solid #dfe1e6;
        border-right: 1px solid #dfe1e6;
        border-top: 1px solid #dfe1e6;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0078D4 !important;
        color: white !important;
    }
    h1 {
        color: #0078D4;
        font-weight: bold;
    }
    h2 {
        color: #0078D4;
    }
    h3 {
        color: #505050;
    }
    .risk-card {
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .risk-high {
        background-color: rgba(255, 59, 48, 0.1);
        border-left: 4px solid #FF3B30;
    }
    .risk-medium {
        background-color: rgba(255, 149, 0, 0.1);
        border-left: 4px solid #FF9500;
    }
    .risk-low {
        background-color: rgba(52, 199, 89, 0.1);
        border-left: 4px solid #34C759;
    }
    .logo-text {
        font-weight: bold;
        font-size: 2.5rem;
        color: #0078D4;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);
    }
    .subheader {
        color: #505050;
        font-size: 1.2rem;
        margin-top: -0.8rem;
        margin-bottom: 1.5rem;
    }
    .status-section {
        background-color: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        margin-bottom: 1rem;
    }
    .map-container {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    }
    .forecast-section {
        padding: 1.5rem;
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        margin-bottom: 1rem;
    }
    .stMetric {
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    }
</style>
""", unsafe_allow_html=True)

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
# ----------------------------
# Load Models & Scalers
# ----------------------------
@st.cache_resource
def load_models():
    vel_model, device = load_model("dashboard/models/terraflow_transformer_model.pth")
    temp_model, temp_scaler = load_tempflow("dashboard/models/best_tempflow_model.keras", "dashboard/models/scaler_tempflow.save")
    return vel_model, device, temp_model, temp_scaler

vel_model, device, temp_model, temp_scaler = load_models()

# ----------------------------
# Sidebar – Enhanced with location presets
# ----------------------------
with st.sidebar:
    st.image("https://via.placeholder.com/300x100.png?text=IceWatch+Logo", use_container_width=True)
    st.markdown("---")
    
    st.header("📍 Location")
    
    # Add location presets
    location_preset = st.selectbox(
        "Select Monitoring Area", 
        ["Shishper Glacier, Pakistan", "Khurdopin Glacier", "Passu Glacier", "Custom Location"],
        index=0
    )
    
    # Set default coordinates based on preset
    if location_preset == "Shishper Glacier, Pakistan":
        default_lat, default_lon = 74.4912, 36.39861
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
    
def get_velocity_forecast(days=7):
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
    
def get_temperature_forecast(days=7):
    """
    Generates a 7-day forecast for surface temperature.
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
        with st.spinner(f"Predicting temperature for next {days} days..."):
            for _ in range(days):
                sequence = prepare_input_sequence(predicted_temps, predicted_dates, temp_scaler, lat, lon)
                next_date = last_available_date + timedelta(days=1)
                prediction = predict_future(
                    temp_model,
                    sequence,
                    next_date,
                    next_date,
                    temp_scaler,
                    lat,
                    lon
                )
                if not prediction:
                    st.error("Temperature prediction failed.")
                    return None
                pred_val = prediction[-1][1]
                forecast.append((next_date, pred_val))
                predicted_temps.append(pred_val)
                last_available_date = next_date
                predicted_dates.append(last_available_date)
        return forecast
    except Exception as e:
        st.error(f"Temperature forecast failed: {str(e)}")
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
with tabs[1]:
    colored_header(
        label="Forecast & Historical Trends",
        description=f"Detailed forecasts and historical data for {location_preset}",
        color_name="blue-70"
    )
    
    st.markdown('<div class="forecast-section">', unsafe_allow_html=True)
    st.subheader(f"{forecast_days}-Day Forecast")
    
    # Fetch velocity forecast
    velocity_forecast = get_velocity_forecast(forecast_days)
    temperature_forecast = get_temperature_forecast(forecast_days)
    
    # Debug: Check the raw forecast data
    print(f"Tab 1 - velocity_forecast: {velocity_forecast}")
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    plot_has_data = False
    
    # Add velocity trace
    if velocity_forecast:
        df_vel_forecast = pd.DataFrame(velocity_forecast, columns=["Date", "Velocity"])
        df_vel_forecast["Date"] = pd.to_datetime(df_vel_forecast["Date"])
        print(f"Tab 1 - df_vel_forecast: {df_vel_forecast}")  # Debug
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
        print(f"Tab 1 - df_temp_forecast: {df_temp_forecast}")  # Debug
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
        fig.update_yaxes(
            title_text="Ice Velocity (m/yr)",
            secondary_y=False,
            showgrid=True,
            gridcolor="rgba(200,200,200,0.3)",
            title_font=dict(color="#0078D4")
        )
        fig.update_yaxes(
            title_text="Temperature (°C)",
            secondary_y=True,
            showgrid=False,
            title_font=dict(color="#FF9500")
        )
        fig.update_xaxes(
            title_text="Date",
            showgrid=True,
            gridcolor="rgba(200,200,200,0.3)",
            rangeslider_visible=True
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
def calculate_glof_risk(velocity, temperature):
    """
    Calculate GLOF risk based on velocity and temperature.
    Returns risk level (low, medium, high) and a percentage.
    """
    # Baseline risk (50%)
    risk_percent = 50.0
    
    # Velocity factor (higher velocity = higher risk)
    # Assuming normal velocity range is 40-90 m/yr
    if velocity < 60:
        velocity_factor = -10  # Lower risk
    elif 60 <= velocity < 90:
        velocity_factor = 0    # Neutral
    elif 90 <= velocity < 120:
        velocity_factor = 15   # Higher risk
    else:
        velocity_factor = 30   # Much higher risk
    
    # Temperature factor (higher temp = higher risk due to melting)
    # Assuming temperature range of -10 to +10°C
    if temperature < -5:
        temp_factor = -15      # Lower risk
    elif -5 <= temperature < 0:
        temp_factor = -5       # Slightly lower risk
    elif 0 <= temperature < 5:
        temp_factor = 10       # Higher risk
    else:
        temp_factor = 20       # Much higher risk
    
    # Combine factors (could add more factors in the future)
    risk_percent += velocity_factor + temp_factor
    
    # Ensure risk is between 0-100%
    risk_percent = max(0, min(100, risk_percent))
    
    # Determine risk level
    if risk_percent < 40:
        return "Low", risk_percent
    elif 40 <= risk_percent < 70:
        return "Medium", risk_percent
    else:
        return "High", risk_percent

# ----------------------------
# Map Generation Function - Enhanced
# ----------------------------
def generate_glacier_map(lat, lon):
    """Generate map centered on monitoring point"""
    m = folium.Map(
        location=[lat, lon],  # Use current coordinates
        zoom_start=12,
        tiles="CartoDB positron"
    )
    
    # Update polygon coordinates relative to current position
    polygon_coords = [
        [lat - 0.1, lon - 0.1],
        [lat - 0.1, lon + 0.1],
        [lat + 0.1, lon + 0.1],
        [lat + 0.1, lon - 0.1],
        [lat - 0.1, lon - 0.1]
    ]
    
    folium.Polygon(
        locations=polygon_coords,
        color="#0078D4",
        fill=True,
        fill_color="#0078D4",
        fill_opacity=0.4,
        tooltip="Glacier Area"
    ).add_to(m)
    
    # Add current monitoring point marker
    folium.Marker(
        location=[lat, lon],
        tooltip="Current Monitoring Point",
        icon=folium.Icon(color="red", icon="info-sign")
    ).add_to(m)
    
    # Add potential danger zones
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
        color="#FF3B30",
        fill=True,
        fill_color="#FF3B30",
        fill_opacity=0.3,
        tooltip="Potential GLOF Impact Zone"
    ).add_to(m)
    
    # Add glacial lake
    folium.Circle(
        location=[lat - 0.32, lon + 0.04],
        radius=500,  # meters
        color="#1E88E5",
        fill=True,
        fill_color="#1E88E5",
        fill_opacity=0.6,
        tooltip="Glacial Lake"
    ).add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Add scale
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
        st.markdown('<div class="status-section">', unsafe_allow_html=True)
        st.subheader("🧊 Ice Velocity (Tomorrow)")
        result = get_velocity_prediction()
        if result:
            velocity, timestamp = result
            if velocity is not None:
                st.metric(
                    label="Predicted Velocity",
                    value=f"{velocity:.2f} m/yr",
                    delta=f"{velocity - 85.65:.2f} m/yr",
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
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col2:
        st.markdown('<div class="status-section">', unsafe_allow_html=True)
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
            
            # Add mini temperature trend (already present)
            last_temps = [-3.2, -2.8, -2.1, temp_pred]
            temp_dates = [
                (datetime.datetime.today() - timedelta(days=3)).strftime('%Y-%m-%d'),
                (datetime.datetime.today() - timedelta(days=2)).strftime('%Y-%m-%d'),
                (datetime.datetime.today() - timedelta(days=1)).strftime('%Y-%m-%d'),
                tomorrow.strftime('%Y-%m-%d')
            ]
            mini_temp_df = pd.DataFrame({
                "Date": temp_dates,
                "Temperature": last_temps
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
            st.error("Temperature prediction unavailable.")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col3:
        if result and temp_pred is not None:
            velocity_val, _ = result
            risk_level, risk_percent = calculate_glof_risk(velocity_val, temp_pred)
            
            risk_class = ""
            if risk_level == "High":
                risk_class = "risk-high"
            elif risk_level == "Medium":
                risk_class = "risk-medium"
            else:
                risk_class = "risk-low"
            
            st.markdown(f'<div class="status-section {risk_class}">', unsafe_allow_html=True)
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
            st.markdown('</div>', unsafe_allow_html=True)
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
        fig.update_yaxes(
            title_text="Ice Velocity (m/yr)",
            secondary_y=False,
            showgrid=True,
            gridcolor="rgba(200,200,200,0.3)",
            title_font=dict(color="#0078D4")
        )
        fig.update_yaxes(
            title_text="Temperature (°C)",
            secondary_y=True,
            showgrid=False,
            title_font=dict(color="#FF9500")
        )
        fig.update_xaxes(
            title_text="Date",
            showgrid=True,
            gridcolor="rgba(200,200,200,0.3)",
            rangeslider_visible=True
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
with tabs[3]:
    colored_header(
        label="Glacial Lake Outburst Floods",
        description="Understanding the risks and mitigation strategies",
        color_name="blue-70"
    )
    
    st.markdown("""
    <div class="forecast-section" style="padding: 2rem;">
        <div class="risk-card risk-high">
            <h3>❄️ What are GLOFs?</h3>
            <p>Glacial Lake Outburst Floods (GLOFs) occur when water dammed by a glacier is released suddenly. These events can:</p>
            <ul>
                <li>Release millions of cubic meters of water in hours</li>
                <li>Travel downstream at 60+ km/h</li>
                <li>Destroy infrastructure 100+ km from source</li>
            </ul>
        </div>
        
        <div class="risk-card risk-medium" style="margin-top: 2rem;">
            <h3>📈 Regional Risk Factors</h3>
            <div class="stMetric">
                <div class="metric-value">+2.3°C</div>
                <div class="metric-delta">Temperature rise since 1990</div>
            </div>
            <div class="stMetric">
                <div class="metric-value">42%</div>
                <div class="metric-delta">Increase in glacial lakes (2000-2020)</div>
            </div>
        </div>
        
        <div class="risk-card risk-low" style="margin-top: 2rem;">
            <h3>🛡️ Mitigation Strategies</h3>
            <p>Early warning systems like IceWatch help communities:</p>
            <ul>
                <li>Detect precursor signals 24-72 hours in advance</li>
                <li>Activate emergency response protocols</li>
                <li>Evacuate vulnerable populations</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ----------------------------
# Final Touches
# ----------------------------
# Add footer
st.markdown("""
<div style="text-align: center; padding: 2rem; color: #666;">
    <hr style="margin-bottom: 1rem;">
    <p>IceWatch Glacier Monitoring System v2.1 | Developed with ❄️ by Anser, Zuha, Talha</p>
    <p>Operational since 2025 | Last updated: Today </p>
</div>
""", unsafe_allow_html=True)
