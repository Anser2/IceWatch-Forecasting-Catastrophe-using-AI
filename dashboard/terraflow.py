# terraflow.py
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from datetime import datetime, timedelta
from itslive import velocity_cubes  # External dependency

def preprocess_velocity_data(historical_df):
    summary_df = historical_df.groupby('mid_date').agg({
        'v [m/yr]': 'mean',
        'lat': 'first',
        'lon': 'first'
    }).reset_index()
    summary_df.rename(columns={'v [m/yr]': 'avg_velocity'}, inplace=True)
    
    summary_df['year'] = summary_df['mid_date'].dt.year
    summary_df['month'] = summary_df['mid_date'].dt.month
    summary_df['day'] = summary_df['mid_date'].dt.day
    
    summary_df['month_sin'] = np.sin(2 * np.pi * summary_df['month'] / 12)
    summary_df['month_cos'] = np.cos(2 * np.pi * summary_df['month'] / 12)
    summary_df['day_sin'] = np.sin(2 * np.pi * summary_df['day'] / summary_df['mid_date'].dt.days_in_month)
    summary_df['day_cos'] = np.cos(2 * np.pi * summary_df['day'] / summary_df['mid_date'].dt.days_in_month)
    
    summary_df = summary_df[['lon', 'lat', 'year', 'month_sin', 'month_cos', 'day_sin', 'day_cos', 'avg_velocity', 'mid_date']]
    summary_df.rename(columns={'mid_date': 'date'}, inplace=True)
    print(f"Preprocessed summary_df shape: {summary_df.shape}")
    return summary_df

class TransformerRegressor(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dropout=0.1):
        super(TransformerRegressor, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, 1000, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        x = self.embedding(x)
        x = x + self.pos_encoder[:, :seq_len, :]
        x = self.transformer_encoder(x)
        x = self.fc(x[:, -1, :])
        return x

def load_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerRegressor(input_dim=8, d_model=256, nhead=8, num_layers=4, dropout=0.1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Model loaded on {device}")
    return model, device

def get_velocity_data(lat, lon):
    print(f"Fetching data for lat={lat}, lon={lon}")
    points = [(lat, lon)]
    velocities = velocity_cubes.get_time_series(points=points)
    print(f"Velocities: {velocities}")
    all_data = []
    for entry in velocities:
        if 'time_series' in entry:
            df = pd.DataFrame({
                'mid_date': pd.to_datetime(entry['time_series']['mid_date'].values),
                'v [m/yr]': entry['time_series']['v'].values,
                'lat': lat,
                'lon': lon
            }).dropna()
            all_data.append(df)
    if not all_data:
        raise ValueError(f"No velocity data available for lat={lat}, lon={lon}")
    historical_df = pd.concat(all_data).sort_values(by='mid_date')
    summary_df = preprocess_velocity_data(historical_df)
    return summary_df

def predict_velocity(model, device, summary_df, target_time, lat, lon):
    model.eval()
    target_date = pd.to_datetime(target_time).date()  # Strip to date only
    earliest_date = summary_df['date'].min().date()
    latest_date = summary_df['date'].max().date()
    print(f"Target date: {target_date}, Data range: {earliest_date} to {latest_date}")

    if target_date < earliest_date:
        raise ValueError(f"Target date {target_date} is before earliest data: {earliest_date}")

    input_features = ['lon', 'lat', 'year', 'month_sin', 'month_cos', 'day_sin', 'day_cos', 'avg_velocity']
    summary_df['date'] = summary_df['date'].dt.date  # Ensure dates are date-only

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
            print(f"Predicted {current_date}: {pred}")

            if current_date == target_date:
                return pred

            next_date = current_date + timedelta(days=1)
            days_in_month = (next_date.replace(month=next_date.month % 12 + 1, day=1) - timedelta(days=1)).day
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
        print(f"Reached end of loop, returning last prediction for {current_date}")
        return pred  # Return last prediction if target_date overshot
