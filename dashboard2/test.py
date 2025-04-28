


#########################
import torch
import numpy as np
import pandas as pd
from collections import defaultdict
import itslive

class TransformerRegressor(torch.nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dropout=0.1):
        super(TransformerRegressor, self).__init__()
        self.embedding = torch.nn.Linear(input_dim, d_model)
        self.pos_encoder = torch.nn.Parameter(torch.zeros(1, 1000, d_model))
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = torch.nn.Linear(d_model, 1)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        x = self.embedding(x)
        x = x + self.pos_encoder[:, :seq_len, :]
        x = self.transformer_encoder(x)
        x = self.fc(x[:, -1, :])
        return x

def load_model(model_path, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerRegressor(input_dim=8, d_model=256, nhead=8, num_layers=4, dropout=0.2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, device

def process_velocity_data(df: pd.DataFrame) -> pd.DataFrame:
    """Process raw velocity data into features with cyclical encoding"""
    proc_df = df.copy()
    
    # Ensure proper datetime conversion
    proc_df['mid_date'] = pd.to_datetime(proc_df['mid_date'])
    proc_df = proc_df.sort_values(['lon', 'lat', 'mid_date'])
    
    # Extract time components
    proc_df['year'] = proc_df['mid_date'].dt.year
    proc_df['month'] = proc_df['mid_date'].dt.month
    proc_df['day'] = proc_df['mid_date'].dt.day
    
    # Cyclical encoding
    proc_df['month_sin'] = np.sin(2 * np.pi * proc_df['month'] / 12)
    proc_df['month_cos'] = np.cos(2 * np.pi * proc_df['month'] / 12)
    proc_df['day_sin'] = np.sin(2 * np.pi * proc_df['day'] / 31)  # Max 31 days in month
    proc_df['day_cos'] = np.cos(2 * np.pi * proc_df['day'] / 31)
    
    # Calculate velocity metrics and preserve mid_date
    summary_df = proc_df.groupby(['lon', 'lat', 'mid_date']).agg({
        'year': 'first',
        'month': 'first',
        'day': 'first',
        'month_sin': 'first',
        'month_cos': 'first',
        'day_sin': 'first',
        'day_cos': 'first',
        'v [m/yr]': ['mean', 'max']
    }).reset_index()
    
    # Flatten column names
    summary_df.columns = ['lon', 'lat', 'mid_date', 'year', 'month', 'day',
                         'month_sin', 'month_cos', 'day_sin', 'day_cos',
                         'avg_velocity', 'max_velocity']
    
    return summary_df

points = [
    (74.8506, 36.3247),
    (74.8355, 36.3197),
    (74.8149, 36.3141),
    (74.8519, 36.3412),
    (74.7895, 36.3164),
    (74.7414, 36.3108),
    (74.7119, 36.3164),
    (74.8471, 36.3634)
    ]

def get_velocity_data():
    velocities = itslive.velocity_cubes.get_time_series(points=points)
    all_data = []
    
    for entry in velocities:
        if 'time_series' not in entry:
            continue
            
        time_series = entry['time_series']
        lat, lon = entry['returned_point_geographic_coordinates']
        
        df = pd.DataFrame({
            'mid_date': pd.to_datetime(time_series['mid_date'].values),
            'v [m/yr]': time_series['v'].values,
            'lat': lat,
            'lon': lon
        }).dropna()
        
        if not df.empty:
            all_data.append(df)
    
    if not all_data:
        return None, None
        
    combined_df = pd.concat(all_data)
    processed_df = process_velocity_data(combined_df)
    
    # Align to common dates
    common_dates = pd.date_range(end=pd.Timestamp.now(), periods=32, freq='D')
    aligned_data = []
    
    for (lat, lon), group in processed_df.groupby(['lat', 'lon']):
        group = group.set_index('mid_date')
        group = group.reindex(common_dates)
        group = group.ffill().bfill()
        group['lat'] = lat
        group['lon'] = lon
        aligned_data.append(group.reset_index())
    
    aligned_df = pd.concat(aligned_data)
    common_timestamps = aligned_df['mid_date'].dt.strftime('%Y-%m-%d %H:%M:%S').values
    features = aligned_df[['lon', 'lat', 'year', 'month_sin', 'month_cos', 
                          'day_sin', 'day_cos', 'avg_velocity']].values
    
    return common_timestamps, features


def predict_velocity(model, device, common_timestamps, target_timestamp, point=(74.8506, 36.3247)):
    try:
        target_dt = pd.to_datetime(target_timestamp)
        common_dates = pd.to_datetime(common_timestamps)
        
        valid_dates = [d for d in common_dates if d <= target_dt][-32:]
        if len(valid_dates) < 32:
            return None, None
        
        # Get features for all points
        _, features = get_velocity_data()
        if features is None:
            return None, None
            
        try:
            point_idx = points.index(point)
            sequence = features[point_idx][-32:]
            
            input_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                prediction = model(input_tensor)
            
            return prediction.item(), valid_dates[-1]
        except ValueError:
            return None, None
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return None, None