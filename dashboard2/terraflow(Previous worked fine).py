# terraflow.py
import torch
import joblib
import numpy as np
import pandas as pd
from collections import defaultdict
import itslive

# ----------------------------
# Define the Transformer model for velocity prediction
# ----------------------------
class TransformerRegressor(torch.nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dropout=0.1):
        super(TransformerRegressor, self).__init__()
        self.embedding = torch.nn.Linear(input_dim, d_model)
        # Learnable positional encoding
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
        # x shape: (batch_size, seq_len, input_dim)
        batch_size, seq_len, _ = x.size()
        x = self.embedding(x)
        x = x + self.pos_encoder[:, :seq_len, :]
        x = self.transformer_encoder(x)
        # Use the last time step's output for prediction
        x = self.fc(x[:, -1, :])
        return x

# ----------------------------
# Load model and scalers
# ----------------------------
def load_model_and_scalers(model_path, scaler_X_path, scaler_y_path, device=None):
    """Loads the pre-trained model and associated scalers."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler_X = joblib.load(scaler_X_path)
    scaler_y = joblib.load(scaler_y_path)
    model = TransformerRegressor(input_dim=3, d_model=128, nhead=8, num_layers=4, dropout=0.2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, scaler_X, scaler_y, device

# ----------------------------
# Fetch and align velocity time-series data
# ----------------------------
def get_velocity_data():
    """
    Fetches and aligns velocity time-series data for a set of coordinates.
    Returns:
      common_timestamps: numpy array of the latest 32 common timestamps (as strings)
      aligned_velocities_array: NumPy array of shape (num_coordinates, 32) containing velocity values
    """
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
    velocities = itslive.velocity_cubes.get_time_series(points=points)
    data_by_coordinate = defaultdict(dict)

    for i, entry in enumerate(velocities):
        time_series = entry['time_series']
        try:
            timestamps = time_series['mid_date'].values  # e.g., array of timestamps
            v_values = time_series['v'].values          # e.g., velocity values
        except KeyError:
            continue

        if len(timestamps) == 0 or len(v_values) == 0:
            continue

        # Remove NaN values
        valid_indices = ~np.isnan(v_values)
        timestamps = timestamps[valid_indices]
        v_values = v_values[valid_indices]

        # Ensure at least 32 valid points exist
        if len(v_values) < 32:
            continue

        lat, lon = entry['returned_point_geographic_coordinates']
        data_by_coordinate[(lat, lon)] = {
            'timestamps': timestamps[-32:],
            'v_values': v_values[-32:]
        }

    if not data_by_coordinate:
        return None, None

    # Determine the common latest 32 timestamps across all coordinates
    all_timestamps = [data['timestamps'] for data in data_by_coordinate.values()]
    common_timestamps = np.sort(np.unique(np.concatenate(all_timestamps)))[-32:]
    # Return as strings for consistency
    common_timestamps = np.array([str(ts) for ts in common_timestamps])

    # Align each coordinate's velocity values to the common timestamps
    aligned_data = {}
    for coord, data in data_by_coordinate.items():
        df = pd.DataFrame({'timestamp': data['timestamps'], 'v': data['v_values']})
        common_dt = pd.to_datetime(common_timestamps)
        aligned_df = df[pd.to_datetime(df['timestamp']).isin(common_dt)]\
              .set_index('timestamp')\
              .reindex(common_dt)
        # Fill missing values using interpolation and forward/backward fill
        aligned_v = aligned_df['v'].interpolate(method='linear')\
                                .bfill()\
                                .ffill().values
        aligned_data[coord] = aligned_v

    aligned_velocities_array = np.array([aligned_data[coord] for coord in sorted(aligned_data.keys())])
    return common_timestamps, aligned_velocities_array

# ----------------------------
# Predict velocity at a given timestamp
# ----------------------------
def predict_velocity_at_timestamp(model, scaler_X, scaler_y, device, common_timestamps, target_timestamp, point=(74.8506, 36.3247)):
    """
    Predicts the velocity at the specified timestamp.
    
    The function finds the last 32 timestamps (from the available common timestamps) that are less than or equal 
    to the target timestamp, builds the model input, and returns both the predicted velocity and the actual 
    timestamp used for prediction.
    
    Args:
        model: Loaded Transformer model.
        scaler_X: Input feature scaler.
        scaler_y: Output scaler.
        device: Torch device.
        common_timestamps: Array of available timestamps (as strings).
        target_timestamp: User-provided timestamp (string in "YYYY-MM-DD HH:MM:SS" format).
        point: (lat, lon) tuple indicating which coordinate to use.
    
    Returns:
        (predicted_velocity, used_timestamp): Predicted velocity (float) and the last timestamp (as a pd.Timestamp)
          of the input sequence. Returns (None, None) if insufficient data is available.
    """
    try:
        target_dt = pd.to_datetime(target_timestamp)
    except Exception as e:
        print(f"Error parsing target timestamp: {e}")
        return None, None

    # Convert available timestamps to a pandas Series of Timestamps
    common_dt = pd.Series(pd.to_datetime(common_timestamps))
    # Find all timestamps that are less than or equal to the target timestamp
    valid_timestamps = common_dt[common_dt <= target_dt]
    if len(valid_timestamps) < 32:
        # Not enough past data available for prediction
        return None, None
    # Take the last 32 valid timestamps
    seq_timestamps = valid_timestamps.iloc[-32:]
    # Convert these timestamps to ordinal values
    ordinals = seq_timestamps.map(lambda x: x.toordinal()).values
    lat, lon = point
    # Build the input features: each row is [ordinal, lat, lon]
    model_input = np.column_stack([ordinals, np.full_like(ordinals, lat), np.full_like(ordinals, lon)])
    model_input_scaled = scaler_X.transform(model_input)
    # Convert to tensor and add batch dimension
    input_tensor = torch.tensor(model_input_scaled, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        prediction = model(input_tensor)
    prediction_unscaled = scaler_y.inverse_transform(prediction.cpu().numpy().reshape(-1, 1))
    # Return the predicted velocity and the last timestamp used in the sequence
    return prediction_unscaled[0, 0], seq_timestamps.iloc[-1]