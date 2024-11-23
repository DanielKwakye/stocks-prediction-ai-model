import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import core
import tensorflow as tf

# Load the saved model and scaler
model = tf.keras.models.load_model("models/stock_prediction_model.keras")
scaler = np.load("models/scaler.npy", allow_pickle=True).item()

# Predict future values for the next n days
def predict_future(data, model, scaler, sma_5, n_days=30, last_date=None):
    """
    Predict future stock prices for n_days based on residual forecasting and SMA adjustment.
    
    Parameters:
    - data: Scaled input data for the LSTM model (3D array).
    - model: Trained LSTM model.
    - scaler: Scaler used to transform the residual data.
    - sma_5: Array of the last SMA values to adjust the residuals back to actual values.
    - n_days: Number of days to predict.
    - last_date: Last date in the dataset to calculate future dates.
    
    Returns:
    - List of tuples with (date, high, low, close) predictions.
    """
    # Start with the most recent 3D sequence
    last_sequence = data  # Already reshaped to (1, sequence_length, num_features)
    predictions = []

    for _ in range(n_days):
        # Predict the next residuals
        next_prediction = model.predict(last_sequence)  # Shape: (1, 3)

        # Append the residual predictions
        predictions.append(next_prediction[0])  # Extract from the batch dimension

        # Update the sequence with the new prediction
        next_prediction_scaled = next_prediction[0].reshape(1, 1, -1)  # Reshape to (1, 1, num_features)
        last_sequence = np.concatenate([last_sequence[:, 1:, :], next_prediction_scaled], axis=1)  # Slide the window

    # Rescale residual predictions back to their original scale
    predictions_rescaled = scaler.inverse_transform(predictions)

    # Dynamically adjust `sma_5` for n_days
    sma_5 = np.append(sma_5, [sma_5[-1]] * max(0, n_days - len(sma_5)))  # Extend for future dates

    # Add back the SMA to convert residuals to actual values
    predictions_actual = predictions_rescaled + sma_5[:len(predictions_rescaled)].reshape(-1, 1)

    # Generate future dates starting from last_date
    future_dates = pd.date_range(start=last_date, periods=n_days + 1, freq='D')[1:]
    return list(zip(future_dates, predictions_actual))

# Load data
train_data, test_data, data, _ = core.get_simulated_data()

# Scale residuals instead of raw values
test_data_scaled = scaler.transform(test_data[['residual_high', 'residual_low', 'residual_close']])

# Dynamically determine sequence length (n_days or available rows)
sequence_length = min(len(test_data_scaled), 30)  # Use the smaller of available rows or 30
test_data_scaled = test_data_scaled[-sequence_length:]  # Take the last `sequence_length` rows
test_data_scaled = test_data_scaled.reshape(1, test_data_scaled.shape[0], test_data_scaled.shape[1])  # Reshape to (1, sequence_length, num_features)

# Dynamically get the last SMA values
sma_5_values = data['sma_5'].values[-sequence_length:]  # Adjust based on available data

# Get the last date from the entire dataset
last_date = data.index[-1]  # Use the most recent date in the combined dataset

# Predict the next 30 days
n_days = 40  # Adjust this value for different prediction horizons
future_predictions = predict_future(test_data_scaled, model, scaler, sma_5=sma_5_values, n_days=n_days, last_date=last_date)

# Print predictions
print(f"Predictions for the next {n_days} days:")
for date, (high, low, close) in future_predictions:
    print(f"{date.date()}, High={high:.2f}, Low={low:.2f}, Close={close:.2f}")
