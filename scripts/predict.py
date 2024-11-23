import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from scripts import core

def predict_future(data, model, scaler, sma_5, n_days=30, last_date=None):
    """
    Predict future stock prices for n_days based on residual forecasting and SMA adjustment.
    """
    last_sequence = data  # Already reshaped to (1, sequence_length, num_features)
    predictions = []

    for _ in range(n_days):
        next_prediction = model.predict(last_sequence)  # Shape: (1, 3)
        predictions.append(next_prediction[0])  # Extract from the batch dimension
        next_prediction_scaled = next_prediction[0].reshape(1, 1, -1)
        last_sequence = np.concatenate([last_sequence[:, 1:, :], next_prediction_scaled], axis=1)

    predictions_rescaled = scaler.inverse_transform(predictions)
    sma_5 = np.append(sma_5, [sma_5[-1]] * max(0, n_days - len(sma_5)))  # Extend for future dates
    predictions_actual = predictions_rescaled + sma_5[:len(predictions_rescaled)].reshape(-1, 1)
    future_dates = pd.date_range(start=last_date, periods=n_days + 1, freq='D')[1:]
    return [{"date": date.date(), "high": pred[0], "low": pred[1], "close": pred[2]} for date, pred in zip(future_dates, predictions_actual)]

def get_predictions(symbol: str, n_days: int):

    # Load the saved model and scaler
    model = tf.keras.models.load_model(f"models/{symbol}_stock_prediction_model.keras")
    scaler = np.load(f"models/{symbol}_scaler.npy", allow_pickle=True).item()

    """
    Orchestrates the loading of data, scaling, and calling predict_future.
    """
    # _, test_data, data, _ = core.get_live_data(symbol=symbol)
    _, test_data, data, _ = core.get_simulated_data()

    # Scale residuals
    test_data_scaled = scaler.transform(test_data[['residual_high', 'residual_low', 'residual_close']])
    sequence_length = min(len(test_data_scaled), 30)  # Use the smaller of available rows or 30
    test_data_scaled = test_data_scaled[-sequence_length:]
    test_data_scaled = test_data_scaled.reshape(1, test_data_scaled.shape[0], test_data_scaled.shape[1])

    # Dynamically get the last SMA values
    sma_5_values = data['sma_5'].values[-sequence_length:]
    last_date = data.index[-1]

    # Call predict_future
    return predict_future(test_data_scaled, model, scaler, sma_5=sma_5_values, n_days=n_days, last_date=last_date)
