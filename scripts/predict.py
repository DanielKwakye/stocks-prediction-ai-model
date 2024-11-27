import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from scripts import core
from datetime import datetime, timedelta

# import core
from sklearn.metrics import mean_squared_error, mean_absolute_error

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
    future_dates = pd.date_range(start=last_date, periods=n_days + 1, freq='B')[1:]
    return [{"date": date.date(), "high": pred[0], "low": pred[1], "close": pred[2]} for date, pred in zip(future_dates, predictions_actual)]

def get_predictions(symbol: str, n_days: int):

    # Load the saved model and scaler
    model = tf.keras.models.load_model(f"models/{symbol}_stock_prediction_model.keras")
    scaler = np.load(f"models/{symbol}_scaler.npy", allow_pickle=True).item()

    """
    Orchestrates the loading of data, scaling, and calling predict_future.
    """
    # _, test_data, data, _, _, _ = core.get_simulated_data()
    _, test_data, data, _, _, _ = core.get_live_data()

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
    

def predict_historical(data, model, scaler, start_date, end_date):
    """
    Predict stock prices for a specific historical date range, excluding weekends.

    Args:
        data (DataFrame): The full dataset with residuals and SMA features.
        model (tf.keras.Model): The trained prediction model.
        scaler (MinMaxScaler): The scaler used during training.
        start_date (str): The start date of the target range in 'YYYY-MM-DD'.
        end_date (str): The end date of the target range in 'YYYY-MM-DD'.

    Returns:
        List[dict]: Predictions for the specified date range.
    """
    # Generate business days only (exclude weekends)
    prediction_dates = pd.date_range(start=start_date, end=end_date, freq='B')

    # Extract the input sequence ending just before start_date
    input_sequence_end_date = pd.to_datetime(start_date) - pd.Timedelta(days=1)
    input_features = ['residual_high', 'residual_low', 'residual_close']
    input_sequence = data.loc[:input_sequence_end_date][input_features]

    # Scale the input sequence
    input_sequence_scaled = scaler.transform(input_sequence)
    test_data_scaled = input_sequence_scaled.reshape(1, input_sequence_scaled.shape[0], input_sequence_scaled.shape[1])

    # Initialize predictions list
    predictions = []

    # Predict for the target date range
    for current_date in prediction_dates:
        try:
            # Predict next step
            next_prediction = model.predict(test_data_scaled)  # Shape: (1, 3)
            predictions.append(next_prediction[0])  # Store raw prediction

            # Update sequence with scaled prediction
            next_prediction_scaled = next_prediction[0].reshape(1, 1, -1)
            test_data_scaled = np.concatenate([test_data_scaled[:, 1:, :], next_prediction_scaled], axis=1)

        except Exception as e:
            print(f"Prediction failed for date {current_date}: {e}")
            break

    # Rescale predictions back to original scale
    predictions_rescaled = scaler.inverse_transform(predictions)

    # Align with SMA values if available (optional step)
    sma_5_values = data.loc[start_date:end_date]['sma_5'].values
    if len(sma_5_values) < len(predictions_rescaled):
        sma_5_values = np.append(sma_5_values, [sma_5_values[-1]] * (len(predictions_rescaled) - len(sma_5_values)))
    predictions_actual = predictions_rescaled + sma_5_values[:len(predictions_rescaled)].reshape(-1, 1)

    # Format predictions with dates
    formatted_predictions = [
        {"date": date.date(), "high": pred[0], "low": pred[1], "close": pred[2]}
        for date, pred in zip(prediction_dates, predictions_actual)
    ]

    return formatted_predictions




def get_predictions_and_mse(symbol: str, start_date: str, end_date: str):
    """
    Predicts stock prices for the specified date range and calculates MSE
    against actual data from core.get_simulated_data.

    Args:
        symbol (str): The stock symbol.
        start_date (str): The start date in 'YYYY-MM-DD' format.
        end_date (str): The end date in 'YYYY-MM-DD' format.

    Returns:
        dict: Predictions, actual values, and MSE scores.
    """
    # Load the saved model and scaler
    model = tf.keras.models.load_model(f"models/{symbol}_stock_prediction_model.keras")
    scaler = np.load(f"models/{symbol}_scaler.npy", allow_pickle=True).item()

    # Load simulated data
    # _, _, data, _, _, _ = core.get_simulated_data()
    _, _, data, _, _, _ = core.get_live_data(symbol=symbol)

    print("start_date: ", start_date)
    print("end_date: ", end_date)

    # Ensure data is within the specified date range
    actual_data = data.loc[start_date:end_date]
    if actual_data.empty:
        raise ValueError(f"No data available for the range {start_date} to {end_date}")

    # Predict for the target date range
    predictions = predict_historical(data, model, scaler, start_date, end_date)

    # Convert predictions to DataFrame
    predictions_df = pd.DataFrame(predictions).set_index('date')

    # # Log missing dates
    # missing_dates = predictions_df.index.difference(actual_data.index)
    # if not missing_dates.empty:
    #     print(f"Warning: The following prediction dates are not in the actual data index: {missing_dates}")

    
    # Align actual data for comparison
    actual_values = actual_data[['2. high', '3. low', '4. close']].rename(
        columns={"2. high": "high", "3. low": "low", "4. close": "close"}
    )

    print(" ------- actual values: data ------")
    print(actual_values)
    print(" ------- End of actual values: data ------")

    print(" ------- predicted values: data ------")
    print(predictions_df)
    print(" ------- End of predicted values: data ------")

    # Ensure date ranges align perfectly
    predictions_df = predictions_df.loc[actual_values.index]

    # Calculate MSE and MAE for high, low, and close
    mse_high = mean_squared_error(actual_values['high'], predictions_df['high'])
    mse_low = mean_squared_error(actual_values['low'], predictions_df['low'])
    mse_close = mean_squared_error(actual_values['close'], predictions_df['close'])

    mae_high = mean_absolute_error(actual_values['high'], predictions_df['high'])
    mae_low = mean_absolute_error(actual_values['low'], predictions_df['low'])
    mae_close = mean_absolute_error(actual_values['close'], predictions_df['close'])

    # Return results
    return {
        "predictions": predictions_df,
        "actual": actual_values,
        "mse_scores": {"mse_high": mse_high, "mse_low": mse_low, "mse_close": mse_close},
        "mae_scores": {"mae_high": mae_high, "mae_low": mae_low, "mae_close": mae_close}
    }

def get_test_date_range(): 
    # _, _, _, _, test_data_start_date, test_data_end_date = core.get_simulated_data()
    _, _, _, _, test_data_start_date, test_data_end_date = core.get_live_data()
    print(f"Test Data Start Date: {test_data_start_date}")
    print(f"Test Data End Date: {test_data_end_date}")
    return {
        "start_date": test_data_start_date,
        "end_date": test_data_end_date
    }

# result = get_predictions_and_mse('AAPL', '2024-09-19', '2024-10-19')
# print("predictions --------------------------------")
# print(result)

# get_test_date_range()



# Function to calculate business days difference
def subtract_business_days(end_date_str, business_days):
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    business_days_counted = 0
    current_date = end_date
    
    while business_days_counted < business_days:
        current_date -= timedelta(days=1)
        if current_date.weekday() < 5:  # Monday to Friday are business days
            business_days_counted += 1

    return current_date.strftime("%Y-%m-%d")