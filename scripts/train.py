import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.layers import Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error
import core

symbol = "AAPL"

try:

    train_data, test_data , _ , _ , _, _= core.get_live_data(symbol=symbol)
    # train_data, test_data , _ , _, _, _ = core.get_simulated_data()

    # Create input sequences and target outputs for multiple columns
    def create_sequences(data, seq_length=30):
        x, y = [], []
        for i in range(len(data) - seq_length):
            x.append(data[i:i+seq_length])
            y.append(data[i+seq_length])  # Target is the next row
        return np.array(x), np.array(y)

    # Scale the data
    # scaler = MinMaxScaler()
    # train_data_scaled = scaler.fit_transform(train_data[['4. close']])

    # Scale the data
    scaler = MinMaxScaler()
    # train_data_scaled = scaler.fit_transform(train_data[['2. high', '3. low', '4. close', 'sma_5', 'sma_20', 'volatility']])
    train_data_scaled = scaler.fit_transform(train_data[['residual_high', 'residual_low', 'residual_close']])


    x_train, y_train = create_sequences(train_data_scaled)

    # Define the model architecture
    # model = Sequential([
    #     LSTM(64, input_shape=(x_train.shape[1], x_train.shape[2]), return_sequences=False),
    #     Dense(3)  # Output layer predicts high, low, and close
    # ])
    model = Sequential([
        LSTM(64, input_shape=(x_train.shape[1], x_train.shape[2]), return_sequences=True),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(3)  # Predict high, low, and close
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='mse')

    # Train the model
    model.fit(x_train, y_train, epochs=50, batch_size=32, validation_split=0.1)

    # Save the model
    model.save(f"models/{symbol}_stock_prediction_model.keras")  # Save in HDF5 format
    np.save(f"models/{symbol}_scaler.npy", scaler)  # Save the scaler for later use
    print("Model and scaler saved.")

except Exception as e:
    print(f"Error {e}")


""" Limited Features
The model may be using insufficient or non-optimal features, such as relying only on residuals (residual_high, residual_low, residual_close) and SMA values. While these features may capture basic price dynamics, they might not fully represent market conditions.
Additional features like trading volume, volatility, or external macroeconomic indicators could improve the model's predictions.
Solutions:
Enrich the feature set with more relevant data (e.g., technical indicators like RSI, MACD, Bollinger Bands, or fundamental features like earnings reports).
"""