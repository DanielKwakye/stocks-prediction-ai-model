import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import os
from alpha_vantage.timeseries import TimeSeries

# Load simulated stock data from JSON
def get_simulated_data(json_path="data/simulated_data.json"):
    with open(json_path, "r") as f:
        mock_data = json.load(f)
    meta_data = mock_data["Meta Data"]
    time_series = mock_data["Time Series (Daily)"]

    # Convert time series data into a DataFrame
    data = pd.DataFrame.from_dict(time_series, orient="index").astype(float)
    data['date'] = pd.to_datetime(data.index)
    data = data.set_index('date')
    data = data.sort_index(ascending=True)  # Ensure chronological order

    print("data => ", data)

    # Add a 'daily_return' feature
    data['daily_return'] = data['4. close'].pct_change()
    data = data.dropna()  # Drop missing values

    # Call add_features() before splitting data
    data = add_features(data)

    # Split into training and testing sets
    train_data, test_data = train_test_split(data, test_size=0.2, shuffle=False)
    return train_data, test_data, data, meta_data

def get_live_data():
    api_key = os.getenv('APP_VANTAGE_API_KEY')
    # Initialize the TimeSeries object
    ts = TimeSeries(key=api_key, output_format='pandas')

    # Retrieve historical data for a stock symbol
    stock_symbol = 'AAPL'  # Replace with the desired stock symbol
    data, meta_data = ts.get_daily(symbol=stock_symbol, outputsize='full')


    # Convert the index to datetime format (if it's not already)
    data['date'] = pd.to_datetime(data.index)
    # Set the date column as the index
    data = data.set_index('date')
    # Handle missing values
    data = data.dropna()
    data = data.sort_index(ascending=True)  # Ensure chronological order

    print("data => ", data)

    # Create a new feature 'daily_return'
    data['daily_return'] = data['4. close'].pct_change()

    # Call add_features() before splitting data
    data = add_features(data)

    # Split the data into training and testing sets
    train_data, test_data = train_test_split(data, test_size=0.2, shuffle=False)
    print("train data => ", train_data)
    print("test data => ", test_data)
    return train_data, test_data, data, meta_data

# get_live_data()

def add_features(data):
    # Add moving averages
    data['sma_5'] = data['4. close'].rolling(window=5).mean()
    data['sma_20'] = data['4. close'].rolling(window=20).mean()
    
    # Add volatility
    data['volatility'] = data['4. close'].rolling(window=5).std()

    # Compute residuals
    data['residual_high'] = data['2. high'] - data['sma_5']
    data['residual_low'] = data['3. low'] - data['sma_5']
    data['residual_close'] = data['4. close'] - data['sma_5']
    
    # Fill missing values created by rolling operations
    data = data.dropna()
    return data