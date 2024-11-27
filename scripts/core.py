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

    # Get the start and end dates of the test_data
    test_data_start_date = test_data.index.min()  # First date in test_data
    test_data_end_date = test_data.index.max()    # Last date in test_data

    return train_data, test_data, data, meta_data, test_data_start_date, test_data_end_date

def get_live_data(symbol: str = 'IBM'):
    """
    Fetches and prepares live stock data, including feature engineering
    and splitting into training and testing datasets.

    Args:
        symbol (str): Stock symbol to fetch data for (default: 'AAPL').

    Returns:
        tuple: train_data, test_data, full_data, meta_data, test_data_start_date, test_data_end_date
    """
    # Fetch API key from environment variables
    api_key = os.getenv('APP_VANTAGE_API_KEY')
    # api_key = "J66LKJPYTVPALNUA"
    if not api_key:
        raise ValueError("API key for Alpha Vantage not found in environment variables.")
    
    # Initialize the TimeSeries object
    ts = TimeSeries(key=api_key, output_format='pandas')

    # Retrieve historical data
    data, meta_data = ts.get_daily(symbol=symbol, outputsize='full')

     # Handle missing data
    data = data.dropna()

    # Ensure the index is in datetime format
    data['date'] = pd.to_datetime(data.index)
    data = data.set_index('date')
    data = data.sort_index(ascending=True)  # Ensure chronological order

    # Add a 'daily_return' feature
    data['daily_return'] = data['4. close'].pct_change()
    data = data.dropna()  # Drop missing values


    # Add additional features
    data = add_features(data)

    # Split into training and testing datasets
    train_data, test_data = train_test_split(data, test_size=0.2, shuffle=False)


    # Get the date range of the testing data
    test_data_start_date = test_data.index.min()  # First date in test_data
    test_data_end_date = test_data.index.max()    # Last date in test_data


    return train_data, test_data, data, meta_data, test_data_start_date, test_data_end_date

# get_live_data()

def add_features(data):
    """
    Adds technical indicators and other features to the stock data.

    Args:
        data (pd.DataFrame): Stock data with '2. high', '3. low', and '4. close' columns.

    Returns:
        pd.DataFrame: Data with additional features.
    """
    # Compute moving averages
    data['sma_5'] = data['4. close'].rolling(window=5).mean()
    data['sma_20'] = data['4. close'].rolling(window=20).mean()

    # Compute volatility (standard deviation of close prices over the last 5 days)
    data['volatility'] = data['4. close'].rolling(window=5).std()

    # Compute residuals for high, low, and close prices against the 5-day SMA
    data['residual_high'] = data['2. high'] - data['sma_5']
    data['residual_low'] = data['3. low'] - data['sma_5']
    data['residual_close'] = data['4. close'] - data['sma_5']

    # Handle missing values resulting from rolling operations
    data = data.dropna()
    return data