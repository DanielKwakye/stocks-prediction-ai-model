from alpha_vantage.timeseries import TimeSeries
import pandas as pd
import json

# Load simulated stock data from JSON
def get_simulated_data(json_path="src/simulated_data.json"):
    with open(json_path, "r") as f:
        mock_data = json.load(f)
    meta_data = mock_data["Meta Data"]
    time_series = mock_data["Time Series (Daily)"]

    # Convert time series data into a DataFrame
    data = pd.DataFrame.from_dict(time_series, orient="index").astype(float)
    data['date'] = pd.to_datetime(data.index)
    data = data.set_index('date')
    data = data.sort_index()  # Ensure chronological order

    # Add a 'daily_return' feature
    data['daily_return'] = data['4. close'].pct_change()
    data = data.dropna()  # Drop missing values

    # Split into training and testing sets
    train_data, test_data = train_test_split(data, test_size=0.2, shuffle=False)
    return train_data, test_data, data, meta_data