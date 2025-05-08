from alpha_vantage.timeseries import TimeSeries
import pandas as pd
import os

# Your Alpha Vantage API key (replace with your actual key)
api_key = 'KSV9IEGZEKYJ89LY'
symbol = 'AAPL'

# Create directory if it doesn't exist
os.makedirs('data/raw_data', exist_ok=True)

# Fetch daily stock data
ts = TimeSeries(key=api_key, output_format='pandas')
data, _ = ts.get_daily(symbol=symbol, outputsize='compact')

# Save to CSV
data.to_csv(f'data/raw_data/{symbol}_raw_data.csv')
print("✅ Data saved successfully!")
