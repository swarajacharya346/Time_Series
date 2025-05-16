from alpha_vantage.timeseries import TimeSeries
import pandas as pd
import os

# Your API key here (make sure to keep it safe and private)
API_KEY = 'KSV9IEGZEKYJ89LY'

# Company to ticker mapping
tickers = {
    'Apple': 'AAPL',
    'Tesla': 'TSLA',
    'Amazon': 'AMZN',
    'Google': 'GOOG',
    'Microsoft': 'MSFT'
}

def fetch_alpha_vantage_data(company_name, outputsize='compact'):
    symbol = tickers.get(company_name)
    if not symbol:
        raise ValueError(f"Company '{company_name}' not supported.")
    
    # Create directory if not exists
    os.makedirs('data/raw_data', exist_ok=True)

    ts = TimeSeries(key=API_KEY, output_format='pandas')
    try:
        print(f"Fetching data for {company_name} ({symbol}) from Alpha Vantage...")
        data, _ = ts.get_daily(symbol=symbol, outputsize=outputsize)
        data = data.rename_axis('Date').reset_index()
        # Alpha Vantage returns columns with spaces and different casing, fix columns to consistent naming
        data.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        data['Date'] = pd.to_datetime(data['Date'])
        # Save CSV
        filepath = f'data/raw_data/{symbol}_raw_data.csv'
        data.to_csv(filepath, index=False)
        print(f"✅ Data saved successfully at {filepath}")
        return filepath
    except Exception as e:
        print(f"❌ Failed to fetch data: {e}")
        return None

# Example usage:
# fetch_alpha_vantage_data('Apple', outputsize='compact')
