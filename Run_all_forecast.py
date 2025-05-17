import os
import pandas as pd
from Prophet_Model import forecast_prophet
from Arima_Model import forecast_arima
from Sarima_Model import forecast_sarima
from LSTM_Model import forecast_lstm

tickers = {
    'Apple': 'AAPL',
    'Tesla': 'TSLA',
    'Amazon': 'AMZN',
    'Google': 'GOOG',
    'Microsoft': 'MSFT'
}

if __name__ == "__main__":
    for company, symbol in tickers.items():
        file_path = f'data/processed_data/{symbol}_cleaned_data.csv'
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, parse_dates=['Date'])
            print(f"\n🚀 Running forecasts for {company} ({symbol})")
            forecast_prophet(df, symbol)
            forecast_arima(df, symbol)
            forecast_sarima(df, symbol)
            forecast_lstm(df, symbol)
        else:
            print(f"❌ Cleaned data file not found for {symbol} at {file_path}")

    print("\n✅ All forecasts completed!")