import os
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

def forecast_arima(df, symbol):
    os.makedirs('data/forecasted_data', exist_ok=True)
    
    model = ARIMA(df['Close'], order=(5, 1, 0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=30)
    
    dates = pd.date_range(df['Date'].iloc[-1], periods=31, freq='B')[1:]
    result = pd.DataFrame({'Date': dates, 'Forecast': forecast})
    
    path = f'data/forecasted_data/{symbol}_arima_forecast.csv'
    result.to_csv(path, index=False)
    print(f"📈 ARIMA forecast saved for {symbol}")

if __name__ == "__main__":
    test_path = 'data/processed_data/AAPL_cleaned_data.csv'
    if os.path.exists(test_path):
        df = pd.read_csv(test_path, parse_dates=['Date'])
        forecast_arima(df, 'AAPL')
