import os
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

def forecast_sarima(df, symbol):
    os.makedirs('data/forecasted_data', exist_ok=True)
    
    model = SARIMAX(df['Close'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    model_fit = model.fit(disp=False)
    forecast = model_fit.forecast(steps=30)
    
    dates = pd.date_range(df['Date'].iloc[-1], periods=31, freq='B')[1:]
    result = pd.DataFrame({'Date': dates, 'Forecast': forecast})
    
    path = f'data/forecasted_data/{symbol}_sarima_forecast.csv'
    result.to_csv(path, index=False)
    print(f"📊 SARIMA forecast saved for {symbol}")

if __name__ == "__main__":
    test_path = 'data/processed_data/AAPL_cleaned_data.csv'
    if os.path.exists(test_path):
        df = pd.read_csv(test_path, parse_dates=['Date'])
        forecast_sarima(df, 'AAPL')
