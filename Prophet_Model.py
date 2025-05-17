import os
import pandas as pd
from prophet import Prophet

def forecast_prophet(df, symbol):
    os.makedirs('data/forecasted_data', exist_ok=True)
    
    df_p = df[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
    model = Prophet()
    model.fit(df_p)
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    
    forecast_clean = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
    forecast_clean.rename(columns={
        'ds': 'Date',
        'yhat': 'Forecast',
        'yhat_lower': 'Forecast_Lower',
        'yhat_upper': 'Forecast_Upper'
    }, inplace=True)
    
    path = f'data/forecasted_data/{symbol}_prophet_forecast.csv'
    forecast_clean.to_csv(path, index=False)
    print(f"🔮 Prophet forecast saved for {symbol}")

if __name__ == "__main__":
    # For quick testing
    test_path = 'data/processed_data/AAPL_cleaned_data.csv'
    if os.path.exists(test_path):
        df = pd.read_csv(test_path, parse_dates=['Date'])
        forecast_prophet(df, 'AAPL')
