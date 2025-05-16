import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

def run_arima_forecast(ticker, data_path, p=5, d=1, q=0, forecast_days=30):
    df = pd.read_csv(data_path, parse_dates=['Date'])
    df.set_index('Date', inplace=True)
    
    model = ARIMA(df['Close'], order=(p, d, q))
    model_fit = model.fit()
    
    forecast = model_fit.forecast(steps=forecast_days)
    
    forecast_df = forecast.reset_index()
    forecast_df.columns = ['Date', 'Forecast_Close']
    
    save_path = f"data/forecasted_data/{ticker}_arima_forecast.csv"
    forecast_df.to_csv(save_path, index=False)
    
    print(f"✅ ARIMA forecast saved to {save_path}")
    return forecast_df
