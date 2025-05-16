import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

def run_sarima_forecast(ticker, data_path, p=1, d=1, q=1, seasonal_order=(1,1,1,12), forecast_days=30):
    df = pd.read_csv(data_path, parse_dates=['Date'])
    df.set_index('Date', inplace=True)
    
    model = SARIMAX(df['Close'], order=(p, d, q), seasonal_order=seasonal_order)
    model_fit = model.fit(disp=False)
    
    forecast = model_fit.forecast(steps=forecast_days)
    
    forecast_df = forecast.reset_index()
    forecast_df.columns = ['Date', 'Forecast_Close']
    
    save_path = f"data/forecasted_data/{ticker}_sarima_forecast.csv"
    forecast_df.to_csv(save_path, index=False)
    
    print(f"✅ SARIMA forecast saved to {save_path}")
    return forecast_df
