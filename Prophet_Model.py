import pandas as pd
from prophet import Prophet

def run_prophet_forecast(ticker, data_path, forecast_days=30):
    df = pd.read_csv(data_path, parse_dates=['Date'])
    df_prophet = df[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})

    model = Prophet()
    model.fit(df_prophet)

    future = model.make_future_dataframe(periods=forecast_days)
    forecast = model.predict(future)

    # Save relevant forecast data
    forecast_save = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    save_path = f"data/forecasted_data/{ticker}_prophet_forecast.csv"
    forecast_save.to_csv(save_path, index=False)

    print(f"✅ Prophet forecast saved to {save_path}")
    return forecast
