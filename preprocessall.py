import os
import pandas as pd
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from keras.models import Sequential  # type: ignore
from keras.layers import LSTM, Dense  # type: ignore
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Your tickers with symbols
tickers = {
    'Apple': 'AAPL',
    'Tesla': 'TSLA',
    'Amazon': 'AMZN',
    'Google': 'GOOG',
    'Microsoft': 'MSFT'
}

def forecast_prophet(df, symbol):
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

def forecast_arima(df, symbol):
    model = ARIMA(df['Close'], order=(5, 1, 0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=30)
    dates = pd.date_range(df['Date'].iloc[-1], periods=31, freq='B')[1:]
    result = pd.DataFrame({'Date': dates, 'Forecast': forecast})
    path = f'data/forecasted_data/{symbol}_arima_forecast.csv'
    result.to_csv(path, index=False)
    print(f"📈 ARIMA forecast saved for {symbol}")

def forecast_sarima(df, symbol):
    model = SARIMAX(df['Close'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    model_fit = model.fit(disp=False)
    forecast = model_fit.forecast(steps=30)
    dates = pd.date_range(df['Date'].iloc[-1], periods=31, freq='B')[1:]
    result = pd.DataFrame({'Date': dates, 'Forecast': forecast})
    path = f'data/forecasted_data/{symbol}_sarima_forecast.csv'
    result.to_csv(path, index=False)
    print(f"📊 SARIMA forecast saved for {symbol}")

def forecast_lstm(df, symbol):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1, 1))
    X, y = [], []
    time_step = 60
    for i in range(time_step, len(scaled_data)):
        X.append(scaled_data[i - time_step:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=5, batch_size=32, verbose=0)

    forecast_input = scaled_data[-time_step:].reshape(1, time_step, 1)
    predictions = []
    for _ in range(30):
        next_pred = model.predict(forecast_input, verbose=0)
        predictions.append(next_pred[0, 0])
        forecast_input = np.append(forecast_input[:, 1:, :], next_pred.reshape(1, 1, 1), axis=1)

    forecast = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    dates = pd.date_range(df['Date'].iloc[-1], periods=31, freq='B')[1:]
    result = pd.DataFrame({'Date': dates, 'Forecast': forecast.flatten()})
    path = f'data/forecasted_data/{symbol}_lstm_forecast.csv'
    result.to_csv(path, index=False)
    print(f"🧠 LSTM forecast saved for {symbol}")

# --- RUN FORECASTS ---

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
