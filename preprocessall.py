import os
import pandas as pd
from alpha_vantage.timeseries import TimeSeries
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# --- SETTINGS ---
api_key = 'KSV9IEGZEKYJ89LY'
tickers = {
    'Apple': 'AAPL',
    'Tesla': 'TSLA',
    'Amazon': 'AMZN',
    'Google': 'GOOG',
    'Microsoft': 'MSFT'
}

# --- DIRECTORIES ---
os.makedirs('data/raw_data', exist_ok=True)
os.makedirs('data/processed_data', exist_ok=True)
os.makedirs('data/forecasted_data', exist_ok=True)

# --- DATA COLLECTION ---
def collect_data(symbol):
    ts = TimeSeries(key=api_key, output_format='pandas')
    try:
        data, _ = ts.get_daily(symbol=symbol, outputsize='compact')
        raw_path = f'data/raw_data/{symbol}_raw_data.csv'
        data.to_csv(raw_path)
        print(f"✅ Collected data for {symbol}")
        return raw_path
    except Exception as e:
        print(f"❌ Failed to fetch {symbol}: {e}")
        return None

# --- DATA CLEANING ---
def clean_data(symbol):
    raw_path = f'data/raw_data/{symbol}_raw_data.csv'
    df = pd.read_csv(raw_path)
    df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values('Date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    cleaned_path = f'data/processed_data/{symbol}_cleaned_data.csv'
    df.to_csv(cleaned_path, index=False)
    print(f"🧼 Cleaned data for {symbol}")
    return df

# --- FORECASTING MODELS ---

def forecast_prophet(df, symbol):
    df_p = df[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
    model = Prophet()
    model.fit(df_p)
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    path = f'data/forecasted_data/{symbol}_prophet_forecasted_data.csv'
    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(path, index=False)
    print(f"🔮 Prophet model done for {symbol}")

def forecast_arima(df, symbol):
    model = ARIMA(df['Close'], order=(5, 1, 0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=30)
    dates = pd.date_range(df['Date'].iloc[-1], periods=31, freq='B')[1:]
    result = pd.DataFrame({'ds': dates, 'yhat': forecast})
    path = f'data/forecasted_data/{symbol}_arima_forecasted_data.csv'
    result.to_csv(path, index=False)
    print(f"📈 ARIMA model done for {symbol}")

def forecast_sarima(df, symbol):
    model = SARIMAX(df['Close'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    model_fit = model.fit(disp=False)
    forecast = model_fit.forecast(steps=30)
    dates = pd.date_range(df['Date'].iloc[-1], periods=31, freq='B')[1:]
    result = pd.DataFrame({'ds': dates, 'yhat': forecast})
    path = f'data/forecasted_data/{symbol}_sarima_forecasted_data.csv'
    result.to_csv(path, index=False)
    print(f"📊 SARIMA model done for {symbol}")

def forecast_lstm(df, symbol):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1, 1))
    X, y = [], []
    time_step = 60
    for i in range(time_step, len(scaled_data)):
        X.append(scaled_data[i-time_step:i, 0])
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
    result = pd.DataFrame({'ds': dates, 'yhat': forecast.flatten()})
    path = f'data/forecasted_data/{symbol}_lstm_forecasted_data.csv'
    result.to_csv(path, index=False)
    print(f"🧠 LSTM model done for {symbol}")

# --- MAIN WORKFLOW ---
for company, symbol in tickers.items():
    print(f"\n🚀 Starting for {company} ({symbol})")
    raw = collect_data(symbol)
    if raw:
        df = clean_data(symbol)
        forecast_prophet(df, symbol)
        forecast_arima(df, symbol)
        forecast_sarima(df, symbol)
        forecast_lstm(df, symbol)
print("\n✅ All companies processed!")
