import os
import pandas as pd
import numpy as np
from keras.models import Sequential  # type: ignore
from keras.layers import LSTM, Dense  # type: ignore
from sklearn.preprocessing import MinMaxScaler

def forecast_lstm(df, symbol):
    os.makedirs('data/forecasted_data', exist_ok=True)
    
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

if __name__ == "__main__":
    test_path = 'data/processed_data/AAPL_cleaned_data.csv'
    if os.path.exists(test_path):
        df = pd.read_csv(test_path, parse_dates=['Date'])
        forecast_lstm(df, 'AAPL')
