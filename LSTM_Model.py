import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential # type: ignore
from keras.layers import LSTM, Dense # type: ignore

def run_lstm_forecast(ticker, data_path, time_step=60, forecast_days=30, epochs=10, batch_size=32):
    df = pd.read_csv(data_path, parse_dates=['Date'])
    close_prices = df['Close'].values.reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_prices)

    X, y = [], []
    for i in range(time_step, len(scaled_data)):
        X.append(scaled_data[i-time_step:i, 0])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=1)

    # Predict next forecast_days
    last_sequence = scaled_data[-time_step:].reshape(1, time_step, 1)
    predictions = []
    for _ in range(forecast_days):
        pred = model.predict(last_sequence)
        predictions.append(pred[0, 0])
        last_sequence = np.append(last_sequence[:,1:,:], [[pred]], axis=1)

    # Inverse transform predictions
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

    forecast_dates = pd.date_range(df['Date'].iloc[-1], periods=forecast_days+1)[1:]
    forecast_df = pd.DataFrame({'Date': forecast_dates, 'Forecast_Close': predictions.flatten()})

    save_path = f"data/forecasted_data/{ticker}_lstm_forecast.csv"
    forecast_df.to_csv(save_path, index=False)

    print(f"✅ LSTM forecast saved to {save_path}")
    return forecast_df
