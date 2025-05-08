import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# Load cleaned data
df = pd.read_csv('data/processed_data/AAPL_cleaned_data.csv')

# Prepare data for Prophet
df_prophet = df[['Date', 'Close']]

# Rename column names

df_prophet.rename(columns={'Date': 'ds', 'Close': 'y'}, inplace=True)

# fit the Prophet model
model = Prophet()
model.fit(df_prophet)

# Prediction
future = model.make_future_dataframe(periods=365)
forecast = model.predict(future)

# Plot the forecast
fig = model.plot(forecast)
plt.title('Apple Stock Price Forecast')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.show()

# Optionally: Plot components (trend, yearly, weekly effects)
fig2 = model.plot_components(forecast)
plt.show()

# Save forecasted data
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv('data/processed_data/AAPL_forecasted_data.csv', index=False)
