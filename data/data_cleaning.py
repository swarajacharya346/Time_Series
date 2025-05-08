import pandas as pd
import os

# Load the raw data
raw_file_path = 'data/raw_data/AAPL_raw_data.csv'
df = pd.read_csv(raw_file_path)

# Rename columns for simplicity
df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']

# Convert 'Date' to datetime format and sort
df['Date'] = pd.to_datetime(df['Date'])
df.sort_values('Date', inplace=True)

# Reset index
df.reset_index(drop=True, inplace=True)

# Save cleaned data
os.makedirs('data/processed_data', exist_ok=True)
df.to_csv('data/processed_data/AAPL_cleaned_data.csv', index=False)

print("✅ Cleaned data saved to data/processed_data/AAPL_cleaned_data.csv")
