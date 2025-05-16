import pandas as pd
import os

def clean_stock_data(ticker):
    raw_file_path = f'data/raw_data/{ticker}_raw_data.csv'
    processed_dir = 'data/processed_data'
    os.makedirs(processed_dir, exist_ok=True)
    
    try:
        df = pd.read_csv(raw_file_path)
        # Rename columns if needed (handle cases where columns might already be named correctly)
        df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        df['Date'] = pd.to_datetime(df['Date'])
        df.sort_values('Date', inplace=True)
        df.reset_index(drop=True, inplace=True)
        
        cleaned_file_path = f'{processed_dir}/{ticker}_cleaned_data.csv'
        df.to_csv(cleaned_file_path, index=False)
        
        print(f"✅ Cleaned data saved to {cleaned_file_path}")
        return cleaned_file_path
    except Exception as e:
        print(f"❌ Error cleaning data for {ticker}: {e}")
        return None

# Example usage:
# clean_stock_data('AAPL')
