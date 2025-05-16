from flask import Flask, render_template, request, jsonify
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import os

app = Flask(__name__)

tickers = {
    'Apple': 'AAPL',
    'Tesla': 'TSLA',
    'Amazon': 'AMZN',
    'Google': 'GOOG',
    'Microsoft': 'MSFT'
}

models = ['ARIMA', 'SARIMA', 'Prophet', 'LSTM']
plot_types = ['line', 'bar', 'pie', 'histogram', 'area']

def get_cleaned_data(company):
    path = f"data/processed_data/{tickers[company]}_cleaned_data.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

def get_forecasted_data(company, model):
    path = f"data/forecasted_data/{tickers[company]}_{model.lower()}_forecasted_data.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

def plot_data(df, plot_type, title):
    plt.clf()
    if plot_type == 'line':
        df.plot(x=df.columns[0], y=df.columns[1])
    elif plot_type == 'bar':
        df.plot.bar(x=df.columns[0], y=df.columns[1])
    elif plot_type == 'pie':
        # pie chart for the first numeric column
        df[df.columns[1]].plot.pie(autopct='%1.1f%%')
    elif plot_type == 'histogram':
        df[df.columns[1]].plot.hist()
    elif plot_type == 'area':
        df.plot.area(x=df.columns[0], y=df.columns[1])
    else:
        df.plot(x=df.columns[0], y=df.columns[1])
    plt.title(title)
    plt.tight_layout()
    
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close()
    return plot_url

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/data', methods=['GET', 'POST'])
def data_page():
    table_data = None
    plot_url = None
    company = None
    if request.method == 'POST':
        company = request.form.get('company')
        df = get_cleaned_data(company)
        if df is not None:
            table_data = df.head(10).to_dict(orient='records')
            plot_url = plot_data(df[['Date', 'Close']], 'line', f"{company} Cleaned Close Price")
    return render_template('data.html', tickers=tickers.keys(), table_data=table_data, plot_url=plot_url, company=company)

@app.route('/predicted', methods=['GET', 'POST'])
def predicted_page():
    forecast_data = None
    company = None
    model = None
    if request.method == 'POST':
        company = request.form.get('company')
        model = request.form.get('model')
        df = get_forecasted_data(company, model)
        if df is not None:
            forecast_data = df.head(10).to_dict(orient='records')
    return render_template('predicted.html', tickers=tickers.keys(), models=models, forecast_data=forecast_data, company=company, model=model)

@app.route('/visualization', methods=['GET', 'POST'])
def visualization_page():
    plot_url = None
    company = None
    model = None
    plot_type = None
    if request.method == 'POST':
        company = request.form.get('company')
        model = request.form.get('model')
        plot_type = request.form.get('plot_type')
        df = get_forecasted_data(company, model)
        if df is not None:
            # For pie charts, ensure data is appropriate (here we assume Date and yhat)
            plot_url = plot_data(df[['ds', 'yhat']], plot_type, f"{company} {model} Forecast ({plot_type})")
    return render_template('visualization.html', tickers=tickers.keys(), models=models, plot_types=plot_types, plot_url=plot_url, company=company, model=model, plot_type=plot_type)

if __name__ == '__main__':
    app.run(debug=True)
