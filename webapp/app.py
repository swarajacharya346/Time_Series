from flask import Flask, render_template, request, redirect, url_for, session, flash
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import os

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # required for session

tickers = {
    'Apple': 'AAPL',
    'Tesla': 'TSLA',
    'Amazon': 'AMZN',
    'Google': 'GOOG',
    'Microsoft': 'MSFT'
}

models = ['ARIMA', 'SARIMA', 'Prophet', 'LSTM']
plot_types = ['line', 'bar', 'pie', 'histogram', 'area']

# Simple user store for demo (username: password)
users = {
    'username': 'password'
}

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

def plot_data(df, plot_type, title, x_col=None, y_col=None):
    plt.clf()
    if x_col is None:
        x_col = df.columns[0]
    if y_col is None:
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        if x_col in numeric_cols:
            numeric_cols.remove(x_col)
        y_col = numeric_cols[0] if numeric_cols else df.columns[1]

    if plot_type == 'line':
        df.plot(x=x_col, y=y_col)
    elif plot_type == 'bar':
        df.plot.bar(x=x_col, y=y_col)
    elif plot_type == 'pie':
        df[y_col].plot.pie(autopct='%1.1f%%')
    elif plot_type == 'histogram':
        df[y_col].plot.hist()
    elif plot_type == 'area':
        df.plot.area(x=x_col, y=y_col)
    else:
        df.plot(x=x_col, y=y_col)

    plt.title(title)
    plt.tight_layout()

    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close()
    return plot_url

# ------------------- AUTH ROUTES -------------------

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if username in users:
            flash('Username already exists! Try another.', 'danger')
        else:
            users[username] = password
            flash('Signup successful! Please login.', 'success')
            return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if username in users and users[username] == password:
            session['username'] = username
            flash('Logged in successfully!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Invalid username or password.', 'danger')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    flash('Logged out.', 'info')
    return redirect(url_for('home'))

# ------------------- MAIN ROUTES -------------------

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/data', methods=['GET', 'POST'])
def data_page():
    if 'username' not in session:
        flash('Please login to access data.', 'warning')
        return redirect(url_for('login'))

    table_data = None
    plot_url = None
    company = None
    if request.method == 'POST':
        company = request.form.get('company')
        df = get_cleaned_data(company)
        if df is not None:
            x_col = df.columns[0]
            numeric_cols = df.select_dtypes(include='number').columns.tolist()
            if x_col in numeric_cols:
                numeric_cols.remove(x_col)
            y_col = numeric_cols[0] if numeric_cols else df.columns[1]

            table_data = df.head(10).to_dict(orient='records')
            plot_url = plot_data(df[[x_col, y_col]], 'line', f"{company} Cleaned {y_col}")
    return render_template('data.html', tickers=tickers.keys(), table_data=table_data, plot_url=plot_url, company=company)

@app.route('/predicted', methods=['GET', 'POST'])
def predicted_page():
    if 'username' not in session:
        flash('Please login to access predicted data.', 'warning')
        return redirect(url_for('login'))

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
    if 'username' not in session:
        flash('Please login to access visualization.', 'warning')
        return redirect(url_for('login'))

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
            x_col = df.columns[0]
            numeric_cols = df.select_dtypes(include='number').columns.tolist()
            if x_col in numeric_cols:
                numeric_cols.remove(x_col)
            y_col = numeric_cols[0] if numeric_cols else df.columns[1]
            plot_url = plot_data(df, plot_type, f"{company} {model} Forecast ({plot_type})", x_col, y_col)
    return render_template('visualization.html', tickers=tickers.keys(), models=models, plot_types=plot_types, plot_url=plot_url, company=company, model=model, plot_type=plot_type)

if __name__ == '__main__':
    app.run(debug=True)
