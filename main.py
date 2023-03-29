import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from flask import Flask, render_template, request
import facebook
from shopify import Shopify
import requests
import json
import pickle
import datetime

# Define function to retrieve daily CPM, CPC, and Cost per acquisition from Facebook API
def get_facebook_ad_metrics(access_token, state):
    params = {
        'date_preset': 'last_7_days',
        'level': 'region',
        'region': state,
        'fields': 'account_id,region,cost_per_acquisition,cpm,cpc'
    }
    graph = facebook.GraphAPI(access_token=access_token, version="3.0")
    ad_insights = graph.get_connections(id='me', connection_name='ad_insights', **params)
    return ad_insights

# Retrieve daily CPM, CPC, and Cost per acquisition from Facebook API for your account by state
access_token = 'YOUR_FACEBOOK_ACCESS_TOKEN' # Replace with your own access token
states = ['Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado', 'Connecticut', 'Delaware', 'Florida', 'Georgia', 'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Kentucky', 'Louisiana', 'Maine', 'Maryland', 'Massachusetts', 'Michigan', 'Minnesota', 'Mississippi', 'Missouri', 'Montana', 'Nebraska', 'Nevada', 'New Hampshire', 'New Jersey', 'New Mexico', 'New York', 'North Carolina', 'North Dakota', 'Ohio', 'Oklahoma', 'Oregon', 'Pennsylvania', 'Rhode Island', 'South Carolina', 'South Dakota', 'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington', 'West Virginia', 'Wisconsin', 'Wyoming']

results = pd.DataFrame(columns=['State', 'ROAS', 'Weather Condition'])

for state in states:
    # Use your model to predict ROAS and weather for this state
    roas, weather_condition = your_model.predict(state)
    
    # Add this information to the 'results' table
    results = results.append({'State': state, 'ROAS': roas, 'Weather Condition': weather_condition}, ignore_index=True)

total_roas = results['ROAS'].sum()

results['Ad Spend Percentage'] = results['ROAS'] / total_roas

total_budget = user_provided_budget
results['Ad Spend'] = results['Ad Spend Percentage'] * total_budget

# Define function to retrieve NOAA weather data
def get_noaa_weather_data(api_key, state, date):
    url = f'https://www.ncdc.noaa.gov/cdo-web/api/v2/data?datasetid=GHCND&datatypeid=TMAX,TMIN,PRCP,SNOW,SNWD,AWND,AWND_ATTRIBUTES,WT01,WT02,WT03,WT04,WT05,WT06,WT07,WT08,WT09,WT10,WT11,WT13,WT14,WT15,WT16,WT17,WT18,WT19,WT21,WT22&units=metric&startdate={date}&enddate={date}&locationid=STATE:{state}&limit=1000'
    headers = {'token': api_key}
    response = requests.get(url, headers=headers)
    data = json.loads(response.text)['results']
    df = pd.DataFrame(data)
    if len(df) == 0:
        return None
    df['date'] = pd.to_datetime(df['date'])
    df['datatype'] = df['datatype'].str.lower()
    df = df.pivot(index='date', columns='datatype', values='value')
    df.columns.name = None
    df.index.name = 'date'
    return df


# Define start and end dates for Shopify sales data
start_date = '2018-01-01'  # update to earliest date with available data
end_date = datetime.date.today().strftime('%Y-%m-%d')

# Retrieve daily CPM, CPC, and Cost per acquisition from Facebook API for your account by state
access_token = 'YOUR_FACEBOOK_ACCESS_TOKEN' # Replace with your own access token
states = ['Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado', 'Connecticut', 'Delaware', 'Florida', 'Georgia', 'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Kentucky', 'Louisiana', 'Maine', 'Maryland', 'Massachusetts', 'Michigan', 'Minnesota', 'Mississippi', 'Missouri', 'Montana', 'Nebraska', 'Nevada', 'New Hampshire', 'New Jersey', 'New Mexico', 'New York', 'North Carolina', 'North Dakota', 'Ohio', 'Oklahoma', 'Oregon', 'Pennsylvania', 'Rhode Island', 'South Carolina', 'South Dakota', 'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington', 'West Virginia', 'Wisconsin', 'Wyoming']
ad_metrics = {}
for state in states:
    ad_metrics[state] = get_facebook_ad_metrics(access_token, state)

# Retrieve API Sales and conversion rates per day from Shopify API for the specified date range
api_key = 'YOUR_SHOPIFY_API_KEY' # Replace with your own API key
password = 'YOUR_SHOPIFY_API_PASSWORD' # Replace with your own API password
shop_url = 'YOUR_SHOPIFY_STORE_URL' # Replace with your own store URL
shopify = Shopify(api_key, password, shop_url)
sales = {}
conversion_rates = {}

# Combine all the data into a single DataFrame
data = pd.DataFrame(columns=['state', 'date', 'cpm', 'cpc', 'cost_per_acquisition', 'sales', 'conversion_rate', 'tmax', 'tmin', 'prcp', 'snow', 'snwd', 'awnd', 'awnd_attributes', 'wt01', 'wt02', 'wt03', 'wt04', 'wt05', 'wt06', 'wt07', 'wt08', 'wt09', 'wt10', 'wt11', 'wt13', 'wt14', 'wt15', 'wt16', 'wt17', 'wt18', 'wt19', 'wt21', 'wt22'])
for state in states:
    ad_data = pd.DataFrame(ad_metrics[state])
    ad_data.columns = ad_data.columns.str.lower()
    ad_data = ad_data[['region', 'date_start', 'cpm', 'cpc', 'cost_per_acquisition']]
    ad_data['region'] = state
    ad_data = ad_data.rename(columns={'region': 'state', 'date_start': 'date'})
    sales_data = pd.DataFrame({'sales': [sales[state]], 'conversion_rate': [conversion_rates[state]]})
    sales_data['state'] = state
    sales_data['date'] = end_date
    weather_data = get_noaa_weather_data(noaa_api_key, state, end_date)
    if weather_data is not None:
        weather_data['state'] = state
        weather_data = weather_data.reset_index()
        data = pd.concat([data, pd.merge(ad_data, sales_data, on=['state', 'date']), weather_data], axis=1)

# Create regression model
model = LinearRegression()
X = data[['cpm', 'cpc', 'cost_per_acquisition', 'tmax', 'tmin', 'prcp', 'snow', 'snwd', 'awnd']]
y = data['sales']
model.fit(X, y)

# Define your NOAA API key
noaa_api_key = 'YOUR_NOAA_API_KEY'

# Define function to load the saved model
def load_model(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

# Define function to retrieve CPI data for a given state and date


def get_cpi_data(state, date):
    cpi_data = pd.read_csv('https://download.bls.gov/pub/time.series/cu/cu.data.1.AllItems', sep='\t', usecols=['series_id', 'year', 'period', 'value'], dtype={'series_id': str})
    cpi_data['period'] = cpi_data['period'].str.lower()
    cpi_data['date'] = pd.to_datetime(cpi_data['year'].astype(str) + cpi_data['period'], format='%Y%B')
    cpi_data = cpi_data[cpi_data['series_id'].str.contains('CUUR' + state.upper() + '1')]
    cpi_data = cpi_data[cpi_data['date'] == date]
    cpi_data = cpi_data.rename(columns={'value': 'all_items'})
    return cpi_data[['all_items']]

# Define scaler object to scale inputs to the model
scaler = joblib.load('scaler.pkl')

# Define Flask app and route for predicting sales
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve inputs from form
    state = request.form['state']
    date = request.form['date']
    ad_spend = float(request.form['ad_spend'])
    # Get weather data and CPI data for state and date
    weather_data = get_noaa_weather_data(noaa_api_key, state, date)
    cpi_data = get_cpi_data(state, date)
    # Calculate predicted sales
    data = pd.DataFrame(columns=['state', 'date', 'cpm', 'cpc', 'cost_per_acquisition', 'sales', 'conversion_rate', 'tmax', 'tmin', 'prcp', 'snow', 'snwd', 'awnd', 'awnd_attributes', 'wt01', 'wt02', 'wt03', 'wt04', 'wt05', 'wt06', 'wt07', 'wt08', 'wt09', 'wt10', 'wt11', 'wt13', 'wt14', 'wt15', 'wt16', 'wt17', 'wt18', 'wt19', 'wt21', 'wt22'])
    ad_metrics = get_facebook_ad_metrics(access_token, state)
    ad_data = pd.DataFrame(ad_metrics)
    ad_data.columns = ad_data.columns.str.lower()
    ad_data['state'] = state
    ad_data['date'] = date
    ad_data['conversion_rate'] = get_shopify_sales_and_conversion_rates(shopify_api_key, shopify_password, shopify_store_url, date, date)[1]
    ad_data = ad_data[['state', 'date', 'cpm', 'cpc', 'cost_per_acquisition', 'conversion_rate']]
    data = data.append(ad_data, ignore_index=True)
    data['date'] = pd.to_datetime(data['date'])
    data = pd.merge(data, cpi_data, how='left', on=['state', 'date'])
    data = pd.merge(data, weather_data, how='left', on=['state', 'date'])
    data = data.fillna(data.mean())
    # Scale inputs using scaler
    inputs_scaled = scaler.transform(data.drop(columns=['state', 'date', 'sales']))
    # Load model and calculate predicted sales
    model = load_model('sales_prediction_model.pkl')
    X = pd.DataFrame(inputs_scaled, columns=data.columns[2:-35])
    y = data['sales']
    sales_pred = model.predict(np.append(inputs_scaled, [[ad_spend]], axis=1))
    # Calculate confidence interval
    X_with_ad_spend = np.append(inputs_scaled, [[ad_spend]], axis=1)
    y_pred = model.predict(X_with_ad_spend)
    se = np.sqrt(np.sum((y - model.predict(X)) ** 2) / (len(X) - len(X.columns) - 1))
    t_value = stats.t.ppf(1 - 0.05 / 2, len(X) - len(X.columns) - 1)
    conf_interval = (y_pred - t_value * se, y_pred + t_value * se)
    conf_interval_str = '${:,.2f} - ${:,.2f}'.format(conf_interval[0][0], conf_interval[1][0])
    # Render results template with predicted sales and confidence interval
    return render_template('results.html', state=state, date=date, ad_spend='${:,.2f}'.format(ad_spend), sales_pred='${:,.2f}'.format(sales_pred[0]), conf_interval=conf_interval_str)


    if weather_data is not None:
        weather_data['state'] = state
        weather_data = weather_data.reset_index()
        data = pd.concat([data, pd.merge(ad_data, sales_data, on=['state', 'date']), weather_data], axis=1)
    data = data.fillna(0)
    # Add CPI data to input
    cpi = cpi_data['all_items']
    inputs = pd.DataFrame({'cpm': [data['cpm'].mean()], 'cpc': [data['cpc'].mean()], 'cost_per_acquisition': [data['cost_per_acquisition'].mean()], 'tmax': [data['tmax'].values[0]], 'tmin': [data['tmin'].values[0]], 'prcp': [data['prcp'].values[0]], 'snow': [data['snow'].values[0]], 'snwd': [data['snwd'].values[0]], 'awnd': [data['awnd'].values[0]], 'cpi': [cpi]})
    # Scale inputs and predict sales
    inputs_scaled = scaler.transform(inputs)
    sales_pred = model.predict(np.append(inputs_scaled, [[ad_spend]], axis=1))[0]
    # Calculate confidence interval

    # Calculate confidence interval
    X_with_ad_spend = np.append(inputs_scaled, [[ad_spend]], axis=1)
    y_pred = model.predict(X_with_ad_spend)
    se = np.sqrt(np.sum((y - model.predict(X)) ** 2) / (len(X) - len(X.columns) - 1))
    t_value = stats.t.ppf(1 - 0.05 / 2, len(X) - len(X.columns) - 1)
    conf_interval = (y_pred - t_value * se, y_pred + t_value * se)
    conf_interval_str = '${:,.2f} - ${:,.2f}'.format(conf_interval[0][0], conf_interval[1][0])
    # Render results template with predicted sales and confidence interval
    return render_template('results.html', state=state, date=date, ad_spend='${:,.2f}'.format(ad_spend), sales_pred='${:,.2f}'.format(sales_pred), conf_interval=conf_interval_str, data=data.to_html(classes='table table-striped'), map_svg=map_svg, state_charts=state_charts, results=results)
