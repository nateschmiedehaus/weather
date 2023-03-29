import os
import json
import pandas as pd
import numpy as np
import datetime
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from flask import Flask, render_template, request, redirect, url_for
from auth import auth, login_manager, User
import folium
from folium.plugins import HeatMap
import requests
from statsmodels.tsa.arima.model import ARIMA

app = Flask(__name__)


state_list = [
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"
]


# Fetch data from Facebook Ads API
def fetch_facebook_ads_data(date):
    access_token = "your_facebook_access_token"
    ad_account_id = "your_facebook_ad_account_id"
    base_url = f"https://graph.facebook.com/v13.0/{ad_account_id}/insights"

    params = {
        "access_token": access_token,
        "date_preset": "yesterday",
        "level": "campaign",
        "fields": "spend, cpm, cpc",
    }

    response = requests.get(base_url, params=params)
    data = response.json()

    # Process the data
    cpm_data = {}  # {state: value}
    cpc_data = {}  # {state: value}
    ad_spend_data = {}  # {state: value}

    # Assume data contains the state abbreviation in campaign name
    for campaign in data["data"]:
        state = campaign["name"][-2:]
        if state in state_list:
            cpm_data[state] = campaign["cpm"]
            cpc_data[state] = campaign["cpc"]
            ad_spend_data[state] = campaign["spend"]

    return cpm_data, cpc_data, ad_spend_data


# Fetch data from Shopify API
def fetch_shopify_conversion_data(date):
    api_key = "your_shopify_api_key"
    password = "your_shopify_password"
    domain = "your_shopify_domain"
    base_url = f"https://{api_key}:{password}@{domain}.myshopify.com/admin/api/2022-04/orders.json"

    start_date = date + "T00:00:00"
    end_date = date + "T23:59:59"
    params = {
        "created_at_min": start_date,
        "created_at_max": end_date,
        "status": "any",
        "limit": 250,
    }

    response = requests.get(base_url, params=params)
    data = response.json()

    # Process the data
    conversion_data = {}  # {state: value}

    for order in data["orders"]:
        state = order["shipping_address"]["province_code"]
        if state not in conversion_data:
            conversion_data[state] = 1
        else:
            conversion_data[state] += 1

    return conversion_data


# Fetch data from NOAA API
# Fetch data from NOAA API
def fetch_weather_data(date):
    api_key = "YOUR_API_KEY"
    base_url = "https://www.ncei.noaa.gov/access/services/data/v1"

    headers = {
        "token": api_key
    }

    # Replace 'datasetid' and 'stationid' with appropriate values for your use case
    params = {
        "dataset": "daily-summaries",
        "dataTypes": "TMAX,PRCP",
        "stations": "GHCND:USW00014735",  # Replace with the correct station ids for each state
        "startDate": date,
        "endDate": date,
        "format": "json",
        "includeAttributes": "false",
        "includeStationName": "true",
        "includeStationLocation": "true",
        "units": "standard",
        "limit": 1000
    }

    response = requests.get(base_url, headers=headers, params=params)
    data = response.json()

    # Process the data
    temperature_data = {}  # {state: value}
    weather_condition_data = {}  # {state: value}
    precipitation_data = {}  # {state: value}

    for record in data:
        state = "STATE_ABBREVIATION"  # Replace with the correct state abbreviation
        if record['datatype'] == 'TMAX':
            temperature_data[state] = record['value']
        elif record['datatype'] == 'PRCP':
            precipitation_data[state] = record['value']

    # Calculate weather_condition_data based on temperature_data and precipitation_data
    for state, temp in temperature_data.items():
        if temp > 60 and precipitation_data[state] < 0.1:
            weather_condition_data[state] = 'sunny'
        elif temp <= 32 and precipitation_data[state] >= 0.1:
            weather_condition_data[state] = 'snowy'
        else:
            weather_condition_data[state] = 'cloudy'

    return temperature_data, weather_condition_data, precipitation_data

# Create ARIMA model for weather variables
def generate_arima_forecasts(data, order=(2, 1, 0)):
    model = ARIMA(data, order=order)
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=1)
    return forecast[0]


# Fetch data from BLS API
def fetch_cpi_data(date):
    api_key = "YOUR_API_KEY"
    series_id = "CUUR0000SA0"  # All items in the U.S. city average, all urban consumers, not seasonally adjusted
    base_url = "https://api.bls.gov/publicAPI/v2/timeseries/data/"

    date_object = datetime.datetime.strptime(date, "%Y-%m-%d")
    year = date_object.year
    month = date_object.month

    start_year = year - 1
    end_year = year

    headers = {
        "Content-type": "application/json"
    }

    data = json.dumps({
        "seriesid": [series_id],
        "startyear": str(start_year),
        "endyear": str(end_year),
        "registrationkey": api_key
    })

    response = requests.post(base_url, headers=headers, data=data)
    result = response.json()

    if result['status'] == 'REQUEST_SUCCEEDED':
        cpi_data_national = None
        for data_point in result['Results']['series'][0]['data']:
            if int(data_point['year']) == year and int(data_point['period'][1:]) == month:
                cpi_data_national = float(data_point['value'])
                break
        if cpi_data_national is not None:
            cpi_data = {state: cpi_data_national for state in state_list}
        else:
            cpi_data = {}  # {state: value}
    else:
        cpi_data = {}

    return cpi_data

# Prepare data for model training
def prepare_data(date):
    cpm_data, cpc_data, ad_spend_data = fetch_facebook_ads_data(date)
    conversion_data = fetch_shopify_conversion_data(date)
    temperature_data, weather_condition_data, precipitation_data = fetch_weather_data(date)
    cpi_data = fetch_cpi_data(date)

    # Add ARIMA forecasts for temperature and precipitation
    temperature_forecasts = {state: generate_arima_forecasts([temp_data]) for state, temp_data in temperature_data.items()}
    precipitation_forecasts = {state: generate_arima_forecasts[(precip_data)] for state, precip_data in precipitation_data.items()}


     # Combine the data into a single DataFrame
    data = pd.DataFrame(
        {
            "CPM": cpm_data,
            "CPC": cpc_data,
            "Ad Spend": ad_spend_data,
            "Conversion": conversion_data,
            "Temperature": temperature_data,
            "Weather Condition": weather_condition_data,
            "Precipitation": precipitation_data,
            "Temperature Forecast": temperature_forecasts,
            "Precipitation Forecast": precipitation_forecasts,
            "CPI": cpi_data,
        }
    )

    return data

# Create machine learning model and generate recommendations
def generate_recommendations(date, roas, total_ad_spend):
    data = prepare_data(date)
    features = ["CPM", "CPC", "Ad Spend", "Temperature", "Precipitation", "CPI"]
    target = "Conversion"

    # Train the model
    model = LinearRegression()
    model.fit(data[features], data[target])

    # Generate predictions for each state
    data['Predicted Conversion'] = model.predict(data[features])

    # Calculate the estimated revenue based on the ad spend
    data['Estimated Revenue'] = data['Predicted Conversion'] * total_ad_spend

    # Rank states based on predicted conversion and ROAS
    data['ROAS'] = data['Estimated Revenue'] / data['Ad Spend']
    data['Rank'] = data['ROAS'].rank(ascending=False)

    # Filter states based on the ROAS requirements
    recommended_states = data[data['ROAS'] >= roas].sort_values(by='Rank', ascending=True)

    # Calculate the regression equation
    coefficients = model.coef_
    intercept = model.intercept_
    regression_equation = f"Conversion = {intercept:.2f} + {coefficients[0]:.2f}(CPM) + {coefficients[1]:.2f}(CPC) + {coefficients[2]:.2f}(Ad Spend) + {coefficients[3]:.2f}(Temperature) + {coefficients[4]:.2f}(Precipitation) + {coefficients[5]:.2f}(CPI)"


    return recommended_states, regression_equation



# Rank states based on population and filter by recommended states
def rank_states_by_population(recommended_states):
    # Population data (replace with the actual population data)
    population_data = {
        "AL": 4903185,
        "AK": 731545,
        "AZ": 7278717,
        # ...
    }

    # Rank states based on population
    recommended_states["Population"] = recommended_states.index.map(population_data)
    recommended_states["Population Rank"] = recommended_states["Population"].rank(ascending=False)

    # Sort by population rank
    recommended_states = recommended_states.sort_values(by="Population Rank", ascending=True)

    return recommended_states

def generate_heat_map(ranked_states):
    # Replace with your own coordinates for the center of the United States
    center_lat, center_lon = 39.8283, -98.5795

    # Create a folium Map object
    m = folium.Map(location=[center_lat, center_lon], zoom_start=4)

    # Prepare heatmap data
    heatmap_data = []
    for _, row in ranked_states.iterrows():
        lat, lon = 0, 0  # Replace with the actual latitude and longitude of the state
        ad_spend = row['Ad Spend']
        heatmap_data.append([lat, lon, ad_spend])

    # Add the HeatMap layer to the folium Map object
    HeatMap(heatmap_data).add_to(m)

    # Save the heatmap as an HTML file
    m.save('templates/heatmap.html')

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        selected_date = request.form["date"]
        data = prepare_data(selected_date)
        model = LinearRegression()
        X = data.drop("Conversion", axis=1)
        y = data["Conversion"]
        model.fit(X, y)
        predictions = model.predict(X)
        mae = mean_absolute_error(y, predictions)

        # Create a folium map with heat map
        heat_map_data = [
            (row["latitude"], row["longitude"], row["Conversion"])
            for index, row in data.iterrows()
        ]
        folium_map = folium.Map(location=[38.9, -77.0], zoom_start=4)
        HeatMap(heat_map_data).add_to(folium_map)

        # Save the map as an HTML file
        folium_map.save("templates/map.html")

        return render_template("results.html", mae=mae, date=selected_date)
    return render_template("index.html")

@app.route("/map")
def map():
    return render_template("map.html")


app = Flask(__name__)
app.secret_key = '12345'  # Replace with a secure secret key

login_manager.init_app(app)
app.register_blueprint(auth)

if __name__ == "__main__":
    app.run(debug=True)