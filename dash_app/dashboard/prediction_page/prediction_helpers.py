
import os
import sys
import json

# Append current directory to system path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import re
import gcsfs
import pytz
import requests
import pandas as pd
from requests.auth import HTTPBasicAuth
from math import radians, sin, cos, sqrt, atan2
from datetime import datetime
from dash import html
from timezonefinder import TimezoneFinder
from dotenv import load_dotenv
from google.oauth2 import service_account

# Loading environment variable with sensitive API keys
load_dotenv()

with open('/etc/secrets/GCP_CREDENTIALS', 'r') as f:
    credentials_info = json.loads(f.read())  # Read and parse the file contents
    credentials = service_account.Credentials.from_service_account_info(credentials_info,
                                                                        scopes=['https://www.googleapis.com/auth/devstorage.read_write',
                                                                                'https://www.googleapis.com/auth/cloud-platform',
                                                                                'https://www.googleapis.com/auth/drive'])

# Initialize Google Cloud Storage FileSystem
fs = gcsfs.GCSFileSystem(project='Flights-Weather-Project', token=credentials)

# Define weather-related features
weather_features = [
    'HourlyDryBulbTemperature', 'HourlyWindSpeed', 'HourlyWindDirection',
    'HourlyDewPointTemperature', 'HourlyRelativeHumidity', 'HourlyVisibility',
    'HourlyStationPressure', 'HourlyWetBulbTemperature'
]

# Function to calculate Haversine distance
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))

# Conversion functions
def convert_pressure_to_inhg(hpa):
    return hpa * 0.02953

def convert_visibility_to_miles(nmi):
    return min(nmi * 1.15078, 10)  # Cap visibility at 10 miles

# API call to fetch weather data
def get_weather_data_for_prediction(latitude, longitude, timestamp, username, password):
    parameter_mapping = {
        "t_2m:C": "DryBulbTemperature", "wind_speed_10m:kmh": "WindSpeed",
        "wind_dir_10m:d": "WindDirection", "dew_point_2m:C": "DewPointTemperature",
        "relative_humidity_2m:p": "RelativeHumidity", "visibility:nmi": "Visibility",
        "msl_pressure:hPa": "StationPressure"
    }
    
    timestamp_str = f"{timestamp.strftime('%Y-%m-%dT%H:%M:%SZ')}"
    parameter_str = ",".join(parameter_mapping.keys())
    url = f"https://api.meteomatics.com/{timestamp_str}/{parameter_str}/{latitude},{longitude}/json?source=ecmwf-ifs"
    
    response = requests.get(url, auth=HTTPBasicAuth(username, password))
    if response.status_code == 200:
        data = response.json()
        weather_data = {model_name: None for model_name in parameter_mapping.values()}

        for forecast in data.get("data", []):
            parameter = forecast.get("parameter")
            model_name = parameter_mapping.get(parameter)
            if model_name:
                value = forecast.get("coordinates", [{}])[0].get("dates", [{}])[0].get("value")
                weather_data[model_name] = (
                    convert_pressure_to_inhg(value) if parameter == "msl_pressure:hPa" else
                    convert_visibility_to_miles(value) if parameter == "visibility:nmi" else value
                )
        return weather_data
    else:
        print("API call failed, using fallback sources.")
        return None

# Enhanced fallback to load weather data from nearest station files
def get_weather_estimates(origin_airport_id, departure_time, closest_weather_airport, 
                          max_distance=100, n_nearest=5, fallback_files=None, origin_state=None):
    # First try: nearest stations within max_distance
    nearest_stations = closest_weather_airport[closest_weather_airport['AIRPORT_ID'] == int(origin_airport_id)]
    if nearest_stations.empty:
        print(f"No nearby stations found for airport {origin_airport_id}.")
        return None

    nearest_stations = nearest_stations[nearest_stations['DISTANCE_KM'] <= max_distance].head(n_nearest)
    features = [
        'HourlyDryBulbTemperature', 'HourlyWindSpeed', 'HourlyWindDirection',
        'HourlyDewPointTemperature', 'HourlyRelativeHumidity',
        'HourlyVisibility', 'HourlyStationPressure', 'HourlyWetBulbTemperature'
    ]
    
    weather_sums = {feature: 0.0 for feature in features}
    valid_counts = {feature: 0 for feature in features}
    
    # Attempt to load data from each nearest station
    for _, station in nearest_stations.iterrows():
        station_id = int(station['STATION_ID'])
        file_path = f'gs://airport-weather-data/ncei-lcd/{station_id}.csv'
        
        try:
            weather_df = pd.read_csv(file_path, storage_options={"token": credentials})
            weather_df['UTC_DATE'] = pd.to_datetime(weather_df['UTC_DATE'])  # Ensure correct datetime parsing
            daily_weather = weather_df[weather_df['UTC_DATE'].dt.date == departure_time.date()]
            
            if daily_weather.empty:
                continue

            closest_time_idx = (daily_weather['UTC_DATE'] - departure_time).abs().idxmin()
            closest_weather = daily_weather.loc[closest_time_idx]
            for feature in features:
                value = closest_weather.get(feature)
                if pd.notnull(value):
                    weather_sums[feature] += value
                    valid_counts[feature] += 1
        except FileNotFoundError:
            print(f"Weather file for station {station_id} not found.")
            continue

    # Calculate average or fall back to state-level aggregates if no valid counts
    station_weather_data = {feature: (weather_sums[feature] / valid_counts[feature]
                                      if valid_counts[feature] > 0 else None) for feature in features}

    # State-level aggregate fallback
    if all(value is None for value in station_weather_data.values()) and fallback_files:
        return load_fallback_summary(fallback_files, origin_state=origin_state, time_key=departure_time.strftime('%H:%M'))
    return station_weather_data

# Enhanced final state-level aggregate fallback
def load_fallback_summary(fallback_files, origin_state, origin_city=None, time_key=None):
    for file_name in fallback_files:
        try:
            fallback_data = pd.read_csv(f"gs://airport-weather-data/aggregate/{file_name}", storage_options={"token": credentials})
            filtered_data = fallback_data[fallback_data["OriginState"] == origin_state]
            if origin_city:
                filtered_data = filtered_data[filtered_data["OriginCity"] == origin_city]
            if time_key and time_key in filtered_data.columns:
                filtered_data = filtered_data[filtered_data[time_key] == time_key]
            if not filtered_data.empty:
                print(f"Loaded fallback data from {file_name}")
                return filtered_data.iloc[0][weather_features].to_dict()
        except Exception as e:
            print(f"Unable to load from gs://airport-weather-data/aggregate/{file_name}: {e}")
    print("No fallback data found.")
    return {feature: 0 for feature in weather_features}  # Default zero values if no data found

# Main weather data function to ensure bulletproof fallback
def fetch_complete_weather_data(latitude, longitude, timestamp, username, password, origin_airport_id, departure_time, closest_weather_airport, origin_state, fallback_files):
    weather_data = get_weather_data_for_prediction(latitude, longitude, timestamp, username, password)
    
    # Proceed with fallback to nearest stations if API fails or data is incomplete
    if weather_data is None or any(value is None for value in weather_data.values()):
        weather_data = get_weather_estimates(
            origin_airport_id, departure_time, closest_weather_airport,
            fallback_files=fallback_files, origin_state=origin_state
        )

    # Final check to ensure no `None` values
    if any(value is None for value in weather_data.values()):
        print("Incomplete weather data; using state-level aggregate as final fallback.")
        weather_data = load_fallback_summary(fallback_files, origin_state, time_key=departure_time.strftime('%H:%M'))

    return weather_data

# Convert local time to UTC
def convert_to_utc(local_time_str, date_str, lat, lon):
    tf = TimezoneFinder()
    local_time_zone = tf.timezone_at(lat=lat, lng=lon)
    local_time = datetime.strptime(f"{date_str} {local_time_str}", "%Y-%m-%d %H:%M")
    local_tz = pytz.timezone(local_time_zone)
    local_dt = local_tz.localize(local_time)
    return local_dt.astimezone(pytz.UTC)

def validate_time_format(time_str):
    return bool(re.match(r'^[0-2]?\d:[0-5]\d$', time_str))

def create_prediction_table(arrival_delay, departure_delay, taxi_delay, total_delay, cancel_msg):
    return html.Div([
        html.H4("Prediction Results", style={"text-align": "center", "margin-bottom": "15px", "font-size": "1.5em"}),

        html.Table([
            html.Tr([html.Th("Delay Type", style={"padding": "10px", "background-color": "#4CAF50", "color": "white"}), html.Th("Expected Delay (Minutes)", style={"padding": "10px", "background-color": "#4CAF50", "color": "white"})]),
            html.Tr([html.Td("Arrival Delay", style={"padding": "10px", "background-color": "#f2f2f2"}), html.Td(f"{arrival_delay:.2f}", style={"padding": "10px", "text-align": "right"})]),
            html.Tr([html.Td("Departure Delay", style={"padding": "10px", "background-color": "#ffffff"}), html.Td(f"{departure_delay:.2f}", style={"padding": "10px", "text-align": "right"})]),
            html.Tr([html.Td("Taxi Delay", style={"padding": "10px", "background-color": "#f2f2f2"}), html.Td(f"{taxi_delay:.2f}", style={"padding": "10px", "text-align": "right"})]),
            html.Tr([html.Td("Total Expected Delay", style={"padding": "10px", "background-color": "#ffffff", "font-weight": "bold"}), html.Td(f"{total_delay:.2f}", style={"padding": "10px", "text-align": "right", "font-weight": "bold"})]),
        ], style={"width": "60%", "margin": "auto", "border-collapse": "collapse", "box-shadow": "0 4px 8px rgba(0,0,0,0.2)", "font-family": "Arial, sans-serif", "font-size": "16px"}),

        html.Div([
            html.Div("Cancellation Likelihood", style={"display": "inline-block", "padding": "10px", "background-color": "#4CAF50", "color": "white", "width": "50%", "text-align": "center", "font-weight": "bold"}),
            html.Div(cancel_msg, style={"display": "inline-block", "padding": "10px", "background-color": "#f2f2f2" if cancel_msg == "No" else "#ffcccb", "width": "50%", "text-align": "center", "font-weight": "bold"})
        ], style={"width": "60%", "margin": "auto", "margin-top": "20px", "box-shadow": "0 4px 8px rgba(0,0,0,0.2)", "font-family": "Arial, sans-serif", "font-size": "16px"})
    ])
