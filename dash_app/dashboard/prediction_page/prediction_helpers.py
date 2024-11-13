
import os
import sys
import json

# Append current directory to system path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import re
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

# Load environment variables
load_dotenv()

credentials_info = os.getenv("GCP_CREDENTIALS")
credentials = service_account.Credentials.from_service_account_info(json.loads(credentials_info),
                                                                    scopes=['https://www.googleapis.com/auth/devstorage.read_write',
                                                                            'https://www.googleapis.com/auth/cloud-platform',
                                                                            'https://www.googleapis.com/auth/drive'])

# Weather-related features
weather_features = [
    'HourlyDryBulbTemperature', 'HourlyWindSpeed', 'HourlyWindDirection',
    'HourlyDewPointTemperature', 'HourlyRelativeHumidity', 'HourlyVisibility',
    'HourlyStationPressure', 'HourlyWetBulbTemperature'
]

# Haversine formula constants
EARTH_RADIUS_KM = 6371  # Earth's radius in kilometers

def haversine(lat1, lon1, lat2, lon2):
    """Calculate Haversine distance between two points."""
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    return EARTH_RADIUS_KM * 2 * atan2(sqrt(a), sqrt(1 - a))

def convert_pressure_to_inhg(hpa):
    """Convert pressure from hPa to inHg."""
    return hpa * 0.02953

def convert_visibility_to_miles(nmi):
    """Convert visibility from nautical miles to miles, capped at 10 miles."""
    return min(nmi * 1.15078, 10)

def get_weather_data_for_prediction(latitude, longitude, timestamp, username, password):
    """Fetch weather data from Meteomatics API."""
    parameter_mapping = {
        "t_2m:C": "DryBulbTemperature",
        "wind_speed_10m:kmh": "WindSpeed",
        "wind_dir_10m:d": "WindDirection",
        "dew_point_2m:C": "DewPointTemperature",
        "relative_humidity_2m:p": "RelativeHumidity",
        "visibility:nmi": "Visibility",
        "msl_pressure:hPa": "StationPressure"
    }

    parameter_str = ",".join(parameter_mapping.keys())
    url = f"https://api.meteomatics.com/{timestamp:%Y-%m-%dT%H:%M:%SZ}/{parameter_str}/{latitude},{longitude}/json?source=ecmwf-ifs"
    
    response = requests.get(url, auth=HTTPBasicAuth(username, password))
    if response.status_code == 200:
        data = response.json()
        weather_data = {value: None for value in parameter_mapping.values()}

        for forecast in data.get("data", []):
            parameter = forecast.get("parameter")
            model_name = parameter_mapping.get(parameter)
            if model_name:
                value = forecast.get("coordinates", [{}])[0].get("dates", [{}])[0].get("value")
                if value is not None:
                    if parameter == "msl_pressure:hPa":
                        value = convert_pressure_to_inhg(value)
                    elif parameter == "visibility:nmi":
                        value = convert_visibility_to_miles(value)
                    weather_data[model_name] = value
        return weather_data

    print("API call failed; fallback required.")
    return None

def load_fallback_summary(fallback_files, origin_state, time_key=None):
    """Load fallback weather data from state-level aggregates using Pandas."""
    for file_name in fallback_files:
        try:
            fallback_data = pd.read_csv(f"gs://airport-weather-data/aggregate/{file_name}", storage_options={"token": credentials})
            filtered_data = fallback_data[fallback_data["OriginState"] == origin_state]
            if time_key:
                filtered_data = filtered_data[filtered_data[time_key] == time_key]
            if not filtered_data.empty:
                print(f"Loaded fallback data from {file_name}")
                return filtered_data.iloc[0][weather_features].to_dict()
        except Exception as e:
            print(f"Failed to load fallback file {file_name}: {e}")

    print("Fallback data not found; returning default values.")
    return {feature: 0 for feature in weather_features}

def get_weather_estimates(origin_airport_id, departure_time, closest_weather_airport, 
                          max_distance=100, n_nearest=5, fallback_files=None, origin_state=None):
    """Estimate weather data from nearby stations using Pandas."""
    # Step 1: Find nearest stations
    nearest_stations = closest_weather_airport[
        closest_weather_airport['AIRPORT_ID'] == int(origin_airport_id)
    ]
    if nearest_stations.empty:
        print(f"No nearby stations found for airport {origin_airport_id}.")
        return None

    # Step 2: Filter stations within max_distance
    nearest_stations = nearest_stations[
        nearest_stations['DISTANCE_KM'] <= max_distance
    ].nsmallest(n_nearest, 'DISTANCE_KM')

    # Step 3: Initialize feature sums and counts
    weather_sums = {feature: 0.0 for feature in weather_features}
    valid_counts = {feature: 0 for feature in weather_features}

    # Step 4: Fetch and aggregate weather data
    for _, station in nearest_stations.iterrows():
        station_id = int(station['STATION_ID'])
        file_path = f'gs://airport-weather-data/ncei-lcd/{station_id}.csv'

        try:
            weather_df = pd.read_csv(file_path, storage_options={"token": credentials})
            weather_df['UTC_DATE'] = pd.to_datetime(weather_df['UTC_DATE'])
            
            # Filter for the specific day
            daily_weather = weather_df[weather_df['UTC_DATE'].dt.date == departure_time.date()]

            if daily_weather.empty:
                continue

            # Find the closest time index to departure time
            daily_weather['time_diff'] = abs(daily_weather['UTC_DATE'] - departure_time)
            closest_weather = daily_weather.nsmallest(1, 'time_diff')

            # Aggregate feature values
            for feature in weather_features:
                value = closest_weather[feature].values[0]
                if pd.notna(value):  # Pandas null check
                    weather_sums[feature] += value
                    valid_counts[feature] += 1
        except FileNotFoundError:
            print(f"Weather file for station {station_id} not found.")
            continue

    # Step 5: Calculate averages or fallback to state-level aggregates
    station_weather_data = {
        feature: (weather_sums[feature] / valid_counts[feature]
                  if valid_counts[feature] > 0 else None)
        for feature in weather_features
    }

    # Fallback if no valid station data
    if all(value is None for value in station_weather_data.values()) and fallback_files:
        return load_fallback_summary(fallback_files, origin_state=origin_state, 
                                     time_key=departure_time.strftime('%H:%M'))

    return station_weather_data

def fetch_complete_weather_data(latitude, longitude, timestamp, username, password, 
                              origin_airport_id, departure_time, closest_weather_airport, 
                              origin_state, fallback_files):
    """Fetch weather data with fallbacks. Guarantees non-None return values."""
    default_weather = {feature: 0 for feature in weather_features}
    try:
        weather_data = get_weather_data_for_prediction(latitude, longitude, timestamp, username, password)

        if weather_data:
            weather_data = {k: weather_data.get(k, default_weather[k]) for k in default_weather}

        if not weather_data or any(v is None for v in weather_data.values()):
            print("Using nearest weather station fallback...")
            weather_data = get_weather_estimates(origin_airport_id, departure_time, closest_weather_airport, max_distance=100, n_nearest=5, fallback_files=fallback_files, origin_state=origin_state)
            
            if weather_data:
                weather_data = {k: weather_data.get(k, default_weather[k]) for k in default_weather}

        if not weather_data or any(v is None for v in weather_data.values()):
            print("Using state-level aggregate fallback...")
            weather_data = load_fallback_summary(fallback_files, origin_state=origin_state, time_key=departure_time.strftime('%H:%M'))
            
            if weather_data:
                weather_data = {k: weather_data.get(k, default_weather[k]) for k in default_weather}

        if not weather_data or any(v is None for v in weather_data.values()):
            print("All fallbacks failed; using default values.")
            weather_data = default_weather.copy()

    except Exception as e:
        print(f"Unexpected error in weather data fetching: {e}")
        weather_data = default_weather.copy()

    return weather_data

def convert_to_utc(local_time_str, date_str, lat, lon):
    """Convert local time to UTC using TimezoneFinder."""
    tf = TimezoneFinder()
    local_tz_name = tf.timezone_at(lat=lat, lng=lon)
    local_time = datetime.strptime(f"{date_str} {local_time_str}", "%Y-%m-%d %H:%M")
    local_tz = pytz.timezone(local_tz_name)
    return local_tz.localize(local_time).astimezone(pytz.UTC)

def validate_time_format(time_str):
    """Validate time format as HH:MM."""
    return bool(re.match(r'^[0-2]?\d:[0-5]\d$', time_str))

def create_prediction_table(arrival_delay, departure_delay, taxi_delay, total_delay, cancel_msg):
    """Create an HTML table for prediction results."""
    rows = [
        ("Arrival Delay", f"{arrival_delay:.2f}"),
        ("Departure Delay", f"{departure_delay:.2f}"),
        ("Taxi Delay", f"{taxi_delay:.2f}"),
        ("Total Expected Delay", f"{total_delay:.2f}")
    ]
    table_rows = [
        html.Tr([html.Td(label), html.Td(value)], style={"background-color": "#f2f2f2" if i % 2 == 0 else "#ffffff"})
        for i, (label, value) in enumerate(rows)
    ]
    cancel_style = {"background-color": "#f2f2f2" if cancel_msg == "No" else "#ffcccb"}
    return html.Div([
        html.Table(
            [html.Tr([html.Th("Delay Type"), html.Th("Expected Delay (Minutes)")])] + table_rows,
            style={"width": "60%", "margin": "auto", "border-collapse": "collapse"}
        ),
        html.Div([
            html.Div("Cancellation Likelihood", style={"font-weight": "bold", "text-align": "center", "width": "50%"}),
            html.Div(cancel_msg, style={"width": "50%", **cancel_style, "text-align": "center"})
        ], style={"display": "flex", "margin-top": "20px"})
    ])
