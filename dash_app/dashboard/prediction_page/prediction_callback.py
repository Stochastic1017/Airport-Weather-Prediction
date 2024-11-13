
import os
import sys

# Add current directory to system path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import json
import pickle
import gcsfs
import dask.dataframe as dd
import pandas as pd
from dash import Output, Input, State, callback
from dotenv import load_dotenv
from google.oauth2 import service_account
from functools import lru_cache
from .prediction_helpers import (haversine, get_weather_data_for_prediction, 
                                 get_weather_estimates, convert_to_utc, 
                                 validate_time_format, create_prediction_table)
import warnings

# Suppress warnings from sklearn
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Load environment variables
load_dotenv()

credentials_info = os.getenv("GCP_CREDENTIALS")
credentials = service_account.Credentials.from_service_account_info(json.loads(credentials_info),
                                                                    scopes=['https://www.googleapis.com/auth/devstorage.read_write',
                                                                            'https://www.googleapis.com/auth/cloud-platform',
                                                                            'https://www.googleapis.com/auth/drive'])

# Initialize GCS filesystem
fs = gcsfs.GCSFileSystem(project='Flights-Weather-Project', token=credentials)

# Cache models using lru_cache
@lru_cache(maxsize=1)
def load_delay_model():
    with fs.open("gs://airport-weather-data/models/random_forest_regressor.pkl", "rb") as f:
        return pickle.load(f)

@lru_cache(maxsize=1)
def load_cancel_model():
    with fs.open("gs://airport-weather-data/models/random_forest_classifier.pkl", "rb") as f:
        return pickle.load(f)

# Load large datasets using Dask
df_airport_metadata = dd.read_csv("gs://airport-weather-data/airports-list-us.csv", 
                                   storage_options={"token": credentials}).persist()
closest_weather_airport = dd.read_csv("gs://airport-weather-data/closest_airport_weather.csv", 
                                       storage_options={"token": credentials}).persist()

# Weather features
weather_features = [
    'HourlyDryBulbTemperature', 'HourlyWindSpeed', 'HourlyWindDirection',
    'HourlyDewPointTemperature', 'HourlyRelativeHumidity', 'HourlyVisibility',
    'HourlyStationPressure', 'HourlyWetBulbTemperature'
]

# Fallback file paths
fallback_paths = [
    "gs://airport-weather-data/aggregate/airport_weather_summary_by_state_city_day.csv",
    "gs://airport-weather-data/aggregate/airport_weather_summary_by_state_city_week.csv",
    "gs://airport-weather-data/aggregate/airport_weather_summary_by_state_city_month.csv",
    "gs://airport-weather-data/aggregate/airport_weather_summary_by_state_city.csv",
    "gs://airport-weather-data/aggregate/airport_weather_summary_by_state.csv"
]

@callback(
    Output("prediction-output", "children"),
    Output("airline-input", "style"),
    Output("origin-airport-input", "style"),
    Output("destination-airport-input", "style"),
    Output("date-input", "style"),
    Output("departure-time-input", "style"),
    Output("arrival-time-input", "style"),
    Input("predict-button", "n_clicks"),
    State("airline-input", "value"),
    State("origin-airport-input", "value"),
    State("destination-airport-input", "value"),
    State("departure-time-input", "value"),
    State("arrival-time-input", "value"),
    State("date-input", "date"),
)
def predict_flight_delay(n_clicks, airline, origin_airport, destination_airport, 
                         departure_time, arrival_time, date):
    if not n_clicks:
        return "", {}, {}, {}, {}, {}, {}

    # Validate inputs
    errors = {}
    required_inputs = {
        "airline-input": airline,
        "origin-airport-input": origin_airport,
        "destination-airport-input": destination_airport,
        "date-input": date,
        "departure-time-input": departure_time,
        "arrival-time-input": arrival_time
    }
    for input_id, value in required_inputs.items():
        if not value:
            errors[input_id] = {"border": "2px solid red"}

    if departure_time and not validate_time_format(departure_time):
        errors["departure-time-input"] = {"border": "2px solid red"}
    if arrival_time and not validate_time_format(arrival_time):
        errors["arrival-time-input"] = {"border": "2px solid red"}

    if errors:
        return "Please fill all fields correctly.", *[errors.get(i, {}) for i in required_inputs]

    # Retrieve airport metadata
    metadata = df_airport_metadata[
        (df_airport_metadata['AIRPORT_ID'] == origin_airport) |
        (df_airport_metadata['AIRPORT_ID'] == destination_airport)
    ].compute()
    origin_data = metadata[metadata['AIRPORT_ID'] == origin_airport].iloc[0]
    dest_data = metadata[metadata['AIRPORT_ID'] == destination_airport].iloc[0]

    origin_latitude, origin_longitude = origin_data['LATITUDE'], origin_data['LONGITUDE']
    dest_latitude, dest_longitude = dest_data['LATITUDE'], dest_data['LONGITUDE']
    distance = haversine(origin_latitude, origin_longitude, dest_latitude, dest_longitude)

    # Convert times to UTC
    departure_time_utc = convert_to_utc(departure_time, date, origin_latitude, origin_longitude)
    arrival_time_utc = convert_to_utc(arrival_time, date, dest_latitude, dest_longitude)

    # Fetch weather data
    weather_forecasts = None
    try:
        weather_forecasts = get_weather_data_for_prediction(
            latitude=origin_latitude, longitude=origin_longitude, timestamp=departure_time_utc,
            username=os.getenv("username_prediction_api"), password=os.getenv("password_prediction_api")
        )
    except Exception:
        weather_forecasts = get_weather_estimates(
            origin_airport_id=origin_airport, departure_time=departure_time_utc,
            closest_weather_airport=closest_weather_airport.compute(), max_distance=100, n_nearest=5
        )

    # Use fallback files
    if not weather_forecasts:
        weather_forecasts = next((
            dd.read_csv(path, storage_options={"token": credentials})
            .loc[
                (lambda df: (df["OriginState"] == origin_data["State"]) &
                            (df["OriginCity"] == origin_data["City"]) &
                            (df["DayOfWeek"] == departure_time_utc.weekday()))
            ][weather_features]
            .mean()
            .compute()
            .to_dict()
            for path in fallback_paths
        ), {feature: 0 for feature in weather_features})

    # Prepare features
    features = {
        "DayOfWeek": departure_time_utc.weekday(),
        "Marketing_Airline_Network": airline,
        "OriginAirportID": origin_airport,
        "DestAirportID": destination_airport,
        "Distance": distance,
        "CRSDepHour": departure_time_utc.hour,
        "CRSArrHour": arrival_time_utc.hour,
        "CRSDepMonth": departure_time_utc.month,
        "CRSDepDayOfWeek": departure_time_utc.weekday(),
        **{feature: weather_forecasts.get(feature, 0) for feature in weather_features}
    }
    feature_df = pd.DataFrame([features])

    # Encode categorical features
    for col in ['Marketing_Airline_Network', 'OriginAirportID', 'DestAirportID']:
        feature_df[col] = feature_df[col].astype('category').cat.codes

    # Perform predictions
    delay_models = load_delay_model()
    cancel_model = load_cancel_model()
    delay_prediction = delay_models.predict(feature_df)[0]
    arrival_delay, departure_delay, taxi_delay, total_delay = map(lambda x: max(0, x), delay_prediction)
    cancel_prediction = cancel_model.predict(feature_df)[0]
    cancel_msg = "Yes" if cancel_prediction == 1 else "No"

    # Create results table
    delay_table = create_prediction_table(arrival_delay, departure_delay, taxi_delay, total_delay, cancel_msg)
    return delay_table, {}, {}, {}, {}, {}, {}
