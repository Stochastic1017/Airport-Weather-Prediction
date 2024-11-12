
import os
import sys

# Add current directory to system path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pickle
import pandas as pd
from dash import Output, Input, State, callback
import gcsfs
from dotenv import load_dotenv
from .prediction_helpers import (haversine, get_weather_data_for_prediction, 
                                 get_weather_estimates, convert_to_utc, 
                                 validate_time_format, create_prediction_table)
import warnings

# Suppress warnings from sklearn
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Loading environment variable with sensitive API keys
load_dotenv()

# Initialize Google Cloud Storage FileSystem
fs = gcsfs.GCSFileSystem(project='Flights-Weather-Project', token=os.getenv("gcs_storage_option"))

# Load airport metadata and closest station data
df_airport_metadata = pd.read_csv("gs://airport-weather-data/airports-list-us.csv", 
                                  storage_options={"token": os.getenv("gcs_storage_option")})
closest_weather_airport = pd.read_csv("gs://airport-weather-data/closest_airport_weather.csv", 
                                      storage_options={"token": os.getenv("gcs_storage_option")})

# Define weather-related features
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
    # Handle the initial case where button hasn't been clicked
    if not n_clicks:
        return "", {}, {}, {}, {}, {}, {}

    # Validate inputs and highlight errors
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

    # Retrieve airport coordinates and calculate distance
    origin_data = df_airport_metadata[df_airport_metadata['AIRPORT_ID'] == origin_airport].iloc[0]
    dest_data = df_airport_metadata[df_airport_metadata['AIRPORT_ID'] == destination_airport].iloc[0]
    origin_latitude, origin_longitude = origin_data['LATITUDE'], origin_data['LONGITUDE']
    dest_latitude, dest_longitude = dest_data['LATITUDE'], dest_data['LONGITUDE']
    distance = haversine(origin_latitude, origin_longitude, dest_latitude, dest_longitude)

    # Convert times to UTC
    departure_time_utc = convert_to_utc(departure_time, date, origin_latitude, origin_longitude)
    arrival_time_utc = convert_to_utc(arrival_time, date, dest_latitude, dest_longitude)
    
    # Try fetching weather data using different fallbacks
    try:
        # 1. API Call
        weather_forecasts = get_weather_data_for_prediction(
            latitude=origin_latitude, longitude=origin_longitude, timestamp=departure_time_utc,
            username=os.getenv("username_prediction_api"), password=os.getenv("password_prediction_api")
        )
    except Exception:
        print("API error: using fallback sources.")
        # 2. Closest Weather Station
        weather_forecasts = get_weather_estimates(
            origin_airport_id=origin_airport, departure_time=departure_time_utc,
            closest_weather_airport=closest_weather_airport, max_distance=100, n_nearest=5
        ) or None

    # 3-7. Fallback to aggregate files if no other data available
    if not weather_forecasts:
        for fallback_path in fallback_paths:
            try:
                fallback_df = pd.read_csv(fallback_path, 
                                          storage_options={"token": os.getenv("gcs_storage_option")})
                weather_forecasts = (
                    fallback_df[
                        (fallback_df["OriginState"] == origin_data["State"]) &
                        (fallback_df["OriginCity"] == origin_data["City"]) &
                        (fallback_df["DayOfWeek"] == departure_time_utc.weekday())  # Adjust filter conditions based on the fallback level
                    ][weather_features].mean().to_dict()
                )
                if weather_forecasts:
                    print(f"Weather data loaded from {fallback_path}")
                    break
            except Exception:
                print(f"Unable to load from {fallback_path}")

    # Build feature dictionary
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
        **{feature: weather_forecasts.get(feature, 0) for feature in weather_features}  # Defaults missing weather data to 0
    }

    # Convert features dictionary to DataFrame
    feature_df = pd.DataFrame([features])

    # Encode categorical features as required for model input
    for col in ['Marketing_Airline_Network', 'OriginAirportID', 'DestAirportID']:
        feature_df[col] = feature_df[col].astype('category').cat.codes

    # Load trained models
    with fs.open("gs://airport-weather-data/models/random_forest_regressor.pkl", "rb") as f:
        delay_models = pickle.load(f)
    with fs.open("gs://airport-weather-data/models/random_forest_classifier.pkl", "rb") as f:
        cancel_model = pickle.load(f)

    # Perform delay prediction
    delay_prediction = delay_models.predict(feature_df)[0]
    arrival_delay, departure_delay, taxi_delay, total_delay = delay_prediction
    
    # Floor the delays to ensure no negative values are reported
    arrival_delay = max(0, arrival_delay)
    departure_delay = max(0, departure_delay)
    taxi_delay = max(0, taxi_delay)
    total_delay = max(0, total_delay)

    # Perform cancellation prediction
    cancel_prediction = cancel_model.predict(feature_df)[0]
    cancel_msg = "Yes" if cancel_prediction == 1 else "No"

    # Use the create_prediction_table function to display results as a table
    delay_table = create_prediction_table(arrival_delay, departure_delay, taxi_delay, total_delay, cancel_msg)

    # Return prediction results
    return delay_table, {}, {}, {}, {}, {}, {}
