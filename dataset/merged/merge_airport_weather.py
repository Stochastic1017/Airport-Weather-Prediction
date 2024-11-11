
import os
import sys
import gcsfs
import pandas as pd
import numpy as np

# Check command-line arguments
if len(sys.argv) != 4:
    print("Usage: python3 merge_airport_weather.py <n_nearest> <max_distance> <output_directory>")
    sys.exit(1)

# Assign command-line arguments to variables
n_nearest = int(sys.argv[1])
max_distance = float(sys.argv[2])
output_directory = sys.argv[3]

def merge_weather_data(weather_data_list, features):
    processed_dfs = []
    for df in weather_data_list:
        df['UTC_DATE'] = pd.to_datetime(df['UTC_DATE'])
        for feature in features:
            df[feature] = pd.to_numeric(df[feature], errors='coerce')
        df = df.groupby('UTC_DATE')[features].mean().reset_index()
        processed_dfs.append(df)
    
    combined_df = pd.concat(processed_dfs).groupby('UTC_DATE')[features].mean().reset_index()
    for feature in features:
        combined_df[feature] = combined_df[feature].ffill()
    
    print(f"Processed {len(processed_dfs)} weather stations and merged data for {len(combined_df)} timestamps.")
    return combined_df

def impute_sky_conditions(weather_data_list):
    sky_conditions_df = pd.concat([df[['UTC_DATE', 'HourlySkyConditions']] for df in weather_data_list])
    sky_conditions_df['UTC_DATE'] = pd.to_datetime(sky_conditions_df['UTC_DATE'])

    def mode_impute(group):
        # Ensure all elements are strings to prevent sorting issues
        group = group.dropna().astype(str)  # Convert non-NaN values to strings
        mode_values = group.mode()
        
        if not mode_values.empty:
            return np.random.choice(mode_values)  # Handle ties by randomly selecting one mode
        else:
            return np.nan  # Return NaN if no mode is found

    sky_conditions_df['HourlySkyConditions'] = sky_conditions_df.groupby('UTC_DATE')['HourlySkyConditions'].transform(mode_impute)
    sky_conditions_df['HourlySkyConditions'] = sky_conditions_df['HourlySkyConditions'].ffill()
    
    print("Imputed sky conditions.")
    return sky_conditions_df[['UTC_DATE', 'HourlySkyConditions']].drop_duplicates()

# Initialize Google Cloud Storage FileSystem
fs = gcsfs.GCSFileSystem(project='Flights-Weather-Project', token='flights-weather-project-f94d306bee1f.json')

# Load metadata for nearest stations and airport list
closest_weather_airport = pd.read_csv(
    f"gs://airport-weather-data/closest_airport_weather.csv",
    storage_options={"token": "flights-weather-project-f94d306bee1f.json"},
    low_memory=False
)
airport_list = pd.read_csv(
    f"gs://airport-weather-data/airports-list-us.csv",
    storage_options={"token": "flights-weather-project-f94d306bee1f.json"},
    low_memory=False
)
print("Loaded closest weather station metadata and airport list.")

# Define the path with wildcard pattern for the files you want to list
weather_data_path_pattern = 'airport-weather-data/ncei-lcd/*.csv'
flight_data_path_pattern = 'airport-weather-data/transtat-bts/*.csv'

# List all flight files
flight_files = fs.glob(flight_data_path_pattern)

# Process each flight file and merge with weather data
os.makedirs(output_directory, exist_ok=True)

for idx, flight_file in enumerate(flight_files):
    print("------------------------------------------------")
    print(f"Currently at: {idx} out of {len(flight_files)}")
    print("\n")
    print(f"Processing flight file: {flight_file}")
    flight_df = pd.read_csv(f"gs://{flight_file}", 
                            storage_options={"token": "flights-weather-project-f94d306bee1f.json"}, low_memory=False
                            )
    print(f"Loaded flight data with {len(flight_df)} rows.")

    # Ensure datetime columns are in datetime format
    datetime_columns = ['UTC_DepTime', 'UTC_ArrTime', 'UTC_CRSDepTime', 'UTC_CRSArrTime']
    for col in datetime_columns:
        flight_df[col] = pd.to_datetime(flight_df[col], errors='coerce')

    origin_airport_id = flight_df["OriginAirportID"].iloc[0]
    print(f"Origin airport ID: {origin_airport_id}")
    
    # Get nearest weather stations, defaulting to the closest available if none are within max_distance
    nearest_stations = closest_weather_airport[
        closest_weather_airport['AIRPORT_ID'] == int(origin_airport_id)
    ]

    if nearest_stations.empty:
        print(f"No nearby stations found for airport {origin_airport_id}. Skipping this file.")
        continue  # Skip processing this file if there are no stations at all

    # If no stations are within max_distance, use the nearest n stations regardless of distance
    if nearest_stations[nearest_stations['DISTANCE_KM'] <= max_distance].empty:
        print(f"No stations within {max_distance} km. Using the closest {n_nearest} stations.")
        nearest_stations = nearest_stations.nsmallest(n_nearest, 'DISTANCE_KM')
    else:
        nearest_stations = nearest_stations[nearest_stations['DISTANCE_KM'] <= max_distance].head(n_nearest)

    print(f"Using {len(nearest_stations)} nearest stations.")

    # Features to process
    features = [
        'HourlyDryBulbTemperature', 'HourlyWindSpeed', 'HourlyWindDirection',
        'HourlyDewPointTemperature', 'HourlyRelativeHumidity',
        'HourlyVisibility', 'HourlyStationPressure', 'HourlyWetBulbTemperature'
    ]

    # Load weather data for each station
    weather_data_list = []
    sky_conditions_data = []
    for station_id, distance in zip(nearest_stations["WEATHER_STATION_ID"], nearest_stations["DISTANCE_KM"]):
        print(f"Loading weather data for station: {station_id} <-> distance: {distance}")
        weather_df = pd.read_csv(
            f"gs://airport-weather-data/ncei-lcd/{station_id}.csv",
            storage_options={"token": "flights-weather-project-f94d306bee1f.json"}, low_memory=False
        )
        weather_data_list.append(weather_df)
        sky_conditions_data.append(weather_df[['UTC_DATE', 'HourlySkyConditions']])

    # Merge numeric weather data by averaging
    merged_weather = merge_weather_data(weather_data_list, features)

    # Impute sky conditions by taking mode and forward-filling if needed
    imputed_sky_conditions = impute_sky_conditions(sky_conditions_data)

    # Merge the imputed sky conditions back with the averaged weather data
    merged_weather = merged_weather.merge(imputed_sky_conditions, on='UTC_DATE', how='left')
    print("Merged weather and sky conditions.")

    # Impute NaN DepTime values with values in the CRSDepTime
    flight_df['UTC_DepTime'] = flight_df['UTC_DepTime'].fillna(flight_df['UTC_CRSDepTime'])
    print("Imputed NaN values in UTC_DepTime with UTC_CRSDepTime.")

    # Drop rows with NaN in the merge columns if they can't be filled
    flight_df = flight_df.dropna(subset=['UTC_DepTime'])
    merged_weather = merged_weather.dropna(subset=['UTC_DATE'])

    # Calculate delays and merge
    flight_df['ArrivalDelay'] = (flight_df['UTC_ArrTime'] - flight_df['UTC_CRSArrTime']).dt.total_seconds() / 60
    flight_df['DepartureDelay'] = (flight_df['UTC_DepTime'] - flight_df['UTC_CRSDepTime']).dt.total_seconds() / 60
    scheduled_flight_time = (flight_df['UTC_CRSArrTime'] - flight_df['UTC_CRSDepTime']).dt.total_seconds() / 60
    actual_flight_time = (flight_df['UTC_ArrTime'] - flight_df['UTC_DepTime']).dt.total_seconds() / 60
    flight_df['TotalFlightDelay'] = scheduled_flight_time - actual_flight_time
    flight_df['TaxiDelay'] = flight_df['TaxiOut'] - flight_df['TaxiIn']
    print("Calculated delay metrics.")

    # Merge the weather data with the flight data
    merged_df = pd.merge_asof(
        flight_df.sort_values('UTC_DepTime'),
        merged_weather.sort_values('UTC_DATE'),
        left_on='UTC_DepTime',
        right_on='UTC_DATE',
        direction='nearest'
    )

    # Merge with airport list to get Origin and Destination City and State
    merged_df = merged_df.merge(airport_list[['AIRPORT_ID', 'City', 'State']], 
                                left_on='OriginAirportID', right_on='AIRPORT_ID', how='left').rename(
                                    columns={'City': 'OriginCity', 'State': 'OriginState'}).drop(columns=['AIRPORT_ID'])
    
    merged_df = merged_df.merge(airport_list[['AIRPORT_ID', 'City', 'State']], 
                                left_on='DestAirportID', right_on='AIRPORT_ID', how='left').rename(
                                    columns={'City': 'DestCity', 'State': 'DestState'}).drop(columns=['AIRPORT_ID'])

    # Define the desired column order
    column_order = [
        # Basic flight details
        'DayOfWeek', 'Marketing_Airline_Network', 'Flight_Number_Operating_Airline',
        
        # Origin and destination information
        'OriginAirportID', 'OriginCity', 'OriginState', 'OriginCityName',
        'DestAirportID', 'DestCity', 'DestState', 'DestCityName', 'Distance',
        
        # Flight timings
        'UTC_CRSDepTime', 'UTC_DepTime', 'UTC_CRSArrTime', 'UTC_ArrTime',
        
        # Delay information
        'ArrivalDelay', 'DepartureDelay', 'TotalFlightDelay', 'TaxiOut', 'TaxiIn', 'TaxiDelay',
        'CarrierDelay', 'WeatherDelay', 'NASDelay', 'SecurityDelay', 'LateAircraftDelay',
        
        # Flight duration
        'CRSElapsedTime', 'ActualElapsedTime', 'Cancelled',
        
        # Weather details
        'HourlyDryBulbTemperature', 'HourlyWindSpeed', 'HourlyWindDirection',
        'HourlyDewPointTemperature', 'HourlyRelativeHumidity', 'HourlyVisibility',
        'HourlyStationPressure', 'HourlyWetBulbTemperature', 'HourlySkyConditions'
    ]

    output_file = f"{output_directory}/{origin_airport_id}_training_data.csv"
    merged_df.to_csv(f"{output_file}", index=False)
    print(f"Data saved to {output_file}")
    print("\n")
