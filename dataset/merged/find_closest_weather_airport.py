
import sys
import pandas as pd
from math import radians, sin, cos, sqrt, atan2

# Function to calculate Haversine distance between two points (lat1, lon1) and (lat2, lon2)
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in kilometers
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    
    return R * c  # Distance in kilometers

# Check if the correct number of arguments is provided
if len(sys.argv) < 2:
    print("Usage: python find_closest_weather_airport.py <number_of_closest_stations>")
    sys.exit(1)

# Get the number of closest stations from command-line arguments
n_closest = int(sys.argv[1])

# Load datasets
airports_df = pd.read_csv("https://raw.githubusercontent.com/Stochastic1017/Airport-Weather-Prediction/refs/heads/main/dataset/transtat-bts/airports-list-us.csv")
weather_stations_df = pd.read_csv("https://raw.githubusercontent.com/Stochastic1017/Airport-Weather-Prediction/refs/heads/main/dataset/ncei-lcd/ncei-lcd-list-us.csv")

# Extract required columns for calculating distances
airports_df = airports_df[['AIRPORT_SEQ_ID', 'AIRPORT_ID', 'AIRPORT', 'DISPLAY_AIRPORT_NAME', 'DISPLAY_AIRPORT_CITY_NAME_FULL',
                           'AIRPORT_COUNTRY_NAME', 'AIRPORT_STATE_NAME', 'LATITUDE', 'LONGITUDE']]

weather_stations_df = weather_stations_df[['station', 'station_name', 'elevation', 'latitude', 'longitude', 'coords', 
                                           'admin1', 'admin2', 'country', 'state']]

# Initialize an empty list to store results
results = []

# Iterate over each airport
for _, airport_row in airports_df.iterrows():
    airport_id = airport_row['AIRPORT_ID']
    airport_name = airport_row['AIRPORT']
    airport_display_name = airport_row['DISPLAY_AIRPORT_NAME']
    airport_city = airport_row['DISPLAY_AIRPORT_CITY_NAME_FULL']
    airport_country = airport_row['AIRPORT_COUNTRY_NAME']
    airport_state = airport_row['AIRPORT_STATE_NAME']
    airport_lat = airport_row['LATITUDE']
    airport_lon = airport_row['LONGITUDE']
    airport_coords = (airport_lat, airport_lon)
    
    # Calculate the distance from the airport to each weather station
    weather_stations_df['distance'] = weather_stations_df.apply(
        lambda row: haversine(airport_lat, airport_lon, row['latitude'], row['longitude']), axis=1
    )
    
    # Get the n closest weather stations
    closest_stations = weather_stations_df.nsmallest(n_closest, 'distance')
    
    # Append results
    for _, station_row in closest_stations.iterrows():
        results.append({
            'AIRPORT_SEQ_ID': airport_row['AIRPORT_SEQ_ID'],
            'AIRPORT_ID': airport_id,
            'AIRPORT_NAME': airport_name,
            'AIRPORT_DISPLAY_NAME': airport_display_name,
            'AIRPORT_CITY': airport_city,
            'AIRPORT_COUNTRY': airport_country,
            'AIRPORT_STATE': airport_state,
            'AIRPORT_COORDINATES': airport_coords,
            'WEATHER_STATION_ID': station_row['station'],
            'WEATHER_STATION_NAME': station_row['station_name'],
            'WEATHER_COORDINATES': (station_row['latitude'], station_row['longitude']),
            'WEATHER_ELEVATION': station_row['elevation'],
            'WEATHER_ADMIN1': station_row['admin1'],
            'WEATHER_ADMIN2': station_row['admin2'],
            'WEATHER_COUNTRY': station_row['country'],
            'WEATHER_STATE': station_row['state'],
            'DISTANCE_KM': station_row['distance']
        })

# Convert results to a DataFrame and save to CSV
closest_df = pd.DataFrame(results)
closest_df.to_csv('closest_airport_weather.csv', index=False)

print("File 'closest_airport_weather.csv' created with the closest weather stations for each airport.")
