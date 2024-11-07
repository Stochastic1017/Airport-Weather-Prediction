
import os
import sys
import pandas as pd
import pytz
from timezonefinder import TimezoneFinder
from glob import glob

"""
This script converts airport-related timestamps from local time to UTC based on each airport's 
latitude and longitude. The process includes the following steps:
1. Load an airport metadata file to get the coordinates (latitude and longitude) for each airport ID.
2. Process each CSV file in a specified input directory:
   - Parse timestamps in the local time zone, handling cases where the time is "24:00" by rolling over to the next day.
   - Use the airport's coordinates to determine the time zone and convert the parsed times to UTC.
3. Save the modified DataFrame with UTC timestamps to a specified output directory.

Usage:
    python3 script_name.py <metadata_path> <input_directory> <output_directory>

Arguments:
    metadata_path (str): Path to the CSV file containing airport metadata, which must include 
                         'AIRPORT_ID', 'LATITUDE', and 'LONGITUDE' columns.
    input_directory (str): Path to the directory containing input CSV files with airport data.
    output_directory (str): Path to save the processed CSV files with UTC timestamps.

CSV File Requirements:
- Input CSV files should contain a 'FlightDate' column and timestamp columns such as 
  'CRSDepTime', 'DepTime', 'CRSArrTime', 'ArrTime', 'WheelsOn', and 'WheelsOff'.
"""

# Check command-line arguments
if len(sys.argv) != 4:
    print("Usage: python3 airport_change_datetime_utc.py <metadata_path> <input_directory> <output_directory>")
    sys.exit(1)

# Assign command-line arguments to variables
metadata_path = sys.argv[1]
input_directory = sys.argv[2]
output_directory = sys.argv[3]

# Initialize TimezoneFinder
tf = TimezoneFinder()

def preprocess_time(row, time_col, date_col):
    """
    Preprocesses and formats times, converting "24:00" to "00:00" of the following day if necessary.
    
    Args:
        row (pd.Series): A row from the DataFrame containing date and time columns.
        time_col (str): The name of the time column.
        date_col (str): The name of the date column.

    Returns:
        pd.Timestamp or None: The processed datetime object or None if conversion fails.
    """
    if pd.isnull(row[time_col]) or pd.isnull(row[date_col]):
        return None

    time_str = f"{int(row[time_col]):04d}"
    
    if time_str == "2400":
        try:
            date_obj = pd.to_datetime(row[date_col]) + pd.Timedelta(days=1)
            return pd.Timestamp(date_obj.strftime('%Y-%m-%d 00:00'))
        except Exception as e:
            print(f"Error handling 24:00 for row {row.name}: {e}")
            return None
    else:
        return pd.to_datetime(f"{row[date_col]} {time_str[:2]}:{time_str[2:]}", format='%Y-%m-%d %H:%M', errors='coerce')

def convert_to_utc(df, time_col, date_col, lat, lon):
    """
    Converts a specified local time column to UTC for a DataFrame, based on latitude and longitude.
    
    Args:
        df (pd.DataFrame): The DataFrame containing the timestamp data.
        time_col (str): The name of the time column to be converted.
        date_col (str): The name of the date column.
        lat (float): The latitude of the location (airport).
        lon (float): The longitude of the location (airport).

    Returns:
        pd.DataFrame: The DataFrame with an additional UTC-converted column.
    """
    print(f"Starting time zone conversion for column: {time_col}")
    
    if time_col not in df.columns:
        print(f"Column {time_col} not found in the DataFrame. Skipping.")
        return df

    df['UTC_' + time_col] = df.apply(lambda row: preprocess_time(row, time_col, date_col), axis=1)
    
    def local_to_utc(row):
        if pd.isnull(row['UTC_' + time_col]) or pd.isnull(lat) or pd.isnull(lon):
            return None
        try:
            local_tz_name = tf.timezone_at(lat=lat, lng=lon)
            if local_tz_name:
                local_tz = pytz.timezone(local_tz_name)
                localized_time = row['UTC_' + time_col].replace(tzinfo=local_tz)
                return localized_time.astimezone(pytz.utc)
            else:
                print(f"Timezone not found for coordinates: LAT={lat}, LON={lon}")
                return None
        except Exception as e:
            print(f"Error converting {time_col} for row {row.name}: {e}")
            return None
    
    df['UTC_' + time_col] = df.apply(local_to_utc, axis=1)
    return df

# Load airport metadata
print("Loading airport metadata...")
airport_meta = pd.read_csv(metadata_path)
print("Airport metadata loaded successfully")

# Ensure the output directory exists
os.makedirs(output_directory, exist_ok=True)

# Process each CSV file in the input directory
for filename in os.listdir(input_directory):
    if filename.endswith('.csv'):
        input_filepath = os.path.join(input_directory, filename)
        output_filepath = os.path.join(output_directory, filename)
        
        print(f"\nProcessing {input_filepath}")
        df = pd.read_csv(input_filepath, low_memory=False)
        
        # Extract the airport ID from the filename
        airport_id = int(filename.split('.')[0])
        
        # Get latitude and longitude for the current airport ID
        if airport_id in airport_meta['AIRPORT_ID'].values:
            lat = airport_meta.loc[airport_meta['AIRPORT_ID'] == airport_id, 'LATITUDE'].values[0]
            lon = airport_meta.loc[airport_meta['AIRPORT_ID'] == airport_id, 'LONGITUDE'].values[0]
            print(f"Found coordinates for airport ID {airport_id}: LAT={lat}, LON={lon}")
            
            # Convert relevant time columns to UTC
            time_columns = ['CRSDepTime', 'DepTime', 'CRSArrTime', 'ArrTime']
            for col in time_columns:
                df = convert_to_utc(df, col, 'FlightDate', lat, lon)
            
            # Drop the original time columns and keep only the UTC versions
            df = df.drop(columns=time_columns + ['FlightDate'], errors='ignore')
            
            # Save the modified DataFrame to the output directory
            df.to_csv(output_filepath, index=False)
            print(f"Processed and saved {output_filepath}")
        else:
            print(f"Coordinates not found for airport ID {airport_id}. Skipping file {filename}.")

print("\nUTC conversion complete. All processed CSV files saved.")
