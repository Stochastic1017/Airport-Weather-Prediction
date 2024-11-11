
import sys
import os
from glob import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import reverse_geocoder as rg
import ast

# Check if year is provided as a command-line argument
if len(sys.argv) < 2:
    print("Usage: python fetch_metadata.py <year>")
    sys.exit(1)

# Get year from command-line argument
year = sys.argv[1]

# Function to safely parse coordinates
def parse_coord(item):
    try:
        # Attempt to parse and return as tuple of floats
        return ast.literal_eval(item)
    except (ValueError, SyntaxError):
        # Return None if the format is incorrect
        return None
    
# Define the main directory
main_dir = os.getcwd()
os.chdir(f"{main_dir}/{year}")

# List CSV files
files = glob("*.csv")

# Function to process each file and extract required data
def process_file(file):
    try:
        df = pd.read_csv(file, low_memory=False)
        return {
            "station": df["STATION"][0],
            "coords": (df["LATITUDE"][0], df["LONGITUDE"][0]),
            "station_name": df["NAME"][0],
            "elevation": df["ELEVATION"][0]
        }
    except Exception as e:
        print(f"Error processing {file}: {e}")
        return None

# Initialize list to store results
results = []

# Parallel processing of files in chunks
with ThreadPoolExecutor(max_workers=10) as executor:
    future_to_file = {executor.submit(process_file, file): file for file in files}
    for i, future in enumerate(as_completed(future_to_file), 1):
        file = future_to_file[future]
        print(f"Currently at file: {file}. At: {i} out of {len(files)}")
        result = future.result()
        if result:
            results.append(result)

# Change back to the main directory
os.chdir(main_dir)

# Create DataFrame from results
result_df = pd.DataFrame(results)

# Filter and parse coordinates
coordinates = []
valid_indices = []
for idx, item in enumerate(result_df["coords"]):
    coord = parse_coord(str(item))  # Ensure item is a string for parsing
    if coord is not None:
        coordinates.append(coord)
        valid_indices.append(idx)

# Filter result to keep only rows with valid coordinates
df_valid = result_df.iloc[valid_indices].reset_index(drop=True)

# Perform reverse geocoding on the valid coordinates
reverse_geocode = rg.search(coordinates)

# Extract geocode results
latitude = [i["lat"] for i in reverse_geocode]
longitude = [i["lon"] for i in reverse_geocode]
names = [i["name"] for i in reverse_geocode]
admin1 = [i["admin1"] for i in reverse_geocode]
admin2 = [i["admin2"] for i in reverse_geocode]
countries = [i["cc"] for i in reverse_geocode]

# Create the final DataFrame with geocoded information
result_df = pd.DataFrame({
    "station": df_valid["station"].values,
    "station_name": df_valid["station_name"].values,
    "coords": df_valid["coords"].values,
    "elevation": df_valid["elevation"].values,
    "latitude": latitude,
    "longitude": longitude,
    "names": names,
    "admin1": admin1,
    "admin2": admin2,
    "country": countries
})

# Filtering to only US data
result_df = result_df[result_df["country"] == "US"]

# Mapping of state names to two-letter codes
state_code_map = {
    'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR', 'California': 'CA', 
    'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE', 'Florida': 'FL', 'Georgia': 'GA', 
    'Hawaii': 'HI', 'Idaho': 'ID', 'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA', 'Kansas': 'KS', 
    'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD', 'Massachusetts': 'MA', 
    'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS', 'Missouri': 'MO', 'Montana': 'MT', 
    'Nebraska': 'NE', 'Nevada': 'NV', 'New Hampshire': 'NH', 'New Jersey': 'NJ', 'New Mexico': 'NM', 
    'New York': 'NY', 'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH', 'Oklahoma': 'OK', 
    'Oregon': 'OR', 'Pennsylvania': 'PA', 'Rhode Island': 'RI', 'South Carolina': 'SC', 'South Dakota': 'SD', 
    'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT', 'Vermont': 'VT', 'Virginia': 'VA', 'Washington': 'WA', 
    'West Virginia': 'WV', 'Wisconsin': 'WI', 'Wyoming': 'WY', 'Washington, D.C.': 'DC'
}

# Map the state names to state codes and add as a new column
result_df['state'] = result_df['admin1'].map(state_code_map)

# Save result to CSV
print("Saving result.")
result_df.to_csv(f"ncei-lcd-list-{year}.csv", index=False)
print("Completed metadata extraction.")