
import os
import sys
import pandas as pd
from glob import glob

"""
This script filters an airport list by removing rows with IDs that do not have corresponding 
data files in a specified directory. The process includes the following steps:
1. Load a CSV file containing a list of U.S. airports and their IDs.
2. Check each ID in the list against a directory containing CSV files for airports.
3. Keep rows in the airport list only if their ID has a matching CSV file in the directory.
4. Save the filtered airport list to a new CSV file.

Usage:
    python3 filter_airport_list.py --airport_list_file: path to file where airport metadata is downloaded from https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FLL&QO_fu146_anzr=
                                   --airport_data_dir: path to directory where "process_airport_data_by_origin.py" was the output directory (processed, cleaned, and splitted airport data).
                                   --output_file: path to file where filtered airport metadata is saved.

Requirements:
- The 'airport_list_file' should contain an 'AIRPORT_ID' column.
"""

# Check if the correct number of arguments is provided
if len(sys.argv) != 5:
    print("Usage: python3 filter_airport_list.py <working_directory> <airport_data_dir> <airport_list_file> <output_file>")
    sys.exit(1)

# Assign command-line arguments to variables
working_dir = sys.argv[1]
airport_data_dir = sys.argv[2]
airport_list_file = sys.argv[3]
output_file = sys.argv[4]

# Load airport list and get list of airport IDs
airport_df = pd.read_csv(airport_list_file)
airport_ids = list(airport_df["AIRPORT_ID"].astype(str))

# Change directory to where airport data files are stored
os.chdir(airport_data_dir)

# List of available airport data files, without the '.csv' extension
available_files = [file_name.strip(".csv") for file_name in glob("*.csv")]

# List to keep track of IDs without matching data files
removed_rows = []

# Check each airport ID and filter out if there is no matching file
for airport_id in airport_ids:
    if str(airport_id) in available_files:
        print(f"Retaining row with airport ID: {airport_id}")
    else:
        print(f"Removing row with airport ID: {airport_id}")
        removed_rows.append(airport_id)

# Return to the original working directory
os.chdir(working_dir)

# Remove rows from airport_df where 'AIRPORT_ID' is in removed_rows
airport_df = airport_df[~airport_df["AIRPORT_ID"].astype(str).isin(removed_rows)]
airport_df = airport_df.drop_duplicates(subset=["AIRPORT_ID"])

# Save the filtered DataFrame to a new CSV
airport_df.to_csv(output_file, index=False)
print(f"Filtered airport list saved to {output_file}.")
