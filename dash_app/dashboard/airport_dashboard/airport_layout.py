
import os
import sys

# Append current directory to system path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import gcsfs
import numpy as np
import pandas as pd
from dash import dcc, html, Output, Input, callback
from dotenv import load_dotenv

# Loading environment variable with sensitive API keys
load_dotenv()

# Initialize Google Cloud Storage FileSystem
fs = gcsfs.GCSFileSystem(project='Flights-Weather-Project', token=os.getenv("gcs_storage_option"))

# Load airport metadata
airport_metdata = f"gs://airport-weather-data/airports-list-us.csv"
df_airport = pd.read_csv(airport_metdata, storage_options={"token": os.getenv("gcs_storage_option")})

# Unique states and cities for dropdown filtering
states = [{'label': state, 'value': state} for state in df_airport['State'].unique()]
years = [2018, 2019, 2020, 2021, 2022, 2023, 2024]
months = {"January": 1, "November": 11, "December": 12}

map_options = [
    {'label': 'Streets', 'value': 'streets-v11'},
    {'label': 'Satellite', 'value': 'satellite-v9'},
    {'label': 'Outdoors', 'value': 'outdoors-v11'},
    {'label': 'Light', 'value': 'light-v10'},
    {'label': 'Dark', 'value': 'dark-v10'},
    {'label': 'Satellite Streets', 'value': 'satellite-streets-v11'},
    {'label': 'Navigation Day', 'value': 'navigation-day-v1'},
    {'label': 'Navigation Night', 'value': 'navigation-night-v1'}
]

airport_color_scales = [
    {'label': 'Viridis', 'value': 'Viridis'},
    {'label': 'Plasma', 'value': 'Plasma'},
    {'label': 'Cividis', 'value': 'Cividis'},
    {'label': 'Inferno', 'value': 'Inferno'},
    {'label': 'Magma', 'value': 'Magma'},
    {'label': 'Turbo', 'value': 'Turbo'},
    {'label': 'Rainbow', 'value': 'Rainbow'},
    {'label': 'Bluered', 'value': 'Bluered'},
    {'label': 'Electric', 'value': 'Electric'}
]

gradient_types = [
    {'label': 'No Gradient', 'value': ''},
    {'label': 'Cancellations', 'value': 'CancellationRate'},
    {'label': 'Average Arrival Delay', 'value': 'AvgArrivalDelay'},
    {'label': 'Average Departure Delay', 'value': 'AvgDepartureDelay'},
    {'label': 'Average Total Flight Delay', 'value': 'AvgTotalFlightDelay'},
    {'label': 'Average Taxi Delay', 'value': 'AvgTaxiDelay'}
]

# Define the layout for the airport dashboard
airport_dashboard_layout = html.Div([
    html.Div([
        # Control panel and layout elements
        html.Div([
            html.H3("Control Panel", className="section-title"),

            # Search Bar Section
            html.Div([
                html.Label('Search Airports', className="label"),
                dcc.Input(
                    id='airport-search-input',
                    type='text',
                    placeholder='Search by name, ID or code...',
                    className="search-input",
                    style={
                        'width': '100%',
                        'padding': '8px',
                        'borderRadius': '4px',
                        'border': '1px solid #ddd',
                        'marginBottom': '10px'
                    }
                ),
                html.Div(
                    id='airport-search-results',
                    className="search-results",
                    style={
                        'maxHeight': '200px',
                        'overflowY': 'auto',
                        'border': '1px solid #ddd',
                        'borderRadius': '4px',
                        'display': 'none'
                    }
                ),
                dcc.Checklist(
                    id='binary_disp_weather_station',
                    options=[{'label': 'Show Nearby Weather Stations', 'value': 'visible'}],
                    value=['visible'],
                    inline=True,
                    className="control-panel-section"
                )
            ], className="control-panel-section"),

            # Location Filters
            html.Div([
                html.Label('Location Filters', className="label"),
                dcc.Dropdown(
                    id='airport-state-selector',
                    options=states,
                    placeholder="Select State",
                    className="dropdown",
                    clearable=True
                ),
                dcc.Dropdown(
                    id='airport-city-selector',
                    placeholder="Select City",
                    className="dropdown",
                    clearable=True
                )
            ], className="control-panel-section"),

            # Marker Settings
            html.Div([
                html.Label('Marker Size', className="label"),
                dcc.Slider(
                    id='airport-marker-size',
                    min=0,
                    max=20,
                    step=1,
                    value=10,
                    marks={i: str(i) for i in range(0, 21, 5)},
                    className='custom-slider'
                ),
                html.Label('Marker Opacity', className="label"),
                dcc.Slider(
                    id='airport-marker-opacity',
                    min=0,
                    max=1,
                    step=0.1,
                    value=0.5,
                    marks={i: str(i) for i in np.linspace(0.0, 1.0, 5)},
                    className='custom-slider'
                )
            ], className="control-panel-section"),

            # Visualization Settings
            html.Div([
                html.Label('Visualization Settings', className="label"),
                dcc.Dropdown(
                    id='airport-mapbox-style-selector',
                    options=map_options,
                    value='navigation-day-v1',
                    className="dropdown",
                    clearable=False
                ),
                dcc.Dropdown(
                    id='gradient-marker-col',
                    options=gradient_types,
                    placeholder="Select Gradient",
                    value='',
                    className="dropdown",
                    clearable=False
                ),
                dcc.Dropdown(
                    id='airport-color-scale-selector',
                    options=airport_color_scales,
                    value='Viridis',
                    className="dropdown",
                    clearable=False
                )
            ], className="control-panel-section"),
        ], className="control-panel", style={'width': '15%', 'float': 'left', 'padding': '10px'}),

        html.Div([
            # Map without loading indicator
            html.Div([
                dcc.Graph(
                    id="airport-enhanced-map",
                    config={"scrollZoom": True},
                    style={'height': '50vh'}
                )
            ], className="map-container", style={'margin-bottom': '20px'}),

            html.Div([
                html.Div(id="airport-info-table", className="station-info-table", style={'margin-bottom': '20px'}),
                html.Div(id="airport-station-info-table", className="station-info-table", style={'margin-bottom': '20px'}),

                # Data Exploration Settings
                html.Div([
                    html.Label('Data Exploration', className="label"),
                    dcc.Dropdown(
                        id='airport-plot-selector',
                        options=[
                            {'label': "Departure vs Arrival Delay", 'value': "Delay Viz"},
                            {'label': "Cancellations", 'value': "Cancel Viz"}
                        ],
                        value="Delay Viz",
                        placeholder="Select Data of Interest",
                        className="dropdown",
                        clearable=False
                    ),
                    dcc.Dropdown(
                        id='airport-year-selector',
                        options=[{'label': str(year), 'value': year} for year in years],
                        value=years[0],
                        placeholder="Select Year",
                        className="dropdown",
                        clearable=False
                    ),
                    dcc.Dropdown(
                        id='airport-month-selector',
                        placeholder="Select Month",
                        value=1,
                        className="dropdown",
                        clearable=False
                    ),
                    html.Button("Update Plot", id="airport-update-plot-button", className="update-plot-button", style={'margin-top': '10px'})
                ], className="time-series-settings-section", style={'margin-bottom': '20px'})
            ], className="station-info-and-settings-container", style={'width': '75%', 'float': 'right', 'padding': '10px'}),

        # Adjust the main dcc.Graph without fixed width/height
        html.Div([
            dcc.Graph(
                id="airport-timeseries-plot",
                style={
                    'width': '100%', 
                    'height': '80vh',  # Adjusts to the viewport height
                    'max-width': '1200px',  # Optionally limit max width for very large screens
                    'margin': '0 auto'  # Center on the page
                },
                config={"responsive": True}
            )
        ], className="timeseries-container", style={'padding': '20px', 'flex-grow': '1'}),
        ], className="main-content", style={'width': '75%', 'float': 'right', 'padding': '10px'})
    ], className="dashboard-container", style={'display': 'flex', 'flex-wrap': 'wrap'})
], style={'max-width': '3000px', 'margin': '0 auto'})

# Update month dropdown options based on the selected year
@callback(
    Output('airport-month-selector', 'options'),
    Input('airport-year-selector', 'value')
)
def update_month_options(selected_year):
    if selected_year == 2024:
        # Only include January for 2024
        available_months = {"January": 1}
    else:
        # Include January, November, and December for other years
        available_months = months
    
    return [{'label': k, 'value': v} for k, v in available_months.items()]
