
import os
import sys
import json

# Append current directory to system path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from dash import dcc, html
from dotenv import load_dotenv
from google.oauth2 import service_account

# Loading environment variable with sensitive API keys
load_dotenv()

credentials_info = os.getenv("GCP_CREDENTIALS")
credentials = service_account.Credentials.from_service_account_info(json.loads(credentials_info),
                                                                    scopes=['https://www.googleapis.com/auth/devstorage.read_write',
                                                                            'https://www.googleapis.com/auth/cloud-platform',
                                                                            'https://www.googleapis.com/auth/drive'])

# Load airport metadata
weather_metdata = f"gs://airport-weather-data/ncei-lcd-list-us.csv"
df_station = pd.read_csv(weather_metdata, storage_options={"token": credentials})

years = [2018, 2019, 2020, 2021, 2022, 2023, 2024]

# Unique states and cities for dropdown filtering
states = [{'label': state, 'value': state} 
          for state in df_station['state'].unique()]

# Available metrics for the time series
metrics = [
    {'label': 'Hourly Dry Bulb Temperature', 'value': 'HourlyDryBulbTemperature'},
    {'label': 'Hourly Wind Speed', 'value': 'HourlyWindSpeed'},
    {'label': 'Hourly Wind Direction', 'value': 'HourlyWindDirection'},
    {'label': 'Hourly Dew Point Temperature', 'value': 'HourlyDewPointTemperature'},
    {'label': 'Hourly Relative Humidity', 'value': 'HourlyRelativeHumidity'},
    {'label': 'Hourly Visibility', 'value': 'HourlyVisibility'},
    {'label': 'Hourly Station Pressure', 'value': 'HourlyStationPressure'},
    {'label': 'Hourly Wet Bulb Temperature', 'value': 'HourlyWetBulbTemperature'},
]

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

weather_color_scales = [
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

# Define the layout for the weather dashboard
weather_dashboard_layout = html.Div([
            html.Div([
                # Control panel and layout elements
                html.Div([
                    html.H3("Control Panel", className="section-title"),

                     html.Div([
                        html.Label('Location Filters', className="label"),
                        dcc.Dropdown(
                            id='weather-state-selector',
                            options=states,
                            placeholder="Select State",
                            className="dropdown",
                            clearable=True,
                            multi=False
                        ),
                        dcc.Dropdown(
                            id='weather-city-selector',
                            placeholder="Select City",
                            className="dropdown",
                            clearable=True,
                            multi=False
                        )
                    ], className="control-panel-section"),
                    
                    html.Div([
                        html.Label('Visualization Settings', className="label"),
                        dcc.Dropdown(
                            id='weather-mapbox-style-selector',
                            options=map_options,
                            value='navigation-day-v1',
                            className="dropdown",
                            clearable=False
                        ),
                        html.Label('Color Scale', className="label"),
                        dcc.Dropdown(
                            id='weather-color-scale-selector',
                            options=weather_color_scales,
                            value='Viridis',
                            className="dropdown",
                            clearable=False
                        )
                    ], className="control-panel-section"),

        # Marker Settings
        html.Div([
            html.Label('Marker Size', className="label"),
            dcc.Slider(
                id='weather-marker-size',
                min=0,
                max=20,
                step=1,
                value=10,
                marks={i: str(i) for i in range(0, 21, 5)},
                className='custom-slider'
            ),
            html.Label('Marker Opacity', className="label"),
            dcc.Slider(
                id='weather-marker-opacity',
                min=0,
                max=1,
                step=0.1,
                value=0.5,
                marks={i: str(i) for i in np.linspace(0.0, 1.0, 5)},
                className='custom-slider'
            )
        ], className="control-panel-section", style={'margin-bottom': '10px'}),
                ], className="control-panel", style={'width': '15%', 'float': 'left', 'padding': '10px'}),

                html.Div([
                    html.Div([
                        dcc.Graph(
                            id="weather-enhanced-map",
                            config={"scrollZoom": True},
                            style={'height': '50vh'}
                        )
                    ], className="map-container", style={'margin-bottom': '20px'}),

                    html.Div([
                        html.Div(id="weather-station-info-table", className="station-info-table", style={'margin-bottom': '20px'}),
                        html.Div([
                            html.Label('Data Exploration Settings', className="label"),
                            dcc.Dropdown(
                                id='weather-year-selector',
                                options=[{'label': str(year), 'value': year} for year in years],
                                value=2018,
                                placeholder="Select Year",
                                className="dropdown time-series-settings-dropdown",
                                searchable=True,
                                clearable=False
                            ),
                            dcc.Dropdown(
                                id='weather-metric-selector',
                                options=metrics,
                                value='HourlyDryBulbTemperature',
                                placeholder="Select Metric",
                                className="dropdown time-series-settings-dropdown",
                                searchable=True,
                                clearable=False
                            ),
                            html.Button("Update Plot", id="weather-update-plot-button", className="update-plot-button", style={'margin-top': '10px'})
                        ], className="time-series-settings-section", style={'margin-top': '20px'})
                    ], className="station-info-and-settings-container", style={'width': '75%', 'float': 'right', 'padding': '10px'}),

                    html.Div([
                        dcc.Graph(
                            id="weather-timeseries-plot",
                            style={'height': '110vh'}
                        )
                    ], className="timeseries-container", style={'margin-top': '20px', 'clear': 'both'})
                ], className="main-content", style={'width': '75%', 'float': 'right', 'padding': '10px'})
            ], className="dashboard-container", style={'display': 'flex', 'flex-wrap': 'wrap'})
        ], style={'max-width': '3000px', 'margin': '0 auto'})
