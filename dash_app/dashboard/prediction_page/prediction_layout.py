
import os
import sys

# Append current directory to system path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import gcsfs
import pandas as pd
from dash import dcc, html, callback, Input, Output
import plotly.express as px
from dotenv import load_dotenv

# Loading environment variable with sensitive API keys
load_dotenv()

# Initialize Google Cloud Storage FileSystem
fs = gcsfs.GCSFileSystem(project='Flights-Weather-Project', token="flights-weather-project-33452f8b2c56.json")

# Load options for prediction from the CSV
df_options = pd.read_csv("gs://airport-weather-data/options_for_prediction.csv", 
                         storage_options={"token": "flights-weather-project-33452f8b2c56.json"})
df_airports = pd.read_csv("gs://airport-weather-data/airports-list-us.csv", 
                          storage_options={"token": "flights-weather-project-33452f8b2c56.json"})

# Merge options for prediction
df_merged = df_options.merge(df_airports[['AIRPORT_ID', 'LATITUDE', 'LONGITUDE']], 
                             left_on='airport_id', 
                             right_on='AIRPORT_ID',
                             how='left')

# Dropdown options
airline_options = [{"label": airline, "value": airline} for airline in df_merged['airline'].dropna().unique()]
airport_options = [
    {
        "label": f"{row['airport_display_name']} ({row['airport_code']}, ID: {int(row['airport_id'])})",
        "value": int(row['airport_id'])
    }
    for _, row in df_merged.dropna(subset=['airport_id', 'airport_display_name', 'airport_code']).iterrows()
]

# Latitude and longitude lookup
airport_coordinates = {
    int(row['airport_id']): {"latitude": row['LATITUDE'], "longitude": row['LONGITUDE']}
    for _, row in df_merged.dropna(subset=['airport_id', 'LATITUDE', 'LONGITUDE']).iterrows()
}

# Define layout for the prediction page
random_forest_prediction_layout = html.Div(
    className="prediction-container",
    children=[
        # Header
        html.H1(
            "Flight Delay & Cancellation Prediction",
            className="prediction-header",
            style={
                "textAlign": "center",
                "color": "white",
                "marginBottom": "30px",
                "fontSize": "2.5rem",
                "fontWeight": "bold",
                "backgroundColor": "#1e2a38",
                "padding": "25px",
                "borderRadius": "8px",
                "boxShadow": "0 4px 10px rgba(0,0,0,0.3)"
            }
        ),
        
        # Input and Map Sections
        html.Div(
            className="prediction-sections",
            style={
                "display": "grid",
                "gridTemplateColumns": "1fr 1fr",
                "gap": "40px",
                "maxWidth": "1400px",
                "margin": "0 auto",
                "padding": "20px"
            },
            children=[
                # Input Section
                html.Div(
                    className="input-section",
                    children=[
                        html.Div(
                            className="input-group",
                            children=[
                                html.Label("Airline", className="input-label"),
                                html.Div(
                                    dcc.Dropdown(
                                        id="airline-input",
                                        options=airline_options,
                                        placeholder="Select Airline",
                                        className="dropdown",
                                        style={"width": "100%", "fontSize": "1rem", "padding": "10px"},
                                        searchable=True
                                    ),
                                    className="dropdown-wrapper",
                                    style={"zIndex": 5}
                                ),
                                html.Div(id="airline-error", className="error-message", style={'color': 'red', "marginTop": "5px"}),

                                html.Label("Flight Date", className="input-label"),
                                html.Div(
                                    dcc.DatePickerSingle(
                                        id="date-input",
                                        placeholder="Select Date",
                                        display_format="YYYY-MM-DD",
                                        min_date_allowed="2024-11-01",
                                        max_date_allowed="2024-12-31",
                                        date="2024-11-01",
                                        initial_visible_month="2024-11-01",
                                        style={
                                            "width": "100%",
                                            "padding": "10px",
                                            "fontSize": "1rem",
                                            "borderRadius": "10px",
                                            "border": "1px solid #334155",
                                            "backgroundColor": "#1e293b",
                                            "color": "#cbd5e1"
                                        }
                                    ),
                                    className="date-picker-wrapper",
                                    style={"zIndex": 3}
                                )
                            ],
                            style={"marginBottom": "20px"}
                        ),

                        html.Div(
                            className="input-group",
                            children=[
                                html.Label("Origin Airport", className="input-label"),
                                dcc.Dropdown(
                                    id="origin-airport-input",
                                    options=airport_options,
                                    placeholder="Select Origin Airport",
                                    className="dropdown",
                                    style={"width": "100%", "fontSize": "1rem", "padding": "10px"},
                                    searchable=True
                                ),
                                html.Label("Departure Time (Local)", className="input-label"),
                                dcc.Input(
                                    id="departure-time-input",
                                    type="text",
                                    placeholder="HH:MM (24-Hour)",
                                    className="time-picker",
                                    style={
                                        "width": "100%",
                                        "padding": "10px",
                                        "fontSize": "1rem",
                                        "borderRadius": "8px",
                                        "border": "1px solid #334155",
                                        "backgroundColor": "#1e293b",
                                        "color": "#cbd5e1"
                                    }
                                )
                            ],
                            style={"marginBottom": "20px"}
                        ),

                        html.Div(
                            className="input-group",
                            children=[
                                html.Label("Destination Airport", className="input-label"),
                                dcc.Dropdown(
                                    id="destination-airport-input",
                                    options=airport_options,
                                    placeholder="Select Destination Airport",
                                    className="dropdown",
                                    style={"width": "100%", "fontSize": "1rem", "padding": "10px"},
                                    searchable=True
                                ),
                                html.Label("Arrival Time (Local)", className="input-label"),
                                dcc.Input(
                                    id="arrival-time-input",
                                    type="text",
                                    placeholder="HH:MM (24-Hour)",
                                    className="time-picker",
                                    style={
                                        "width": "100%",
                                        "padding": "10px",
                                        "fontSize": "1rem",
                                        "borderRadius": "8px",
                                        "border": "1px solid #334155",
                                        "backgroundColor": "#1e293b",
                                        "color": "#cbd5e1"
                                    }
                                )
                            ],
                            style={"marginBottom": "20px"}
                        ),
                        
                        # Prediction button
                        html.Div(
                            html.Button(
                                "Predict Delays and Cancellation",
                                id="predict-button",
                                className="predict-button",
                                n_clicks=0,
                                style={
                                    "backgroundColor": "#5c6bc0",
                                    "color": "white",
                                    "fontWeight": "bold",
                                    "border": "none",
                                    "padding": "15px 30px",
                                    "fontSize": "1.2rem",
                                    "cursor": "pointer",
                                    "borderRadius": "8px",
                                    "boxShadow": "0 4px 8px rgba(0,0,0,0.2)"
                                }
                            ),
                            style={"textAlign": "center", "marginTop": "20px"}
                        )
                    ],
                    style={
                        "padding": "20px",
                        "border": "1px solid #e0e0e0",
                        "borderRadius": "8px",
                        "backgroundColor": "#1e293b",
                        "width": "100%"
                    }
                ),

                # Map Section with Loading Spinner
                dcc.Loading(
                    id="loading-map",
                    type="cube",
                    children=html.Div(
                        dcc.Graph(
                            id="flight-map",
                            style={
                                "width": "100%",
                                "height": "500px",
                                "border": "1px solid #e0e0e0",
                                "borderRadius": "8px"
                            }
                        ),
                        style={
                            "padding": "20px",
                            "border": "1px solid #e0e0e0",
                            "borderRadius": "8px",
                            "boxShadow": "0 4px 8px rgba(0,0,0,0.1)",
                            "backgroundColor": "#1e293b",
                            "width": "100%",
                            "marginTop": "20px",
                            "textAlign": "center"  # Center-align the map in the container
                        }
                    )
                )
            ]
        ),

        # Prediction Output Section with Cube Loading Animation
        dcc.Loading(
            id="loading-prediction",
            type="cube",
            children=html.Div(
                id="prediction-output",
                style={
                    "padding": "30px",
                    "color": "#2b3e50",
                    "fontSize": "1.3rem",
                    "backgroundColor": "#e0f7fa",
                    "borderRadius": "8px",
                    "boxShadow": "0 4px 8px rgba(0,0,0,0.1)",
                    "textAlign": "center",
                    "marginTop": "20px"
                }
            ),
            style={"marginTop": "50px"}
        )
    ]
)


@callback(
    Output("flight-map", "figure"),
    Input("origin-airport-input", "value"),
    Input("destination-airport-input", "value")
)
def update_map(origin_airport, destination_airport):
    # Retrieve airport coordinates from the dictionary
    origin_data = airport_coordinates.get(origin_airport)
    dest_data = airport_coordinates.get(destination_airport)

    if origin_data and dest_data:
        # Set up map data for origin and destination
        map_data = pd.DataFrame({
            'lat': [origin_data['latitude'], dest_data['latitude']],
            'lon': [origin_data['longitude'], dest_data['longitude']],
            'text': ['Origin Airport', 'Destination Airport']
        })
        
        # Create the map figure with an arrowed line
        fig = px.scatter_mapbox(
            map_data,
            lat='lat',
            lon='lon',
            text='text',
            zoom=4,
            mapbox_style="carto-positron"
        )
        
        # Add line trace with theme color
        fig.add_trace(
            px.line_mapbox(
                map_data,
                lat='lat',
                lon='lon'
            ).data[0].update(line=dict(color="#5c6bc0"))  # Set line color to match theme
        )

        return fig

    # Return an empty figure if either airport is missing
    return {}
