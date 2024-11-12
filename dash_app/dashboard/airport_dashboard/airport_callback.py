
import os
import sys

# Append current directory to system path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from dash import callback, callback_context, Output, State, Input, html, ALL, ctx
from dash.exceptions import PreventUpdate
import plotly.express as px
import json
from dotenv import load_dotenv
from google.oauth2 import service_account
from .airport_helpers import (create_default_plot, create_airport_map_figure, 
                              create_delay_plots, create_cancellation_plot, get_closest_weather_stations)

# Loading environment variable with sensitive API keys
load_dotenv()

# Mapbox token setup
px.set_mapbox_access_token(os.getenv("mapbox_token"))

with open('/etc/secrets/GCP_CREDENTIALS', 'r') as f:
    credentials_info = json.loads(f.read())  # Read and parse the file contents
    credentials = service_account.Credentials.from_service_account_info(credentials_info,
                                                                        scopes=['https://www.googleapis.com/auth/devstorage.read_write',
                                                                                'https://www.googleapis.com/auth/cloud-platform',
                                                                                'https://www.googleapis.com/auth/drive'])

# Load airport metadata
airport_metdata = f"gs://airport-weather-data/airports-list-us.csv"
df_airport = pd.read_csv(airport_metdata, storage_options={"token": credentials})

# Load airport metadata
weather_metdata = f"gs://airport-weather-data/closest_airport_weather.csv"
df_weather = pd.read_csv(weather_metdata, storage_options={"token": credentials})

@callback(
    [Output('airport-search-results', 'children'),
     Output('airport-search-results', 'style')],
    [Input('airport-search-input', 'value')],
    [State('airport-search-results', 'style')]
)
def update_search_results(search_value, current_style):
    if not search_value:
        return [], {'display': 'none'}
    
    # Filter airports based on search value
    search_value = search_value.lower()
    
    # Convert columns to string type for searching
    matching_airports = df_airport[
        df_airport['DISPLAY_AIRPORT_NAME'].astype(str).str.lower().str.contains(search_value, na=False) |
        df_airport['AIRPORT_ID'].astype(str).str.lower().str.contains(search_value, na=False) |
        df_airport['AIRPORT'].astype(str).str.lower().str.contains(search_value, na=False) |
        df_airport['AIRPORT_SEQ_ID'].astype(str).str.lower().str.contains(search_value, na=False)
    ].head(10)  # Limit to top 10 results
    
    if matching_airports.empty:
        return [html.Div("No matches found", style={'padding': '10px'})], {'display': 'block'}
    
    results = []
    for _, airport in matching_airports.iterrows():
        results.append(
            html.Div(
                f"{airport['DISPLAY_AIRPORT_NAME']} ({airport['AIRPORT_ID']})",
                id={'type': 'airport-search-result', 'index': airport['AIRPORT_ID']},
                className='search-result-item',
                style={
                    'padding': '8px',
                    'cursor': 'pointer',
                    'hover': {'backgroundColor': '#f0f0f0'}
                }
            )
        )
    
    return results, {'display': 'block'}

@callback(
    Output('airport-enhanced-map', 'clickData'),
    [Input({'type': 'airport-search-result', 'index': ALL}, 'n_clicks')],
    [State({'type': 'airport-search-result', 'index': ALL}, 'id')]
)
def handle_search_selection(n_clicks, ids):
    if not ctx.triggered:
        raise PreventUpdate
    
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    airport_id = json.loads(triggered_id)['index']
    
    # Get the airport's data
    airport_data = df_airport[df_airport['AIRPORT_ID'] == airport_id].iloc[0]
    
    # Create click data in the format expected by your existing callback
    click_data = {
        'points': [{
            'lat': airport_data['LATITUDE'],
            'lon': airport_data['LONGITUDE'],
            'hovertext': airport_id  # This matches the format used in your update_map_and_station_info callback
        }]
    }
    
    return click_data

@callback(
    [Output("airport-city-selector", "options"),
     Output("airport-enhanced-map", "figure"),
     Output("airport-info-table", "children")],
    [Input("airport-mapbox-style-selector", "value"),
     Input("airport-marker-size", "value"),
     Input("airport-marker-opacity", "value"),
     Input("gradient-marker-col", "value"),
     Input("airport-color-scale-selector", "value"),
     Input("airport-state-selector", "value"),
     Input("airport-city-selector", "value"),
     Input("airport-enhanced-map", "clickData"),
     Input("binary_disp_weather_station", "value")]
)
def update_map_and_station_info(mapbox_style, marker_size, marker_opacity, gradient_type, color_scale,
                                selected_state, selected_city, click_data, show_weather_station):
    
    # Determine which input triggered the callback
    ctx = callback_context
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    # Filter city options based on the selected state
    if selected_state:
        city_options = [{'label': city, 'value': city} 
                        for city in df_airport[df_airport['State'] == selected_state]['City'].unique()]
    else:
        city_options = [{'label': city, 'value': city} 
                        for city in df_airport['City'].unique()]

    # Filter DataFrame based on selected state and city
    filtered_df = df_airport
    if selected_state:
        filtered_df = filtered_df[filtered_df['State'] == selected_state]
    if selected_city:
        filtered_df = filtered_df[filtered_df['City'] == selected_city]

    # Initialize default map center and zoom level
    center = dict(lat=39.8283, lon=-98.5795)  # Center of the U.S.
    zoom = 3  # Default zoom level for the U.S. view

    # Check if callback was triggered by a marker click
    airport_info = None
    if trigger_id == "airport-enhanced-map" and click_data:
        # Zoom in to the clicked airport's location
        airport_id = click_data['points'][0]['hovertext']
        airport_info = df_airport[df_airport['AIRPORT_ID'] == airport_id].iloc[0]
        try:
            airport_lat, airport_lon = airport_info['LATITUDE'], airport_info['LONGITUDE']
            center = dict(lat=airport_lat, lon=airport_lon)
            zoom = 8  # Closer zoom level for selected airport
        except Exception as e:
            print(f"Error parsing coordinates: {e}")

    # Check if the callback was triggered by marker size or opacity change
    elif trigger_id in ["airport-marker-size", "airport-marker-opacity"]:
        # Reset to show the entire U.S. view
        center = dict(lat=39.8283, lon=-98.5795)
        zoom = 3

    # Create the map figure
    fig = create_airport_map_figure(
        mapbox_style=mapbox_style,
        marker_size=marker_size,
        marker_opacity=marker_opacity,
        filtered_df=filtered_df,
        color_scale=color_scale,
        color_by_metric=gradient_type
    )

    # Update map layout with the determined center and zoom
    fig.update_layout(
        mapbox=dict(
            center=center,
            zoom=zoom
        )
    )

    # Add weather station trails if 'Show Nearby Weather Stations' is selected
    if 'visible' in show_weather_station and airport_info is not None:
        airport_id = airport_info['AIRPORT_ID']
        weather_stations = get_closest_weather_stations(airport_id, df_weather)
        
        if not weather_stations.empty:
            # Add weather stations as scatter points with enhanced hover data
            fig.add_trace(
                px.scatter_mapbox(
                    weather_stations, lat="WEATHER_COORDINATES_Lat", lon="WEATHER_COORDINATES_Lon",
                    hover_name="WEATHER_STATION_NAME",
                    hover_data={
                        "WEATHER_ELEVATION": True,
                        "WEATHER_COUNTRY": True,
                        "WEATHER_STATE": True,
                        "DISTANCE_KM": True
                    }
                ).data[0].update(marker=dict(color="black", size=8))
            )
            # Draw black trails between airport and weather stations
            for _, station in weather_stations.iterrows():
                fig.add_trace(px.line_mapbox(
                    pd.DataFrame({
                        'lat': [airport_info['LATITUDE'], station['WEATHER_COORDINATES_Lat']],
                        'lon': [airport_info['LONGITUDE'], station['WEATHER_COORDINATES_Lon']]
                    }),
                    lat="lat", lon="lon"
                ).data[0].update(line=dict(color="black", width=2)))

    # Generate airport info table if airport_info is available
    if airport_info is not None:
        station_info_table = html.Table([
            html.Tr([html.Th("Airport Info")]),
            html.Tr([html.Td("Name:"), html.Td(airport_info.get("DISPLAY_AIRPORT_NAME"))]),
            html.Tr([html.Td("Airport Code:"), html.Td(airport_info.get("AIRPORT"))]),
            html.Tr([html.Td("Coordinates:"), html.Td(f"({airport_info.get('LATITUDE')}, {airport_info.get('LONGITUDE')})")]),
            html.Tr([html.Td("State:"), html.Td(airport_info.get("State"))]),
            html.Tr([html.Td("City:"), html.Td(airport_info.get("City"))])
        ])
    else:
        station_info_table = html.Table([
            html.Tr([html.Th("No Station Selected")])
        ])

    return city_options, fig, station_info_table

@callback(
    Output("airport-timeseries-plot", "figure"),
    [Input("airport-update-plot-button", "n_clicks")],
    [State("airport-enhanced-map", "clickData"),
     State("airport-year-selector", "value"),
     State("airport-month-selector", "value"),
     State("airport-plot-selector", "value")]
)
def update_visualization(n_clicks, click_data, selected_year, selected_month, selected_plot_type):
    # Validate input: button click, map selection, year, and plot type
    if not n_clicks or not click_data:
        fig = create_default_plot()
        fig.add_annotation(
            text="Please select a airport on the map.",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14)
        )
        return fig

    # Retrieve airport ID and information from the map click data
    airport_id = click_data['points'][0]['hovertext']
    try:
        airport_info = df_airport[df_airport['AIRPORT_ID'] == airport_id].iloc[0]
        airport_name = airport_info['DISPLAY_AIRPORT_NAME']
        airport_city = airport_info['City']
        airport_state = airport_info['State']
        title_info = f"{airport_name} ({airport_id}) - {airport_city}, {airport_state}"
    except IndexError:
        # Handle the case where the airport ID is not found in the dataframe
        fig = create_default_plot()
        fig.add_annotation(
            text="Selected airport data not available.",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14, color="red")
        )
        return fig

    if selected_plot_type == "Delay Viz":
        return create_delay_plots(airport_id, selected_year, selected_month)
    
    if selected_plot_type == "Cancel Viz":
        return  create_cancellation_plot(airport_id, selected_year, selected_month)
