
import os
import sys
import json

# Append current directory to system path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import plotly.graph_objects as go
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
from dotenv import load_dotenv
from google.oauth2 import service_account

# Loading environment variable with sensitive API keys
load_dotenv()

credentials_info = os.getenv("GCP_CREDENTIALS")
credentials = service_account.Credentials.from_service_account_info(json.loads(credentials_info),
                                                                    scopes=['https://www.googleapis.com/auth/devstorage.read_write',
                                                                            'https://www.googleapis.com/auth/cloud-platform',
                                                                            'https://www.googleapis.com/auth/drive'])

# Initial Plot Message
def create_default_plot():
    fig = go.Figure()
    fig.update_layout(
        plot_bgcolor="black",
        paper_bgcolor="black",
        title=None,
        autosize=True,
        template="plotly_dark",
        showlegend=False,
        hovermode='closest',
        margin=dict(t=100, b=50, l=50, r=50)
    )
    return fig

def create_airport_map_figure(mapbox_style, marker_size, marker_opacity, filtered_df, color_scale, color_by_metric=''):
    color_column = color_by_metric if color_by_metric and color_by_metric in filtered_df.columns else None
    hover_data = {
        "DISPLAY_AIRPORT_CITY_NAME_FULL": True,
        "AIRPORT": True,  # Using 3-letter code for better space efficiency
        "State": True,
        "City": True,
        color_column: True if color_column else False
    }

    fig = px.scatter_mapbox(
        filtered_df,
        lat="LATITUDE",
        lon="LONGITUDE",
        hover_name="AIRPORT_ID",
        hover_data=hover_data,
        color=color_column,
        color_continuous_scale=color_scale if color_column else None
    ).update_traces(marker=dict(size=marker_size, opacity=marker_opacity))

    fig.update_layout(
        mapbox=dict(
            style=f"mapbox://styles/mapbox/{mapbox_style}",
            zoom=3.5,
            center={"lat": 37.0902, "lon": -95.7129}
        ),
        coloraxis_colorbar=dict(
            title=color_by_metric.replace("Avg", "Average ").replace("CancellationRate", "Cancellation Rate"),
            ticksuffix="%" if color_by_metric == "CancellationRate" else " min",
        ) if color_column else None,
        autosize=True,
        margin=dict(t=20, b=20, l=20, r=20),
        showlegend=False
    )
    return fig

def create_delay_plots(airport_id, year, month):
    try:
        file_path = f"gs://airport-weather-data/merged_data/{airport_id}_training_data.csv"
        df = pd.read_csv(file_path, storage_options={"token": credentials}, low_memory=False)
        df["UTC_DATE"] = pd.to_datetime(df["UTC_DATE"], errors="coerce")
        df = df[(df["UTC_DATE"].dt.year == year) & (df["UTC_DATE"].dt.month == month)]

        if df.empty:
            fig = go.Figure()
            fig.add_annotation(
                text=f"No data available for airport {airport_id}, year {year}, month {month}",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=18, color="red")
            )
            fig.update_layout(
                template="plotly_dark",
                autosize=True
            )
            return fig

        # Load airport metadata
        airport_metadata = f"gs://airport-weather-data/airports-list-us.csv"
        df_airport = pd.read_csv(airport_metadata, storage_options={"token": credentials})
        df = df.merge(df_airport[["AIRPORT_ID", "AIRPORT"]], left_on="DestAirportID", right_on="AIRPORT_ID", how="left") 

        # Mapping days of the week for better readability
        day_map = {1: "Mon", 2: "Tue", 3: "Wed", 4: "Thu", 5: "Fri", 6: "Sat", 7: "Sun"}
        df["DayOfWeek"] = df["DayOfWeek"].map(day_map)

        delay_types = [
            {"column": "ArrivalDelay", "label": "Arrival Delay", "color": "#1f77b4"},
            {"column": "DepartureDelay", "label": "Departure Delay", "color": "#ff7f0e"},
            {"column": "TotalFlightDelay", "label": "Total Flight Delay", "color": "#2ca02c"},
            {"column": "TaxiDelay", "label": "Taxi Delay", "color": "#d62728"}
        ]

        fig = make_subplots(
            rows=4, cols=4,
            vertical_spacing=0.08, horizontal_spacing=0.03,
            subplot_titles=["Arrival Distribution", "Departure Distribution", "Total Flight Distribution", "Taxi Distribution",
                            "Arrival by Day", "Departure by Day", "Total Flight by Day", "Taxi by Day",
                            "Arrival by Airline", "Departure by Airline", "Total Flight by Airline", "Taxi by Airline",
                            "Arrival by Destination", "Departure by Destination", "Total Flight by Destination", "Taxi by Destination"]
        )

        for idx, dt in enumerate(delay_types):
            col = idx + 1
            delay = dt["column"]
            label = dt["label"]
            color = dt["color"]

            # Row 1: Histogram (Count)
            fig.add_trace(
                go.Histogram(
                    x=df[delay].dropna(),
                    marker=dict(color=color),
                    name=f"{label} Distribution",
                    showlegend=False
                ),
                row=1, col=col
            )

            # Row 2: Box plot by Day of Week (outliers hidden)
            fig.add_trace(
                go.Box(
                    y=df[delay].dropna(),
                    x=df["DayOfWeek"],
                    marker=dict(color=color),
                    name=f"{label} by Day",
                    boxpoints=False,
                    showlegend=False
                ),
                row=2, col=col
            )

            # Row 3: Box plot by Marketing Airline (outliers hidden)
            fig.add_trace(
                go.Box(
                    y=df[delay].dropna(),
                    x=df["Marketing_Airline_Network"],
                    marker=dict(color=color),
                    name=f"{label} by Airline",
                    boxpoints=False,
                    showlegend=False
                ),
                row=3, col=col
            )

            # Row 4: Box plot by Destination Airport (outliers hidden)
            fig.add_trace(
                go.Box(
                    y=df[delay].dropna(),
                    x=df["AIRPORT"],
                    marker=dict(color=color),
                    name=f"{label} by Destination",
                    boxpoints=False,
                    showlegend=False
                ),
                row=4, col=col
            )

        fig.update_layout(
            template="plotly_dark",
            autosize=True,
            margin=dict(t=30, b=20, l=20, r=20),
            showlegend=True,
            legend=dict(
                title="Delay Types",
                orientation="h",
                yanchor="top",
                y=1.1,  # Place above the top of the chart
                xanchor="center",
                x=0.5
            ),
            font=dict(size=8)  # Adjust subplot font size for better fit
        )
        
        return fig

    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error loading data for airport {airport_id}:<br>{str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14, color="red")
        )
        fig.update_layout(template="plotly_dark", autosize=True)
        return fig
    
def create_cancellation_plot(airport_id, year, month):
    try:
        file_path = f"gs://airport-weather-data/merged_data/{airport_id}_training_data.csv"
        df = pd.read_csv(file_path, storage_options={"token": credentials}, low_memory=False)
        df["UTC_DATE"] = pd.to_datetime(df["UTC_DATE"], errors='coerce')
        
        df_airport = pd.read_csv(f"gs://airport-weather-data/airports-list-us.csv", storage_options={"token": credentials})
        df = df[(df["UTC_DATE"].dt.year == year) & (df["UTC_DATE"].dt.month == month)]

        if df.empty:
            fig = go.Figure()
            fig.add_annotation(
                text=f"No data available for airport {airport_id}, year {year}, month {month}",
                xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font=dict(size=18, color="red")
            )
            fig.update_layout(title=f"Data Unavailable - {airport_id} ({year}-{month})", template='plotly_dark', autosize=True)
            return fig
        
        df = df.merge(df_airport[['AIRPORT_ID', 'AIRPORT']], left_on='DestAirportID', right_on='AIRPORT_ID', how='left')  # Using AIRPORT code
        plot_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=["Cancelled vs Not Cancelled Flights", "Cancellations by Day of the Week", "Cancellations by Marketing Airline", "Cancellations by Destination Airport"]
        )
        
        cancel_counts = df['Cancelled'].value_counts()
        fig.add_trace(go.Bar(x=['Not Cancelled', 'Cancelled'], y=[cancel_counts.get(0, 0), cancel_counts.get(1, 0)], marker=dict(color=plot_colors[0])), row=1, col=1)

        day_of_week_cancel = df[df['Cancelled'] == 1]['DayOfWeek'].value_counts().sort_index()
        fig.add_trace(go.Bar(x=day_of_week_cancel.index, y=day_of_week_cancel.values, marker=dict(color=plot_colors[1])), row=1, col=2)

        airline_cancel = df[df['Cancelled'] == 1]['Marketing_Airline_Network'].value_counts()
        fig.add_trace(go.Bar(x=airline_cancel.index, y=airline_cancel.values, marker=dict(color=plot_colors[2])), row=2, col=1)

        destination_cancel = df[df['Cancelled'] == 1]['AIRPORT'].value_counts()  # Using AIRPORT code
        fig.add_trace(go.Bar(x=destination_cancel.index, y=destination_cancel.values, marker=dict(color=plot_colors[3])), row=2, col=2)

        fig.update_layout(
            title=None,
            template="plotly_dark",
            autosize=True,
            margin=dict(t=20, b=20, l=20, r=20),
            showlegend=False,
            font=dict(size=10)  # Adjust subplot title font size
        )
        return fig

    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f"Error loading data for airport {airport_id}:<br>{str(e)}", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font=dict(size=14, color="red"))
        fig.update_layout(title=f"Error - {airport_id} ({year}-{month})", template="plotly_dark", autosize=True)
        return fig

def get_closest_weather_stations(airport_id, df_weather, max_distance=100, max_stations=5):  
    closest_stations = df_weather[df_weather['AIRPORT_ID'] == airport_id]
    closest_stations = closest_stations[closest_stations['DISTANCE_KM'] <= max_distance].nsmallest(max_stations, 'DISTANCE_KM')
    
    # Split WEATHER_COORDINATES column into latitude and longitude for plotting
    closest_stations[['WEATHER_COORDINATES_Lat', 'WEATHER_COORDINATES_Lon']] = closest_stations['WEATHER_COORDINATES'].str.strip('()').str.split(',', expand=True).astype(float)
    
    return closest_stations
    
