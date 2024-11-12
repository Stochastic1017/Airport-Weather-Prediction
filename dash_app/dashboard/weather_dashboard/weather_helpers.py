
import os
import sys
import json

# Append current directory to system path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import gcsfs
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from scipy import stats
from plotly.subplots import make_subplots
from dotenv import load_dotenv
from google.oauth2 import service_account

import warnings
warnings.simplefilter("ignore", category=FutureWarning)

with open('/etc/secrets/GCP_CREDENTIALS', 'r') as f:
    credentials_info = json.loads(f.read())  # Read and parse the file contents
    credentials = service_account.Credentials.from_service_account_info(credentials_info,
                                                                        scopes=['https://www.googleapis.com/auth/devstorage.read_write',
                                                                                'https://www.googleapis.com/auth/cloud-platform',
                                                                                'https://www.googleapis.com/auth/drive'])

# Loading environment variable with sensitive API keys
load_dotenv()

# Initialize Google Cloud Storage FileSystem
fs = gcsfs.GCSFileSystem(project='Flights-Weather-Project', token=credentials)

def create_weather_map_figure(mapbox_style, marker_size, marker_opacity, 
                              weather_color_scale, filtered_df, center=None, zoom=3.5):
    fig = px.scatter_mapbox(
        filtered_df,
        lat="latitude",
        lon="longitude",
        hover_name="station",
        hover_data={"station_name": True, "elevation": True, "admin1": True, "admin2": True},
        color="elevation",
        color_continuous_scale=weather_color_scale,
        range_color=[filtered_df['elevation'].min(), filtered_df['elevation'].max()],
    ).update_traces(marker=dict(size=marker_size, opacity=marker_opacity))
    fig.update_layout(
        mapbox=dict(
            style=f"mapbox://styles/mapbox/{mapbox_style}",
            zoom=zoom,
            center=center or {"lat": 37.0902, "lon": -95.7129},  # Default center for the US
        ),
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        showlegend=False
    )
    return fig

def create_timeseries_plot(station, year, metric, title_info):
    try:
        file_path = f"gs://airport-weather-data/ncei-lcd/{station}.csv"
        df = pd.read_csv(file_path, storage_options={"token": credentials}, low_memory=False)
        df["UTC_DATE"] = pd.to_datetime(df["UTC_DATE"], errors='coerce')
        months_to_plot = [1, 11, 12]  # January, November, December
        filtered_df = df[(df["UTC_DATE"].dt.month.isin(months_to_plot)) & 
                         (df["UTC_DATE"].dt.year == year)].copy()
        
        if filtered_df.empty:
            raise ValueError(f"Data not available for {metric}: {station}, {year}")        
       
        filtered_df[metric] = pd.to_numeric(filtered_df[metric], errors='coerce')
        
        fig = make_subplots(
            rows=3, cols=3,
            shared_xaxes=False,
            column_widths=[0.75, 0.15, 0.15],
            vertical_spacing=0.1,
            horizontal_spacing=0.02,
            subplot_titles=[
                "January Time Series", "January Distribution", "January Stats",
                "November Time Series", "November Distribution", "November Stats",
                "December Time Series", "December Distribution", "December Stats"
            ],
            specs=[[{"type": "scatter"}, {"type": "xy"}, {"type": "table"}],
                    [{"type": "scatter"}, {"type": "xy"}, {"type": "table"}],
                    [{"type": "scatter"}, {"type": "xy"}, {"type": "table"}]]
        )
        
        colors = {1: '#00B4D8', 11: '#4C9A2A', 12: '#EE6C4D'}  # Blue, Green, Red
        
        for i, month in enumerate(months_to_plot, 1):
            month_df = filtered_df[filtered_df["UTC_DATE"].dt.month == month].copy()
            
            if not month_df.empty:
                kde_y = month_df[metric].dropna().astype(float)
                mean_y = kde_y.mean()
                std_y = kde_y.std()
                
                if len(kde_y) > 1:
                    month_df["UTC_DATE"] = pd.to_datetime(month_df["UTC_DATE"], errors="coerce")
                    plot_dates = np.array(month_df["UTC_DATE"].dt.to_pydatetime(), dtype='datetime64[ns]')
                    
                    kde = stats.gaussian_kde(kde_y)
                    kde_x = np.linspace(kde_y.min() - std_y, kde_y.max() + std_y, 100)
                    kde_points = kde(kde_x)
                    
                    # Time Series Plot
                    fig.add_trace(
                        go.Scatter(
                            x=plot_dates,
                            y=kde_y,
                            mode="lines",
                            line=dict(color=colors[month], width=2)
                        ),
                        row=i, col=1
                    )
                    
                    # Histogram and KDE
                    fig.add_trace(
                        go.Histogram(
                            y=kde_y,
                            histnorm='probability density',
                            showlegend=False,
                            marker=dict(color=colors[month], opacity=0.3),
                            nbinsy=30
                        ),
                        row=i, col=2
                    )
                    
                    fig.add_trace(
                        go.Scatter(
                            x=kde_points,
                            y=kde_x,
                            mode="lines",
                            showlegend=False,
                            line=dict(color=colors[month], width=2)
                        ),
                        row=i, col=2
                    )
                    
                    # Create summary table
                    summary_table_data = [
                        ['Mean', f"{mean_y:.2f}"],
                        ['Median', f"{kde_y.median():.2f}"],
                        ['Std Dev', f"{std_y:.2f}"],
                        ['Min', f"{kde_y.min():.2f}"],
                        ['Max', f"{kde_y.max():.2f}"],
                        ['Missing Data (%)', f"{(kde_y.isna().sum() / len(kde_y)) * 100:.2f}%"],
                        ['Skewness', f"{kde_y.skew():.2f}"],
                        ['Kurtosis', f"{kde_y.kurtosis():.2f}"],
                    ]
                    fig.add_trace(
                        go.Table(
                            header=dict(
                                values=["<b>Statistic<b>", "<b>Value<B>"],
                                fill_color=colors[month],  # Header color matches line color
                                align='center',
                                font=dict(color='white', size=11),
                                line_color=colors[month],
                            ),
                            cells=dict(
                                values=[list(x) for x in zip(*summary_table_data)],
                                fill_color=colors[month],
                                align='center',
                                font=dict(color="white", size=10),  # Cell font color matches line color
                                line_color=colors[month],
                            )
                        ),
                        row=i, col=3
                    )
        
        fig.update_layout(
            title=dict(
                text=f"{metric} Analysis<br><sup>{title_info}</sup>",
                font=dict(size=22),
                y=0.98, 
                x=0.5
            ),
            autosize=True,  # Enable autosizing
            template="plotly_dark",
            showlegend=False,
            hovermode='closest',
            margin=dict(t=100, b=50, l=50, r=50)
        )
        
        return fig
        
    except Exception as e:
        print(f"Error: {str(e)}")
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error loading data for station {station}:<br>{str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14, color="red")
        )
        fig.update_layout(
            title=f"Error - {station} ({year})",
            template='plotly_dark',
            autosize=True,  # Enable autosizing for error plot as well
            margin=dict(t=100, b=50, l=50, r=50),
        )
        return fig
