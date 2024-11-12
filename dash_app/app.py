
import os
import sys
from dash import Dash, html, dcc
from dash.dependencies import Input, Output

# Append current directory to system path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import layouts for each dashboard
from dashboard.prediction_page.prediction_layout import random_forest_prediction_layout
from dashboard.weather_dashboard.weather_layout import weather_dashboard_layout
from dashboard.airport_dashboard.airport_layout import airport_dashboard_layout

# Import callbacks for each dashboard to ensure they are registered
import dashboard. prediction_page.prediction_callback
import dashboard.weather_dashboard.weather_callback
import dashboard.airport_dashboard.airport_callback

# Initialize the app
app = Dash(__name__, suppress_callback_exceptions=True, 
           external_scripts=['https://cdn.plot.ly/plotly-latest.min.js'])
server = app.server

# Main layout with tabs for different dashboards
app.layout = html.Div([
    dcc.Tabs(id="tabs-dashboard", value='prediction-dashboard', 
             children=[
                dcc.Tab(label='Make Prediction', value='prediction-dashboard', 
                        children=[random_forest_prediction_layout]),
                dcc.Tab(label='Explore Weather Data', value='weather-dashboard', 
                        children=[weather_dashboard_layout]),
                dcc.Tab(label='Explore Airport Data', value='airport-dashboard', 
                        children=[airport_dashboard_layout]),
    ]),
])

# Callback to update the content based on the selected tab
@app.callback(Output('page-content', 'children'), Input('tabs-dashboard', 'value'))
def render_content(tab):
    if tab == 'prediction-dashboard':
        return random_forest_prediction_layout
    elif tab == 'weather-dashboard':
        return weather_dashboard_layout
    elif tab == 'airport-dashboard':
        return airport_dashboard_layout

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
