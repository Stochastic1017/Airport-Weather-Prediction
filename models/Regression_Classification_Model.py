
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# Load your data (assume it's in CSV format)
data = pd.read_csv("your_data_file.csv")

# Feature Engineering
data['CRSDepHour'] = pd.to_datetime(data['UTC_CRSDepTime']).dt.hour
data['CRSArrHour'] = pd.to_datetime(data['UTC_CRSArrTime']).dt.hour
data['CRSDepMonth'] = pd.to_datetime(data['UTC_CRSDepTime']).dt.month
data['CRSDepDayOfWeek'] = pd.to_datetime(data['UTC_CRSDepTime']).dt.dayofweek

# Encode categorical features
data['Marketing_Airline_Network'] = data['Marketing_Airline_Network'].astype('category').cat.codes
data['OriginAirportID'] = data['OriginAirportID'].astype('category').cat.codes
data['DestAirportID'] = data['DestAirportID'].astype('category').cat.codes
data['HourlySkyConditions'] = data['HourlySkyConditions'].astype('category').cat.codes

# Handling missing values for weather features
weather_features = [
    'HourlyDryBulbTemperature', 'HourlyWindSpeed', 'HourlyWindDirection', 
    'HourlyDewPointTemperature', 'HourlyRelativeHumidity', 'HourlyVisibility', 
    'HourlyStationPressure', 'HourlyWetBulbTemperature', 'HourlySkyConditions'
]

# Impute missing values and scale weather features
imputer = SimpleImputer(strategy='mean')
scaler = StandardScaler()
data[weather_features] = imputer.fit_transform(data[weather_features])
data[weather_features] = scaler.fit_transform(data[weather_features])

# Prepare the feature set for model training
X = data[['DayOfWeek', 'Marketing_Airline_Network', 'OriginAirportID', 'DestAirportID', 'Distance', 
          'CRSDepHour', 'CRSArrHour', 'CRSDepMonth', 'CRSDepDayOfWeek'] + weather_features]

# Split dataset for delay predictions
y_delay = data[['ArrivalDelay', 'DepartureDelay', 'TotalFlightDelay', 'TaxiDelay']].fillna(0)
X_train_delay, X_test_delay, y_train_delay, y_test_delay = train_test_split(X, y_delay, test_size=0.2, random_state=42)

# Fit random forest regressor for delay prediction
regressor = RandomForestRegressor(n_estimators=100, random_state=42)
regressor.fit(X_train_delay, y_train_delay)

# Predict and evaluate delay model
y_pred_delay = regressor.predict(X_test_delay)
mse_delay = mean_squared_error(y_test_delay, y_pred_delay)
print(f"Mean Squared Error for Delay Prediction: {mse_delay}")

# Split dataset for cancellation predictions
y_cancel = data['Cancelled'].fillna(0).astype(int)  # Convert to binary if needed
X_train_cancel, X_test_cancel, y_train_cancel, y_test_cancel = train_test_split(X, y_cancel, test_size=0.2, random_state=42)

# Fit random forest classifier for cancellation prediction
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train_cancel, y_train_cancel)

# Predict and evaluate cancellation model
y_pred_cancel = classifier.predict(X_test_cancel)
accuracy_cancel = accuracy_score(y_test_cancel, y_pred_cancel)
print(f"Accuracy for Cancellation Prediction: {accuracy_cancel}")
