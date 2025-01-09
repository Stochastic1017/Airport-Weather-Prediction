
import pandas as pd
import numpy as np
import time
import pickle
import warnings
from concurrent.futures import ThreadPoolExecutor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectFromModel

warnings.filterwarnings('ignore')

def create_interaction_features(data):
    """Create interaction features between weather conditions"""
    print("Creating weather interaction features...")
    interactions = pd.DataFrame()
    interactions['temp_humidity'] = data['HourlyDryBulbTemperature'] * data['HourlyRelativeHumidity']
    interactions['temp_dewpoint_diff'] = data['HourlyDryBulbTemperature'] - data['HourlyDewPointTemperature']
    interactions['visibility_wind'] = data['HourlyVisibility'] * data['HourlyWindSpeed']
    interactions['wind_pressure'] = data['HourlyWindSpeed'] * data['HourlyStationPressure']
    interactions['wind_direction_speed'] = np.sin(np.radians(data['HourlyWindDirection'])) * data['HourlyWindSpeed']
    interactions['temp_range'] = data['HourlyDryBulbTemperature'] - data['HourlyWetBulbTemperature']
    return interactions

def process_data_chunk(chunk, label_encoders):
    """Process a single chunk of data"""
    # Convert datetime columns to numerical features
    chunk['UTC_CRSDepTime'] = pd.to_datetime(chunk['UTC_CRSDepTime'])
    chunk['UTC_CRSArrTime'] = pd.to_datetime(chunk['UTC_CRSArrTime'])
    
    # Create numerical features from datetime
    chunk['CRSDepHour'] = chunk['UTC_CRSDepTime'].dt.hour
    chunk['CRSArrHour'] = chunk['UTC_CRSArrTime'].dt.hour
    chunk['CRSDepMonth'] = chunk['UTC_CRSDepTime'].dt.month
    chunk['CRSDepDayOfWeek'] = chunk['UTC_CRSDepTime'].dt.dayofweek
    
    # Drop original datetime columns
    chunk.drop(columns=['UTC_CRSDepTime', 'UTC_CRSArrTime'], inplace=True)
    
    # Encode categorical features
    print(chunk.columns)
    categorical_features = ['Marketing_Airline_Network', 'OriginAirportID', 
                            'DestAirportID', 'HourlySkyConditions']
    
    for feature in categorical_features:
        chunk[feature] = label_encoders[feature].transform(chunk[feature].astype(str))
    
    # Create weather interaction features
    interactions = create_interaction_features(chunk)
    chunk = pd.concat([chunk, interactions], axis=1)
    
    return chunk

def load_and_preprocess_data(file_path, chunk_size=100000):
    """Load and preprocess data in chunks"""
    print("Reading and processing data in chunks...")
    dtypes = {'Marketing_Airline_Network': 'category', 'OriginAirportID': 'category', 
              'DestAirportID': 'category', 'Distance': 'float32', 'Cancelled': 'int8'}
    
    label_encoders = {feature: LabelEncoder() for feature in dtypes if feature != 'Cancelled'}
    for feature in label_encoders:
        unique_values = pd.read_csv(file_path, usecols=[feature])
        label_encoders[feature].fit(unique_values[feature].astype(str))
    
    chunks = [process_data_chunk(chunk, label_encoders) 
              for chunk in pd.read_csv(file_path, dtype=dtypes, chunksize=chunk_size)]
    return pd.concat(chunks, axis=0), label_encoders

def create_stacking_model():
    """Create a stacking model with multiple base estimators"""
    base_estimators = [
        ('xgb', XGBRegressor(tree_method='hist', random_state=42)),
        ('lgbm', LGBMRegressor(random_state=42)),
        ('rf', RandomForestRegressor(n_estimators=100, random_state=42))
    ]
    return StackingRegressor(
        estimators=base_estimators,
        final_estimator=CatBoostRegressor(verbose=False, random_state=42),
        cv=5,
        n_jobs=-1
    )

def select_features(X, y, feature_names):
    """Select important features using multiple models in parallel"""
    print("Performing feature selection...")
    models = [XGBRegressor(tree_method='hist', random_state=42),
              LGBMRegressor(random_state=42),
              RandomForestRegressor(n_estimators=100, random_state=42)]
    
    def fit_and_get_importance(model):
        selector = SelectFromModel(model)
        selector.fit(X, y)
        return selector.estimator_.feature_importances_
    
    # Execute in parallel
    with ThreadPoolExecutor() as executor:
        feature_importances = list(executor.map(fit_and_get_importance, models))
    
    # Average feature importances
    feature_importance = np.mean(feature_importances, axis=0)
    
    threshold = np.mean(feature_importance)
    selected_features = feature_importance > threshold
    print("\nSelected features and their importance scores:")
    for feat, imp in zip(feature_names, feature_importance):
        if imp > threshold:
            print(f"{feat}: {imp:.4f}")
    return selected_features

def evaluate_with_cv(model, X, y, cv=5):
    """Evaluate model using cross-validation"""
    cv_scores = cross_val_score(
        model, X, y, 
        cv=KFold(n_splits=cv, shuffle=True, random_state=42),
        scoring='neg_root_mean_squared_error',
        n_jobs=-1
    )
    return -cv_scores.mean(), cv_scores.std()

def main():
    weather_features = ['HourlyDryBulbTemperature', 'HourlyWindSpeed', 'HourlyWindDirection',
                        'HourlyDewPointTemperature', 'HourlyRelativeHumidity', 'HourlyVisibility',
                        'HourlyStationPressure', 'HourlyWetBulbTemperature']
    base_features = ['DayOfWeek', 'Marketing_Airline_Network', 'OriginAirportID', 
                     'DestAirportID', 'Distance', 'CRSDepHour', 'CRSArrHour', 
                     'CRSDepMonth', 'CRSDepDayOfWeek'] + weather_features
    
    data, label_encoders = load_and_preprocess_data("airport-weather-data.csv")
    
    interaction_features = [col for col in data.columns 
                            if col not in base_features + ['ArrivalDelay', 'DepartureDelay', 
                                                           'TotalFlightDelay', 'TaxiDelay', 'Cancelled']]
    all_features = base_features + interaction_features
    X = data[all_features].astype(np.float32)
    y_delay = data[['ArrivalDelay', 'DepartureDelay', 'TotalFlightDelay', 'TaxiDelay']].fillna(0).astype(np.float32)
    
    selected_features = select_features(X, y_delay['ArrivalDelay'], all_features)
    X_selected = X.iloc[:, selected_features]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y_delay, test_size=0.2, random_state=42)
    
    delay_types = ['ArrivalDelay', 'DepartureDelay', 'TotalFlightDelay', 'TaxiDelay']
    stacking_models = {}
    
    print("\nTraining and evaluating models:")
    for delay_type in delay_types:
        print(f"\nProcessing {delay_type}:")
        stacking_model = create_stacking_model()
        
        cv_rmse, cv_std = evaluate_with_cv(stacking_model, X_train, y_train[delay_type])
        print(f"Cross-validation RMSE: {cv_rmse:.2f} (+/- {cv_std:.2f})")
        
        stacking_model.fit(X_train, y_train[delay_type])
        y_pred = stacking_model.predict(X_test)
        test_mse = mean_squared_error(y_test[delay_type], y_pred)
        print(f"Test set RMSE: {test_mse**(1/2):.2f}")
        
        stacking_models[delay_type] = stacking_model
    
    print("\nSaving models and feature information...")
    with open("stacking_models.pkl", "wb") as file:
        pickle.dump({
            'models': stacking_models,
            'selected_features': selected_features,
            'feature_names': all_features
        }, file)

if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"\nTotal execution time: {(time.time() - start_time) / 60:.2f} minutes")
