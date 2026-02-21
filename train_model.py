import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import warnings

warnings.filterwarnings('ignore')

def load_and_preprocess_data(file_path):
    print("Loading dataset...")
    df = pd.read_csv(file_path)
    
    # 1. Handle Timestamp columns
    print("Processing timestamps...")
    time_cols = ['Scheduled_Departure', 'Scheduled_Arrival', 'Actual_Departure', 'Actual_Arrival']
    for col in time_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    # 2. Derive Features
    # Extract hour and minute from scheduled departure
    if 'Scheduled_Departure' in df.columns:
        df['Scheduled_Dep_Hour'] = df['Scheduled_Departure'].dt.hour
        df['Scheduled_Dep_Minute'] = df['Scheduled_Departure'].dt.minute
        df['Scheduled_Dep_Month'] = df['Scheduled_Departure'].dt.month
        df['Scheduled_Dep_DayOfWeek'] = df['Scheduled_Departure'].dt.dayofweek

    # 3. Handle Target Variable (Arrival_Delay_min)
    if 'Arrival_Delay_min' not in df.columns:
        raise ValueError("Target column 'Arrival_Delay_min' not found.")
    
    # Drop rows where target is missing (if any)
    df = df.dropna(subset=['Arrival_Delay_min'])

    # 4. Feature Selection
    # Select numeric and categorical features
    numeric_features = [
        'Distance_km', 'Scheduled_Travel_Time_min', 'Previous_Train_Delay_min', 
        'Number_of_Stops', 'Passenger_Load_pct', 'Temperature_C', 'Humidity_pct',
        'Precipitation_mm', 'WindSpeed_kmph', 'Visibility_km', 'Pressure_hPa'
    ]
    categorical_features = [
        'Day_of_Week', 'Is_Holiday', 'Festive_Period', 'Weather', 
        'Track_Maintenance', 'Signal_Failure', 'Engine_Breakdown', 
        'Crew_Change', 'Loco_Type', 'WeatherCondition'
    ]
    
    # Keep only available columns
    numeric_features = [c for c in numeric_features if c in df.columns]
    categorical_features = [c for c in categorical_features if c in df.columns]

    # Force numeric columns to be numeric (coerce errors to NaN)
    for col in numeric_features:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    features = numeric_features + categorical_features
    X = df[features].copy()
    y = df['Arrival_Delay_min']

    # 5. Impute Missing Values
    print("Imputing missing values...")
    # Numeric: Median
    num_imputer = SimpleImputer(strategy='median')
    X[numeric_features] = num_imputer.fit_transform(X[numeric_features])
    
    # Categorical: Mode (Constant 'Unknown' if preferred, but mode is safer for generic)
    cat_imputer = SimpleImputer(strategy='most_frequent')
    X[categorical_features] = cat_imputer.fit_transform(X[categorical_features])

    # 6. Encode Categorical Variables
    print("Encoding categorical variables...")
    label_encoders = {}
    for col in categorical_features:
        le = LabelEncoder()
        # Ensure all types are string for encoding
        X[col] = X[col].astype(str)
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le

    return X, y, num_imputer, cat_imputer, label_encoders, features

def train_model(X, y):
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training Random Forest Regressor (this may take a moment)...")
    # Reduced n_estimators for speed in this demo, increase for production
    model = RandomForestRegressor(n_estimators=50, max_depth=15, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print("-" * 30)
    print(f"Model Performance:")
    print(f"MAE: {mae:.2f} minutes")
    print(f"RMSE: {rmse:.2f} minutes")
    print(f"R2 Score: {r2:.4f}")
    print("-" * 30)
    
    return model

def main():
    file_path = 'stratquest_dataset.csv'
    
    try:
        # Preprocess
        X, y, num_imputer, cat_imputer, encoders, feature_names = load_and_preprocess_data(file_path)
        
        # Train
        model = train_model(X, y)
        
        # Save Artifacts
        print("Saving model artifacts...")
        artifacts = {
            'model': model,
            'num_imputer': num_imputer,
            'cat_imputer': cat_imputer,
            'encoders': encoders,
            'feature_names': feature_names
        }
        joblib.dump(artifacts, 'train_delay_model_artifacts.pkl')
        print("Model saved to 'train_delay_model_artifacts.pkl'")
        
    except FileNotFoundError:
        print("Error: Dataset 'stratquest_dataset.csv' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
