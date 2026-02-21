import pandas as pd
import joblib
import numpy as np

def load_model_artifacts(filename='train_delay_model_artifacts.pkl'):
    print(f"Loading model artifacts from {filename}...")
    try:
        artifacts = joblib.load(filename)
        return artifacts
    except FileNotFoundError:
        print(f"Error: {filename} not found. Please run train_model.py first.")
        return None

def predict_delay(input_data, artifacts):
    """
    Predicts delay for a single dictionary of input data.
    """
    model = artifacts['model']
    num_imputer = artifacts['num_imputer']
    cat_imputer = artifacts['cat_imputer']
    encoders = artifacts['encoders']
    feature_names = artifacts['feature_names']
    
    # Convert input to DataFrame
    df = pd.DataFrame([input_data])
    
    # 1. Handle Derived Features just like training
    if 'Scheduled_Departure' in df.columns:
        df['Scheduled_Departure'] = pd.to_datetime(df['Scheduled_Departure'])
        df['Scheduled_Dep_Hour'] = df['Scheduled_Departure'].dt.hour
        df['Scheduled_Dep_Minute'] = df['Scheduled_Departure'].dt.minute
        df['Scheduled_Dep_Month'] = df['Scheduled_Departure'].dt.month
        df['Scheduled_Dep_DayOfWeek'] = df['Scheduled_Departure'].dt.dayofweek
    
    # Ensure all feature columns exist, fill with NaN if missing
    for col in feature_names:
        if col not in df.columns:
            df[col] = np.nan
            
    X = df[feature_names].copy()
    
    # Split features into numeric and categorical again
    # We can infer from the imputer feature_names_in_ if available, or just use try/except logic
    # Detailed way: recreate lists based on what we had.
    # Simpler way for this demo: use the column types from the dataframe after imputation
    
    # But to be robust, we should align with how we trained.
    # The artifacts object works, but sklearn 1.0+ imputers have feature_names_in_
    # Let's trust the column names match what we used in training logic.
    
    numeric_features = [
        'Distance_km', 'Scheduled_Travel_Time_min', 'Previous_Train_Delay_min', 
        'Number_of_Stops', 'Passenger_Load_pct', 'Temperature_C', 'Humidity_pct',
        'Precipitation_mm', 'WindSpeed_kmph', 'Visibility_km', 'Pressure_hPa'
    ]
    # Filter to what is actually in feature_names
    numeric_features = [f for f in numeric_features if f in feature_names]
    
    categorical_features = [
        'Day_of_Week', 'Is_Holiday', 'Festive_Period', 'Weather', 
        'Track_Maintenance', 'Signal_Failure', 'Engine_Breakdown', 
        'Crew_Change', 'Loco_Type', 'WeatherCondition'
    ]
    categorical_features = [f for f in categorical_features if f in feature_names]

    # Impute
    X[numeric_features] = num_imputer.transform(X[numeric_features])
    X[categorical_features] = cat_imputer.transform(X[categorical_features])
    
    # Encode
    for col in categorical_features:
        le = encoders[col]
        X[col] = X[col].astype(str)
        
        # Handle unseen labels
        # A simple trick: map to a known class or simple error handling
        # For this demo, we'll try transform and if error, fallback to 0 or valid class
        try:
             X[col] = le.transform(X[col])
        except ValueError:
            # If label seen in test is new, assigning the most common (0 usually if strictly encoded)
            # or we can assume the encoder handles unknown if configured (LabelEncoder doesn't natively)
            # We will just warn and set to 0
            print(f"Warning: Unseen label in column {col}. Defaulting to 0.")
            X[col] = 0

    # Predict
    prediction = model.predict(X)[0]
    return prediction

def main():
    artifacts = load_model_artifacts()
    if artifacts is None:
        return

    # Sample Input (Modify this to test different scenarios)
    sample_input = {
        'Distance_km': 530,
        'Scheduled_Departure': '2024-06-24 21:37:52', # Will be parsed
        'Scheduled_Travel_Time_min': 818,
        'Day_of_Week': 'Monday',
        'Month': 6,
        'Is_Holiday': 'No',
        'Festive_Period': 'No',
        'Weather': 'Clear',
        'Track_Maintenance': 'No',
        'Signal_Failure': 'No',
        'Engine_Breakdown': 'No',
        'Previous_Train_Delay_min': 8,
        'Number_of_Stops': 10,
        'Crew_Change': 'No',
        'Passenger_Load_pct': 85.6,
        'Loco_Type': 'Electric',
        'Humidity_pct': 82.79,
        'Precipitation_mm': 2.75,
        'WindSpeed_kmph': 12.25,
        'Visibility_km': 12.28,
        'Pressure_hPa': 1017.49,
        'WeatherCondition': 'Rain'
    }
    
    print("\nPredicting for sample input:")
    for k, v in sample_input.items():
        print(f"  {k}: {v}")
        
    delay = predict_delay(sample_input, artifacts)
    print(f"\nPredicted Arrival Delay: {delay:.2f} minutes")

if __name__ == "__main__":
    main()
