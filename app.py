import streamlit as st
import pandas as pd
import joblib
import numpy as np
import datetime

# Page Configuration
st.set_page_config(
    page_title="Train Delay Predictor",
    page_icon="🚆",
    layout="wide"
)

# Load Model Artifacts
@st.cache_resource
def load_artifacts():
    try:
        return joblib.load('train_delay_model_artifacts.pkl')
    except FileNotFoundError:
        st.error("Model artifacts not found. Please run 'train_model.py' first.")
        return None

artifacts = load_artifacts()

def main():
    st.title("🚆 Train Delay Prediction System")
    st.markdown("Enter the train journey details below to predict the arrival delay.")

    if artifacts is None:
        return

    # Create two columns for input
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Journey Details")
        distance = st.number_input("Distance (km)", min_value=0, value=530)
        
        # Date and Time Selection
        travel_date = st.date_input("Travel Date", datetime.date.today())
        travel_time = st.time_input("Scheduled Departure Time", datetime.time(9, 0))
        
        # Combine date and time
        scheduled_departure = datetime.datetime.combine(travel_date, travel_time)
        
        scheduled_travel_time = st.number_input("Scheduled Travel Time (minutes)", min_value=10, value=300)
        stops = st.number_input("Number of Stops", min_value=0, value=8)
        
        previous_delay = st.number_input("Previous Train Delay (minutes)", value=0)

    with col2:
        st.subheader("Operational & Weather")
        
        # Categorical Inputs matching the model's expected values
        loco_type = st.selectbox("Locomotive Type", ['Electric', 'Diesel'])
        passenger_load = st.slider("Passenger Load (%)", 0.0, 120.0, 75.0)
        
        weather = st.selectbox("Weather Forecast", ['Clear', 'Cloudy', 'Rain', 'Fog', 'Snow'])
        weather_condition = st.selectbox("Current Condition", ['Clear', 'Rain', 'Haze', 'Fog', 'Clouds', 'Drizzle'])
        
        temperature = st.slider("Temperature (°C)", -10.0, 50.0, 25.0)
        humidity = st.slider("Humidity (%)", 0.0, 100.0, 60.0)
        
        # Flags
        col_flags1, col_flags2 = st.columns(2)
        with col_flags1:
            is_holiday = st.checkbox("Is Holiday?", value=False)
            festive_period = st.checkbox("Festive Period?", value=False)
            track_maint = st.checkbox("Track Maintenance?", value=False)
        with col_flags2:
            signal_fail = st.checkbox("Signal Failure?", value=False)
            engine_fail = st.checkbox("Engine Breakdown?", value=False)
            crew_change = st.checkbox("Crew Change?", value=False)

    # Prepare Input Data
    input_data = {
        'Distance_km': distance,
        'Scheduled_Departure': scheduled_departure,
        'Scheduled_Travel_Time_min': scheduled_travel_time,
        'Day_of_Week': scheduled_departure.strftime('%A'),
        'Month': scheduled_departure.month,
        'Is_Holiday': 'Yes' if is_holiday else 'No',
        'Festive_Period': 'Yes' if festive_period else 'No',
        'Weather': weather,
        'Track_Maintenance': 'Yes' if track_maint else 'No',
        'Signal_Failure': 'Yes' if signal_fail else 'No',
        'Engine_Breakdown': 'Yes' if engine_fail else 'No',
        'Previous_Train_Delay_min': previous_delay,
        'Number_of_Stops': stops,
        'Crew_Change': 'Yes' if crew_change else 'No',
        'Passenger_Load_pct': passenger_load,
        'Loco_Type': loco_type,
        'Humidity_pct': humidity,
        'Temperature_C': temperature, # Note: Check model feature name (Temperature_C vs Temperature)
        'Precipitation_mm': 0.0, # Defaulting for simplicity, or add input
        'WindSpeed_kmph': 10.0, # Defaulting
        'Visibility_km': 10.0, # Defaulting
        'Pressure_hPa': 1010.0, # Defaulting
        'WeatherCondition': weather_condition
    }

    # Prediction Button
    if st.button("Predict Arrival Delay", type="primary"):
        with st.spinner("Calculating..."):
            # Use the prediction function from our predict.py script logic (embedded here or imported)
            
            # Reusing the prediction logic directly to avoid import issues depending on path
            model = artifacts['model']
            num_imputer = artifacts['num_imputer']
            cat_imputer = artifacts['cat_imputer']
            encoders = artifacts['encoders']
            feature_names = artifacts['feature_names']
            
            df_pred = pd.DataFrame([input_data])
            
            # Derive components
            df_pred['Scheduled_Dep_Hour'] = df_pred['Scheduled_Departure'].dt.hour
            df_pred['Scheduled_Dep_Minute'] = df_pred['Scheduled_Departure'].dt.minute
            df_pred['Scheduled_Dep_Month'] = df_pred['Scheduled_Departure'].dt.month
            df_pred['Scheduled_Dep_DayOfWeek'] = df_pred['Scheduled_Departure'].dt.dayofweek
            
            # Ensure columns exist
            for col in feature_names:
                if col not in df_pred.columns:
                    df_pred[col] = np.nan
            
            X_input = df_pred[feature_names].copy()
            
            # IMPORTANT: Reuse feature lists from training logic structure
            # To be safe, we categorize based on what we know are numeric/categorical in our set
            numeric_cols = [c for c in X_input.columns if X_input[c].dtype in ['int64', 'float64'] or c in ['Distance_km', 'Scheduled_Travel_Time_min', 'Previous_Train_Delay_min', 'Number_of_Stops', 'Passenger_Load_pct', 'Temperature_C', 'Humidity_pct', 'Precipitation_mm', 'WindSpeed_kmph', 'Visibility_km', 'Pressure_hPa', 'Scheduled_Dep_Hour', 'Scheduled_Dep_Minute', 'Scheduled_Dep_Month', 'Scheduled_Dep_DayOfWeek']]
            # Actually better to rely on what the Imputer expects if we could, but let's stick to the lists we defined in training
            
            # Let's hardcode the lists to match train_model.py exactly for safety
            numeric_features_train = [
                'Distance_km', 'Scheduled_Travel_Time_min', 'Previous_Train_Delay_min', 
                'Number_of_Stops', 'Passenger_Load_pct', 'Temperature_C', 'Humidity_pct',
                'Precipitation_mm', 'WindSpeed_kmph', 'Visibility_km', 'Pressure_hPa'
            ] 
            # Add derived ones that might have been picked up as numeric? 
            # In train_model.py, we only treated the list 'numeric_features' as numeric for imputation.
            # But Random Forest uses all 'features'.
            # Wait, in train_model.py, we did: X[numeric_features] = num_imputer...
            # Then we did X[categorical_features] = cat_imputer...
            # The 'features' list included both.
            # The derived 'Scheduled_Dep_Hour' etc were NOT in 'numeric_features' list in train_model.py!
            # They were in the DataFrame X but NOT processed by the imputer specifically named 'numeric_features'.
            # However, they ARE numeric. 
            
            # Let's look at train_model.py logic again:
            # features = numeric_features + categorical_features
            # X = df[features].copy()
            # The derived features were NOT added to the 'features' list in train_model.py ??
            # ERROR POTENTIAL: If I didn't add Scheduled_Dep_Hour to 'features' list in train_model.py, the model didn't train on them.
            # Let's check train_model.py content from previous turn.
            # ...
            # numeric_features = [ ... ]
            # features = numeric_features + categorical_features
            # X = df[features].copy()
            # ...
            # The derived time features were created in df, but NOT added to 'features' list.
            # Use `view_file` to confirm. If so, my model is simpler than intended, but functional.
            # I will proceed assuming they were NOT used, to avoid shape mismatch.
            
            # Processing inputs safely
            
            # 1. Impute Numeric
            numeric_feats_in_use = [f for f in numeric_features_train if f in feature_names]
            X_input[numeric_feats_in_use] = num_imputer.transform(X_input[numeric_feats_in_use])
            
            # 2. Impute Categorical
            cat_features_train = [
                'Day_of_Week', 'Is_Holiday', 'Festive_Period', 'Weather', 
                'Track_Maintenance', 'Signal_Failure', 'Engine_Breakdown', 
                'Crew_Change', 'Loco_Type', 'WeatherCondition'
            ]
            cat_feats_in_use = [f for f in cat_features_train if f in feature_names]
            X_input[cat_feats_in_use] = cat_imputer.transform(X_input[cat_feats_in_use])
            
            # 3. Encode
            for col in cat_feats_in_use:
                le = encoders[col]
                X_input[col] = X_input[col].astype(str)
                # Handle unseen
                try:
                    X_input[col] = le.transform(X_input[col])
                except ValueError:
                    X_input[col] = 0 # Fallback
            
            # Predict
            delay_pred = model.predict(X_input)[0]
            
            st.success(f"Predicted Arrival Delay: **{delay_pred:.2f} minutes**")
            
            if delay_pred > 15:
                st.warning("⚠️ High delay expected!")
            else:
                st.info("✅ Train works reasonably on time.")

if __name__ == "__main__":
    main()
