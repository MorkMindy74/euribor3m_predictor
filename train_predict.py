import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import os

# Load data
base_path = r"c:\Users\marco.rossi\Desktop\Euribor_data\euribor_dev"
df = pd.read_csv(os.path.join(base_path, "master_dataset_full.csv"), index_col='date', parse_dates=True)

# Filter only numeric columns
df = df.select_dtypes(include=[np.number])

# Feature Engineering
def create_features(data):
    df_feat = data.copy()
    
    # Target Lags (Autoregression)
    target = 'Euribor_3m'
    for lag in [1, 2, 5, 10, 22]:
        df_feat[f'{target}_lag_{lag}'] = df_feat[target].shift(lag)
    
    # Rolling stats on Target
    df_feat[f'{target}_roll_mean_5'] = df_feat[target].shift(1).rolling(window=5).mean()
    df_feat[f'{target}_roll_std_5'] = df_feat[target].shift(1).rolling(window=5).std()
    
    # External Feature Lags
    # We take Lag 1 of all other features to use them as predictors
    # (We assume we know yesterday's value of X to predict today's Y)
    feature_cols = [c for c in df_feat.columns if c != target and 'lag' not in c and 'roll' not in c]
    
    for col in feature_cols:
        df_feat[f'{col}_lag_1'] = df_feat[col].shift(1)
        # Maybe Lag 5 too for weekly trends
        df_feat[f'{col}_lag_5'] = df_feat[col].shift(5)
        
    return df_feat, feature_cols

print("Feature Engineering...")
df_processed, raw_feature_cols = create_features(df)
df_processed.dropna(inplace=True)

# Define Features for Model
# We use all created lags and rolling stats. We DO NOT use current day values of other features 
# because we won't have them at inference time (unless we forecast them too).
model_features = [c for c in df_processed.columns if 'lag' in c or 'roll' in c]
target = 'Euribor_3m'

print(f"Model Features ({len(model_features)}): {model_features[:10]} ...")

# Split Data
test_days = 30
train = df_processed.iloc[:-test_days]
test = df_processed.iloc[-test_days:]

print(f"Training on {len(train)} samples, Testing on {len(test)} samples")

# Train Model
model = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.01, max_depth=6, subsample=0.8, colsample_bytree=0.8, early_stopping_rounds=50)
model.fit(train[model_features], train[target], eval_set=[(test[model_features], test[target])], verbose=False)

# Evaluate
preds = model.predict(test[model_features])
rmse = np.sqrt(mean_squared_error(test[target], preds))
mae = mean_absolute_error(test[target], preds)
print(f"Test RMSE: {rmse:.4f}")
print(f"Test MAE: {mae:.4f}")

# Feature Importance
importance = pd.DataFrame({'feature': model_features, 'importance': model.feature_importances_})
importance = importance.sort_values('importance', ascending=False).head(10)
print("\nTop 10 Important Features:")
print(importance)

# --- Future Prediction (Recursive) ---
print("\nRetraining on full dataset for future prediction...")
full_model = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.01, max_depth=6, subsample=0.8, colsample_bytree=0.8)
full_model.fit(df_processed[model_features], df_processed[target])

future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=30, freq='B')
future_preds = []

# Initial State
last_row = df_processed.iloc[-1]
# We need a history buffer to calculate rolling stats and lags dynamically
# We'll append predictions to this history
history_df = df_processed.copy()

print("Generating recursive forecasts...")
for date in future_dates:
    # 1. Create a new row for the 'tomorrow' we want to predict
    # We start by copying the last row to get the static/exogenous structure
    new_row = history_df.iloc[-1].copy()
    new_row.name = date
    
    # 2. Update Lags for Target (Euribor_3m)
    # The 'current' Euribor_3m is unknown (we are predicting it), but we need its lags
    # lag_1 is the prediction from the previous step (or actual data if step 0)
    new_row['Euribor_3m_lag_1'] = history_df.iloc[-1][target]
    new_row['Euribor_3m_lag_2'] = history_df.iloc[-2][target]
    new_row['Euribor_3m_lag_5'] = history_df.iloc[-5][target]
    new_row['Euribor_3m_lag_10'] = history_df.iloc[-10][target]
    new_row['Euribor_3m_lag_22'] = history_df.iloc[-22][target]
    
    # 3. Update Rolling Stats for Target
    # We need the last 5 values of the target
    last_5_values = history_df[target].iloc[-5:].values
    new_row['Euribor_3m_roll_mean_5'] = np.mean(last_5_values)
    new_row['Euribor_3m_roll_std_5'] = np.std(last_5_values)
    
    # 4. Update Lags for External Features
    # For external features (e.g. ECB rates), we assume they stay constant (Naive Forecast)
    # or we could use their own lags if we were forecasting them.
    # Here: new_row['X_lag_1'] should be history_df.iloc[-1]['X']
    # But wait, 'X' in history_df is the actual value.
    # Since we are NOT predicting X, we assume X_future = X_last_known
    # So X_lag_1 for tomorrow is X_last_known.
    for col in raw_feature_cols:
        # We assume the value of the feature 'col' remains constant at its last known value
        last_val = history_df.iloc[-1][col]
        new_row[f'{col}_lag_1'] = last_val
        new_row[f'{col}_lag_5'] = history_df.iloc[-5][col] # This is technically correct if we assume constant future
        
        # Update the 'actual' column in the new row to this constant value so next iteration picks it up
        new_row[col] = last_val

    # 5. Predict
    # Extract features as dataframe (single row)
    X_input = pd.DataFrame([new_row[model_features]])
    pred = full_model.predict(X_input)[0]
    
    # 6. Store Prediction
    future_preds.append(pred)
    
    # 7. Update the 'Target' in new_row with the prediction so it becomes history
    new_row[target] = pred
    
    # Append to history
    history_df = pd.concat([history_df, pd.DataFrame([new_row])])

# Save Predictions
df_future = pd.DataFrame({'Date': future_dates, 'Predicted_Euribor_3m': future_preds})
df_future.set_index('Date', inplace=True)
output_file = os.path.join(base_path, "euribor_predictions_30days_enhanced.csv")
df_future.to_csv(output_file)
print(f"Predictions saved to {output_file}")
print(df_future.head())

# Plot
plt.figure(figsize=(12, 6))
plt.plot(df.index[-90:], df['Euribor_3m'][-90:], label='Historical')
plt.plot(df_future.index, df_future['Predicted_Euribor_3m'], label='Forecast (Enhanced)', color='green', linestyle='--')
plt.title('Euribor 3M Prediction (Enhanced with All Data)')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(base_path, "prediction_plot_enhanced.png"))
print("Plot saved to prediction_plot_enhanced.png")
