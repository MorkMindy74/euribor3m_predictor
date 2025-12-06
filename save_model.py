import pandas as pd
import numpy as np
import xgboost as xgb
import os
import json

# Load data
base_path = r"c:\Users\marco.rossi\Desktop\Euribor_data\euribor_dev"
df = pd.read_csv(os.path.join(base_path, "master_dataset_full.csv"), index_col='date', parse_dates=True)
df = df.select_dtypes(include=[np.number])

# Feature Engineering (Same as before)
def create_features(data):
    df_feat = data.copy()
    target = 'Euribor_3m'
    for lag in [1, 2, 5, 10, 22]:
        df_feat[f'{target}_lag_{lag}'] = df_feat[target].shift(lag)
    df_feat[f'{target}_roll_mean_5'] = df_feat[target].shift(1).rolling(window=5).mean()
    df_feat[f'{target}_roll_std_5'] = df_feat[target].shift(1).rolling(window=5).std()
    feature_cols = [c for c in df_feat.columns if c != target and 'lag' not in c and 'roll' not in c]
    for col in feature_cols:
        df_feat[f'{col}_lag_1'] = df_feat[col].shift(1)
        df_feat[f'{col}_lag_5'] = df_feat[col].shift(5)
    return df_feat

df_processed = create_features(df)
df_processed.dropna(inplace=True)

model_features = [c for c in df_processed.columns if 'lag' in c or 'roll' in c]
target = 'Euribor_3m'

# Train Full Model
print("Training final model...")
model = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.01, max_depth=6, subsample=0.8, colsample_bytree=0.8)
model.fit(df_processed[model_features], df_processed[target])

# Save Model
model_file = os.path.join(base_path, "euribor_xgb_model.json")
model.save_model(model_file)
print(f"Model saved to {model_file}")

# Save Feature Names (important for loading later)
feature_names_file = os.path.join(base_path, "model_features.json")
with open(feature_names_file, 'w') as f:
    json.dump(model_features, f)
print(f"Feature names saved to {feature_names_file}")
