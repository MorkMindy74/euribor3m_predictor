# Euribor 3M Predictor 📈

This project implements a machine learning model to predict the **Euribor 3-Month** interest rate for the next 30 business days. It uses an **XGBoost Regressor** trained on historical data and various macroeconomic indicators.

## 🚀 Features

- **Data Integration**: Merges multiple data sources including:
  - Historical Euribor rates (1w, 1m, 3m, 6m, 12m)
  - ECB Policy Rates (Deposit Facility, Marginal Lending, Main Refinancing)
  - Inflation Data (HICP Euro Area & Germany)
  - Market Volatility (VIX)
  - Sovereign Bond Yields (German Bund 10Y)
  - Monetary Aggregates (M3 Growth)
  - OIS Rates & Yield Curves
- **Advanced Modeling**: Uses XGBoost with recursive forecasting to predict future values.
- **Feature Engineering**: Incorporates lags, rolling averages, and standard deviations to capture temporal dynamics.

## 📊 Methodology

The model treats the forecasting problem as a regression task. Key predictive features identified by the model include:
1. **ECB Deposit Facility Rate** (Primary driver)
2. **Lagged Euribor 3M** (Autoregression)
3. **Euribor 6M** (Term structure)
4. **ECB Marginal Lending Rate**

The system performs a **recursive forecast**:
1. Predicts $t+1$ using known history.
2. Appends the prediction to the history.
3. Uses the updated history to predict $t+2$, and so on.

## 🛠️ Installation & Usage

### Prerequisites
- Python 3.8+
- Required libraries: `pandas`, `numpy`, `xgboost`, `scikit-learn`, `matplotlib`

```bash
pip install pandas numpy xgboost scikit-learn matplotlib
```

### Running the Project

1. **Prepare Data**:
   Place your raw CSV files in the project root.
   Run the preparation script to clean and merge data:
   ```bash
   python prepare_data.py
   ```
   *Output: `master_dataset_full.csv`*

2. **Train & Predict**:
   Train the model and generate a 30-day forecast:
   ```bash
   python train_predict.py
   ```
   *Output: `euribor_predictions_30days_enhanced.csv`, `prediction_plot_enhanced.png`*

3. **Save Model Artifacts**:
   To save the trained model for later use:
   ```bash
   python save_model.py
   ```
   *Output: `euribor_xgb_model.json`, `model_features.json`*

## 📂 File Structure

- `prepare_data.py`: Data ingestion, cleaning, merging, and correlation analysis.
- `train_predict.py`: Model training, evaluation, and recursive forecasting script.
- `save_model.py`: Utility to train the final model and export it to JSON.
- `euribor_xgb_model.json`: The trained XGBoost model artifact.
- `master_dataset_full.csv`: (Generated) The processed dataset used for training.

## ⚠️ Disclaimer

This tool is for **educational and research purposes only**. It does not constitute financial advice. Predictions are based on historical correlations and may not reflect future market shocks or policy changes.
