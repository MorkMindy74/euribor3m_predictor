import pandas as pd
import os
import glob
import numpy as np

# Define paths
base_path = r"c:\Users\marco.rossi\Desktop\Euribor_data\euribor_dev"
target_file = os.path.join(base_path, "euribor_3m_all-years.csv")

print("--- Starting Data Preparation ---")

# 1. Load Target
print("Loading Target (Euribor 3M)...")
df_target = pd.read_csv(target_file)
df_target['date'] = pd.to_datetime(df_target['date'])
df_target.set_index('date', inplace=True)
df_target.rename(columns={'3m': 'Euribor_3m'}, inplace=True)

# 2. Load Other Euribor Tenors
print("Loading Other Euribor Tenors...")
tenors_file = glob.glob(os.path.join(base_path, "euribor_3m-1w*.csv"))[0]
df_tenors = pd.read_csv(tenors_file)
df_tenors['date'] = pd.to_datetime(df_tenors['date'])
df_tenors.set_index('date', inplace=True)
# Keep relevant tenors
keep_cols = ['1w', '1m', '6m', '12m']
df_tenors = df_tenors[[c for c in keep_cols if c in df_tenors.columns]]
df_tenors.columns = [f'Euribor_{c}' for c in df_tenors.columns]

# 3. VIX
print("Loading VIX...")
vix_file = os.path.join(base_path, "VIX_History.csv")
df_vix = pd.read_csv(vix_file)
df_vix['DATE'] = pd.to_datetime(df_vix['DATE'])
df_vix.set_index('DATE', inplace=True)
df_vix = df_vix[['CLOSE']].rename(columns={'CLOSE': 'VIX'})

# 4. Euro Stoxx Banks (chart.csv)
print("Loading Euro Stoxx Banks...")
stoxx_file = os.path.join(base_path, "chart.csv")
df_stoxx = pd.read_csv(stoxx_file)
# Format is MM/YYYY, need to handle
df_stoxx['Data'] = pd.to_datetime(df_stoxx['Data'], format='%m/%Y') + pd.offsets.MonthEnd(0)
df_stoxx.set_index('Data', inplace=True)
df_stoxx.rename(columns={'EURO STOXX Banks': 'EuroStoxx_Banks'}, inplace=True)
df_stoxx = df_stoxx.resample('D').ffill()

# 5. German Bund (IRLTLT01DEM156N.csv)
print("Loading German Bund...")
bund_file = os.path.join(base_path, "IRLTLT01DEM156N.csv")
df_bund = pd.read_csv(bund_file)
df_bund['observation_date'] = pd.to_datetime(df_bund['observation_date'])
df_bund.set_index('observation_date', inplace=True)
df_bund.rename(columns={'IRLTLT01DEM156N': 'Bund_10Y'}, inplace=True)
df_bund = df_bund.resample('D').ffill()

# 6. ECB Policy Rates
print("Loading ECB Policy Rates...")
ecb_rates_file = os.path.join(base_path, "ECB Data Portal_20251206071852.csv")
df_ecb_rates = pd.read_csv(ecb_rates_file)
df_ecb_rates['DATE'] = pd.to_datetime(df_ecb_rates['DATE'])
df_ecb_rates.set_index('DATE', inplace=True)
# Map columns
rate_cols = {}
for col in df_ecb_rates.columns:
    if 'Deposit facility' in col: rate_cols[col] = 'ECB_Deposit'
    elif 'Marginal lending' in col: rate_cols[col] = 'ECB_Marginal'
    elif 'Main refinancing' in col: rate_cols[col] = 'ECB_Main'
df_ecb_rates = df_ecb_rates[list(rate_cols.keys())].rename(columns=rate_cols)

# 7. ECB Yield Curve (AAA)
print("Loading ECB Yield Curve...")
ecb_yc_file = os.path.join(base_path, "ECB Data Portal_20251206084417.csv")
df_yc = pd.read_csv(ecb_yc_file)
df_yc['DATE'] = pd.to_datetime(df_yc['DATE'])
df_yc.set_index('DATE', inplace=True)
# Filter for 10Y Yield or similar if possible, otherwise take all numeric
# The file likely has many columns. Let's pick a few key ones based on "10-year" or "1-year" in name if possible, 
# but column names are codes. Let's just take all and clean later.
# Actually, let's just take the first few columns as proxies if we can't identify them easily.
# Or better, let's try to identify "Yield curve... 10-year" from the header we saw earlier.
# The header had "Yield curve instantaneous forward rate, 10-year maturity...".
yc_cols = {}
for col in df_yc.columns:
    if '10-year' in col and 'Yield curve' in col:
        yc_cols[col] = 'ECB_Yield_10Y'
    elif '1-year' in col and 'Yield curve' in col:
        yc_cols[col] = 'ECB_Yield_1Y'
if yc_cols:
    df_yc = df_yc[list(yc_cols.keys())].rename(columns=yc_cols)
else:
    # Fallback: take first 2 numeric columns
    df_yc = df_yc.select_dtypes(include=[np.number]).iloc[:, :2]
    df_yc.columns = [f'ECB_YC_{i}' for i in range(len(df_yc.columns))]

# 8. Monetary Aggregates (M3)
print("Loading Monetary Aggregates...")
m3_file = os.path.join(base_path, "ECB Data Portal_20251206072849.csv")
df_m3 = pd.read_csv(m3_file)
df_m3['DATE'] = pd.to_datetime(df_m3['DATE'])
df_m3.set_index('DATE', inplace=True)
# Look for M3 Annual Growth or similar
m3_cols = {}
for col in df_m3.columns:
    if 'M3' in col:
        m3_cols[col] = 'M3_Growth'
if m3_cols:
    df_m3 = df_m3[list(m3_cols.keys())].rename(columns=m3_cols)
    df_m3 = df_m3.resample('D').ffill()

# 9. OIS Rates (Daily)
print("Loading OIS Rates...")
ois_file = os.path.join(base_path, "ECB Data Portal daily_20251206083820.csv")
df_ois = pd.read_csv(ois_file)
df_ois['DATE'] = pd.to_datetime(df_ois['DATE'])
df_ois.set_index('DATE', inplace=True)
ois_cols = {}
for col in df_ois.columns:
    if 'OIS' in col and '3 m' in col: ois_cols[col] = 'OIS_3m'
    elif 'OIS' in col and '1 month' in col: ois_cols[col] = 'OIS_1m'
if ois_cols:
    df_ois = df_ois[list(ois_cols.keys())].rename(columns=ois_cols)

# --- MERGE ALL ---
print("\nMerging all datasets...")
dfs_to_merge = [df_tenors, df_vix, df_stoxx, df_bund, df_ecb_rates, df_yc, df_m3, df_ois]

df_master = df_target.copy()
for d in dfs_to_merge:
    # Join using left index (Target dates)
    # We use merge_asof or reindex/join. Since most are daily, join is fine.
    # For monthly data resampled to daily, we already did resample('D').ffill()
    df_master = df_master.join(d, how='left')

# Forward fill missing values (propagating last known value)
df_master.ffill(inplace=True)

# Drop rows where target is still NaN (shouldn't happen if target is base)
df_master.dropna(subset=['Euribor_3m'], inplace=True)

# Drop rows where features are largely missing (e.g. early years if some data starts later)
# Let's drop rows where > 50% of columns are NaN
df_master.dropna(thresh=len(df_master.columns)//2, inplace=True)

# Fill remaining NaNs (e.g. at start) with backfill or 0
df_master.bfill(inplace=True)
df_master.fillna(0, inplace=True)

print(f"\nFinal Dataset Shape: {df_master.shape}")
print("Columns:", df_master.columns.tolist())

# --- CORRELATION ANALYSIS ---
print("\n--- Correlation Analysis with Euribor 3m ---")
# Ensure we only correlate numeric columns
df_numeric = df_master.select_dtypes(include=[np.number])
correlations = df_numeric.corr()['Euribor_3m'].sort_values(ascending=False)
print(correlations)

# Save
df_master.to_csv(os.path.join(base_path, "master_dataset_full.csv"))
print("\nSaved master_dataset_full.csv")
