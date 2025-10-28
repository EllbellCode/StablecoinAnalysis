"""
Granger Causality

USed to capture linear causalities
"""

import pandas as pd
from pathlib import Path
from statsmodels.tsa.stattools import grangercausalitytests
import itertools

# === Step 1: Load data ===
data_dir = Path("Data/Verified")
files = list(data_dir.glob("*.csv"))
volatility = "RV"
max_lag = 1

coin_data = {}
for file in files:
    coin_name = file.stem.replace("Verif_", "")
    df = pd.read_csv(file)

    # Ensure ascending date order
    df = df.sort_values("Date").reset_index(drop=True)
    
    # Drop NaN rows caused by pct_change
    df = df.dropna(subset=["Log Returns", "VolLogChange"])

    train = df[(df['Date'] >= '2020-01-01') & (df['Date'] <= '2023-12-31')]
    
    coin_data[coin_name] = train

# === Step 2: Create variable list ===
# For each coin, we have two variables: Returns and VolChange
variables = []
for coin in coin_data.keys():
    variables.append((coin, volatility))
    variables.append((coin, "VolLogChange"))
    variables.append((coin, "Log Returns"))

# === Step 3: Run Granger tests for all ordered pairs ===
results = []

for (coin_x, var_x), (coin_y, var_y) in itertools.product(variables, repeat=2):
    if coin_x == coin_y and var_x == var_y:
        continue  # skip same variable to same variable

    df_x = coin_data[coin_x][var_x]
    df_y = coin_data[coin_y][var_y]
    
    # Align lengths and drop NaNs
    combined = pd.concat([df_x, df_y], axis=1).dropna()
    combined.columns = ["X", "Y"]
    
    if len(combined) > max_lag:
    
        test_result = grangercausalitytests(combined, maxlag=max_lag)
        for lag in range(1, max_lag + 1):
            f_pvalue = test_result[lag][0]['ssr_ftest'][1]
            results.append({
                "source_coin": coin_x,
                "source_var": var_x,
                "target_coin": coin_y,
                "target_var": var_y,
                "lag": lag,
                "F-test p-value": f_pvalue
            })

# === Step 4: Store results ===
results_df = pd.DataFrame(results)

sig_results = results_df[(results_df["F-test p-value"] < 0.05) & 
                         (results_df["source_coin"].isin(["USDT", "USDC", "DAI"])) &
                         (results_df["target_coin"].isin(["BTC", "ETH", "XRP", "BNB"])) &
                         (results_df["source_var"].isin(["VolLogChange", volatility])) &
                         (results_df["target_var"].isin(["Log Returns", volatility]))
                         ]
sig_results.to_csv("grangerSig.csv", index=False)
