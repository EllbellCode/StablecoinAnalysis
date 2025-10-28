"""
ADF test on our metrics
"""


import pandas as pd
from statsmodels.tsa.stattools import adfuller
from pathlib import Path
import numpy as np


"""
Runs the ADF test on a given series and returns a dictionary of results.
"""
def run_adf_test(series, autolag='AIC'):

        series = series.dropna()
        if series.empty:
            return {
                "ADF Statistic": np.nan,
                "p-value": np.nan,
                "Number of Lags": 0,
                "Result": "Series is empty"
            }
            
        result = adfuller(series, autolag=autolag)
        
        return {
            "ADF Statistic": result[0],
            "p-value": result[1],
            "Number of Lags": result[2],
            "Result": "Stationary" if result[1] < 0.05 else "Non-Stationary"
        }

"""
Main test
"""
data_dir = Path("Data/Verified")
files = list(data_dir.glob("*.csv"))

stablecoins = ['DAI', 'USDC', 'USDT']
cryptos = ['BNB', 'BTC', 'ETH', 'XRP']

tests_to_run = (
    
    [(coin, 'LogVolChange') for coin in stablecoins] +
    [(coin, 'Delta_LogRV') for coin in stablecoins] +
    
    [(coin, 'Log Returns') for coin in cryptos] +
    [(coin, 'Delta_LogRV') for coin in cryptos]
)

coin_data = {}
for file in files:
    coin_name = file.stem.replace("Verif_", "")
    if coin_name in stablecoins or coin_name in cryptos:
        df = pd.read_csv(file, parse_dates=['Date'], index_col='Date').sort_index()
        coin_data[coin_name] = df

all_results = []

print("Running ADF tests for the Four-Factor Analysis...")
print("=" * 50)

for coin, var in tests_to_run:
    if coin in coin_data and var in coin_data[coin].columns:
        
        print(f"Testing {coin} - {var}...")
        series = coin_data[coin][var]
        
        test_result = run_adf_test(series, autolag='AIC')
        
        test_result['Coin'] = coin
        test_result['Variable'] = var
        all_results.append(test_result)
        
    else:
        print(f"Skipping {coin} - {var} (Data not found)")

results_df = pd.DataFrame(all_results)
results_df = results_df[['Coin', 'Variable', 'ADF Statistic', 'p-value', 'Number of Lags', 'Result']]

print("\n--- ADF Test Results ---")
print(results_df.to_string())

results_df.to_csv("adf_results.csv", index=False)