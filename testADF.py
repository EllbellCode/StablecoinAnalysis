"""
Runs the Augnented Dickey Fuller test on our data to confirm stationarity
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from pathlib import Path


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

def main():
    
    data_dir = Path("Data/Verified")
    output_dir = Path("Results/ADF/")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    files = list(data_dir.glob("*.csv"))
    if not files:
        print("No CSV files found in Data/Verified")
        return

    stablecoins = ['DAI', 'USDC', 'USDT']
    cryptos = ['BNB', 'BTC', 'ETH', 'XRP']

    tests_to_run = (
        [(coin, 'LogVolChange') for coin in stablecoins] +
        [(coin, 'Log Returns') for coin in stablecoins] +
        [(coin, 'Upside_Vol') for coin in stablecoins] +
        [(coin, 'Downside_Vol') for coin in stablecoins] +

        [(coin, 'LogVolChange') for coin in cryptos] +
        [(coin, 'Log Returns') for coin in cryptos] +
        [(coin, 'Upside_Vol') for coin in cryptos] +
        [(coin, 'Downside_Vol') for coin in cryptos]
    )

    coin_data = {}
    for file in files:
        coin_name = file.stem.replace("Verif_", "")
        if coin_name in stablecoins or coin_name in cryptos:
            try:
                df = pd.read_csv(file, parse_dates=['Date'], index_col='Date').sort_index()
                coin_data[coin_name] = df
            except Exception as e:
                print(f"Error loading {file.name}: {e}")

    all_results = []
    print("Running ADF tests...")
    print("=" * 50)

    for coin, var in tests_to_run:
        if coin in coin_data and var in coin_data[coin].columns:
        
            series = coin_data[coin][var]
            
            test_result = run_adf_test(series, autolag='AIC')
            test_result['Coin'] = coin
            test_result['Variable'] = var
            all_results.append(test_result)
        else:
            print(f"Skipping {coin} - {var} (Data not found)")

    results_df = pd.DataFrame(all_results)
    
    name_map = {
        'Log Returns': 'Log Returns',
        'LogVolChange': 'Log Volume Change',
        'Upside_Vol': 'Delta Upside Volatility',
        'Downside_Vol': 'Delta Downside Volatility'
    }
    results_df['Display Name'] = results_df['Variable'].map(name_map).fillna(results_df['Variable'])

    summary_df = results_df.groupby('Display Name').agg(
        Min_ADF=('ADF Statistic', 'min'),
        Max_ADF=('ADF Statistic', 'max'),
        Max_P_Value=('p-value', 'max')
    ).reset_index()

    summary_df['ADF Statistic Range'] = (
        summary_df['Min_ADF'].map('{:.2f}'.format) + " to " + 
        summary_df['Max_ADF'].map('{:.2f}'.format)
    )

    summary_df['Conclusion'] = np.where(summary_df['Max_P_Value'] < 0.05, 'Stationary', 'Non-Stationary')

    final_table = summary_df[['Display Name', 'ADF Statistic Range', 'Max_P_Value', 'Conclusion']]
    final_table.columns = ['Variable', 'ADF Statistic (Range)', 'Max p-value', 'Conclusion']

    print("\n--- Summary ADF Test Results (Option 2) ---")
    print(final_table.to_string(index=False))

    out_path = output_dir / "adf_summary_option2.csv"
    final_table.to_csv(out_path, index=False)
    print(f"\nSummary table saved to: {out_path}")

    results_df.drop(columns=['Display Name'], inplace=True)
    results_df.to_csv(output_dir / "adf_raw_results.csv", index=False)

if __name__ == "__main__":
    main()