"""
Fits all variables of a single stablecoin as exogenous
"""

import pandas as pd
import numpy as np
from arch import arch_model
from pathlib import Path

dir = Path("Data/Verified")
error_dist = 'skewt'
models = ['Garch', 'eGarch']
variables = ['RV', 'LogVolChange']   # both variables together
max_lags = 1
cutoff_date = pd.Timestamp("2024-01-01")

# Load stablecoins
stablecoins = ['Verif_USDT.csv', 'Verif_USDC.csv', 'Verif_DAI.csv']
stablecoin_dfs = {}
for sc_file in stablecoins:
    df_sc = pd.read_csv(dir / sc_file, parse_dates=['Date']).set_index('Date').sort_index()
    sc_name = sc_file.replace('Verif_', '').replace('.csv', '')
    stablecoin_dfs[sc_name] = df_sc

cryptos = ['Verif_BNB.csv', 'Verif_BTC.csv', 'Verif_ETH.csv', 'Verif_XRP.csv']
cryptoPaths = [dir / f for f in cryptos]

results = []

# Iterate over all cryptos, models, and stablecoins
for file in cryptoPaths:
    df = pd.read_csv(file, parse_dates=['Date']).set_index('Date').sort_index()
    
    for model in models:
        for sc_name, sc_data in stablecoin_dfs.items():
            # Join crypto data with stablecoin vars
            data = df.copy()
            for var in variables:
                data = data.join(sc_data[[var]].rename(columns={var: f'{sc_name}_{var}'}), how='inner')
            
            # Create lagged features for BOTH variables
            lag_cols = []
            for var in variables:
                for lag in range(1, max_lags+1):
                    col_name = f'{sc_name}_{var}_lag{lag}'
                    data[col_name] = data[f'{sc_name}_{var}'].shift(lag)
                    lag_cols.append(col_name)
            
            # Drop NA rows
            data = data.dropna(subset=['Log Returns'] + lag_cols)
            if data.empty:
                continue
            
            train_data = data[data.index < cutoff_date]
            test_data  = data[data.index >= cutoff_date]
            
            y = train_data['Log Returns']
            X = train_data[lag_cols]
            y_scaled = y * 10  # rescale to avoid DataScaleWarning
            
            # Baseline GARCH(1,1)
            m0 = arch_model(y_scaled, vol=model, p=1, q=1, mean='Constant', dist=error_dist)
            r0 = m0.fit(disp='off')
            
            # GARCH(1,1) with AR(1) + both vars + lags
            m1 = arch_model(y_scaled, vol=model, p=1, q=1, mean='ARX', lags=1, x=X, dist=error_dist)
            r1 = m1.fit(disp='off')
            
            # Store baseline results
            results.append({
                'Crypto': file.stem,
                'Stablecoin': sc_name,
                'Model': f'{model} Baseline',
                'Omega': r0.params.get('omega', np.nan),
                'Alpha[1]': r0.params.get('alpha[1]', np.nan),
                'Beta[1]': r0.params.get('beta[1]', np.nan),
            })
            
            # Identify significant lags
            significant_lags = []
            for col in lag_cols:
                pval = r1.pvalues.get(col, np.nan)
                if pval < 0.05:
                    significant_lags.append(f'{col} (p={pval:.3g})')
            
            # Store ARX results
            results.append({
                'Crypto': file.stem,
                'Stablecoin': sc_name,
                'Model': f'{model} + AR(1) + {sc_name} RV+LogVol lags 1-{max_lags}',
                'Omega': r1.params.get('omega', np.nan),
                'Alpha[1]': r1.params.get('alpha[1]', np.nan),
                'Beta[1]': r1.params.get('beta[1]', np.nan),
                'Significant_lags': ', '.join(significant_lags) if significant_lags else 'None'
            })

# Convert to DataFrame
summary_table = pd.DataFrame(results)

# Print only significant results
sig_lags_df = summary_table[
    (summary_table['Model'].str.contains(r'AR\(1\) \+')) & 
    (summary_table['Significant_lags'] != 'None')
]

for idx, row in sig_lags_df.iterrows():
    print(f"{row['Crypto']} ({row['Model']}): {row['Significant_lags']}")

"""
USDC Vol/RV combo for BNB and BTC
"""

