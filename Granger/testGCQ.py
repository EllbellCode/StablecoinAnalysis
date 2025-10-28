import pandas as pd
from pathlib import Path
import itertools
import numpy as np
import statsmodels.api as sm
from statsmodels.regression.quantile_regression import QuantReg

# === Step 1: Load data ===
data_dir = Path("Data/Verified")
files = list(data_dir.glob("*.csv"))
volatility = "RV"
max_lag = 1

#Quantiles of target coin variable
quantiles = [0.1, 0.5, 0.9]

coin_data = {}
for file in files:
    coin_name = file.stem.replace("Verif_", "")
    df = pd.read_csv(file)

    df = df.sort_values("Date").reset_index(drop=True)
    df = df.dropna(subset=["Log Returns", "VolLogChange"])
    train = df[(df['Date'] >= '2020-01-01') & (df['Date'] <= '2023-12-31')]
    coin_data[coin_name] = train

# === Step 2: Create variable list ===
variables = []
for coin in coin_data.keys():
    variables.append((coin, volatility))
    variables.append((coin, "VolLogChange"))
    variables.append((coin, "Log Returns"))

def quantile_granger_test(x, y, max_lag, tau):
    """
    Runs quantile Granger causality test for given series x -> y at quantile tau.
    Returns p-value from Wald test.
    """
    df = pd.concat([y, x], axis=1).dropna()
    df.columns = ["Y", "X"]

    # Create lagged dataframe
    for lag in range(1, max_lag+1):
        df[f"Y_lag{lag}"] = df["Y"].shift(lag)
        df[f"X_lag{lag}"] = df["X"].shift(lag)
    df = df.dropna()

    # Restricted model: Y ~ own lags
    y_dep = df["Y"]
    X_unres = sm.add_constant(df[[f"Y_lag{lag}" for lag in range(1, max_lag+1)] +
                                 [f"X_lag{lag}" for lag in range(1, max_lag+1)]])

    # Fit quantile regressions
    mod_unres = QuantReg(y_dep, X_unres).fit(q=tau, max_iter=2500)

    # Wald test: test if coefficients on X lags are jointly zero
    # We use Wald Test in Quantiles instead of F-test as 
    # F assumes mean-based OLS regression, which assumes normally distributed and homoskedastic residuals
    # F is also Linear
    # Wald is non-linear and residuals are not necessarily normal or homoskedastic
    R = np.zeros((max_lag, len(mod_unres.params)))
    param_names = list(mod_unres.params.index)
    for i in range(max_lag):
        R[i, param_names.index(f"X_lag{i+1}")] = 1
    wald_res = mod_unres.wald_test(R, scalar=True)

    # Collect coefficients of X lags
    x_coefs = mod_unres.params[[f"X_lag{i+1}" for i in range(max_lag)]].values.flatten()

    return float(wald_res.pvalue), x_coefs

# === Step 3: Run tests ===
results = []

for (coin_x, var_x), (coin_y, var_y) in itertools.product(variables, repeat=2):
    if coin_x == coin_y and var_x == var_y:
        continue  

    df_x = coin_data[coin_x][var_x]
    df_y = coin_data[coin_y][var_y]

    if len(df_x) > max_lag and len(df_y) > max_lag:
        for tau in quantiles:

            pval, coefs = quantile_granger_test(df_x, df_y, max_lag, tau)
            results.append({
                "source_coin": coin_x,
                "source_var": var_x,
                "target_coin": coin_y,
                "target_var": var_y,
                "quantile": tau,
                "lag": max_lag,
                "p_value": pval,
                "Coef": coefs[0]
            })

# === Step 4: Store results ===
results_df = pd.DataFrame(results)

# Significant at 5% level
sig_results = results_df[(results_df["p_value"] < 0.05) &
                         (results_df["source_coin"].isin(["USDT", "USDC", "DAI"])) &
                         (results_df["target_coin"].isin(["BTC", "ETH", "XRP", "BNB"])) &
                         (results_df["source_var"].isin(["VolLogChange", volatility])) &
                         (results_df["target_var"].isin(["Log Returns", volatility]))
                         ]

sig_results.to_csv("grangerQuantileSig.csv", index=False)