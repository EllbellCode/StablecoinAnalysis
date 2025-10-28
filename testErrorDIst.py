"""
Code for testing best error distributions in GARCH models
"""

import numpy as np
import pandas as pd
from arch import arch_model
from pathlib import Path
from scipy.stats import norm

dir = Path("Data/Verified")
usdt = pd.read_csv(dir / "Verif_USDT.csv", parse_dates=['Date']).set_index('Date').sort_index()
usdt = usdt[['RV']].rename(columns={'RV':'USDT_RV'})

# Example cryptos
cryptos = ['Verif_BNB.csv', 'Verif_BTC.csv', 'Verif_ETH.csv', 'Verif_XRP.csv']
cryptoPaths = [dir / f for f in cryptos]

dists = ['normal', 't', 'skewt', 'ged']
summary_rows = []

for file in cryptoPaths:
    df = pd.read_csv(file, parse_dates=['Date']).set_index('Date').sort_index()
    crypto_name = file.stem.replace("Verif_", "")

    results = {}
    for d in dists:
        m = arch_model(df['Log Returns'].dropna()*10, vol='Garch', p=1, q=1, mean='Constant', dist=d)
        res = m.fit(disp='off')

        # Standardized residuals
        std_resid = res.std_resid.dropna()
        n = len(std_resid)
        skew_val = std_resid.skew()

        # Skewness z-test
        skew_se = np.sqrt(6/n)
        z_score = skew_val / skew_se
        p_value = 2 * (1 - norm.cdf(abs(z_score)))
        skew_flag = p_value < 0.05  # True if skew significantly different from 0

        results[d] = {
            "loglik": res.loglikelihood,
            "aic": res.aic,
            "bic": res.bic,
            "skew": skew_val,
            "skew_z": z_score,
            "skew_p": p_value,
            "skew_flag": skew_flag
        }

    # Convert results to DataFrame
    stats_df = pd.DataFrame(results).T

    # Find best for each criterion
    best_loglik = stats_df["loglik"].idxmax()   # higher is better
    best_aic = stats_df["aic"].idxmin()         # lower is better
    best_bic = stats_df["bic"].idxmin()         # lower is better

    summary_rows.append({
        "Crypto": crypto_name,
        "Best (LogLik)": best_loglik,
        "Best (AIC)": best_aic,
        "Best (BIC)": best_bic,
        "Skew (LogLik)": stats_df.loc[best_loglik, "skew"],
        "Skew z-test p (LogLik)": stats_df.loc[best_loglik, "skew_p"],
        "Skew Flag (LogLik)": stats_df.loc[best_loglik, "skew_flag"],
        "Skew (AIC)": stats_df.loc[best_aic, "skew"],
        "Skew z-test p (AIC)": stats_df.loc[best_aic, "skew_p"],
        "Skew Flag (AIC)": stats_df.loc[best_aic, "skew_flag"],
        "Skew (BIC)": stats_df.loc[best_bic, "skew"],
        "Skew z-test p (BIC)": stats_df.loc[best_bic, "skew_p"],
        "Skew Flag (BIC)": stats_df.loc[best_bic, "skew_flag"]
    })

# Build summary table
summary_df = pd.DataFrame(summary_rows)
print(summary_df)
"""
Shows that AIC/BIC use t distirbution for best results in each coin
Log Likelihood use skewt for best results

Likely because AIC/BIC penalises extra parameters and skewt has the extra skew paramter.

We choose skew t because we show significant pvalues for skew in each test
"""
