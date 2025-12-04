import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
from statsmodels.tsa.api import VAR
import warnings

warnings.filterwarnings('ignore')

# ===================================================================
# CONFIGURATION
# ===================================================================
DATA_DIR = Path("Data/Verified")
OUTPUT_DIR = Path("Results/GrangerLinear")
START_DATE = '2020-01-01'
END_DATE = '2024-01-01'
STATIONARY_VOL = "Delta_LogRV"
MAXLAGS = 1
WINSOR_QUANTILE = 0.01 

# ===================================================================
# Helper Functions
# ===================================================================

def check_stationarity(series):
    try:
        result = adfuller(series.dropna())
        return result[1] < 0.05
    except:
        return False

def load_and_process_factors(data_dir, start_date, end_date, winsor_limit):
    print("Loading and processing data...")
    coin_data = {}
    if not data_dir.exists(): raise FileNotFoundError(f"Data directory not found: {data_dir}")

    for file in data_dir.glob("*.csv"):
        if (c := file.stem.replace("Verif_", "")) in ["DAI", "USDC", "USDT", "BNB", "BTC", "ETH", "XRP"]:
            df = pd.read_csv(file, parse_dates=['Date']).sort_values("Date")
            coin_data[c] = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)].set_index('Date')

    def get_fac(coins, var, name):
        df_list = [coin_data[c][var] for c in coins if c in coin_data]
        if not df_list: return pd.Series(dtype=float, name=name)
        df = pd.concat(df_list, axis=1, keys=coins, join='inner').dropna()
        for col in df.columns: df[col] = winsorize(df[col], limits=[winsor_limit, winsor_limit])
        pca = PCA(1).fit(StandardScaler().fit_transform(df))
        return pd.Series(pca.transform(StandardScaler().fit_transform(df)).ravel(), index=df.index, name=name)

    factors = {
        "Stable_Volume": get_fac(["DAI", "USDC", "USDT"], "LogVolChange", "Stable_Volume"),
        "Stable_Volatility": get_fac(["DAI", "USDC", "USDT"], STATIONARY_VOL, "Stable_Volatility"),
        "Stable_Returns": get_fac(["DAI", "USDC", "USDT"], "Log Returns", "Stable_Returns"),
        "Crypto_Volume": get_fac(["BNB", "BTC", "ETH", "XRP"], "LogVolChange", "Crypto_Volume"),
        "Crypto_Volatility": get_fac(["BNB", "BTC", "ETH", "XRP"], STATIONARY_VOL, "Crypto_Volatility"),
        "Crypto_Returns": get_fac(["BNB", "BTC", "ETH", "XRP"], "Log Returns", "Crypto_Returns"),
    }
    return factors

def run_batch_granger(factors, pairs, description):
    results = []
    print(f"\n--- {description} ---")
    print(f"{'Source':<20} | {'Target':<20} | {'Lag':<3} | {'F-Stat':<8} | {'p-value':<8} | {'Sig?'}")
    print("-" * 90)

    for src_name, tgt_name in pairs:
        src_series = factors[src_name]
        tgt_series = factors[tgt_name]
        if src_series.empty or tgt_series.empty: continue

        df = pd.concat([tgt_series, src_series], axis=1, join='inner').dropna()
        df.columns = ['Target', 'Source']

        # Silent stationarity check (can add logging if needed)
        if not check_stationarity(df['Target']) or not check_stationarity(df['Source']): pass 

        try:
            model = VAR(df)
            lag_results = model.select_order(maxlags=MAXLAGS)
            best_lag = lag_results.aic if lag_results.aic > 0 else 1
            best_lag = MAXLAGS
        except:
            best_lag = 1

        try:
            gc_res = grangercausalitytests(df[['Target', 'Source']], maxlag=[best_lag], verbose=False)
            f_stat = gc_res[best_lag][0]['ssr_ftest'][0]
            p_value = gc_res[best_lag][0]['ssr_ftest'][1]
            is_sig = "YES" if p_value < 0.05 else "No"
            
            print(f"{src_name:<20} | {tgt_name:<20} | {best_lag:<3} | {f_stat:<8.4f} | {p_value:<8.4f} | {is_sig}")
            
            results.append({
                "Source": src_name, "Target": tgt_name, "Optimal_Lag": best_lag,
                "F_Statistic": f_stat, "p_value": p_value, "Significant": is_sig
            })
        except Exception as e:
            print(f"Error: {src_name}->{tgt_name}: {e}")
            
    return pd.DataFrame(results)

# ===================================================================
# Main Execution
# ===================================================================
if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    factors = load_and_process_factors(DATA_DIR, START_DATE, END_DATE, WINSOR_QUANTILE)
    
    stable_vars = ["Stable_Volume", "Stable_Volatility", "Stable_Returns"]
    crypto_vars = ["Crypto_Volume", "Crypto_Volatility", "Crypto_Returns"]
    
    # 1. Stable -> Crypto
    pairs_forward = []
    for s in stable_vars:
        for c in crypto_vars:
            pairs_forward.append((s, c))
            
    df_forward = run_batch_granger(factors, pairs_forward, "Stable -> Crypto Causality")
    save_fwd = OUTPUT_DIR / "Linear_GC_Stable_to_Crypto.csv"
    df_forward.to_csv(save_fwd, index=False)

    # 2. Crypto -> Stable
    pairs_reverse = []
    for c in crypto_vars:
        for s in stable_vars:
            pairs_reverse.append((c, s))
            
    df_reverse = run_batch_granger(factors, pairs_reverse, "Crypto -> Stable Causality")
    save_rev = OUTPUT_DIR / "Linear_GC_Crypto_to_Stable.csv"
    df_reverse.to_csv(save_rev, index=False)
    
    print("-" * 90)
    print(f"Analysis Complete. Results saved to:")
    print(f"1. {save_fwd.resolve()}")
    print(f"2. {save_rev.resolve()}")