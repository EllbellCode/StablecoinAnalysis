import numpy as np
import pandas as pd
from pathlib import Path
from scipy.special import betaln
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import json
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Script to Test Copula Granger Causality on Daily/Weekly/Monthly Horizons
# "Does todays/the last week/the last month have causality for tomorrow?"
# Use an expanding window PCA to prevent data leakage

# ===================================================================
# USER CONFIGURATION
# ===================================================================

# 1. INPUT SCOPE (The Predictor)
#    - "Day":   Uses Daily metrics (Today -> Tomorrow)
#    - "Week":  Uses Weekly metrics (Last 7 Days Rolling -> Tomorrow)
#    - "Month": Uses Monthly metrics (Last 30 Days Rolling -> Tomorrow)
INPUT_SCOPE = "Month" 

# 2. TEST DIRECTION
RUN_FORWARD_TEST = True

if RUN_FORWARD_TEST:
    DIRECTION_NAME = "StableCrypto"
    TESTS_TO_RUN = [
        # Original Tests
        ("Stable_Volume", "Crypto_Volatility"),
        ("Stable_Volume", "Crypto_Returns"),
        ("Stable_Volume", "Crypto_Upside_Vol"),
        ("Stable_Volume", "Crypto_Downside_Vol"),

        ("Stable_Volatility", "Crypto_Returns"),
        ("Stable_Volatility", "Crypto_Volatility"),
        ("Stable_Volatility", "Crypto_Upside_Vol"),
        ("Stable_Volatility", "Crypto_Downside_Vol"),
        
        ("Stable_Upside_Vol", "Crypto_Returns"),
        ("Stable_Upside_Vol", "Crypto_Volatility"),
        ("Stable_Upside_Vol", "Crypto_Downside_Vol"),
        ("Stable_Upside_Vol", "Crypto_Upside_Vol"),

        ("Stable_Downside_Vol", "Crypto_Upside_Vol"),
        ("Stable_Downside_Vol", "Crypto_Downside_Vol"),
        ("Stable_Downside_Vol", "Crypto_Volatility"),
        ("Stable_Downside_Vol", "Crypto_Returns"),
    ]
else:
    DIRECTION_NAME = "CryptoStable"
    TESTS_TO_RUN = [
        ("Crypto_Returns", "Stable_Volume"),
        ("Crypto_Returns", "Stable_Volatility"),
        ("Crypto_Upside_Vol", "Stable_Downside_Vol"), 
        ("Crypto_Downside_Vol", "Stable_Upside_Vol"),
    ]

# 3. SETTINGS
DATA_DIR = Path("Data/Verified")
OUTPUT_DIR = Path("Results/GrangerCopula")
START_DATE = '2020-01-01'
END_DATE = '2024-01-01'
N_BOOT = 200
WINSOR_QUANTILE = 0.01
RANDOM_STATE = 123
MAXLAGS = 1
MIN_PCA_PERIODS = 30 

# ===================================================================
# Data Loading & Processing
# ===================================================================

SCOPE_SUFFIXES = {
    "Day": "",
    "Week": "_Weekly",
    "Month": "_Monthly"
}

# --- MODIFIED: Added Mappings for New Upside/Downside Columns ---
VAR_MAP = {
    "Volume": "LogVolChange",
    #"Volatility": "Delta_LogGK",
    "Volatility" : "RS",
    "Returns": "Log Returns",
    "Upside_Vol": "Upside_Vol",     
    "Downside_Vol": "Downside_Vol"
}

def get_column_name(metric_type, scope):
    base = VAR_MAP.get(metric_type, metric_type)
    suffix = SCOPE_SUFFIXES.get(scope, "")
    return f"{base}{suffix}"

# ===================================================================
# Expanding PCA Logic (Fixes Lookahead Bias)
# ===================================================================
def get_expanding_pca(df, min_periods=30, winsor_limit=0.01):
    """
    Calculates the first Principal Component using an expanding window
    to prevent lookahead bias.
    """
    n_samples, n_features = df.shape
    factors = np.full(n_samples, np.nan)
    prev_component = None
    
    iter_range = range(min_periods, n_samples + 1)
    
    for t in iter_range:
        window = df.iloc[:t].copy()
        for col in window.columns:
            window[col] = winsorize(window[col].values, limits=[winsor_limit, winsor_limit])
            
        scaler = StandardScaler()
        window_std = scaler.fit_transform(window)
        
        pca = PCA(n_components=1)
        pca.fit(window_std)
        current_vec = pca.components_[0]
        
        if prev_component is not None:
            if np.dot(current_vec, prev_component) < 0:
                current_vec = -current_vec
        else:
            if np.sum(current_vec) < 0:
                current_vec = -current_vec

        prev_component = current_vec
        last_row_scaled = window_std[-1]
        factors[t-1] = np.dot(last_row_scaled, current_vec)

    return pd.Series(factors, index=df.index)

# ===================================================================
# Core Copula Granger Causality Functions
# ===================================================================
def kernelWeights(query, samples, h):
    q2 = np.sum(query**2, axis=1, keepdims=True)
    s2 = np.sum(samples**2, axis=1, keepdims=True).T
    cross = query @ samples.T
    dist2 = np.maximum(q2 + s2 - 2.0 * cross, 0.0)
    return np.exp(-0.5 * dist2 / (h**2))

def calcBandwidth(z, d):
    n = z.shape[0]
    return np.power(max(n, 2), -1.0 / (d + 4.0))

def yCMD(y_next, y_lags, h, query_y_next=None, query_y_lags=None):
    if query_y_lags is None: query_y_lags, query_y_next = y_lags, y_next
    W = kernelWeights(query_y_lags, y_lags, h)
    W_sum = W.sum(axis=1, keepdims=True) + 1e-12
    num = (W * (y_next[None, :] <= query_y_next[:, None])).sum(axis=1, keepdims=True)
    return np.clip((num / W_sum).ravel(), 1e-6, 1.0 - 1e-6)

def xCMD(x_lags, y_lags, h, query_x_lags=None, query_y_lags=None):
    T, m = x_lags.shape
    if query_y_lags is None: query_y_lags, query_x_lags = y_lags, x_lags
    W = kernelWeights(query_y_lags, y_lags, h)
    W_sum = W.sum(axis=1, keepdims=True) + 1e-12
    Q = query_x_lags.shape[0]
    ind = np.ones((Q, T), dtype=bool)
    for j in range(m): ind &= (x_lags[None, :, j] <= query_x_lags[:, None, j])
    return np.clip(((W * ind).sum(axis=1, keepdims=True) / W_sum).ravel(), 1e-6, 1.0 - 1e-6)

def calcBernstein(u, v, m=10):
    u, v = np.clip(np.asarray(u), 1e-6, 1-1e-6), np.clip(np.asarray(v), 1e-6, 1-1e-6)
    T = len(u); bins = np.linspace(0.0, 1.0, m + 1)
    ui, vi = np.clip(np.digitize(u, bins) - 1, 0, m - 1), np.clip(np.digitize(v, bins) - 1, 0, m - 1)
    W = np.zeros((m, m), dtype=float)
    for k in range(T): W[ui[k], vi[k]] += 1.0
    W /= max(T, 1)
    i = np.arange(1, m + 1); alpha, beta = i[None, :], (m - i + 1)[None, :]
    log_norm = -betaln(alpha, beta)
    def get_B(points): return np.exp((alpha - 1) * np.log(points[:, None]) + (beta - 1) * np.log(1 - points[:, None]) + log_norm)
    return np.clip(np.einsum('tm,tm->t', get_B(u) @ W, get_B(v)), 1e-12, None)

def calcGC(x, y, lag=1, m_bern=10, h=None):
    x, y = np.asarray(x), np.asarray(y)
    valid = (~np.isnan(x)) & (~np.isnan(y))
    x, y = x[valid], y[valid]
    n = len(x)
    if n <= lag + 1: return np.nan
    y_lags = np.column_stack([y[i:n - lag + i] for i in range(lag)])
    x_lags = np.column_stack([x[i:n - lag + i] for i in range(lag)])
    y_lags_std = StandardScaler().fit_transform(y_lags)
    h = calcBandwidth(y_lags_std, y_lags_std.shape[1]) if h is None else float(h)
    u, v = yCMD(y[lag:], y_lags_std, h), xCMD(x_lags, y_lags_std, h)
    return float(np.mean(np.log(calcBernstein(u, v, m=m_bern))))

def bootstrapGC(x, y, lag=1, n_boot=200, m_bern=10, h=None, random_state=None):
    rng = np.random.default_rng(random_state)
    x, y = np.asarray(x), np.asarray(y)
    valid = (~np.isnan(x)) & (~np.isnan(y)); x, y = x[valid], y[valid]
    n = len(x); T = n - lag
    if n <= lag + 1: return np.full(n_boot, np.nan)
    y_lags = np.column_stack([y[i:n - lag + i] for i in range(lag)])
    x_lags = np.column_stack([x[i:n - lag + i] for i in range(lag)])
    y_next = y[lag:]
    y_lags_std = StandardScaler().fit_transform(y_lags)
    h = calcBandwidth(y_lags_std, y_lags_std.shape[1]) if h is None else float(h)
    W_full = kernelWeights(y_lags_std, y_lags_std, h)
    row_sums = W_full.sum(axis=1, keepdims=True)
    W_probs = W_full / np.where(row_sums == 0, 1.0, row_sums)
    gc_null = np.empty(n_boot, dtype=float)
    for b in tqdm(range(n_boot), desc="Bootstrapping", leave=False):
        idx_next = np.argmax(np.cumsum(W_probs[rng.integers(0, T, size=T)], axis=1) >= rng.random(size=(T, 1)), axis=1)
        y_star = np.concatenate([y[:lag], y_next[idx_next]])
        x_star = np.concatenate([x[:lag], x_lags[idx_next][:, 0]])
        gc_null[b] = calcGC(x_star, y_star, lag=lag, m_bern=m_bern, h=None)
    return gc_null

# ===================================================================
# Main Execution
# ===================================================================
if __name__ == "__main__":
    print(f"Starting {DIRECTION_NAME} Analysis")
    print(f"Prediction: {INPUT_SCOPE} Metrics -> Daily Metrics (Tomorrow)")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    print("Loading Data...")
    coin_data = {}
    for file in DATA_DIR.glob("*.csv"):
        if (c := file.stem.replace("Verif_", "")) in ["DAI", "USDC", "USDT", "BNB", "BTC", "ETH", "XRP"]:
            df = pd.read_csv(file, parse_dates=['Date']).sort_values("Date")
            coin_data[c] = df[(df['Date'] >= START_DATE) & (df['Date'] <= END_DATE)].set_index('Date')

    # 2. Factor Creation
    def get_fac(coins, metric_type, scope, name):
        col_name = get_column_name(metric_type, scope)
        if coins and col_name not in coin_data[coins[0]].columns:
            print(f"Warning: Column {col_name} not found.")
            return pd.Series(dtype=float)
        df_list = [coin_data[c][col_name] for c in coins if c in coin_data]
        if not df_list: return pd.Series(dtype=float)
        df = pd.concat(df_list, axis=1, keys=coins, join='inner').dropna()
        if df.empty: return pd.Series(dtype=float)
        print(f"  Calculating Expanding PCA for {name} ({len(df)} rows)...")
        factor_series = get_expanding_pca(df, min_periods=MIN_PCA_PERIODS, winsor_limit=WINSOR_QUANTILE)
        factor_series.name = name
        return factor_series

    # 3. Build Factor Dictionary
    print(f"\nExtracting Factors (Input Scope: {INPUT_SCOPE})...")
    
    factors = {}
    stable_coins = ["DAI", "USDC", "USDT"]
    crypto_coins = ["BNB", "BTC", "ETH", "XRP"]
    
    # --- MODIFIED: Added Upside_Vol and Downside_Vol to metrics list ---
    metrics = ["Volume", "Volatility", "Returns", "Upside_Vol", "Downside_Vol"]
    
    for m in metrics:
        # 1. Always calculate the Input (Predictor)
        factors[f"Stable_{m}_Input"] = get_fac(stable_coins, m, INPUT_SCOPE, f"Stable_{m}_{INPUT_SCOPE}")
        factors[f"Crypto_{m}_Input"] = get_fac(crypto_coins, m, INPUT_SCOPE, f"Crypto_{m}_{INPUT_SCOPE}")

        # 2. Handle the Daily (Target)
        if INPUT_SCOPE == "Day":
            # OPTIMIZATION: If Input is already 'Day', just point to the existing data
            factors[f"Stable_{m}_Daily"] = factors[f"Stable_{m}_Input"]
            factors[f"Crypto_{m}_Daily"] = factors[f"Crypto_{m}_Input"]
        else:
            # Otherwise, we need to calculate 'Day' separately (e.g., if Input was 'Week')
            factors[f"Stable_{m}_Daily"] = get_fac(stable_coins, m, "Day", f"Stable_{m}_Day")
            factors[f"Crypto_{m}_Daily"] = get_fac(crypto_coins, m, "Day", f"Crypto_{m}_Day")

    # 4. Run Tests
    results, nulls = [], []
    print("-" * 60)
    
    for src_type, tgt_type in TESTS_TO_RUN:
        src_key = f"{src_type}_Input"
        tgt_key = f"{tgt_type}_Daily"
        
        print(f"Testing: {src_key} -> {tgt_key}")
        
        if src_key not in factors or tgt_key not in factors:
            print(f"  Skipping: Factors missing.")
            continue
            
        x_series = factors[src_key]
        y_series = factors[tgt_key]
        
        df = pd.concat([x_series, y_series], axis=1, join='inner').dropna()
        if df.empty:
            print("  Skipping: Empty data after alignment.")
            continue
            
        x, y = df.iloc[:, 0].values, df.iloc[:, 1].values

        if len(x) > MAXLAGS + 1:
            gc = calcGC(x, y, lag=MAXLAGS)
            print("  Bootstrapping...")
            null_dist = bootstrapGC(x, y, lag=MAXLAGS, n_boot=N_BOOT, random_state=RANDOM_STATE)
            pval = np.mean(null_dist >= gc)
            print(f"  -> GC: {gc:.4f}, p-val: {pval:.4f}\n")
            
            results.append({
                "Source": src_type, 
                "Target": tgt_type, 
                "InputScope": INPUT_SCOPE,
                "GC": gc, 
                "p-value": pval
            })
            nulls.append({
                "Source": src_type, 
                "Target": tgt_type, 
                "InputScope": INPUT_SCOPE,
                "Nulls": json.dumps(null_dist.tolist())
            })

    fname_res = OUTPUT_DIR / f"GC_Results_{DIRECTION_NAME}_{INPUT_SCOPE}.csv"
    fname_null = OUTPUT_DIR / f"GC_Nulls_{DIRECTION_NAME}_{INPUT_SCOPE}.csv"
    
    pd.DataFrame(results).to_csv(fname_res, index=False)
    pd.DataFrame(nulls).to_csv(fname_null, index=False)
    print(f"Done. Results saved to:\n  {fname_res}")