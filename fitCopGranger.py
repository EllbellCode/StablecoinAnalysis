import numpy as np
import pandas as pd
from pathlib import Path
from scipy.special import betaln
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from statsmodels.tsa.ar_model import ar_select_order
from scipy.stats.mstats import winsorize 
import json
import warnings
from tqdm import tqdm 

warnings.filterwarnings('ignore')

# ===================================================================
# USER CONFIGURATION
# ===================================================================

# Set to True for Forward (Stable -> Crypto), False for Reverse (Crypto -> Stable)
RUN_FORWARD_TEST = True

if RUN_FORWARD_TEST:
    DIRECTION_NAME = "StableCrypto"
    TESTS_TO_RUN = [
        
        ("Stable_Volume", "Crypto_Volatility"),
        ("Stable_Volatility", "Crypto_Volatility"),
        ("Stable_Volume", "Crypto_Returns"),
        ("Stable_Volatility", "Crypto_Returns"),
        
        
        # ("Stable_Returns", "Crypto_Volatility"),
        # ("Stable_Returns", "Crypto_Returns"),
        # ("Stable_Returns", "Crypto_Volume"),
        
        
        # ("Stable_Volume", "Crypto_Volume"),
        # ("Stable_Volatility", "Crypto_Volume"),
    ]
else:
    DIRECTION_NAME = "CryptoStable"
    TESTS_TO_RUN = [
        # Original Crypto -> Stable
        ("Crypto_Returns", "Stable_Volume"),
        ("Crypto_Returns", "Stable_Volatility"),
        ("Crypto_Volatility", "Stable_Volume"),
        ("Crypto_Volatility", "Stable_Volatility"),

        # ("Crypto_Volume", "Stable_Volume"),
        # ("Crypto_Volume", "Stable_Volatility"),
        # ("Crypto_Volume", "Stable_Returns"),

        # ("Crypto_Returns", "Stable_Returns"),
        # ("Crypto_Volatility", "Stable_Returns"),
    ]

# Common Settings
DATA_DIR = Path("Data/Verified")
OUTPUT_DIR = Path("Results/GrangerCopula")
START_DATE = '2020-01-01'
END_DATE = '2024-01-01'
N_BOOT = 200
STATIONARY_VOL = "Delta_LogGK"
MAXLAGS = 1
INFO_CRITERION = 'aic' #At maxlag = 1 this has no effect as it will always choose the single lag
WINSOR_QUANTILE = 0.01
RANDOM_STATE = 123

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
    T, d = y_lags.shape
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
    print(f"Starting {DIRECTION_NAME} Copula Granger Causality Analysis (Winsorized)...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    coin_data = {}
    for file in DATA_DIR.glob("*.csv"):
        if (c := file.stem.replace("Verif_", "")) in ["DAI", "USDC", "USDT", "BNB", "BTC", "ETH", "XRP"]:
            df = pd.read_csv(file, parse_dates=['Date']).sort_values("Date")
            coin_data[c] = df[(df['Date'] >= START_DATE) & (df['Date'] <= END_DATE)].set_index('Date')

    # 2. Create Factors (MODIFIED TO INCLUDE WINSORIZATION)
    def get_fac(coins, var, name):
        # Concatenate raw data
        df_list = [coin_data[c][var] for c in coins if c in coin_data]
        if not df_list: 
             print(f"Warning: Missing data for {name}")
             return pd.Series(dtype=float)

        df = pd.concat(df_list, axis=1, keys=coins, join='inner').dropna()
        
        # Winsorize before PCA
        for col in df.columns:
            df[col] = winsorize(df[col], limits=[WINSOR_QUANTILE, WINSOR_QUANTILE])

        # Fit PCA
        pca = PCA(1).fit(StandardScaler().fit_transform(df))
        print(f"{name} Exp. Var: {pca.explained_variance_ratio_[0]:.2%}")
        
        # Transform and return
        return pd.Series(pca.transform(StandardScaler().fit_transform(df)).ravel(), index=df.index, name=name)

    factors = {
        "Stable_Volume": get_fac(["DAI", "USDC", "USDT"], "LogVolChange", "PC1_S_Vol"),
        "Stable_Volatility": get_fac(["DAI", "USDC", "USDT"], STATIONARY_VOL, "PC1_S_Vola"),
        "Stable_Returns": get_fac(["DAI", "USDC", "USDT"], "Log Returns", "PC1_S_Ret"), # NEW
        
        "Crypto_Volume": get_fac(["BNB", "BTC", "ETH", "XRP"], "LogVolChange", "PC1_C_Vol"), # NEW
        "Crypto_Returns": get_fac(["BNB", "BTC", "ETH", "XRP"], "Log Returns", "PC1_C_Ret"),
        "Crypto_Volatility": get_fac(["BNB", "BTC", "ETH", "XRP"], STATIONARY_VOL, "PC1_C_Vola")
    }

    # 3. Run Tests
    results, nulls = [], []
    print("-" * 50)
    for src, tgt in TESTS_TO_RUN:
        print(f"Testing: {src} -> {tgt}")
        if src not in factors or tgt not in factors:
            print(f"  Skipping: {src} or {tgt} not found in factors.")
            continue
            
        y_ser = factors[tgt]
        if y_ser.empty: 
            print(f"  Skipping: Target {tgt} is empty.")
            continue

        # Note: We force MAXLAGS here per your previous config
        opt_lag = MAXLAGS 
        print(f"  Lag: {opt_lag}")

        df = pd.concat([factors[src], y_ser], axis=1, join='inner').dropna()
        x, y = df.iloc[:, 0].values, df.iloc[:, 1].values

        if len(x) > opt_lag + 1:
            gc = calcGC(x, y, lag=opt_lag)
            print("  Bootstrapping...")
            null_dist = bootstrapGC(x, y, lag=opt_lag, n_boot=N_BOOT, random_state=RANDOM_STATE)
            pval = np.mean(null_dist >= gc)
            print(f"  -> GC: {gc:.4f}, p-val: {pval:.4f}\n")
            
            results.append({"Source": src, "Target": tgt, "Lag": opt_lag, "GC": gc, "p-value": pval})
            nulls.append({"Source": src, "Target": tgt, "Nulls": json.dumps(null_dist.tolist())})

    # 4. Save
    pd.DataFrame(results).to_csv(OUTPUT_DIR / f"GC_Results_{DIRECTION_NAME}_{MAXLAGS}.csv", index=False)
    pd.DataFrame(nulls).to_csv(OUTPUT_DIR / f"GC_Nulls_{DIRECTION_NAME}_{MAXLAGS}.csv", index=False)
    print(f"Done. Saved to {OUTPUT_DIR}")