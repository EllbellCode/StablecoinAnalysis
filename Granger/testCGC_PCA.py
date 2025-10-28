"""
Test For Copula Granger Causality
Used to capture non-linear causalities
Rather than test each stablecoin metric/ crypto metric pair individually
We combine the stablecoin metrics into a single time series using PCA (one for volatility, another for volume)
and do the same for cryptos
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.special import betaln
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from statsmodels.tsa.ar_model import ar_select_order
import json
import warnings

"""
Core Copula Granger Causality Functions
"""

"""
Gaussian Kernel Weights

Returned weights are unnormalised
"""
def kernelWeights(query, samples, h):
    q2 = np.sum(query**2, axis=1, keepdims=True)
    s2 = np.sum(samples**2, axis=1, keepdims=True).T
    cross = query @ samples.T
    dist2 = np.maximum(q2 + s2 - 2.0 * cross, 0.0)
    w = np.exp(-0.5 * dist2 / (h**2))
    return w

"""
Calculating Kernel bandwidth using Silvermans Rule
"""
def calcBandwidth(z, d):
    n = z.shape[0]
    return np.power(max(n, 2), -1.0 / (d + 4.0))

"""
Estimates the Conditional Marginal Distribution F(y_{t+1} | y_t^n)

Uses Kernel Density Estimation
"""
def yCMD(y_next, y_lags, h, query_y_next=None, query_y_lags=None):
    T, d = y_lags.shape
    if query_y_lags is None:
        query_y_lags = y_lags
        query_y_next = y_next
    W = kernelWeights(query_y_lags, y_lags, h)
    W_sum = W.sum(axis=1, keepdims=True) + 1e-12
    indicators = (y_next[None, :] <= query_y_next[:, None])
    num = (W * indicators).sum(axis=1, keepdims=True)
    F = (num / W_sum).ravel()
    eps = 1e-6
    return np.clip(F, eps, 1.0 - eps)

"""
Estimates the Conditional Marginal Distribution G(x_t^m | y_t^n)

Uses Kernel Density Estimation
"""
def xCMD(x_lags, y_lags, h, query_x_lags=None, query_y_lags=None):
    T, m = x_lags.shape
    if query_y_lags is None:
        query_y_lags = y_lags
        query_x_lags = x_lags
    W = kernelWeights(query_y_lags, y_lags, h)
    W_sum = W.sum(axis=1, keepdims=True) + 1e-12
    Q = query_x_lags.shape[0]
    ind = np.ones((Q, T), dtype=bool)
    for j in range(m):
        ind &= (x_lags[None, :, j] <= query_x_lags[:, None, j])
    num = (W * ind).sum(axis=1, keepdims=True)
    G = (num / W_sum).ravel()
    eps = 1e-6
    return np.clip(G, eps, 1.0 - eps)

"""
Computes the grid of Beta Densities for Bernstein Approximation

Uses log Beta for stability
"""
def betaGrid(points, m):
    points = np.clip(points, 1e-6, 1 - 1e-6)
    T = len(points)
    i = np.arange(1, m + 1)
    alpha = i[None, :]
    beta = (m - i + 1)[None, :]
    log_norm = -betaln(alpha, beta)
    log_pdf = (alpha - 1) * np.log(points[:, None]) + (beta - 1) * np.log(1 - points[:, None]) + log_norm
    return np.exp(log_pdf)

"""
Estimate Copula Density using Bernstein Approximation

c(u,v) = sum_{i=1}^m sum_{j=1}^m w_ij * Beta_i(u) * Beta_j(v)
    where w_ij are 2D histogram frequencies of (u,v) on an m×m grid.
"""
def calcBernstein(u, v, m=10):
    u = np.clip(np.asarray(u), 1e-6, 1 - 1e-6)
    v = np.clip(np.asarray(v), 1e-6, 1 - 1e-6)
    T = len(u)
    bins = np.linspace(0.0, 1.0, m + 1)
    ui = np.clip(np.digitize(u, bins) - 1, 0, m - 1)
    vi = np.clip(np.digitize(v, bins) - 1, 0, m - 1)
    W = np.zeros((m, m), dtype=float)
    for k in range(T):
        W[ui[k], vi[k]] += 1.0
    W /= max(T, 1)
    Bu = betaGrid(u, m)
    Bv = betaGrid(v, m)
    mid = Bu @ W
    dens = np.einsum('tm,tm->t', mid, Bv)
    return np.clip(dens, 1e-12, None)

"""
Calculates the Granger Causality using the Bernstein Approximated copula density
      1) conditional kernel CDFs to get u = F(y_{t+1}|y_lags), v = G(x_lags|y_lags)
      2) Bernstein copula density estimator c(u,v)
      3) GC = mean(log c(u,v))
"""
def calcGC(x, y, lag=1, m_bern=10, h=None):
    x = np.asarray(x)
    y = np.asarray(y)
    valid = (~np.isnan(x)) & (~np.isnan(y))
    x, y = x[valid], y[valid]
    n = len(x)
    if n <= lag + 1:
        return np.nan
    T = n - lag
    y_lags = np.column_stack([y[i:n - lag + i] for i in range(lag)])
    x_lags = np.column_stack([x[i:n - lag + i] for i in range(lag)])
    y_next = y[lag:]
    scaler_y = StandardScaler().fit(y_lags)
    y_lags_std = scaler_y.transform(y_lags)
    d = y_lags_std.shape[1]
    h = calcBandwidth(y_lags_std, d) if h is None else float(h)
    u = yCMD(y_next, y_lags_std, h)
    v = xCMD(x_lags, y_lags_std, h)
    c_hat = calcBernstein(u, v, m=m_bern)
    gc = float(np.mean(np.log(c_hat)))
    return gc

"""
Uses Bootstrapping to generate n_boot synthetic datasets
Computes GC for each dataset and builds null distribution
Returns (n_boot,) array of GC statistics to form null distribution.
"""
def bootstrapGC(x, y, lag=1, n_boot=200, m_bern=10, h=None, random_state=None):
    rng = np.random.default_rng(random_state)
    x = np.asarray(x)
    y = np.asarray(y)
    valid = (~np.isnan(x)) & (~np.isnan(y))
    x, y = x[valid], y[valid]
    n = len(x)
    if n <= lag + 1:
        return np.full(n_boot, np.nan)
    T = n - lag
    y_lags = np.column_stack([y[i:n - lag + i] for i in range(lag)])
    x_lags = np.column_stack([x[i:n - lag + i] for i in range(lag)])
    y_next = y[lag:]
    scaler_y = StandardScaler().fit(y_lags)
    y_lags_std = scaler_y.transform(y_lags)
    d = y_lags_std.shape[1]
    h = calcBandwidth(y_lags_std, d) if h is None else float(h)
    
    def draw_y_next_given(y_lag_q_std):
        w = kernelWeights(y_lag_q_std[None, :], y_lags_std, h).ravel()
        if not np.any(w): w = np.ones_like(w)
        w /= w.sum()
        idx = rng.choice(T, size=1, replace=True, p=w)
        return y_next[idx][0]

    def draw_x_lags_given(y_lag_q_std):
        w = kernelWeights(y_lag_q_std[None, :], y_lags_std, h).ravel()
        if not np.any(w): w = np.ones_like(w)
        w /= w.sum()
        idx = rng.choice(T, size=1, replace=True, p=w)
        return x_lags[idx][0]

    gc_null = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx_y = rng.integers(0, T, size=T)
        y_lags_star = y_lags[idx_y]
        y_lags_star_std = scaler_y.transform(y_lags_star)
        y_next_star = np.array([draw_y_next_given(y_lags_star_std[t]) for t in range(T)])
        x_lags_star = np.vstack([draw_x_lags_given(y_lags_star_std[t]) for t in range(T)])
        y_star = np.concatenate([y[:lag], y_next_star])
        x_star = np.concatenate([x[:lag], x_lags_star[:, 0]])
        gc_null[b] = calcGC(x_star, y_star, lag=lag, m_bern=m_bern, h=None)
    return gc_null


"""
Data Prelims
"""

# --- 1. Data Setup and Loading ---
data_dir = Path("Data/Verified")
files = list(data_dir.glob("*.csv"))

# Use the stationary variables we confirmed
STATIONARY_VOL = "Delta_LogRV" 

source_coins = ["DAI", "USDC", "USDT"]
target_coins = ["BNB", "BTC", "ETH", "XRP"]
n_boot = 1000

# Load all data into a dictionary of DataFrames, indexed by Date
coin_data = {}
for file in files:
    coin_name = file.stem.replace("Verif_", "")
    if coin_name in source_coins or coin_name in target_coins:
        df = pd.read_csv(file).sort_values("Date")
        df['Date'] = pd.to_datetime(df['Date'])
        train = df[(df['Date'] >= '2020-01-01') & (df['Date'] <= '2023-12-31')]
        coin_data[coin_name] = train.set_index('Date')

print("All coin data loaded.")


"""
PCA functions
"""

"""
Builds a data matrix, standardizes, and runs PCA.
Returns a pd.Series (the PC1 factor)
"""
def create_pca_factor(coins, var, data_dict, factor_name):
    
    df_list = [data_dict[coin][var] for coin in coins]
    data_matrix = pd.concat(df_list, axis=1, keys=coins, join='inner')
    data_matrix.dropna(inplace=True)
    
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_matrix)
    
    pca = PCA(n_components=1)
    factor = pca.fit_transform(data_scaled)
    
    print(f"Created Factor '{factor_name}'. PCA Explained Variance: {pca.explained_variance_ratio_[0]:.2%}")
    # Print component loadings (the "drill-down")
    print(f"  Loadings: {dict(zip(coins, pca.components_[0]))}\n")
    
    return pd.Series(factor.ravel(), index=data_matrix.index, name=factor_name)

"""
Uses AutoReg to find the optimal lag for a series based on AIC
"""
def find_optimal_lag(series, max_lags=25, ic='aic'):
    
    series = series.dropna()
    safe_max_lags = min(max_lags, len(series) // 2 - 1)
    
    if safe_max_lags < 1:
        return 1
        
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        # Use the modern, standalone lag selection function
        sel = ar_select_order(series, maxlag=safe_max_lags, ic=ic)
        
    optimal_lag = max(sel.ar_lags) if sel.ar_lags else 1
    return optimal_lag

"""
Create the PCA factors
"""
print("Creating PCA Factors...")

factors = {}
factors["Stable_Volume"] = create_pca_factor(
    source_coins, "LogVolChange", coin_data, "PC1_Stable_Volume"
)
factors["Stable_Volatility"] = create_pca_factor(
    source_coins, STATIONARY_VOL, coin_data, "PC1_Stable_Volatility"
)
factors["Crypto_Returns"] = create_pca_factor(
    target_coins, "Log Returns", coin_data, "PC1_Crypto_Returns"
)
factors["Crypto_Volatility"] = create_pca_factor(
    target_coins, STATIONARY_VOL, coin_data, "PC1_Crypto_Volatility"
)


"""
Run the PCA Copula Granger 
"""
tests_to_run = [
    # (Source Factor Name, Target Factor Name)
    ("Stable_Volume", "Crypto_Volatility"),
    ("Stable_Volatility", "Crypto_Volatility"),
    ("Stable_Volume", "Crypto_Returns"),
    ("Stable_Volatility", "Crypto_Returns"),
]

results = []
null_dist_data = []
max_aic_lags = 30 # Max lags to check for AIC (based on your ADF results)

for source_key, target_key in tests_to_run:
    
    x_series = factors[source_key]
    y_series = factors[target_key]
    
    print(f"\nProcessing Test: {source_key} -> {target_key}")

    # 1. Find the optimal lag for the TARGET (y) series to control for its memory
    optimal_lag = find_optimal_lag(y_series, max_lags=max_aic_lags, ic='aic')
    print(f"  Optimal lag for {target_key} (AIC): {optimal_lag}")

    # 2. Align the two factor time series by date and drop NaNs
    aligned_df = pd.concat([x_series, y_series], axis=1, join='inner')
    aligned_df.dropna(inplace=True)
    
    x = aligned_df[x_series.name].values
    y = aligned_df[y_series.name].values

    if len(x) > optimal_lag + 1:
        # 3. Run the GC test using the CORRECT lag
        gc = calcGC(x, y, lag=optimal_lag, m_bern=10, h=None)
        
        # 4. Run the bootstrap for the null distribution
        null_dist = bootstrapGC(x, y, lag=optimal_lag, n_boot=n_boot, m_bern=10, h=None, random_state=123)
        
        # 5. Calculate p-value
        p_value = np.mean(null_dist >= gc)
        
        print(f"  --> GC={gc:.6f}, p-value={p_value:.4f}")
        
        results.append({
            "Source_Factor": source_key,
            "Target_Factor": target_key,
            "lag": optimal_lag,
            "Copula GC": gc,
            "p-value": p_value,
        })
        
        null_dist_data.append({
            "Source_Factor": source_key,
            "Target_Factor": target_key,
            "lag": optimal_lag,
            "GC_test": gc,
            "null_dist": json.dumps(null_dist.tolist())
        })
    else:
        print(f"  Skipping {source_key} -> {target_key}, not enough data after alignment.")

# --- 5. Save Results ---
results_df = pd.DataFrame(results)
results_df.to_csv("grangerCopula_PCA_Results.csv", index=False)

sig_results = results_df[results_df["p-value"] < 0.05]
sig_results.to_csv("grangerCopula_PCA_Significant.csv", index=False)

null_df = pd.DataFrame(null_dist_data)
null_df.to_csv("grangerCopula_PCA_Nulls.csv", index=False)

print("\nAnalysis complete. Results saved to 'grangerCopula_PCA_Results.csv'.")