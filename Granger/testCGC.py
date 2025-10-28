"""
Test For Copula Granger Causality
Used to capture non-linear causalities
"""


import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product
from scipy.special import betaln
from sklearn.preprocessing import StandardScaler
import json

"""
Data Setup and Prelims
"""
data_dir = Path("Data/Verified")
files = list(data_dir.glob("*.csv"))

volatility = "RV"
source_coins = ["DAI", "USDC", "USDT"]
#source_coins = ["USDT"]
source_vars  = [volatility, "LogVolChange"]
target_coins = ["BNB", "BTC", "ETH", "XRP"]
#target_coins = ["BTC"]
target_vars  = [volatility, "Log Returns"]
n_boot = 200
max_lag = 1

coin_data = {}
for file in files:
    coin_name = file.stem.replace("Verif_", "")
    df = pd.read_csv(file).sort_values("Date").dropna(subset=["Log Returns", "LogVolChange", volatility])
    train = df[(df['Date'] >= '2020-01-01') & (df['Date'] <= '2023-12-31')]
    coin_data[coin_name] = train

"""
Gaussian Kernel Weights

Returned weights are unnormalised
"""
def kernelWeights(query, samples, h):
    
    # squared L2 distance: ||q - s||^2 = |q|^2 + |s|^2 - 2 q.s
    q2 = np.sum(query**2, axis=1, keepdims=True)        # (Q,1)
    s2 = np.sum(samples**2, axis=1, keepdims=True).T    # (1,N)
    cross = query @ samples.T                            # (Q,N)
    dist2 = np.maximum(q2 + s2 - 2.0 * cross, 0.0)       # (Q,N)
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
    W = kernelWeights(query_y_lags, y_lags, h)           # (Q,T)
    W_sum = W.sum(axis=1, keepdims=True) + 1e-12
    indicators = (y_next[None, :] <= query_y_next[:, None])  # (Q,T)
    num = (W * indicators).sum(axis=1, keepdims=True)
    F = (num / W_sum).ravel()
    # clip to (0,1) to avoid boundary issues in beta pdf later
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

    W = kernelWeights(query_y_lags, y_lags, h)          # (Q,T)
    W_sum = W.sum(axis=1, keepdims=True) + 1e-12

    # For each query q, indicator_i = 1[ x_i^m <= x_q^m (componentwise) ]
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
    alpha = i[None, :]                 # (1,m)
    beta = (m - i + 1)[None, :]        # (1,m)

    log_norm = -betaln(alpha, beta)    # (1,m)
    # log pdf = (alpha-1)log x + (beta-1)log(1-x) - logB(alpha,beta)
    log_pdf = (alpha - 1) * np.log(points[:, None]) + (beta - 1) * np.log(1 - points[:, None]) + log_norm
    return np.exp(log_pdf)             # (T,m)

"""
Estimate Copula Density using Bernstein Approximation

c(u,v) = sum_{i=1}^m sum_{j=1}^m w_ij * Beta_i(u) * Beta_j(v)
    where w_ij are 2D histogram frequencies of (u,v) on an m×m grid.
"""
def calcBernstein(u, v, m=10):
    
    u = np.clip(np.asarray(u), 1e-6, 1 - 1e-6)
    v = np.clip(np.asarray(v), 1e-6, 1 - 1e-6)
    T = len(u)

    # 2D histogram (weights sum to 1)
    bins = np.linspace(0.0, 1.0, m + 1)
    ui = np.clip(np.digitize(u, bins) - 1, 0, m - 1)  # 0..m-1
    vi = np.clip(np.digitize(v, bins) - 1, 0, m - 1)
    W = np.zeros((m, m), dtype=float)
    for k in range(T):
        W[ui[k], vi[k]] += 1.0
    W /= max(T, 1)

    Bu = betaGrid(u, m)  # (T,m)
    Bv = betaGrid(v, m)  # (T,m)

    # density_t = Bu[t,:] @ W @ Bv[t,:]^T
    # Compute efficiently for all t
    mid = Bu @ W                # (T,m)
    dens = np.einsum('tm,tm->t', mid, Bv)  # (T,)
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

    # build lags
    T = n - lag
    y_lags = np.column_stack([y[i:n - lag + i] for i in range(lag)])   # (T,lag)
    x_lags = np.column_stack([x[i:n - lag + i] for i in range(lag)])   # (T,lag)
    y_next = y[lag:]                                                   # (T,)

    # standardize y_lags for kernel distances (x_lags not needed for weights)
    scaler_y = StandardScaler().fit(y_lags)
    y_lags_std = scaler_y.transform(y_lags)

    d = y_lags_std.shape[1]
    h = calcBandwidth(y_lags_std, d) if h is None else float(h)

    # conditional PITs
    u = yCMD(y_next, y_lags_std, h)
    v = xCMD(x_lags, y_lags_std, h)

    # copula density & GC
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

    # original lagged data (used to estimate the conditionals for resampling)
    T = n - lag
    y_lags = np.column_stack([y[i:n - lag + i] for i in range(lag)])   # (T,lag)
    x_lags = np.column_stack([x[i:n - lag + i] for i in range(lag)])   # (T,lag)
    y_next = y[lag:]                                                   # (T,)

    scaler_y = StandardScaler().fit(y_lags)
    y_lags_std = scaler_y.transform(y_lags)
    d = y_lags_std.shape[1]
    h = calcBandwidth(y_lags_std, d) if h is None else float(h)

    # Precompute weights between any query and the original sample via a function
    def draw_y_next_given(y_lag_q_std):
        # weights vs all observed y_lags
        w = kernelWeights(y_lag_q_std[None, :], y_lags_std, h).ravel()
        if not np.any(w):  # safety
            w = np.ones_like(w)
        w /= w.sum()
        # kernel-weighted resampling from observed y_next
        idx = rng.choice(T, size=1, replace=True, p=w)
        return y_next[idx][0]

    def draw_x_lags_given(y_lag_q_std):
        w = kernelWeights(y_lag_q_std[None, :], y_lags_std, h).ravel()
        if not np.any(w):
            w = np.ones_like(w)
        w /= w.sum()
        idx = rng.choice(T, size=1, replace=True, p=w)
        return x_lags[idx][0]  # returns (lag,)

    #null distribution to store the n_boot GC statistics
    gc_null = np.empty(n_boot, dtype=float)

    for b in range(n_boot):
        # 1) sample y_lags* from empirical (either KDE sampling or resample). We resample indices.
        idx_y = rng.integers(0, T, size=T)
        y_lags_star = y_lags[idx_y]                       # (T,lag)
        y_lags_star_std = scaler_y.transform(y_lags_star) # standardize into same space

        # 2) conditionally sample y_next* and x_lags* independently for each row
        y_next_star = np.array([draw_y_next_given(y_lags_star_std[t]) for t in range(T)])
        x_lags_star = np.vstack([draw_x_lags_given(y_lags_star_std[t]) for t in range(T)])

        # Step 3: Reconstruct full bootstrap sequences aligned in length
        y_star = np.concatenate([y[:lag], y_next_star])  # initial lag + resampled next
        x_star = np.concatenate([x[:lag], x_lags_star[:, 0]])  # initial lag + first component of x_lags_star

        # Step 4: Compute single GC value on bootstrap sample
        gc_null[b] = calcGC(
            x_star,
            y_star,
            lag=lag,
            m_bern=m_bern,
            h=None
        )

    return gc_null


"""
Runs the Copula Granger Causality on our data
"""
def runCGC(coin_data, source_coins, source_vars, target_coins, target_vars, max_lag=1, n_boot=200, m_bern=10, bw=None, seed=42):
    results = []
    for coin_x, var_x, coin_y, var_y in product(source_coins, source_vars, target_coins, target_vars):
        print(f"Processing {coin_x}-{var_x} -> {coin_y}-{var_y}")
        x = coin_data[coin_x][var_x].values
        y = coin_data[coin_y][var_y].values
        min_len = min(len(x), len(y))
        x, y = x[-min_len:], y[-min_len:]
        if len(x) > max_lag + 1:
            gc = calcGC(x, y, lag=max_lag, m_bern=m_bern, h=bw)
            null_dist = bootstrapGC(x, y, lag=max_lag, n_boot=n_boot, m_bern=m_bern, h=bw, random_state=seed)
            p_value = np.mean(null_dist >= gc)
            results.append({
                "source_coin": coin_x,
                "source_var": var_x,
                "target_coin": coin_y,
                "target_var": var_y,
                "lag": max_lag,
                "Copula GC": gc,
                "p-value": p_value,
                "null_dist": null_dist
            })
            print(f"GC={gc:.6f}, p-value={p_value:.4f}")
    return results


results = runCGC(
    coin_data, source_coins, source_vars, target_coins, target_vars,
    max_lag=max_lag, n_boot=n_boot, m_bern=10, bw=None, seed=123
)

results_df = pd.DataFrame([{k:v for k,v in r.items() if k != "null_dist"} for r in results])
results_df.to_csv("grangerCopulaRes.csv", index=False)

sig_results = results_df[results_df["p-value"] < 0.05]
sig_results.to_csv("grangerCopulaSig.csv", index=False)

null_df = pd.DataFrame([{
    "source_coin": r["source_coin"],
    "source_var": r["source_var"],
    "target_coin": r["target_coin"],
    "target_var": r["target_var"],
    "lag": r["lag"],
    "GC_test": r["Copula GC"],
    "null_dist": json.dumps(r["null_dist"].tolist())  # store as JSON string
} for r in results])

null_df.to_csv("grangerCopulaNulls.csv", index=False)