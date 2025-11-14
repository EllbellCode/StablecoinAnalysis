import pandas as pd
import numpy as np
from pathlib import Path
from arch import arch_model
from scipy.stats import t as t_dist, norm, chi2, multivariate_t
from scipy.optimize import minimize, brentq
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from statsmodels.distributions.empirical_distribution import ECDF
import matplotlib.pyplot as plt
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIGURATION
# ==============================================================================
DATA_DIR = Path("Data/Verified")
START_DATE = '2020-01-01'
BACKTEST_START_DATE = '2024-01-01'
END_DATE = '2025-01-01'

VAR_ALPHA = 0.05

# Model Settings
GARCH_P = 1
GARCH_Q = 1
# CHANGED to 't' for consistency with Copula mapping function
GARCH_DIST = 't' 
GARCH_VOL = 'EGARCH'

# Test Parameters
MA_WINDOWS = [1]

# TEST_PAIRS = [
#     ("PC1_Crypto_Volatility", "PC1_Stable_Volatility"),
#     ("PC1_Stable_Volatility", "PC1_Crypto_Volatility"),
#     ("PC1_Crypto_Returns", "PC1_Stable_Volatility"),
#     ("PC1_Stable_Volatility", "PC1_Crypto_Returns"),
#     ("PC1_Crypto_Volatility", "PC1_Stable_Volume"),
#     ("PC1_Stable_Volume", "PC1_Crypto_Volatility"),
#     ("PC1_Crypto_Returns", "PC1_Stable_Volume"),
#     ("PC1_Stable_Volume", "PC1_Crypto_Returns"),
# ]

TEST_PAIRS = [("PC1_Stable_Volatility", "PC1_Crypto_Volatility")]

# ==============================================================================
# CORE HELPER FUNCTIONS
# ==============================================================================
def kupiec_pof_test(actual, var_series, alpha):
    """Calculates Kupiec POF test statistics."""
    if alpha > 0.5: # Right tail (Spikes)
        breaches = actual > var_series
        p_exp = 1 - alpha
    else: # Left tail (Losses)
        breaches = actual < var_series
        p_exp = alpha

    T = len(breaches)
    N = breaches.sum()
    rate = N / T if T > 0 else np.nan
    
    if T == 0 or N == 0 or N == T:
         return {'N': T, 'Exceptions': N, 'Rate': rate, 'p_value': np.nan}

    lr_stat = -2 * ((T - N) * np.log(1 - p_exp) + N * np.log(p_exp) - 
                    ((T - N) * np.log(1 - rate) + N * np.log(rate)))
    return {'N': T, 'Exceptions': N, 'Rate': rate, 'p_value': 1 - chi2.cdf(lr_stat, 1)}

def plot_uniform_diagnostics(u_series, v_series, title_suffix=""):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.hist(u_series, bins=20, color='skyblue', edgecolor='k', density=True)
    plt.axhline(y=1.0, color='r', linestyle='--', label="Ideal")
    plt.title(f"Target (U) - {title_suffix}")
    
    plt.subplot(1, 2, 2)
    plt.hist(v_series, bins=20, color='lightgreen', edgecolor='k', density=True)
    plt.axhline(y=1.0, color='r', linestyle='--', label="Ideal")
    plt.title(f"Source (V_lag) - {title_suffix}")
    
    plt.tight_layout()
    print(f"[DIAGNOSTIC] Showing plot for {title_suffix}. Close window to continue...")
    plt.show()

# ==============================================================================
# COPULA FUNCTIONS
# ==============================================================================
def gaussian_copula_logpdf(u, v, rho):
    if not -0.999 < rho < 0.999: return -np.inf
    u = np.clip(u, 1e-6, 1-1e-6); v = np.clip(v, 1e-6, 1-1e-6)
    z_u = norm.ppf(u); z_v = norm.ppf(v)
    return -0.5 * np.log(1 - rho**2) - (rho**2 * (z_u**2 + z_v**2) - 2 * rho * z_u * z_v) / (2 * (1 - rho**2))

def t_copula_logpdf(u, v, rho, nu):
    if not -0.999 < rho < 0.999 or nu <= 2.001: return -np.inf
    u = np.clip(u, 1e-6, 1-1e-6); v = np.clip(v, 1e-6, 1-1e-6)
    t_u = t_dist.ppf(u, df=nu); t_v = t_dist.ppf(v, df=nu)
    data = np.column_stack([t_u, t_v])
    return multivariate_t.logpdf(data, shape=[[1, rho], [rho, 1]], df=nu) - t_dist.logpdf(t_u, df=nu) - t_dist.logpdf(t_v, df=nu)

def fit_best_copula(u_series, v_series):
    best_aic, best_params = np.inf, None
    u, v = u_series.values, v_series.values
    
    # Gaussian
    try:
        res_g = minimize(lambda p: -np.sum(gaussian_copula_logpdf(u, v, p[0])), [0.1], bounds=[(-0.99, 0.99)])
        if res_g.success and (aic_g := 2 + 2*res_g.fun) < best_aic:
            best_aic, best_params = aic_g, {'type': 'gaussian', 'rho': res_g.x[0]}
    except: pass

    # Student-t
    try:
        res_t = minimize(lambda p: -np.sum(t_copula_logpdf(u, v, p[0], p[1])), [0.1, 4.0], bounds=[(-0.99, 0.99), (2.01, 50)])
        if res_t.success and (aic_t := 4 + 2*res_t.fun) < best_aic:
             best_params = {'type': 'student-t', 'rho': res_t.x[0], 'nu': res_t.x[1]}
    except: pass
            
    return best_params

def get_conditional_quantile(v_val, copula_params, alpha):
    if not copula_params: return alpha
    try:
        if copula_params['type'] == 'gaussian':
            rho = copula_params['rho']; z_v = norm.ppf(v_val)
            func = lambda u: norm.cdf((norm.ppf(u) - rho*z_v)/np.sqrt(1-rho**2)) - alpha
        else:
            rho, nu = copula_params['rho'], copula_params['nu']; t_v = t_dist.ppf(v_val, df=nu)
            func = lambda u: t_dist.cdf((t_dist.ppf(u, df=nu) - rho*t_v)*np.sqrt((nu+1)/((nu+t_v**2)*(1-rho**2))), df=nu+1) - alpha
        return brentq(func, 1e-6, 1-1e-6)
    except: return alpha

def run_garch_benchmark(y_series):
    results = []
    dates = y_series.loc[BACKTEST_START_DATE:END_DATE].index
    print(f"  Running GARCH Benchmark on {len(dates)} days...")
    
    for t_date in tqdm(dates, desc="Benchmark"):
        train_y = y_series.loc[:t_date - pd.Timedelta(days=1)]
        try:
            # FIXED: Standard GARCH(1,1) with Constant Mean
            am = arch_model(train_y, vol=GARCH_VOL, p=GARCH_P, q=GARCH_Q, dist=GARCH_DIST, mean='AR', lags=1)
            res = am.fit(disp='off', show_warning=False, options={'ftol': 1e-3})
            
            fc = res.forecast(horizon=1, reindex=False)
            mu, var = fc.mean.iloc[0,0], fc.variance.iloc[0,0]
            nu = res.params.get('nu', np.inf)
            q = t_dist.ppf(VAR_ALPHA, df=nu)
            
            results.append({
                'date': t_date, 'actual': y_series.loc[t_date],
                'mu': mu, 'sigma': np.sqrt(var), 'nu': nu,
                'VaR_Bench': mu + np.sqrt(var) * q,
                'std_resid': res.std_resid
            })
        except Exception: continue
    return pd.DataFrame(results).set_index('date')

def run_copula_backtest(target_bench_res, source_series, ma_window):
    copula_forecasts = []
    common_dates = target_bench_res.index.intersection(source_series.index)
    test_dates = common_dates[common_dates >= BACKTEST_START_DATE]
    DIAGNOSTIC_SHOWN = False

    print(f"    -> Running Copula (MA{ma_window}) on {len(test_dates)} days...")
    # ADDED TQDM here for progress bar on Copula runs
    for t_date in tqdm(test_dates, desc=f"Copula MA{ma_window}", leave=False):
        bench_row = target_bench_res.loc[t_date]
        
        # Fit Source GARCH up to t-1 (Standard Constant Mean)
        train_x = source_series.loc[:t_date - pd.Timedelta(days=1)]
        try:
            am_x = arch_model(train_x, vol=GARCH_VOL, p=GARCH_P, q=GARCH_Q, dist=GARCH_DIST, mean='AR', lags=1)
            res_x = am_x.fit(disp='off', show_warning=False, options={'ftol': 1e-3})
            source_avg_resids = res_x.std_resid.rolling(window=ma_window).mean().dropna()
        except Exception: continue

        common_idx = bench_row['std_resid'].index.intersection(source_avg_resids.index)
        if len(common_idx) < 100: continue
        
        u_raw = bench_row['std_resid'].loc[common_idx]
        v_raw = source_avg_resids.loc[common_idx]

        # Transform to Uniform (ECDF)
        ecdf_u, ecdf_v = ECDF(u_raw), ECDF(v_raw)
        copula_data = pd.DataFrame({
            'u': ecdf_u(u_raw),
            'v': ecdf_v(v_raw)
        }, index=common_idx)
        
        # Lag: V(t-1) predicts U(t)
        copula_data['v_lag'] = copula_data['v'].shift(1)
        copula_data = copula_data.dropna()

        # if not DIAGNOSTIC_SHOWN:
        #      plot_uniform_diagnostics(copula_data['u'], copula_data['v_lag'], f"MA{ma_window} on {t_date.date()}")
        #      DIAGNOSTIC_SHOWN = True

        best_copula = fit_best_copula(copula_data['u'], copula_data['v_lag'])
        
        # Forecast
        if (t_date - pd.Timedelta(days=1)) in common_idx:
             # Find the rank of yesterday's actual averaged residual
             v_lag_val = ecdf_v(source_avg_resids.loc[t_date - pd.Timedelta(days=1)])
        else:
             v_lag_val = 0.5

        u_star = get_conditional_quantile(v_lag_val, best_copula, VAR_ALPHA)
        copula_var = bench_row['mu'] + bench_row['sigma'] * t_dist.ppf(u_star, df=bench_row['nu'])
        copula_forecasts.append({'date': t_date, 'VaR_Copula': copula_var})

    return pd.DataFrame(copula_forecasts).set_index('date')

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
print("--- Initializing Data & Factors ---")
coin_data = {}
for f in DATA_DIR.glob("*.csv"):
    try:
        df = pd.read_csv(f, parse_dates=['Date']).set_index('Date').sort_index()
        coin_data[f.stem.replace("Verif_", "")] = df[START_DATE:END_DATE]
    except: pass

def create_pca_factor(coins, var, data_dict, fn):
    dl = [data_dict[c][var] for c in coins if c in data_dict and var in data_dict[c].columns]
    if not dl: return pd.Series(dtype=float, name=fn)
    dm = pd.concat(dl, axis=1, keys=coins, join='inner').dropna()
    return pd.Series(PCA(1).fit_transform(StandardScaler().fit_transform(dm)).ravel(), index=dm.index, name=fn)

factors = {
    "PC1_Stable_Volume": create_pca_factor(["DAI","USDC","USDT"], "LogVolChange", coin_data, "PC1_Stable_Volume"),
    "PC1_Stable_Volatility": create_pca_factor(["DAI","USDC","USDT"], "Delta_LogRV", coin_data, "PC1_Stable_Volatility"),
    "PC1_Crypto_Returns": create_pca_factor(["BNB","BTC","ETH","XRP"], "Log Returns", coin_data, "PC1_Crypto_Returns"),
    "PC1_Crypto_Volatility": create_pca_factor(["BNB","BTC","ETH","XRP"], "Delta_LogRV", coin_data, "PC1_Crypto_Volatility")
}
F_DF = pd.concat(factors.values(), axis=1).dropna()

# --- MAIN LOOP ---
benchmark_cache = {}
all_summary_stats = []

print(f"\n--- Starting Copula Test Battery ({len(TEST_PAIRS)} Pairs) ---")

for source, target in TEST_PAIRS:
    print(f"\n[Target: {target}]")
    
    if target not in benchmark_cache:
        bench_res = run_garch_benchmark(F_DF[target])
        benchmark_cache[target] = bench_res
        # Save Benchmark Stats once
        if not bench_res.empty:
            b_stats = kupiec_pof_test(bench_res['actual'], bench_res['VaR_Bench'], VAR_ALPHA)
            b_stats.update({'Target': target, 'Source': 'None', 'Model': 'Benchmark', 'MA': '-'})
            all_summary_stats.append(b_stats)
    else:
        bench_res = benchmark_cache[target]

    print(f"  [Source: {source}]")
    for ma in MA_WINDOWS:
        copula_res = run_copula_backtest(bench_res, F_DF[source], ma)
        
        if not copula_res.empty:
            full = bench_res[['actual', 'VaR_Bench']].join(copula_res, how='inner')
            full.columns = ['Actual_Y', 'VaR_Benchmark', 'VaR_Challenger']
            
            # Directional Breaches for Plotting
            if VAR_ALPHA > 0.5:
                full['Breach_Benchmark'] = full['Actual_Y'] > full['VaR_Benchmark']
                full['Breach_Challenger'] = full['Actual_Y'] > full['VaR_Challenger']
            else:
                full['Breach_Benchmark'] = full['Actual_Y'] < full['VaR_Benchmark']
                full['Breach_Challenger'] = full['Actual_Y'] < full['VaR_Challenger']

            # Save Plot Data
            sd = Path("Results/GARCH/VaR/"); sd.mkdir(parents=True, exist_ok=True)
            full.to_csv(sd / f"Copula_VaR_backtest_{source}_to_{target}_{VAR_ALPHA}_MA{ma}.csv")
            
            # Calculate & Store Copula Stats
            c_stats = kupiec_pof_test(full['Actual_Y'], full['VaR_Challenger'], VAR_ALPHA)
            c_stats.update({'Target': target, 'Source': source, 'Model': 'Copula-GARCH', 'MA': ma})
            all_summary_stats.append(c_stats)

# --- FINAL SUMMARY ---
if all_summary_stats:
    summary_df = pd.DataFrame(all_summary_stats)
    cols = ['Target', 'Source', 'Model', 'MA', 'N', 'Exceptions', 'Rate', 'p_value']
    summary_df = summary_df[[c for c in cols if c in summary_df.columns]]
    
    out_path = Path("Results/GARCH/Copula_VaR_Summary.csv")
    summary_df.to_csv(out_path, index=False)
    print(f"\n=== FINAL SUMMARY (Saved to {out_path}) ===")
    print(summary_df.to_string(index=False))
else:
    print("\nNo results generated.")