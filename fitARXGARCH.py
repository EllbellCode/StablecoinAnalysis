import pandas as pd
import numpy as np
from pathlib import Path
from arch import arch_model
from scipy.stats import chi2
from arch.univariate import SkewStudent, StudentsT, Normal
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIGURATION
# ==============================================================================
DATA_DIR = Path("Data/Verified")
START_DATE = '2020-01-01'
BACKTEST_START_DATE = '2024-01-01'
END_DATE = '2025-01-01'

# VaR Setting (0.95 for 95% Confidence Interval)
# NOTE: Use >0.5 for Volatility (testing spikes), <0.5 for Returns (testing losses)
VAR_ALPHA = 0.95

# Model Settings
GARCH_P = 1
GARCH_Q = 1
GARCH_DIST = 'skewt'
GARCH_VOL = 'EGARCH'

# Test Parameters
MA_WINDOWS = [1, 7, 30]

# Define the 8 requested combinations (Source -> Target)
TEST_PAIRS = [
    # Crypto Volatility <-> Stable Volatility (2 tests)
    ("PC1_Crypto_Volatility", "PC1_Stable_Volatility"),
    ("PC1_Stable_Volatility", "PC1_Crypto_Volatility"),

    # Crypto Returns <-> Stable Volatility (2 tests)
    ("PC1_Crypto_Returns", "PC1_Stable_Volatility"),
    ("PC1_Stable_Volatility", "PC1_Crypto_Returns"),
    
    # Crypto Volatility <-> Stable Volume (2 tests)
    ("PC1_Crypto_Volatility", "PC1_Stable_Volume"),
    ("PC1_Stable_Volume", "PC1_Crypto_Volatility"),

    # Crypto Returns <-> Stable Volume (2 tests)
    ("PC1_Crypto_Returns", "PC1_Stable_Volume"),
    ("PC1_Stable_Volume", "PC1_Crypto_Returns"),
]

# ==============================================================================
# CORE FUNCTIONS
# ==============================================================================
def create_pca_factor(coins, var, data_dict, factor_name):
    df_list = [data_dict[c][var] for c in coins if c in data_dict and var in data_dict[c].columns]
    if not df_list: return pd.Series(dtype=float, name=factor_name)
    data_matrix = pd.concat(df_list, axis=1, keys=coins, join='inner').dropna()
    if data_matrix.empty: return pd.Series(dtype=float, name=factor_name)

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_matrix)
    pca = PCA(n_components=1)
    factor = pca.fit_transform(data_scaled)
    return pd.Series(factor.ravel(), index=data_matrix.index, name=factor_name)

def run_expanding_window_backtest(y, x_exog, garch_dist_name, alpha):
    forecasts = []
    backtest_dates = y.loc[BACKTEST_START_DATE:END_DATE].index
    
    if garch_dist_name == 't': dist_obj = StudentsT()
    elif garch_dist_name == 'skewt': dist_obj = SkewStudent()
    else: dist_obj = Normal()

    # Determine model type for logging
    is_challenger = x_exog is not None
    model_type = "Challenger (ARX)" if is_challenger else "Benchmark (GARCH)"
    print(f"  Running {model_type} backtest on {len(backtest_dates)} days...")

    for i, t_date in enumerate(backtest_dates):
        train_end = t_date - pd.DateOffset(days=1)
        curr_y = y.loc[START_DATE:train_end].values
        
        curr_x, x_val = None, None
        if is_challenger:
            curr_x = x_exog.loc[START_DATE:train_end].values
            x_val = np.array([x_exog.loc[t_date]]).reshape(1, 1)
            if len(curr_y) != len(curr_x): continue

        try:
            am = arch_model(curr_y, x=curr_x, mean='ARX' if is_challenger else 'Constant',
                            lags=0 if is_challenger else None, p=GARCH_P, q=GARCH_Q, 
                            dist=garch_dist_name, vol=GARCH_VOL)
            res = am.fit(update_freq=0, disp='off', options={'maxiter': 200})
            
            fc = res.forecast(x=x_val, horizon=1, reindex=False)
            mu, var = fc.mean.iloc[0, 0], fc.variance.iloc[0, 0]

            # Robust quantile
            d_name = res.model.distribution.name.lower()
            if 'skew' in d_name: q = dist_obj.ppf(alpha, [res.params['eta'], res.params['lambda']])
            elif 't' in d_name: q = dist_obj.ppf(alpha, [res.params['nu']])
            else: q = dist_obj.ppf(alpha)

            forecasts.append({
                'date': t_date, 'actual': y.loc[t_date], 
                'VaR': mu + np.sqrt(var) * q
            })
        except: continue

    return pd.DataFrame(forecasts).set_index('date') if forecasts else pd.DataFrame()

def kupiec_pof_test(actual, var_series, alpha):
    # Auto-detect tail: if alpha > 0.5 we are looking for right-tail spikes
    if alpha > 0.5:
        breaches = actual > var_series
        p_exp = 1 - alpha
    else:
        breaches = actual < var_series
        p_exp = alpha

    T, N = len(breaches), breaches.sum()
    if T == 0: return {'N': 0, 'Rate': np.nan, 'p_value': np.nan}
    
    pi_hat = N / T
    if N == 0 or N == T: return {'N': T, 'Exceptions': N, 'Rate': pi_hat, 'p_value': np.nan}

    lr_stat = -2 * ((T - N) * np.log(1 - p_exp) + N * np.log(p_exp) - 
                    ((T - N) * np.log(1 - pi_hat) + N * np.log(pi_hat)))
    return {'N': T, 'Exceptions': N, 'Rate': pi_hat, 'LR_Stat': lr_stat, 
            'p_value': 1 - chi2.cdf(lr_stat, 1)}

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
# 1. LOAD DATA & CREATE FACTORS
print("--- Initializing Data & Factors ---")
coin_data = {}
for f in DATA_DIR.glob("*.csv"):
    try:
        df = pd.read_csv(f, parse_dates=['Date']).set_index('Date').sort_index()
        coin_data[f.stem.replace("Verif_", "")] = df[START_DATE:END_DATE]
    except: pass

factors = {}
# Note: "Delta_LogRV" is assumed to be your stationary volatility measure based on previous scripts
factors["PC1_Stable_Volume"] = create_pca_factor(["DAI","USDC","USDT"], "LogVolChange", coin_data, "PC1_Stable_Volume")
factors["PC1_Stable_Volatility"] = create_pca_factor(["DAI","USDC","USDT"], "Delta_LogRV", coin_data, "PC1_Stable_Volatility")
factors["PC1_Crypto_Returns"] = create_pca_factor(["BNB","BTC","ETH","XRP"], "Log Returns", coin_data, "PC1_Crypto_Returns")
factors["PC1_Crypto_Volatility"] = create_pca_factor(["BNB","BTC","ETH","XRP"], "Delta_LogRV", coin_data, "PC1_Crypto_Volatility")
F_DF = pd.concat(factors.values(), axis=1).dropna()

# 2. MAIN LOOP
all_summary_stats = []
benchmark_cache = {} # Cache to avoid re-running same benchmark multiple times

print(f"\n--- Starting Test Battery: {len(TEST_PAIRS)} Pairs x {len(MA_WINDOWS)} MAs ---")

for source, target in TEST_PAIRS:
    if source not in F_DF.columns or target not in F_DF.columns:
        print(f"Skipping {source}->{target}: Factors missing.")
        continue
        
    print(f"\n[Target: {target}]")
    
    # A. Run Benchmark (only once per target)
    if target not in benchmark_cache:
        print(f"  New target. Running Benchmark GARCH...")
        bench_res = run_expanding_window_backtest(F_DF[target], None, GARCH_DIST, VAR_ALPHA)
        benchmark_cache[target] = bench_res
        
        # Save benchmark stats
        if not bench_res.empty:
            b_stats = kupiec_pof_test(bench_res['actual'], bench_res['VaR'], VAR_ALPHA)
            b_stats.update({'Target': target, 'Source': 'None (Benchmark)', 'Model': 'GARCH', 'MA': np.nan})
            all_summary_stats.append(b_stats)
    else:
        bench_res = benchmark_cache[target]

    # B. Run Challengers (combinations of Source + MA)
    print(f"  [Source: {source}]")
    for ma in MA_WINDOWS:
        print(f"    -> Testing MA({ma})...")
        # Create lagged exogenous feature
        x_ma = F_DF[source].rolling(window=ma).mean().shift(1)
        data = pd.concat([F_DF[target], x_ma], axis=1, keys=['y', 'x']).dropna()
        
        if data.empty: continue
            
        chall_res = run_expanding_window_backtest(data['y'], data['x'], GARCH_DIST, VAR_ALPHA)
        
        if not chall_res.empty:
            # 1. Save Stats
            c_stats = kupiec_pof_test(chall_res['actual'], chall_res['VaR'], VAR_ALPHA)
            c_stats.update({'Target': target, 'Source': source, 'Model': 'ARX-GARCH', 'MA': ma})
            all_summary_stats.append(c_stats)
            
            # 2. Save Plot Data
            plot_df = pd.DataFrame(index=bench_res.index)
            plot_df['Actual_Y'] = bench_res['actual']
            plot_df['VaR_Benchmark'] = bench_res['VaR']
            plot_df['VaR_Challenger'] = chall_res['VaR'] # Aligns by index automatically
            
            # Define breaches based on tail direction
            if VAR_ALPHA > 0.5: # Right tail (Spikes)
                plot_df['Breach_Benchmark'] = plot_df['Actual_Y'] > plot_df['VaR_Benchmark']
                plot_df['Breach_Challenger'] = plot_df['Actual_Y'] > plot_df['VaR_Challenger']
            else: # Left tail (Losses)
                plot_df['Breach_Benchmark'] = plot_df['Actual_Y'] < plot_df['VaR_Benchmark']
                plot_df['Breach_Challenger'] = plot_df['Actual_Y'] < plot_df['VaR_Challenger']

            # Save file for plotter
            save_dir = Path("Results/GARCH/VaR/")
            save_dir.mkdir(parents=True, exist_ok=True)
            fname = f"ARX_VaR_backtest_{source}_to_{target}_{VAR_ALPHA}_MA{ma}.csv"
            plot_df.dropna().to_csv(save_dir / fname)

# 3. FINAL SUMMARY
if all_summary_stats:
    summary_df = pd.DataFrame(all_summary_stats)
    # Reorder columns for readability
    cols = ['Target', 'Source', 'Model', 'MA', 'N', 'Exceptions', 'Rate', 'p_value']
    summary_df = summary_df[[c for c in cols if c in summary_df.columns]]
    
    out_path = Path("Results/GARCH/ARX_VaR_Summary_Combinations.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out_path, index=False)
    print(f"\nValues saved to {out_path}")
    print(summary_df.to_string(index=False))
else:
    print("\nNo results generated.")