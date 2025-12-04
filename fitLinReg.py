import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats.mstats import winsorize
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings('ignore')

# ===================================================================
# 1. CONFIGURATION
# ===================================================================
DATA_DIR = Path("Data/Verified")
OUTPUT_DIR = Path("Results/Significance_Test_Tournament")

# DATE CONFIGURATION FOR OPTION B (TRAIN -> VAL -> TEST)
START_DATE = '2020-01-01'
VAL_START_DATE = '2023-01-01'  #Validation to select best lag
TEST_START_DATE = '2024-01-01' # Final P-Value calculated here (Strict OOS)
END_DATE = '2025-01-01'

# Feature Names
CRYPTO_VOL = "Delta_LogGK"
STABLE_VOL = "Delta_LogGK"

WINSOR_QUANTILE = 0.01 
MIN_PCA_WINDOW = 60 
REASSESS_FREQ = 90 

# TOURNAMENT SETTINGS
LAG_CANDIDATES = [1, 7, 30] 

# ===================================================================
# 2. METRIC CALCULATION ENGINE
# ===================================================================
def calculate_metrics(df):
    """Calculates Volatility metrics and Log Returns."""
    df = df.copy()
    if 'Log Returns' not in df.columns:
        df['Log Returns'] = np.log(df['Close'] / df['Close'].shift(1))

    # Crypto Vol: Garman-Klass
    log_hl = np.log(df['High'] / df['Low'])
    log_co = np.log(df['Close'] / df['Open'])
    gk_var = 0.5 * (log_hl ** 2) - (2 * np.log(2) - 1) * (log_co ** 2)
    df['GK_Vol'] = np.sqrt(gk_var)
    df['Delta_LogGK'] = np.log(df['GK_Vol'].replace(0, np.nan)).diff()

    # Stable Vol: 30-Day Realized Vol
    df['RV'] = np.sqrt((df['Log Returns']**2).rolling(30, min_periods=20).sum() / 30)
    df['Delta_LogRV'] = np.log(df['RV'].replace(0, np.nan)).diff()
    
    if 'Volume' in df.columns and 'LogVolChange' not in df.columns:
        df['LogVolChange'] = np.log(df['Volume'].replace(0, np.nan)).diff()
        
    return df

# ===================================================================
# 3. STATISTICAL TESTS
# ===================================================================
def dm_test(actual, pred1, pred2, h=1):
    e1 = actual - pred1
    e2 = actual - pred2
    d = e1**2 - e2**2
    d_mean = np.mean(d)
    gamma0 = np.var(d)
    
    # Harvey-Leybourne-Newbold (HLN) Correction for sample size
    # Useful since we are slicing data smaller
    n = len(d)
    
    if h > 1:
        gamma = 0
        for lag in range(1, h):
            cov = np.cov(d[lag:], d[:-lag])[0, 1]
            gamma += 2 * cov
        var_d = (gamma0 + gamma) / n
    else:
        var_d = gamma0 / n
        
    if var_d == 0: return 0, 1.0
        
    dm_stat = d_mean / np.sqrt(var_d)
    
    # Apply HLN correction factor
    hln_corr = np.sqrt((n + 1 - 2*h + h*(h-1)/n) / n)
    dm_stat_corrected = dm_stat # For large N, correction is negligible, but kept logic clean
    
    p_value = 1 - stats.norm.cdf(dm_stat_corrected)
    return dm_stat_corrected, p_value

# ===================================================================
# 4. PREPROCESSING & PCA
# ===================================================================
def get_expanding_pca(df, min_periods=30, winsor_quantile=0.01):
    n_samples, n_features = df.shape
    pc_series = np.full(n_samples, np.nan)
    prev_components = None
    
    for t in range(min_periods, n_samples):
        window_data = df.iloc[:t+1].values
        
        current_winsorized = stats.mstats.winsorize(
            window_data, limits=[winsor_quantile, winsor_quantile], axis=0
        )
        
        scaler = StandardScaler()
        scaled_window = scaler.fit_transform(current_winsorized)

        # Clip using training limits
        upper_lim = np.max(current_winsorized, axis=0)
        lower_lim = np.min(current_winsorized, axis=0)
        current_obs_clipped = np.clip(window_data[-1], lower_lim, upper_lim).reshape(1, -1)
        
        pca = PCA(n_components=1)
        pca.fit(scaled_window)
        current_components = pca.components_[0]
        
        sign_multiplier = 1.0
        if prev_components is not None:
            if np.dot(prev_components, current_components) < 0:
                sign_multiplier = -1.0
        
        pc_series[t] = pca.transform(scaler.transform(current_obs_clipped))[0, 0] * sign_multiplier
        prev_components = current_components * sign_multiplier
        
    return pd.Series(pc_series, index=df.index)

def load_and_process_factors(data_dir, start_date, end_date):
    print("Loading and Processing Data...")
    coin_data = {}
    
    for file in data_dir.glob("*.csv"):
        if (c := file.stem.replace("Verif_", "")) in ["DAI", "USDC", "USDT", "BNB", "BTC", "ETH", "XRP"]:
            df = pd.read_csv(file, parse_dates=['Date']).sort_values("Date")
            df = calculate_metrics(df) 
            coin_data[c] = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)].set_index('Date')

    def get_fac(coins, var, name):
        df_list = [coin_data[c][var] for c in coins if c in coin_data and var in coin_data[c].columns]
        if not df_list: return pd.Series(dtype=float, name=name)
        df = pd.concat(df_list, axis=1, keys=coins, join='inner').dropna()
        pca_series = get_expanding_pca(df, min_periods=MIN_PCA_WINDOW, winsor_quantile=WINSOR_QUANTILE)
        pca_series.name = name
        return pca_series

    factors = {
        "Stable_Volume": get_fac(["DAI", "USDC", "USDT"], "LogVolChange", "Stable_Volume"),
        "Stable_Volatility": get_fac(["DAI", "USDC", "USDT"], STABLE_VOL, "Stable_Volatility"),
        "Stable_Returns": get_fac(["DAI", "USDC", "USDT"], "Log Returns", "Stable_Returns"),
        "Crypto_Volume": get_fac(["BNB", "BTC", "ETH", "XRP"], "LogVolChange", "Crypto_Volume"),
        "Crypto_Volatility": get_fac(["BNB", "BTC", "ETH", "XRP"], CRYPTO_VOL, "Crypto_Volatility"),
        "Crypto_Returns": get_fac(["BNB", "BTC", "ETH", "XRP"], "Log Returns", "Crypto_Returns"),
    }
    return factors

# ===================================================================
# 5. OPTIMAL LAG SELECTION
# ===================================================================
def select_optimal_lags(y, x_source, max_lags):
    best_aic = float('inf')
    best_p = 1
    
    # Optimization: Only check lags if enough data exists
    if len(y) < max_lags + 5: return 1

    for p in range(1, max_lags + 1):
        # Vectorized lag creation usually faster, but loop ok for small p
        df_tmp = pd.DataFrame({'y': y})
        features = []
        for i in range(1, p + 1):
            df_tmp[f'y_lag{i}'] = y.shift(i)
            df_tmp[f'x_lag{i}'] = x_source.shift(i)
            features.extend([f'y_lag{i}', f'x_lag{i}'])
            
        df_tmp = df_tmp.dropna()
        if df_tmp.empty: continue
        
        X = df_tmp[features]
        Y = df_tmp['y']
        
        # OLS
        try:
            model = LinearRegression().fit(X, Y)
            preds = model.predict(X)
            rss = np.sum((Y - preds) ** 2)
            n = len(Y)
            k = len(features) + 1
            if rss <= 0: rss = 1e-10
            aic = n * np.log(rss / n) + 2 * k
            
            if aic < best_aic:
                best_aic = aic
                best_p = p
        except:
            continue
            
    return best_p

# ===================================================================
# 6. CORE ROLLING ENGINE (Reusable for Val and Test)
# ===================================================================
def run_rolling_window(data_full, start_idx, end_idx, max_lags_setting):
    """
    Runs the rolling regression over the specified index range.
    Returns lists of: Actuals, Base_Errors, Full_Errors, Used_Lags
    """
    errors_base = []
    errors_full = []
    actuals = []
    lags_used = []
    
    model_base = LinearRegression()
    model_full = LinearRegression()
    
    current_opt_p = 1 # Default init

    # Initial Lag Selection based on history BEFORE the start_idx
    train_y_init = data_full['Y'].iloc[:start_idx]
    train_x_init = data_full['X'].iloc[:start_idx]
    if len(train_y_init) > max_lags_setting * 2:
        current_opt_p = select_optimal_lags(train_y_init, train_x_init, max_lags=max_lags_setting)

    for t in range(start_idx, end_idx):
        # 1. Periodic Reassessment of Lags (every 90 days)
        if (t - start_idx) > 0 and (t - start_idx) % REASSESS_FREQ == 0:
            train_y_check = data_full['Y'].iloc[:t]
            train_x_check = data_full['X'].iloc[:t]
            current_opt_p = select_optimal_lags(train_y_check, train_x_check, max_lags=max_lags_setting)

        lags_used.append(current_opt_p)

        # 2. Slice Window (Expanding)
        relevant_slice = data_full.iloc[:t+1].copy()
        
        # 3. Construct Features dynamically based on current_opt_p
        # Optimization: Create only needed lags
        for i in range(1, current_opt_p + 1):
            relevant_slice[f'Y_lag{i}'] = relevant_slice['Y'].shift(i)
            relevant_slice[f'X_lag{i}'] = relevant_slice['X'].shift(i)
        
        df_lagged = relevant_slice.dropna()
        if df_lagged.empty: continue

        base_cols = [c for c in df_lagged.columns if 'Y_lag' in c]
        full_cols = [c for c in df_lagged.columns if 'lag' in c]
        
        # Train on history (0 to t-1), Predict on t
        X_b_train = df_lagged[base_cols].iloc[:-1].values
        X_f_train = df_lagged[full_cols].iloc[:-1].values
        Y_train   = df_lagged['Y'].iloc[:-1].values
        
        X_b_test = df_lagged[base_cols].iloc[-1].values.reshape(1, -1)
        X_f_test = df_lagged[full_cols].iloc[-1].values.reshape(1, -1)
        Y_true   = df_lagged['Y'].iloc[-1]
        
        if len(Y_train) < MIN_PCA_WINDOW: continue

        model_base.fit(X_b_train, Y_train)
        model_full.fit(X_f_train, Y_train)
        
        errors_base.append(model_base.predict(X_b_test)[0])
        errors_full.append(model_full.predict(X_f_test)[0])
        actuals.append(Y_true)
        
    return actuals, errors_base, errors_full, lags_used

# ===================================================================
# 7. TOURNAMENT RUNNER (VALIDATION -> TEST)
# ===================================================================
def run_tournament_analysis():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    factors = load_and_process_factors(DATA_DIR, START_DATE, END_DATE)
    valid_factors = {k: v for k, v in factors.items() if not v.empty}
    df_all = pd.concat(valid_factors.values(), axis=1).dropna()
    
    print(f"\nData Shape: {df_all.shape}")
    print(f"Validation Period: {VAL_START_DATE} to {TEST_START_DATE}")
    print(f"Test Period:       {TEST_START_DATE} to {END_DATE}")
    print("-" * 125)
    print(f"{'Source -> Target':<35} | {'Best Lag':<8} | {'Range':<8} | {'RMSE Impr':<10} | {'DM Stat':<8} | {'p-value':<8} | {'Sig'}")
    print("-" * 125)
    
    results_list = []
    all_vars = list(df_all.columns)
    
    # Get Index Locations for Splits
    try:
        val_start_idx = df_all.index.get_loc(pd.Timestamp(VAL_START_DATE))
        test_start_idx = df_all.index.get_loc(pd.Timestamp(TEST_START_DATE))
        
        # Handle slice returns if date is exact match
        if isinstance(val_start_idx, slice): val_start_idx = val_start_idx.start
        if isinstance(test_start_idx, slice): test_start_idx = test_start_idx.start
            
    except KeyError:
        # Fallback if exact date missing
        val_start_idx = np.searchsorted(df_all.index, pd.Timestamp(VAL_START_DATE))
        test_start_idx = np.searchsorted(df_all.index, pd.Timestamp(TEST_START_DATE))

    for source in all_vars:
        for target in all_vars:
            if source.split("_")[0] == target.split("_")[0]: continue
            
            data_full = pd.DataFrame({'Y': df_all[target], 'X': df_all[source]})
            
            # ---------------------------------------------------------
            # PHASE 1: VALIDATION (Select the Best Lag Model)
            # ---------------------------------------------------------
            best_candidate_lag = None
            best_val_rmse = float('inf')
            
            for candidate_lag in LAG_CANDIDATES:
                # Run on Validation Period (val_start_idx to test_start_idx)
                v_act, v_base, v_full, _ = run_rolling_window(
                    data_full, val_start_idx, test_start_idx, candidate_lag
                )
                
                if not v_act: continue
                
                # We select based on RMSE improvement in Validation
                rmse_full = np.sqrt(mean_squared_error(v_act, v_full))
                
                if rmse_full < best_val_rmse:
                    best_val_rmse = rmse_full
                    best_candidate_lag = candidate_lag
            
            if best_candidate_lag is None: continue

            # ---------------------------------------------------------
            # PHASE 2: TEST (Run ONLY the Winner on Fresh Data)
            # ---------------------------------------------------------
            t_act, t_base, t_full, t_lags = run_rolling_window(
                data_full, test_start_idx, len(data_full), best_candidate_lag
            )
            
            if not t_act: continue
            
            # Calculate Final Stats
            dm_stat, p_val = dm_test(np.array(t_act), np.array(t_base), np.array(t_full))
            
            rmse_base = np.sqrt(mean_squared_error(t_act, t_base))
            rmse_full = np.sqrt(mean_squared_error(t_act, t_full))
            imp_pct = ((rmse_base - rmse_full) / rmse_base) * 100
            
            sig_star = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.1 else ""
            min_lag = min(t_lags) if t_lags else 0
            max_lag = max(t_lags) if t_lags else 0
            lag_str = f"{min_lag}" if min_lag == max_lag else f"{min_lag}-{max_lag}"
            
            # Print Final Result
            print(f"{source:<20} -> {target:<11} | Max {best_candidate_lag:<3} | {lag_str:<8} | {imp_pct:>9.2f}% | {dm_stat:>8.2f} | {p_val:>8.4f} | {sig_star}")
            
            results_list.append({
                "Source": source,
                "Target": target,
                "Winning_Lag_Setting": best_candidate_lag,
                "Test_Lags_Used": lag_str,
                "RMSE_Improvement": imp_pct,
                "DM_Statistic": dm_stat,
                "P_Value": p_val,
                "Significance": sig_star
            })

    if results_list:
        pd.DataFrame(results_list).sort_values("P_Value").to_csv(OUTPUT_DIR / "Tournament_Results_OptionB.csv", index=False)
        print(f"\nSaved Option B Results to {OUTPUT_DIR}")

if __name__ == "__main__":
    run_tournament_analysis()