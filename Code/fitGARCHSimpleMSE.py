import pandas as pd
import numpy as np
from pathlib import Path
from arch import arch_model
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
from tqdm import tqdm
from scipy import stats

warnings.filterwarnings('ignore')

# --- Configuration ---
DATA_DIR = Path("Data/Verified")
INPUT_RESULTS_DIR = Path("StableCryptoResults/MSE")
OUTPUT_DIR = Path("Results/MSE_EGARCH_Baseline")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

START_DATE = '2020-01-01'
TRAIN_END_DATE = '2024-01-01'
FULL_END_DATE = '2025-01-01'
MAX_ARMA_ORDER = 1 
MAX_GARCH_ORDER = 1 
GARCH_DIST = 'skewt'
GARCH_MODEL = 'EGARCH' 
WINDOW_SIZE = 9999 
WIN_LIMITS = [0.01, 0.01]
VOLATILITY = "RS"

# Define the single pair to test
SOURCE_FACTOR = "Stable_Downside"
TARGET_FACTOR = "Crypto_Downside"

stablecoins = ["DAI", "USDC", "USDT"]
cryptos = ["BNB", "BTC", "ETH", "XRP"]

# --- Statistical Tests ---
def diebold_mariano_test(real, pred_bench, pred_chall, h=1, metric='mse', nested=False):
    T = len(real)
    e1 = (real - pred_bench)
    e2 = (real - pred_chall)

    if metric == 'mse':
        if nested:
            d = e1**2 - (e2**2 - (pred_bench - pred_chall)**2) # Clark-West
        else:
            d = e1**2 - e2**2
    elif metric == 'direction':
        L1 = (np.sign(real) != np.sign(pred_bench)).astype(int)
        L2 = (np.sign(real) != np.sign(pred_chall)).astype(int)
        d = L1 - L2
    
    mean_d = np.mean(d)
    
    def autocovariance(xi, k):
        if k == 0: return np.var(xi)
        return np.mean((xi[k:] - np.mean(xi)) * (xi[:-k] - np.mean(xi)))

    gamma = [autocovariance(d, i) for i in range(h)]
    v_d = gamma[0] + 2 * sum(gamma[1:])
    
    if v_d <= 0: return 0.0, 1.0
    
    dm_stat = mean_d / np.sqrt(v_d / T)
    
    # HLN Modification
    hln_multiplier = np.sqrt((T + 1 - 2*h + (h*(h-1)/T)) / T)
    dm_stat_hln = hln_multiplier * dm_stat
    
    p_value = 2 * (1 - stats.t.cdf(abs(dm_stat_hln), df=T-1))
    
    return dm_stat_hln, p_value

def pesaran_timmermann_test(real, pred):
    real = np.array(real)
    pred = np.array(pred)
    
    y = (real > 0).astype(int)
    y_hat = (pred > 0).astype(int)
    
    T = len(y)
    P = np.mean(y == y_hat) 
    py = np.mean(y)     
    py_hat = np.mean(y_hat) 
    P_star = py * py_hat + (1 - py) * (1 - py_hat)
    var_P = (P_star * (1 - P_star)) / T
    
    if var_P == 0:
        return np.nan, np.nan, P
        
    pt_stat = (P - P_star) / np.sqrt(var_P)
    p_value = 2 * (1 - stats.norm.cdf(abs(pt_stat)))
    
    return pt_stat, p_value, P

# --- Econometric Models ---
def select_best_arma(series, max_order=MAX_ARMA_ORDER):
    best_aic = np.inf
    best_order = (0, 0)
    series = series.dropna()

    if series.empty: return best_order

    for p in range(max_order + 1):
        for q in range(max_order + 1):
            if p == 0 and q == 0: 
                continue
            try:
                model = ARIMA(series, order=(p, 0, q)).fit(method_kwargs={"warn_convergence": False})
                if model.aic < best_aic:
                    best_aic = model.aic
                    best_order = (p, q)
            except Exception:
                continue
    return best_order

def fit_best_garch(series, p, q, mean_p, mean_q, dist=GARCH_DIST):
    series = series.dropna()
    if series.empty: return None
    
    mean_model = 'Constant' if mean_p == 0 else 'AR'
    ar_lags = None if mean_p == 0 else mean_p

    try:
        am = arch_model(series, vol=GARCH_MODEL, p=p, q=q,
                        mean=mean_model, lags=ar_lags, dist=dist)
        res = am.fit(update_freq=0, disp='off', options={'maxiter': 200})
        return res
    except Exception:
        return None

# --- PCA Generation ---
def calculate_pca_for_window(coins, var, data_dict, factor_name, current_date):
    current_date = pd.to_datetime(current_date)
    yesterday = current_date - pd.Timedelta(days=1)
    
    if WINDOW_SIZE < 100000:
        window_start_date = current_date - pd.Timedelta(days=WINDOW_SIZE)
        effective_start = max(window_start_date, pd.to_datetime(START_DATE))
    else:
        effective_start = pd.to_datetime(START_DATE)
    
    df_list = [data_dict[coin][var] for coin in coins if coin in data_dict and var in data_dict[coin].columns]
    if not df_list: return pd.Series(dtype=float, name=factor_name)
    
    raw_df = pd.concat(df_list, axis=1, keys=coins, join='inner').dropna()
    train_data = raw_df.loc[effective_start:yesterday]
    test_data = raw_df.loc[[current_date]]
    
    if train_data.empty: return pd.Series(dtype=float, name=factor_name)

    lower_limit = train_data.quantile(WIN_LIMITS[0])
    upper_limit = train_data.quantile(1 - WIN_LIMITS[1])
    
    train_data = train_data.clip(lower=lower_limit, upper=upper_limit, axis=1)
    test_data = test_data.clip(lower=lower_limit, upper=upper_limit, axis=1)

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_data) 
    test_scaled = scaler.transform(test_data) if not test_data.empty else np.empty((0, train_data.shape[1]))

    pca = PCA(n_components=1)
    train_factor = pca.fit_transform(train_scaled)
    
    loadings = pca.components_[0]
    if np.sum(loadings) < 0:
        loadings = -loadings
        train_factor = -train_factor
        pca.components_[0] = loadings
    
    test_factor = pca.transform(test_scaled) if not test_data.empty else np.array([])
    full_index = train_data.index.union(test_data.index)
    full_values = np.concatenate([train_factor.ravel(), test_factor.ravel()])
    
    return pd.Series(full_values, index=full_index, name=factor_name)

def generate_factors_window(data_dict, current_end_date):
    f = {}
    f["Stable_Volume"] = calculate_pca_for_window(stablecoins, "LogVolChange", data_dict, "PC1_Stable_Volume", current_end_date)
    f["Stable_Volatility"] = calculate_pca_for_window(stablecoins, VOLATILITY, data_dict, "PC1_Stable_Volatility", current_end_date)
    f["Stable_Upside"] = calculate_pca_for_window(cryptos, "Upside_Vol", data_dict, "PC1_Stable_Upside", current_end_date)
    f["Stable_Downside"] = calculate_pca_for_window(cryptos, "Downside_Vol", data_dict, "PC1_Stable_Downside", current_end_date)
    f["Stable_Returns"] = calculate_pca_for_window(stablecoins, "Log Returns", data_dict, "PC1_Stable_Returns", current_end_date)
    f["Crypto_Returns"] = calculate_pca_for_window(cryptos, "Log Returns", data_dict, "PC1_Crypto_Returns", current_end_date)
    f["Crypto_Volatility"] = calculate_pca_for_window(cryptos, VOLATILITY, data_dict, "PC1_Crypto_Volatility", current_end_date)
    f["Crypto_Volume"] = calculate_pca_for_window(cryptos, "LogVolChange", data_dict, "PC1_Crypto_Volume", current_end_date)
    f["Crypto_Upside"] = calculate_pca_for_window(cryptos, "Upside_Vol", data_dict, "PC1_Crypto_Upside", current_end_date)
    f["Crypto_Downside"] = calculate_pca_for_window(cryptos, "Downside_Vol", data_dict, "PC1_Crypto_Downside", current_end_date)
    return f

# --- Main Execution ---
def run_egarch_baseline():
    print("Loading and preparing raw verified data...")
    coin_data = {}
    all_coins = stablecoins + cryptos
    for file in DATA_DIR.glob("*.csv"):
        coin_name = file.stem.replace("Verif_", "")
        if coin_name in all_coins:
            df = pd.read_csv(file, parse_dates=['Date']).sort_values("Date")
            df = df[(df['Date'] >= START_DATE) & (df['Date'] <= FULL_END_DATE)]
            coin_data[coin_name] = df.set_index('Date')

    # Get initial ARMA orders for the target factor to freeze them, matching original script logic
    print("Calculating Initial PCA Factors for ARMA Order Selection...")
    initial_factors = generate_factors_window(coin_data, TRAIN_END_DATE)
    y_series_init = initial_factors[TARGET_FACTOR].dropna()
    fixed_arma_order_y = select_best_arma(y_series_init, max_order=MAX_ARMA_ORDER)
    print(f"Fixed ARMA Order for Target: {fixed_arma_order_y}")

    full_dates = coin_data[cryptos[0]].index
    test_dates = full_dates[(full_dates > TRAIN_END_DATE) & (full_dates <= FULL_END_DATE)]

    print(f"\nRunning Pure EGARCH Backtest for {len(test_dates)} out-of-sample points...")
    oos_predictions_egarch = []

    for current_date in tqdm(test_dates):
        yesterday = current_date - pd.Timedelta(days=1)
        
        current_factors_full = generate_factors_window(coin_data, current_date)
        y_series_now = current_factors_full[TARGET_FACTOR].dropna()
        
        if current_date not in y_series_now.index: continue
        actual_y_value = y_series_now.loc[current_date]
        
        train_y = y_series_now.loc[:yesterday]
        if train_y.empty: continue

        # Fit pure EGARCH
        model_y = fit_best_garch(train_y, p=MAX_GARCH_ORDER, q=MAX_GARCH_ORDER,
                                 mean_p=fixed_arma_order_y[0], mean_q=fixed_arma_order_y[1], dist=GARCH_DIST)
        
        if model_y is None: continue

        # Forecast the conditional mean for t
        forecast = model_y.forecast(horizon=1, reindex=False)
        mu_next = forecast.mean.iloc[0, 0]
        
        oos_predictions_egarch.append({
            'Date': current_date,
            'Actual': actual_y_value,
            'Pred_EGARCH': mu_next
        })

    df_egarch_preds = pd.DataFrame(oos_predictions_egarch).set_index('Date')
    
    # --- Load Advanced Framework Results ---
    results_filename = f"OOS_Res_GARCH_{SOURCE_FACTOR}_to_{TARGET_FACTOR}.csv" 
    results_path = INPUT_RESULTS_DIR / results_filename
    
    if not results_path.exists():
        print(f"\nWarning: Advanced result file {results_path} not found.")
        return
        
    df_res = pd.read_csv(results_path, index_col='Date', parse_dates=True)
    
    
    # --- Align and Evaluate ---
    # The join adds suffixes if 'Actual' exists in both dfs
    df_eval = df_res.join(df_egarch_preds, how='inner', lsuffix='_chall', rsuffix='_egarch')
    
    # Use the suffixed names created by the join
    actual = df_eval['Actual_chall'] # Or 'Actual_egarch', they are the same
    pred_chall = df_eval['Pred_Chall']
    pred_egarch = df_eval['Pred_EGARCH']
    
    mse_egarch = mean_squared_error(actual, pred_egarch)
    mse_chall = mean_squared_error(actual, pred_chall)
    mse_red_pct = ((mse_egarch - mse_chall) / mse_egarch) * 100
    
    dm_stat, dm_p_val = diebold_mariano_test(actual, pred_egarch, pred_chall, metric='mse', nested=False)
    pt_stat_e, pt_p_e, acc_e = pesaran_timmermann_test(actual, pred_egarch)
    pt_stat_c, pt_p_c, acc_c = pesaran_timmermann_test(actual, pred_chall)
    dm_stat_dir, dm_p_dir = diebold_mariano_test(actual, pred_egarch, pred_chall, metric='direction', nested=False)
    
    # --- Output Summary ---
    summary_df = pd.DataFrame([{
        'Source_Factor': SOURCE_FACTOR,
        'Target_Factor': TARGET_FACTOR,
        'OOS_Days': len(df_eval),
        'MSE_Pure_EGARCH': mse_egarch,
        'MSE_Challenger': mse_chall,
        'MSE_Reduction_Pct': f"{mse_red_pct:.2f}%",
        'DM_MSE_Stat_vs_EGARCH': dm_stat,
        'DM_MSE_P_Value': dm_p_val,
        'EGARCH_Directional_Acc': acc_e,
        'Challenger_Directional_Acc': acc_c,
        'DM_Directional_Stat': dm_stat_dir,
        'DM_Directional_P_Value': dm_p_dir
    }])
    
    output_file = OUTPUT_DIR / f"EGARCH_vs_Challenger_Summary_{SOURCE_FACTOR}_to_{TARGET_FACTOR}.csv"
    summary_df.to_csv(output_file, index=False)
    
    print("\n" + "="*50)
    print("PURE EGARCH VS CHALLENGER SUMMARY")
    print("="*50)
    print(summary_df.to_string(index=False))
    print(f"\nSummary successfully saved to {output_file}")

if __name__ == "__main__":
    run_egarch_baseline()