import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import stats
from pathlib import Path
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# --- Configuration ---
DATA_DIR = Path("Data/Verified")
INPUT_RESULTS_DIR = Path("StableCryptoResults/MSE")
OUTPUT_DIR = Path("Results/MSE_Baseline_Comparison")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

START_DATE = '2020-01-01'
TRAIN_END_DATE = '2024-01-01'
FULL_END_DATE = '2025-01-01'
WINDOW_SIZE = 9999 
WIN_LIMITS = [0.01, 0.01]
VOLATILITY = "RS"
RANDOM_STATE = 123

# Define the single pair to test (matches original script)
SOURCE_FACTOR = "Stable_Volume"
TARGET_FACTOR = "Crypto_Downside"

XGB_PARAMS = { 
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'random_state': RANDOM_STATE,
    'n_jobs': -1,
    'n_estimators': 100, 
    'learning_rate': 0.05, 
    'max_depth': 3
}

stablecoins = ["DAI", "USDC", "USDT"]
cryptos = ["BNB", "BTC", "ETH", "XRP"]

# --- Modified Diebold-Mariano Test ---
def modified_diebold_mariano_test(actual, pred_base, pred_chall):
    e_base = actual - pred_base
    e_chall = actual - pred_chall
    
    d_t = (e_base ** 2) - (e_chall ** 2)
    
    N = len(d_t)
    mean_d = np.mean(d_t)
    gamma_0 = np.var(d_t, ddof=1)
    var_d = gamma_0 / N
    
    if var_d == 0:
        return 0.0, 1.0
        
    dm_stat = mean_d / np.sqrt(var_d)
    
    h = 1
    modification = np.sqrt((N + 1 - 2*h + (h/N)*(h-1)) / N)
    mod_dm_stat = dm_stat * modification
    
    p_value = 1 - stats.t.cdf(mod_dm_stat, df=N-1)
    
    return mod_dm_stat, p_value

# --- PCA Generation Functions (From Original Script) ---
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
    
    if not test_data.empty:
        test_factor = pca.transform(test_scaled)
    else:
        test_factor = np.array([])

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
def run_baseline_comparison():
    print("Loading and preparing raw verified data...")
    coin_data = {}
    all_coins = stablecoins + cryptos
    for file in DATA_DIR.glob("*.csv"):
        coin_name = file.stem.replace("Verif_", "")
        if coin_name in all_coins:
            df = pd.read_csv(file, parse_dates=['Date']).sort_values("Date")
            df = df[(df['Date'] >= START_DATE) & (df['Date'] <= FULL_END_DATE)]
            coin_data[coin_name] = df.set_index('Date')
            
    if not coin_data:
        print("Error: No data loaded. Check DATA_DIR path.")
        return

    full_dates = coin_data[cryptos[0]].index
    test_dates = full_dates[(full_dates > TRAIN_END_DATE) & (full_dates <= FULL_END_DATE)]

    print(f"\nRunning Simple Baseline Backtest for {len(test_dates)} out-of-sample points...")
    print(f"Source: {SOURCE_FACTOR}, Target: {TARGET_FACTOR}")
    
    oos_predictions_baseline = []

    for current_date in tqdm(test_dates):
        yesterday = current_date - pd.Timedelta(days=1)
        
        # Dynamically calculate PCA factors up to 'current_date' to prevent lookahead
        current_factors_full = generate_factors_window(coin_data, current_date)
        
        df_model = pd.DataFrame({
            'Actual_Target': current_factors_full[TARGET_FACTOR],
            'Source': current_factors_full[SOURCE_FACTOR]
        }).dropna()
        
        # Create Lag 1 features
        df_model['Lag1_Target'] = df_model['Actual_Target'].shift(1)
        df_model['Lag1_Source'] = df_model['Source'].shift(1)
        df_model.dropna(inplace=True)
        
        if current_date not in df_model.index:
            continue
            
        # Train on expanding window up to yesterday
        train_data = df_model.loc[:yesterday]
        if train_data.empty:
            continue
            
        X_train = train_data[['Lag1_Target', 'Lag1_Source']]
        y_train = train_data['Actual_Target']
        
        # Test on exactly current_date
        X_test = df_model.loc[[current_date], ['Lag1_Target', 'Lag1_Source']]
        actual_y_value = df_model.loc[current_date, 'Actual_Target']
        
        # Fit simple XGBoost directly on the raw factors
        base_model = xgb.XGBRegressor(**XGB_PARAMS)
        base_model.fit(X_train, y_train)
        
        baseline_pred = base_model.predict(X_test)[0]
        
        oos_predictions_baseline.append({
            'Date': current_date,
            'Pred_Baseline': baseline_pred
        })

    df_base_preds = pd.DataFrame(oos_predictions_baseline).set_index('Date')
    
    # --- Load Advanced Framework Results ---
    results_filename = f"OOS_Res_GARCH_{SOURCE_FACTOR}_to_{TARGET_FACTOR}.csv" 
    results_path = INPUT_RESULTS_DIR / results_filename
    
    if not results_path.exists():
        print(f"\nWarning: Advanced result file {results_path} not found.")
        print("Cannot calculate comparison metrics without the advanced results.")
        return
        
    df_res = pd.read_csv(results_path, index_col='Date', parse_dates=True)
    
    # --- Align and Evaluate ---
    df_eval = df_res.join(df_base_preds, how='inner')
    
    actual = df_eval['Actual']
    pred_chall = df_eval['Pred_Chall']
    pred_base = df_eval['Pred_Baseline']
    
    mse_base = mean_squared_error(actual, pred_base)
    mse_chall = mean_squared_error(actual, pred_chall)
    mse_red_pct = ((mse_base - mse_chall) / mse_base) * 100
    
    dm_stat, dm_p_val = modified_diebold_mariano_test(actual, pred_base, pred_chall)
    
    # --- Output Summary ---
    summary_df = pd.DataFrame([{
        'Source_Factor': SOURCE_FACTOR,
        'Target_Factor': TARGET_FACTOR,
        'OOS_Days': len(df_eval),
        'MSE_Simple_Baseline': mse_base,
        'MSE_Challenger': mse_chall,
        'MSE_Reduction_Pct': f"{mse_red_pct:.2f}%",
        'DM_Stat_vs_Baseline': dm_stat,
        'DM_P_Value_vs_Baseline': dm_p_val
    }])
    
    output_file = OUTPUT_DIR / f"Baseline_vs_Challenger_Summary_{SOURCE_FACTOR}_to_{TARGET_FACTOR}.csv"
    summary_df.to_csv(output_file, index=False)
    
    print("\n" + "="*50)
    print("BASELINE VS CHALLENGER SUMMARY")
    print("="*50)
    print(summary_df.to_string(index=False))
    print(f"\nSummary successfully saved to {output_file}")

if __name__ == "__main__":
    run_baseline_comparison()