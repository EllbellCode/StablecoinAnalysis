"""
Used to plot the feature importance of our models.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from arch import arch_model
from statsmodels.tsa.arima.model import ARIMA
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import warnings
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats


warnings.filterwarnings('ignore')

PARAM_GRID = {
    'n_estimators': [50, 100],                
    'learning_rate': [0.05, 0.1],             
    'max_depth': [3, 4, 5],                     
    'subsample': [0.7, 0.8],                   
    'colsample_bytree': [0.7, 0.8],               
    'reg_alpha': [0, 0.05],                   
    'reg_lambda': [0.5, 1.5]                  
}

DATA_DIR = Path("Data/Verified")
OUTPUT_DIR = Path("Results/MSE")
PLOT_DIR = Path("Plots/MSE")
START_DATE = '2020-01-01'
TRAIN_END_DATE = '2024-01-01'
FULL_END_DATE = '2025-01-01'
MAX_ARMA_ORDER = 1 
MAX_GARCH_ORDER = 1 
GARCH_DIST = 'skewt'
GARCH_MODEL = 'EGARCH' 
SPECIFIC_LAGS = [1, 7, 30]
DAYS_NUMBER = 366 
WIN_LIMITS = [0.01, 0.01]
WINDOW_SIZE = 9999 # Selects the rolling window size for PCA. Set to a high number (99999) to switch to expanding window
RANDOM_STATE = 123
VOLATILITY = "RS"
PLOT_FONT_SIZE = 24  

plt.rcParams.update({'font.size': PLOT_FONT_SIZE})

SOURCE_FACTOR = "Stable_Upside"
TARGET_FACTOR = "Crypto_Upside"

XGB_PARAMS = { 
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'random_state': RANDOM_STATE,
    'n_jobs': -1 
}

def select_best_arma(series, max_order=MAX_ARMA_ORDER):
    
    best_aic = np.inf
    best_order = (0, 0)
    series = series.dropna()

    if series.empty: return best_order
    for p in range(max_order + 1):
        for q in range(max_order + 1):
            if p == 0 and q == 0: continue
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
    if mean_p == 0:
        mean_model = 'Constant'
        ar_lags = None
    else:
        mean_model = 'AR'
        ar_lags = mean_p
    try:
        am = arch_model(series, vol=GARCH_MODEL, p=p, q=q,
                        mean=mean_model, lags=ar_lags,
                        dist=dist)
        res = am.fit(update_freq=0, disp='off', options={'maxiter': 200})
        return res
    except Exception as e:
        return None

def get_conditional_volatility(model_result):
    if model_result is None: return pd.Series(dtype=float)
    return model_result.conditional_volatility

def transform_to_uniform(model_result):
    if model_result is None: 
        return pd.Series(dtype=float)
    std_resid = model_result.std_resid.dropna()
    if std_resid.empty:
        return pd.Series(dtype=float)
    dist = model_result.model.distribution 
    all_params = model_result.params
    dist_param_names = dist.parameter_names()
    dist_params = [all_params[name] for name in dist_param_names]
    uniform_shocks = dist.cdf(std_resid, parameters=dist_params)
    return pd.Series(uniform_shocks, index=std_resid.index)

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

def format_feature_name(name):
    
    if 'source' in name:
        prefix = 'Stable'
    elif 'target' in name:
        prefix = 'Crypto'
    else:
        prefix = 'Unknown'
        
    if 'vol' in name:
        metric = 'Cond Vol.'
    elif 'resid' in name:
        metric = 'Res.'
    else:
        metric = 'Res.'
        
    suffix = ''
    if 'lag' in name:
        
        parts = name.split('_')
        lag_num = parts[-1]
        suffix = f'Lag {lag_num}'
    elif 'ma' in name:
        parts = name.split('_')
        win_num = parts[-1]
        suffix = f'MA {win_num}'
        
    clean_name = f"{prefix} {metric} {suffix}".strip()
    return clean_name

def create_garch_features(u_target_series, u_vol_series, specific_lags, v_resid_series=None, v_vol_series=None):
    u_target_series = u_target_series.replace([np.inf, -np.inf], np.nan)
    u_vol_series = u_vol_series.replace([np.inf, -np.inf], np.nan)
    if v_resid_series is not None:
        v_resid_series = v_resid_series.replace([np.inf, -np.inf], np.nan)
    if v_vol_series is not None:
        v_vol_series = v_vol_series.replace([np.inf, -np.inf], np.nan)

    df = pd.DataFrame({'u_target': u_target_series})
    df['u_vol'] = u_vol_series
    
    
    for i in specific_lags:
        df[f'target_lag_{i}'] = df['u_target'].shift(i)
    for i in specific_lags:
        df[f'target_vol_lag_{i}'] = df['u_vol'].shift(i)
        
    if v_resid_series is not None and v_vol_series is not None:
        df['v_resid'] = v_resid_series
        df['v_vol'] = v_vol_series
        for i in specific_lags:
            df[f'source_resid_lag_{i}'] = df['v_resid'].shift(i)
        for i in specific_lags:
            df[f'source_vol_lag_{i}'] = df['v_vol'].shift(i)
            
        df['source_resid_ma_7'] = df['v_resid'].rolling(window=7).mean().shift(1)
        df['source_resid_ma_30'] = df['v_resid'].rolling(window=30).mean().shift(1)
        df['source_vol_ma_7'] = df['v_vol'].rolling(window=7).mean().shift(1)
        df['source_vol_ma_30'] = df['v_vol'].rolling(window=30).mean().shift(1)
        
        df = df.drop(columns=['v_resid', 'v_vol']) 

    df = df.drop(columns='u_vol')
    
    bench_cols = [col for col in df.columns if col.startswith('target_')]
    
    df_bench = df[['u_target'] + bench_cols].dropna()
    y_bench = df_bench['u_target']
    X_bench = df_bench.drop(columns='u_target')
    
    if v_resid_series is not None:
        df_chall_garch = df.dropna() 
        y_chall_garch = df_chall_garch['u_target']
        X_chall_garch = df_chall_garch.drop(columns='u_target')
    else:
        X_chall_garch, y_chall_garch = (None, None)

    return X_bench, y_bench, X_chall_garch, y_chall_garch

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading and preparing data...")
coin_data = {}
stablecoins = ["DAI", "USDC", "USDT"]
cryptos = ["BNB", "BTC", "ETH", "XRP"]
all_coins = stablecoins + cryptos
for file in DATA_DIR.glob("*.csv"):
    coin_name = file.stem.replace("Verif_", "")
    if coin_name in all_coins:
        df = pd.read_csv(file, parse_dates=['Date']).sort_values("Date")
        df = df[(df['Date'] >= START_DATE) & (df['Date'] <= FULL_END_DATE)]
        coin_data[coin_name] = df.set_index('Date')


print("Calculating Initial PCA Factors for Hyperparameter Tuning...")
initial_factors = generate_factors_window(coin_data, TRAIN_END_DATE)

print("-" * 50)
mode_str = "ROLLING" if WINDOW_SIZE < 2000 else "EXPANDING"
print(f"Running Tuning and Plotting (No Backtest) ({mode_str} PCA, Window: {WINDOW_SIZE})")
print(f"Source: {SOURCE_FACTOR}, Target: {TARGET_FACTOR}, Lags: {SPECIFIC_LAGS}")
print("-" * 50)

ml_results = []
source_key = SOURCE_FACTOR
target_key = TARGET_FACTOR
specific_lags = SPECIFIC_LAGS

y_series_init = initial_factors[target_key].dropna()
x_series_init = initial_factors[source_key].dropna()
common_idx_init = y_series_init.index.intersection(x_series_init.index)
y_series_init = y_series_init.loc[common_idx_init]
x_series_init = x_series_init.loc[common_idx_init]

if len(y_series_init) < 50:
    print(f"  Skipping test: Not enough data.")
    exit()

print(f"  Finding initial ARMA orders for GARCH filtering...")
fixed_arma_order_y = select_best_arma(y_series_init, max_order=MAX_ARMA_ORDER)
fixed_arma_order_x = select_best_arma(x_series_init, max_order=MAX_ARMA_ORDER)

initial_target_garch_fit = fit_best_garch(y_series_init, p=MAX_GARCH_ORDER, q=MAX_GARCH_ORDER,
                                          mean_p=fixed_arma_order_y[0], mean_q=fixed_arma_order_y[1],
                                          dist=GARCH_DIST)
initial_source_garch_fit = fit_best_garch(x_series_init, p=MAX_GARCH_ORDER, q=MAX_GARCH_ORDER,
                                          mean_p=fixed_arma_order_x[0], mean_q=fixed_arma_order_x[1],
                                          dist=GARCH_DIST)

if initial_target_garch_fit is None or initial_source_garch_fit is None:
    print("  Skipping test: Failed to fit initial GARCH models.")
    exit()

u_target_full = transform_to_uniform(initial_target_garch_fit).clip(1e-6, 1-1e-6)
u_vol_full = get_conditional_volatility(initial_target_garch_fit)
v_resid_full = transform_to_uniform(initial_source_garch_fit).clip(1e-6, 1-1e-6)
v_vol_full = get_conditional_volatility(initial_source_garch_fit)

X_train_initial_bench_g, y_train_initial_bench_g, X_train_initial_chall_g, y_train_initial_chall_g = create_garch_features(
    u_target_series=u_target_full, u_vol_series=u_vol_full, specific_lags=specific_lags,   
    v_resid_series=v_resid_full, v_vol_series=v_vol_full        
)

if X_train_initial_bench_g.empty or X_train_initial_chall_g.empty:
    print("  Skipping tuning: Not enough initial data.")
    exit()

tscv = TimeSeriesSplit(n_splits=5)

print("  Tuning Benchmark GARCH Model...")
grid_search_benchmark_garch = GridSearchCV(xgb.XGBRegressor(**XGB_PARAMS), PARAM_GRID, scoring='neg_mean_squared_error', cv=tscv, n_jobs=-1)
grid_search_benchmark_garch.fit(X_train_initial_bench_g, y_train_initial_bench_g)
best_params_benchmark_garch = {**XGB_PARAMS, **grid_search_benchmark_garch.best_params_}

print("  Tuning Challenger GARCH Model...")
grid_search_challenger_garch = GridSearchCV(xgb.XGBRegressor(**XGB_PARAMS), PARAM_GRID, scoring='neg_mean_squared_error', cv=tscv, n_jobs=-1)
grid_search_challenger_garch.fit(X_train_initial_chall_g, y_train_initial_chall_g)
best_params_challenger_garch = {**XGB_PARAMS, **grid_search_challenger_garch.best_params_}

print("\n" + "-"*30)
print("FINAL HYPERPARAMETER CONFIGURATIONS")
print("-"*30)
print(f"Best Benchmark Params:  {grid_search_benchmark_garch.best_params_}")
print(f"Best Challenger Params: {grid_search_challenger_garch.best_params_}")
print("-"*30 + "\n")


print("  Generating Feature Importance Plot...")
try:

    best_model = grid_search_challenger_garch.best_estimator_
    importances = best_model.feature_importances_
    
    feature_names = [format_feature_name(col) for col in X_train_initial_chall_g.columns]
    
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=True).tail(10)

    plt.figure(figsize=(14, 10))
    
    colors = []
    for feat in feature_importance_df['Feature']:
        if 'Stable' in feat:
            colors.append('midnightblue')
        else:
            colors.append('red')
    
    plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color=colors)
    plt.xlabel('Feature Gain')
    
    stable_patch = mpatches.Patch(color='midnightblue', label='Stablecoin')
    crypto_patch = mpatches.Patch(color='red', label='Crypto')
    plt.legend(handles=[stable_patch, crypto_patch])
    
    plt.tight_layout()

    plot_path = PLOT_DIR / f"Feature_Importance_{SOURCE_FACTOR}_to_{TARGET_FACTOR}.png"
    plt.savefig(plot_path)
    print(f"  Feature importance plot saved to: {plot_path}")

except Exception as e:
    print(f"  Warning: Could not generate importance plot: {e}")
    import traceback
    traceback.print_exc()

print("\nFeature importance generation complete. Backtest skipped as requested.")