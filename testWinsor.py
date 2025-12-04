import pandas as pd
import numpy as np
from pathlib import Path
from arch import arch_model
from statsmodels.tsa.arima.model import ARIMA
from scipy.stats import t, norm, distributions
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import warnings
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
import matplotlib.pyplot as plt
from scipy.stats.mstats import winsorize # <--- NEW IMPORT

warnings.filterwarnings('ignore')

# --- OOS Hyperparameter Tuning: Define Grid (Smaller Version) ---
PARAM_GRID = {
    'n_estimators': [50, 100],                
    'learning_rate': [0.05, 0.1],             
    'max_depth': [3, 4, 5],                     
    'subsample': [0.7, 0.8],                   
    'colsample_bytree': [0.7, 0.8],               
    'reg_alpha': [0, 0.05],                   
    'reg_lambda': [0.5, 1.5]                  
}

# --- Constants / Settings ---
DATA_DIR = Path("Data/Verified")
OUTPUT_DIR = Path("Results/ML/XG")
PLOT_DIR = Path("Plots/ML/Gain/XG")
START_DATE = '2020-01-01'
TRAIN_END_DATE = '2024-01-01'
FULL_END_DATE = '2025-01-01'
MAX_ARMA_ORDER = 1 
MAX_GARCH_ORDER = 1 
GARCH_DIST = 'skewt'
GARCH_MODEL = 'EGARCH' 
SPECIFIC_LAGS = [1, 7, 30]
DAYS_NUMBER = 366 
TRADING_THRESHOLD = 0.5
WIN_LIMITS = [0.01, 0.01]
RANDOM_STATE = 123
VOLATILITY = "Delta_LogGK"

# Define the single pair to test
SOURCE_FACTOR = "Stable_Volatility"
TARGET_FACTOR = "Crypto_Volatility"
# --- *** END NEW SETTINGS *** ---


XGB_PARAMS = { 
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'random_state': RANDOM_STATE,
    'n_jobs': -1 
}

RISK_FREE_RATE = 0.0 

# ===================================================================
# Helper Functions 
# ===================================================================

def get_oos_forecast_params(fitted_model, actual_value):
    """Calculates 1-step-ahead forecast parameters."""
    forecast = fitted_model.forecast(horizon=1, reindex=False)
    mean_forecast = forecast.mean.iloc[0, 0]
    var_forecast = forecast.variance.iloc[0, 0]
    scale_forecast = np.sqrt(var_forecast)
    try:
        std_shock = (actual_value - mean_forecast) / scale_forecast 
        dist = fitted_model.model.distribution 
        all_params = fitted_model.params
        dist_param_names = dist.parameter_names()
        dist_params = [all_params[name] for name in dist_param_names]
        uniform_transform = dist.cdf(std_shock, parameters=dist_params)
        nu = all_params.get('nu', np.inf)
        return mean_forecast, scale_forecast, nu, uniform_transform
    except Exception as e:
        return np.nan, np.nan, np.nan, np.nan

def select_best_arma(series, max_order=MAX_ARMA_ORDER):
    """Selects the best ARMA(p,q) order based on AIC."""
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
    """Fits the specified ARMA(mean_p, mean_q)-GARCH(p,q) model."""
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
        print(f"!!! GARCH FIT FAILED WITH EXCEPTION !!!")
        print(f"  Error on series ending {series.index[-1]}: {e}")
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

# --- MODIFIED PCA function (Includes Winsorization) ---
def create_pca_factor(coins, var, data_dict, factor_name, train_end_date):
    df_list = [data_dict[coin][var] for coin in coins if coin in data_dict and var in data_dict[coin].columns]
    if not df_list: return pd.Series(dtype=float, name=factor_name)
    data_matrix = pd.concat(df_list, axis=1, keys=coins, join='inner').dropna()
    if data_matrix.empty: return pd.Series(dtype=float, name=factor_name)
    
    # Split Logic
    train_matrix = data_matrix.loc[START_DATE:train_end_date].copy() # Use .copy()
    test_matrix = data_matrix.loc[train_end_date:].iloc[1:].copy()
    
    if train_matrix.empty or test_matrix.empty: return pd.Series(dtype=float, name=factor_name)

    # --- NEW: Winsorize Training Data BEFORE PCA ---
    # We do this col by col to preserve the DataFrame structure
    for col in train_matrix.columns:
        train_matrix[col] = winsorize(train_matrix[col], limits=WIN_LIMITS)
    # -----------------------------------------------

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_matrix)
    
    pca = PCA(n_components=1)
    train_factor_scaled = pca.fit_transform(train_scaled)
    
    print(f"Created Factor '{factor_name}'. PCA Explained Variance: {pca.explained_variance_ratio_[0]:.2%}")
    
    # Apply weights to test data (We do NOT winsorize test data inputs for PCA projection, we treat them as raw arrivals)
    test_scaled = scaler.transform(test_matrix)
    test_factor_scaled = pca.transform(test_scaled)
    
    train_series = pd.Series(train_factor_scaled.ravel(), index=train_matrix.index, name=factor_name)
    test_series = pd.Series(test_factor_scaled.ravel(), index=test_matrix.index, name=factor_name)
    
    return pd.concat([train_series, test_series])

# ===================================================================
# *** ML Feature Creation Functions (Unchanged) ***
# ===================================================================
# ... [Keep create_garch_features, create_raw_on_raw_features, 
#      create_vol_on_garch_features, calculate_sharpe_ratio exactly as they were] ...

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
        df[f'u_target_lag_{i}'] = df['u_target'].shift(i)
    for i in specific_lags:
        df[f'u_vol_lag_{i}'] = df['u_vol'].shift(i)
        
    if v_resid_series is not None and v_vol_series is not None:
        df['v_resid'] = v_resid_series
        df['v_vol'] = v_vol_series
        for i in specific_lags:
            df[f'v_resid_lag_{i}'] = df['v_resid'].shift(i)
        for i in specific_lags:
            df[f'v_vol_lag_{i}'] = df['v_vol'].shift(i)
            
        df['v_resid_ma_7_lag_1'] = df['v_resid'].rolling(window=7).mean().shift(1)
        df['v_resid_ma_30_lag_1'] = df['v_resid'].rolling(window=30).mean().shift(1)
        df['v_vol_ma_7_lag_1'] = df['v_vol'].rolling(window=7).mean().shift(1)
        df['v_vol_ma_30_lag_1'] = df['v_vol'].rolling(window=30).mean().shift(1)
        
        df = df.drop(columns=['v_resid', 'v_vol']) 

    df = df.drop(columns='u_vol')
    
    bench_cols = [col for col in df.columns if col.startswith('u_') and col != 'u_target']
    
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

def create_raw_on_raw_features(y_raw_series, x_raw_series, specific_lags):
    y_raw_series = y_raw_series.replace([np.inf, -np.inf], np.nan)
    x_raw_series = x_raw_series.replace([np.inf, -np.inf], np.nan)

    df = pd.DataFrame({'raw_target': y_raw_series})
    df['raw_source'] = x_raw_series
    
    for i in specific_lags:
        df[f'raw_target_lag_{i}'] = df['raw_target'].shift(i)
    df['raw_target_ma_7_lag_1'] = df['raw_target'].rolling(window=7).mean().shift(1)
    df['raw_target_ma_30_lag_1'] = df['raw_target'].rolling(window=30).mean().shift(1)

    for i in specific_lags:
        df[f'raw_source_lag_{i}'] = df['raw_source'].shift(i)
    df['raw_source_ma_7_lag_1'] = df['raw_source'].rolling(window=7).mean().shift(1)
    df['raw_source_ma_30_lag_1'] = df['raw_source'].rolling(window=30).mean().shift(1)

    df = df.drop(columns=['raw_source'])

    bench_cols = [col for col in df.columns if col.startswith('raw_target_') and col != 'raw_target']
    
    df_bench_raw = df[['raw_target'] + bench_cols].dropna()
    y_bench_raw = df_bench_raw['raw_target']
    X_bench_raw = df_bench_raw.drop(columns='raw_target')
    
    df_chall_raw = df.dropna()
    y_chall_raw = df_chall_raw['raw_target']
    X_chall_raw = df_chall_raw.drop(columns='raw_target')

    return X_bench_raw, y_bench_raw, X_chall_raw, y_chall_raw

def create_vol_on_garch_features(y_vol_series, u_shock_series, v_shock_series, v_vol_series, specific_lags):
    y_vol_series = y_vol_series.replace([np.inf, -np.inf], np.nan)
    u_shock_series = u_shock_series.replace([np.inf, -np.inf], np.nan) 
    if v_shock_series is not None:
        v_shock_series = v_shock_series.replace([np.inf, -np.inf], np.nan)
    if v_vol_series is not None:
        v_vol_series = v_vol_series.replace([np.inf, -np.inf], np.nan)

    df = pd.DataFrame({'target_vol': y_vol_series})
    
    df['u_shock'] = u_shock_series
    df['v_shock'] = v_shock_series
    df['v_vol'] = v_vol_series
    
    for i in specific_lags:
        df[f'target_vol_lag_{i}'] = df['target_vol'].shift(i)
    for i in specific_lags:
        df[f'u_shock_lag_{i}'] = df['u_shock'].shift(i)
        
    if v_shock_series is not None and v_vol_series is not None:
        for i in specific_lags:
            df[f'v_shock_lag_{i}'] = df['v_shock'].shift(i)
        for i in specific_lags:
            df[f'v_vol_lag_{i}'] = df['v_vol'].shift(i)

        df['v_shock_ma_7_lag_1'] = df['v_shock'].rolling(window=7).mean().shift(1)
        df['v_shock_ma_30_lag_1'] = df['v_shock'].rolling(window=30).mean().shift(1)
        df['v_vol_ma_7_lag_1'] = df['v_vol'].rolling(window=7).mean().shift(1)
        df['v_vol_ma_30_lag_1'] = df['v_vol'].rolling(window=30).mean().shift(1)
        
    df = df.drop(columns=['u_shock', 'v_shock', 'v_vol'], errors='ignore')

    bench_cols = [col for col in df.columns if col.startswith('target_vol_lag_') or col.startswith('u_shock_lag_')]
    
    df_bench_vol = df[['target_vol'] + bench_cols].dropna()
    y_bench_vol = df_bench_vol['target_vol']
    X_bench_vol = df_bench_vol.drop(columns='target_vol')
    
    df_chall_vol = df.dropna()
    y_chall_vol = df_chall_vol['target_vol']
    X_chall_vol = df_chall_vol.drop(columns='target_vol')

    return X_bench_vol, y_bench_vol, X_chall_vol, y_chall_vol

def calculate_sharpe_ratio(returns, risk_free_rate=RISK_FREE_RATE):
    excess_returns = returns - risk_free_rate / DAYS_NUMBER
    mean_er = excess_returns.mean()
    std_er = excess_returns.std()
    if std_er == 0: return 0.0
    sharpe = (mean_er / std_er) * np.sqrt(DAYS_NUMBER)
    return sharpe

# ===================================================================
# Main ML Backtest Logic
# ===================================================================

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

print("Creating PCA Factors (Train/Test Split aware + Winsorized)...")
factors = {}
factors["Stable_Volume"] = create_pca_factor(stablecoins, "LogVolChange", coin_data, "PC1_Stable_Volume", TRAIN_END_DATE)
factors["Stable_Volatility"] = create_pca_factor(stablecoins, VOLATILITY, coin_data, "PC1_Stable_Volatility", TRAIN_END_DATE)
factors["Stable_Returns"] = create_pca_factor(stablecoins, "Log Returns", coin_data, "PC1_Stable_Returns", TRAIN_END_DATE)
factors["Crypto_Returns"] = create_pca_factor(cryptos, "Log Returns", coin_data, "PC1_Crypto_Returns", TRAIN_END_DATE)
factors["Crypto_Volatility"] = create_pca_factor(cryptos, VOLATILITY, coin_data, "PC1_Crypto_Volatility", TRAIN_END_DATE)
factors["Crypto_Volume"] = create_pca_factor(cryptos, "LogVolChange", coin_data, "PC1_Crypto_Volume", TRAIN_END_DATE)


print("-" * 50)
print(f"Running Out-of-Sample ML Backtests (WINSORIZED)")
print(f"Source: {SOURCE_FACTOR}, Target: {TARGET_FACTOR}, Lags: {SPECIFIC_LAGS}")
print("-" * 50)

ml_results = []
source_key = SOURCE_FACTOR
target_key = TARGET_FACTOR
specific_lags = SPECIFIC_LAGS
max_lag_needed = max(specific_lags)

y_series_all = factors[target_key].dropna()
x_series_all = factors[source_key].dropna()
common_idx = y_series_all.index.intersection(x_series_all.index)
y_series_all = y_series_all.loc[common_idx]
x_series_all = x_series_all.loc[common_idx]

initial_train_y_raw = y_series_all.loc[:TRAIN_END_DATE]
initial_train_x_raw = x_series_all.loc[:TRAIN_END_DATE]
test_dates = y_series_all.loc[TRAIN_END_DATE:].iloc[1:].index

min_obs = max(max_lag_needed, 30) + 50
if len(initial_train_y_raw) < min_obs or len(test_dates) < 50:
    print(f"  Skipping test: Not enough data.")
    exit()

print(f"  Finding initial ARMA orders for GARCH filtering...")
fixed_arma_order_y = select_best_arma(initial_train_y_raw, max_order=MAX_ARMA_ORDER)
fixed_arma_order_x = select_best_arma(initial_train_x_raw, max_order=MAX_ARMA_ORDER)

# --- NEW: Winsorize Initial Training Data for Hyperparam Tuning ---
init_y_win = pd.Series(winsorize(initial_train_y_raw, limits=WIN_LIMITS), index=initial_train_y_raw.index)
init_x_win = pd.Series(winsorize(initial_train_x_raw, limits=WIN_LIMITS), index=initial_train_x_raw.index)
# ------------------------------------------------------------------

initial_target_garch_fit = fit_best_garch(init_y_win, p=MAX_GARCH_ORDER, q=MAX_GARCH_ORDER,
                                          mean_p=fixed_arma_order_y[0], mean_q=fixed_arma_order_y[1],
                                          dist=GARCH_DIST)
initial_source_garch_fit = fit_best_garch(init_x_win, p=MAX_GARCH_ORDER, q=MAX_GARCH_ORDER,
                                          mean_p=fixed_arma_order_x[0], mean_q=fixed_arma_order_x[1],
                                          dist=GARCH_DIST)

print(initial_source_garch_fit)
print(initial_target_garch_fit)

if initial_target_garch_fit is None or initial_source_garch_fit is None:
    print("  Skipping test: Failed to fit initial GARCH models.")
    exit()

u_target_full = transform_to_uniform(initial_target_garch_fit).clip(1e-6, 1-1e-6)
u_vol_full = get_conditional_volatility(initial_target_garch_fit)
v_resid_full = transform_to_uniform(initial_source_garch_fit).clip(1e-6, 1-1e-6)
v_vol_full = get_conditional_volatility(initial_source_garch_fit)

# Clean infs
u_target_full.replace([np.inf, -np.inf], np.nan, inplace=True)
v_resid_full.replace([np.inf, -np.inf], np.nan, inplace=True)
u_vol_full.replace([np.inf, -np.inf], np.nan, inplace=True)
v_vol_full.replace([np.inf, -np.inf], np.nan, inplace=True)

# Note: We use the winsorized raw data (init_y_win) for the "Raw on Raw" feature creation
# to ensure the tree doesn't learn from extreme outliers during hyperparam tuning.
X_train_initial_bench_g, y_train_initial_bench_g, X_train_initial_chall_g, y_train_initial_chall_g = create_garch_features(
    u_target_series=u_target_full, 
    u_vol_series=u_vol_full,     
    specific_lags=specific_lags,   
    v_resid_series=v_resid_full,   
    v_vol_series=v_vol_full        
)
X_train_initial_bench_r, y_train_initial_bench_r, X_train_initial_chall_r, y_train_initial_chall_r = create_raw_on_raw_features(
    y_raw_series=init_y_win, # Using Winsorized
    x_raw_series=init_x_win, # Using Winsorized
    specific_lags=specific_lags
)
X_train_initial_bench_v, y_train_initial_bench_v, X_train_initial_chall_v, y_train_initial_chall_v = create_vol_on_garch_features(
    y_vol_series=u_vol_full,
    u_shock_series=u_target_full,
    v_shock_series=v_resid_full,
    v_vol_series=v_vol_full,
    specific_lags=specific_lags
)

if X_train_initial_bench_g.empty or X_train_initial_chall_g.empty or \
   X_train_initial_bench_r.empty or X_train_initial_chall_r.empty or \
   X_train_initial_bench_v.empty or X_train_initial_chall_v.empty:
    
    print("  Skipping tuning: Not enough initial data for features.")
    best_params_benchmark_garch = XGB_PARAMS
    best_params_challenger_garch = XGB_PARAMS
    best_params_benchmark_raw = XGB_PARAMS
    best_params_challenger_raw = XGB_PARAMS
    best_params_benchmark_vol = XGB_PARAMS
    best_params_challenger_vol = XGB_PARAMS
    class DummyEstimator:
        def __init__(self, params): self.best_estimator_ = xgb.XGBRegressor(**params).fit(np.array([[0,1],[1,0]]), np.array([0,1]))
    grid_search_challenger_garch = DummyEstimator(XGB_PARAMS)
    grid_search_challenger_vol = DummyEstimator(XGB_PARAMS)

else:
    # ... [Hyperparameter Tuning Blocks (Unchanged)] ...
    # (I am skipping repeating the GridSearch code here to save space, it remains identical)
    common_initial_idx_garch = X_train_initial_bench_g.index.intersection(X_train_initial_chall_g.index)
    y_train_initial_g = y_train_initial_bench_g.loc[common_initial_idx_garch]
    X_train_initial_bench_g = X_train_initial_bench_g.loc[common_initial_idx_garch]
    X_train_initial_chall_g = X_train_initial_chall_g.loc[common_initial_idx_garch]
    
    common_initial_idx_raw = X_train_initial_bench_r.index.intersection(X_train_initial_chall_r.index)
    y_train_initial_r = y_train_initial_bench_r.loc[common_initial_idx_raw]
    X_train_initial_bench_r = X_train_initial_bench_r.loc[common_initial_idx_raw]
    X_train_initial_chall_r = X_train_initial_chall_r.loc[common_initial_idx_raw]

    common_initial_idx_vol = X_train_initial_bench_v.index.intersection(X_train_initial_chall_v.index)
    y_train_initial_v = y_train_initial_bench_v.loc[common_initial_idx_vol]
    X_train_initial_bench_v = X_train_initial_bench_v.loc[common_initial_idx_vol]
    X_train_initial_chall_v = X_train_initial_chall_v.loc[common_initial_idx_vol]

    tscv = TimeSeriesSplit(n_splits=5)

    # GARCH Tuning
    xgb_benchmark_garch_tune = xgb.XGBRegressor(**XGB_PARAMS)
    grid_search_benchmark_garch = GridSearchCV(xgb_benchmark_garch_tune, PARAM_GRID, scoring='neg_mean_squared_error', cv=tscv, n_jobs=-1)
    grid_search_benchmark_garch.fit(X_train_initial_bench_g, y_train_initial_g)
    best_params_benchmark_garch = {**XGB_PARAMS, **grid_search_benchmark_garch.best_params_}

    xgb_challenger_garch_tune = xgb.XGBRegressor(**XGB_PARAMS)
    grid_search_challenger_garch = GridSearchCV(xgb_challenger_garch_tune, PARAM_GRID, scoring='neg_mean_squared_error', cv=tscv, n_jobs=-1)
    grid_search_challenger_garch.fit(X_train_initial_chall_g, y_train_initial_g)
    best_params_challenger_garch = {**XGB_PARAMS, **grid_search_challenger_garch.best_params_}

    # Raw Tuning
    xgb_benchmark_raw_tune = xgb.XGBRegressor(**XGB_PARAMS)
    grid_search_benchmark_raw = GridSearchCV(xgb_benchmark_raw_tune, PARAM_GRID, scoring='neg_mean_squared_error', cv=tscv, n_jobs=-1)
    grid_search_benchmark_raw.fit(X_train_initial_bench_r, y_train_initial_r)
    best_params_benchmark_raw = {**XGB_PARAMS, **grid_search_benchmark_raw.best_params_}

    xgb_challenger_raw_tune = xgb.XGBRegressor(**XGB_PARAMS)
    grid_search_challenger_raw = GridSearchCV(xgb_challenger_raw_tune, PARAM_GRID, scoring='neg_mean_squared_error', cv=tscv, n_jobs=-1)
    grid_search_challenger_raw.fit(X_train_initial_chall_r, y_train_initial_r)
    best_params_challenger_raw = {**XGB_PARAMS, **grid_search_challenger_raw.best_params_}

    # Vol Tuning
    xgb_benchmark_vol_tune = xgb.XGBRegressor(**XGB_PARAMS)
    grid_search_benchmark_vol = GridSearchCV(xgb_benchmark_vol_tune, PARAM_GRID, scoring='neg_mean_squared_error', cv=tscv, n_jobs=-1)
    grid_search_benchmark_vol.fit(X_train_initial_bench_v, y_train_initial_v)
    best_params_benchmark_vol = {**XGB_PARAMS, **grid_search_benchmark_vol.best_params_}

    xgb_challenger_vol_tune = xgb.XGBRegressor(**XGB_PARAMS)
    grid_search_challenger_vol = GridSearchCV(xgb_challenger_vol_tune, PARAM_GRID, scoring='neg_mean_squared_error', cv=tscv, n_jobs=-1)
    grid_search_challenger_vol.fit(X_train_initial_chall_v, y_train_initial_v)
    best_params_challenger_vol = {**XGB_PARAMS, **grid_search_challenger_vol.best_params_}
    
    print("\n" + "="*50)
print("Generating Feature Importance Plots (Test 1 Challenger vs Test 3 Challenger)...")

try:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))

    model_chall_garch = grid_search_challenger_garch.best_estimator_
    feature_names_garch = model_chall_garch.feature_names_in_
    importances_garch = model_chall_garch.feature_importances_
    importance_df_garch = pd.DataFrame({
        'Feature': feature_names_garch,
        'Importance': importances_garch
    }).sort_values(by='Importance', ascending=True)
    ax1.barh(importance_df_garch['Feature'], importance_df_garch['Importance'], color='skyblue')
    ax1.set_title(f'Test 1: Challenger GARCH (Predicts Shocks)\n{SOURCE_FACTOR} -> {TARGET_FACTOR}', fontsize=16)
    ax1.set_xlabel('Feature Importance (Gain)', fontsize=12)
    ax1.set_ylabel('Feature', fontsize=12)
    ax1.grid(axis='x', linestyle='--', alpha=0.7)

    model_chall_vol = grid_search_challenger_vol.best_estimator_
    feature_names_vol = model_chall_vol.feature_names_in_
    importances_vol = model_chall_vol.feature_importances_
    importance_df_vol = pd.DataFrame({
        'Feature': feature_names_vol,
        'Importance': importances_vol
    }).sort_values(by='Importance', ascending=True)
    ax2.barh(importance_df_vol['Feature'], importance_df_vol['Importance'], color='salmon')
    ax2.set_title(f'Test 3: Challenger VOL (Predicts Volatility)\n{SOURCE_FACTOR} -> {TARGET_FACTOR}', fontsize=16)
    ax2.set_xlabel('Feature Importance (Gain)', fontsize=12)
    ax2.set_ylabel('Feature', fontsize=12)
    ax2.grid(axis='x', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    chall_plot_filename = f"gain_{SOURCE_FACTOR}_to_{TARGET_FACTOR}_GarchVsVol.png"
    output_path = PLOT_DIR / chall_plot_filename
    plt.savefig(output_path)
    print(f"Saved Challenger importance plot to {output_path}")
    plt.close(fig)

except Exception as e:
    print(f"Could not generate feature importance plot. Error: {e}")

print("="*50 + "\n")


print("="*50 + "\n")

oos_predictions_garch = []
oos_predictions_raw = []
oos_predictions_vol = []

print(f"  Running expanding window forecast for {len(test_dates)} test points...")
for current_date in tqdm(test_dates):
    yesterday = current_date - pd.Timedelta(days=1)
    
    # --- NEW: Extract AND Winsorize the Expanding Window ---
    # 1. Extract History
    current_train_y_raw = y_series_all.loc[:yesterday].copy()
    current_train_x_raw = x_series_all.loc[:yesterday].copy()
    
    # 2. Get Actual for Evaluation (DO NOT WINSORIZE THIS)
    actual_y_raw = y_series_all.loc[current_date] 

    # 3. Winsorize the History used for Training models
    # This prevents the GARCH/XGBoost from overfitting to past outliers
    train_y_win = pd.Series(winsorize(current_train_y_raw, limits=WIN_LIMITS), index=current_train_y_raw.index)
    train_x_win = pd.Series(winsorize(current_train_x_raw, limits=WIN_LIMITS), index=current_train_x_raw.index)
    # -------------------------------------------------------

    # 4. Fit GARCH models (Use Winsorized Data)
    bench_model_fit = fit_best_garch(train_y_win, p=MAX_GARCH_ORDER, q=MAX_GARCH_ORDER,
                                     mean_p=fixed_arma_order_y[0], mean_q=fixed_arma_order_y[1],
                                     dist=GARCH_DIST)
    x_model_fit = fit_best_garch(train_x_win, p=MAX_GARCH_ORDER, q=MAX_GARCH_ORDER,
                                 mean_p=fixed_arma_order_x[0], mean_q=fixed_arma_order_x[1],
                                 dist=GARCH_DIST)
    if bench_model_fit is None or x_model_fit is None: continue
        
    # 5. Create and CLEAN all GARCH-filtered series
    u_target_is_full = transform_to_uniform(bench_model_fit).clip(1e-6, 1-1e-6)
    u_vol_is_full = get_conditional_volatility(bench_model_fit)
    v_resid_is_full = transform_to_uniform(x_model_fit).clip(1e-6, 1-1e-6)
    v_vol_is_full = get_conditional_volatility(x_model_fit)

    u_target_is_full.replace([np.inf, -np.inf], np.nan, inplace=True)
    v_resid_is_full.replace([np.inf, -np.inf], np.nan, inplace=True)
    u_vol_is_full.replace([np.inf, -np.inf], np.nan, inplace=True)
    v_vol_is_full.replace([np.inf, -np.inf], np.nan, inplace=True)

    if u_target_is_full.empty: continue

    # 6. Create Feature Sets (Training)
    X_train_bench_g, y_train_bench_g, X_train_chall_g, y_train_chall_g = create_garch_features(
        u_target_is_full, u_vol_is_full, specific_lags, 
        v_resid_is_full, v_vol_is_full
    )
    
    # Use winsorized raw data for Test 2 training to be consistent
    X_train_bench_r, y_train_bench_r, X_train_chall_r, y_train_chall_r = create_raw_on_raw_features(
        train_y_win, train_x_win, specific_lags
    )

    X_train_bench_v, y_train_bench_v, X_train_chall_v, y_train_chall_v = create_vol_on_garch_features(
        y_vol_series=u_vol_is_full,
        u_shock_series=u_target_is_full,
        v_shock_series=v_resid_is_full,
        v_vol_series=v_vol_is_full,
        specific_lags=specific_lags
    )

    if X_train_bench_g.empty or X_train_chall_g.empty or \
       X_train_bench_r.empty or X_train_chall_r.empty or \
       X_train_bench_v.empty or X_train_chall_v.empty:
        continue

    # --- Align Training Data ---
    common_train_idx_g = X_train_bench_g.index.intersection(X_train_chall_g.index)
    if common_train_idx_g.empty: continue
    y_train_g = y_train_bench_g.loc[common_train_idx_g]
    X_train_bench_g = X_train_bench_g.loc[common_train_idx_g]
    X_train_chall_g = X_train_chall_g.loc[common_train_idx_g]

    common_train_idx_r = X_train_bench_r.index.intersection(X_train_chall_r.index)
    if common_train_idx_r.empty: continue
    y_train_r = y_train_bench_r.loc[common_train_idx_r]
    X_train_bench_r = X_train_bench_r.loc[common_train_idx_r]
    X_train_chall_r = X_train_chall_r.loc[common_train_idx_r]

    common_train_idx_v = X_train_bench_v.index.intersection(X_train_chall_v.index)
    if common_train_idx_v.empty: continue
    y_train_v = y_train_bench_v.loc[common_train_idx_v]
    X_train_bench_v = X_train_bench_v.loc[common_train_idx_v]
    X_train_chall_v = X_train_chall_v.loc[common_train_idx_v]

    # 7. Train XGBoost Models
    model_bench_garch = xgb.XGBRegressor(**best_params_benchmark_garch)
    model_chall_garch = xgb.XGBRegressor(**best_params_challenger_garch)
    model_bench_garch.fit(X_train_bench_g, y_train_g)
    model_chall_garch.fit(X_train_chall_g, y_train_g)
    
    model_bench_raw = xgb.XGBRegressor(**best_params_benchmark_raw)
    model_chall_raw = xgb.XGBRegressor(**best_params_challenger_raw)
    model_bench_raw.fit(X_train_bench_r, y_train_r)
    model_chall_raw.fit(X_train_chall_r, y_train_r)

    model_bench_vol = xgb.XGBRegressor(**best_params_benchmark_vol)
    model_chall_vol = xgb.XGBRegressor(**best_params_challenger_vol)
    model_bench_vol.fit(X_train_bench_v, y_train_v)
    model_chall_vol.fit(X_train_chall_v, y_train_v)

    # 8. Create Features for Today's Prediction
    # NOTE: For today's feature, we usually use the 'real' latest values (residuals/vol),
    # which are derived from the model we just fit.
    if len(u_target_is_full) < max_lag_needed or len(v_resid_is_full) < 30 or len(current_train_y_raw) < 30:
        continue 
        
    try:
        # ... [Feature Construction Logic (Unchanged)] ...
        # (The logic here uses .iloc[-i] from u_target_is_full, which comes from the fitted model)
        # (This is correct: The model was fit on winsorized history, but generates "clean" residuals for today)
        u_resid_lags = np.array([u_target_is_full.iloc[-i] for i in specific_lags])
        u_vol_lags = np.array([u_vol_is_full.iloc[-i] for i in specific_lags])
        features_today_bench_g = np.concatenate([u_resid_lags, u_vol_lags])
        
        v_resid_lags = np.array([v_resid_is_full.iloc[-i] for i in specific_lags])
        v_vol_lags = np.array([v_vol_is_full.iloc[-i] for i in specific_lags])
        v_resid_ma_7 = v_resid_is_full.rolling(window=7).mean().iloc[-1]
        v_resid_ma_30 = v_resid_is_full.rolling(window=30).mean().iloc[-1]
        v_vol_ma_7 = v_vol_is_full.rolling(window=7).mean().iloc[-1]
        v_vol_ma_30 = v_vol_is_full.rolling(window=30).mean().iloc[-1]
        
        features_today_garch_v = np.concatenate([
            v_resid_lags, v_vol_lags,
            np.array([v_resid_ma_7, v_resid_ma_30, v_vol_ma_7, v_vol_ma_30])
        ])
        features_today_garch = np.concatenate([features_today_bench_g, features_today_garch_v]).reshape(1, -1)
        
        # For Raw Features, we use the raw data (current_train_y_raw) because that's what arrives today.
        # However, if today is a massive outlier, we should theoretically clip it to the training bounds.
        # For simplicity/safety, we will use the winsorized version 'train_y_win' here too
        # to ensure we don't feed a 10-sigma event into a tree that's never seen >3-sigma.
        raw_target_lags = np.array([train_y_win.iloc[-i] for i in specific_lags])
        raw_target_ma_7 = train_y_win.rolling(window=7).mean().iloc[-1]
        raw_target_ma_30 = train_y_win.rolling(window=30).mean().iloc[-1]
        features_today_bench_r_1D = np.concatenate([
            raw_target_lags,
            np.array([raw_target_ma_7, raw_target_ma_30])
        ])
        
        v_raw_lags = np.array([train_x_win.iloc[-i] for i in specific_lags])
        v_raw_ma_7 = train_x_win.rolling(window=7).mean().iloc[-1]
        v_raw_ma_30 = train_x_win.rolling(window=30).mean().iloc[-1]
        features_today_raw_v = np.concatenate([
            v_raw_lags,
            np.array([v_raw_ma_7, v_raw_ma_30])
        ])
        features_today_chall_r = np.concatenate([features_today_bench_r_1D, features_today_raw_v]).reshape(1, -1)

        #--- Test 3: VOL Features ---
        target_vol_lags = np.array([u_vol_is_full.iloc[-i] for i in specific_lags])
        features_today_bench_v_1D = np.concatenate([target_vol_lags, u_resid_lags])
        
        features_today_vol_v = np.concatenate([
            v_resid_lags, v_vol_lags,
            np.array([v_resid_ma_7, v_resid_ma_30, v_vol_ma_7, v_vol_ma_30])
        ])
        features_today_chall_v = np.concatenate([features_today_bench_v_1D, features_today_vol_v]).reshape(1, -1)

        if not np.all(np.isfinite(features_today_bench_g)) or \
           not np.all(np.isfinite(features_today_garch)) or \
           not np.all(np.isfinite(features_today_bench_r_1D)) or \
           not np.all(np.isfinite(features_today_chall_r)) or \
           not np.all(np.isfinite(features_today_bench_v_1D)) or \
           not np.all(np.isfinite(features_today_chall_v)):
            continue

        # ... [DataFrame Creation and Prediction logic (Unchanged)] ...
        bench_g_cols = [f'u_target_lag_{i}' for i in specific_lags] + [f'u_vol_lag_{i}' for i in specific_lags]
        X_pred_bench_g = pd.DataFrame(features_today_bench_g.reshape(1, -1), columns=bench_g_cols)
        
        chall_garch_v_cols = [f'v_resid_lag_{i}' for i in specific_lags] + \
                             [f'v_vol_lag_{i}' for i in specific_lags] + \
                             ['v_resid_ma_7_lag_1', 'v_resid_ma_30_lag_1', 'v_vol_ma_7_lag_1', 'v_vol_ma_30_lag_1']
        chall_garch_cols = bench_g_cols + chall_garch_v_cols
        X_pred_chall_garch = pd.DataFrame(features_today_garch, columns=chall_garch_cols)

        bench_r_cols = [f'raw_target_lag_{i}' for i in specific_lags] + \
                       ['raw_target_ma_7_lag_1', 'raw_target_ma_30_lag_1']
        X_pred_bench_r = pd.DataFrame(features_today_bench_r_1D.reshape(1, -1), columns=bench_r_cols)
        
        chall_raw_v_cols = [f'raw_source_lag_{i}' for i in specific_lags] + \
                           ['raw_source_ma_7_lag_1', 'raw_source_ma_30_lag_1']
        chall_r_cols = bench_r_cols + chall_raw_v_cols
        X_pred_chall_r = pd.DataFrame(features_today_chall_r, columns=chall_r_cols)
        
        bench_v_cols = [f'target_vol_lag_{i}' for i in specific_lags] + \
                       [f'u_shock_lag_{i}' for i in specific_lags]
        X_pred_bench_v = pd.DataFrame(features_today_bench_v_1D.reshape(1, -1), columns=bench_v_cols)

        chall_vol_v_cols = [f'v_shock_lag_{i}' for i in specific_lags] + \
                           [f'v_vol_lag_{i}' for i in specific_lags] + \
                           ['v_shock_ma_7_lag_1', 'v_shock_ma_30_lag_1', 'v_vol_ma_7_lag_1', 'v_vol_ma_30_lag_1']
        chall_v_cols = bench_v_cols + chall_vol_v_cols
        X_pred_chall_v = pd.DataFrame(features_today_chall_v, columns=chall_v_cols)

        pred_bench_g = model_bench_garch.predict(X_pred_bench_g)[0]
        pred_chall_g = model_chall_garch.predict(X_pred_chall_garch)[0]
        pred_bench_r = model_bench_raw.predict(X_pred_bench_r)[0]
        pred_chall_r = model_chall_raw.predict(X_pred_chall_r)[0]
        pred_bench_v = model_bench_vol.predict(X_pred_bench_v)[0]
        pred_chall_v = model_chall_vol.predict(X_pred_chall_v)[0]

        # 12. Get Actual Shock AND Vol for Today
        # IMPORTANT: We compare against 'actual_y_raw' (Real return), NOT winsorized return
        mean_fc, scale_fc, nu, actual_u_c_t = get_oos_forecast_params(bench_model_fit, actual_y_raw)

        if not np.isfinite(actual_u_c_t) or not np.isfinite(scale_fc): continue

        actual_volatility_t = scale_fc
        last_actual_vol = u_vol_is_full.iloc[-1]
        
        if not np.isfinite(last_actual_vol): continue

        oos_predictions_garch.append({
            "Date": current_date,
            "Actual_Shock": actual_u_c_t,
            "Pred_Shock_Bench_G": pred_bench_g,
            "Pred_Shock_Chall_G": pred_chall_g,
            "Actual_Return": actual_y_raw
        })
        
        oos_predictions_raw.append({
            "Date": current_date,
            "Actual_Raw": actual_y_raw,
            "Pred_Raw_Bench": pred_bench_r,
            "Pred_Raw_Chall": pred_chall_r
        })

        oos_predictions_vol.append({
            "Date": current_date,
            "Actual_Vol": actual_volatility_t,
            "Pred_Vol_Bench": pred_bench_v,
            "Pred_Vol_Chall": pred_chall_v,
            "Last_Actual_Vol": last_actual_vol,
            "Actual_Return": actual_y_raw 
        })

    except IndexError:
        continue

# [Evaluation logic remains unchanged] ...

# --- Evaluate OOS Predictions (after loop) ---
if len(oos_predictions_garch) < 20 or len(oos_predictions_raw) < 20 or len(oos_predictions_vol) < 20:
    print("  Skipping Evaluation: Not enough valid OOS forecasts generated.")
    exit()

print("  OOS loop finished. Evaluating models...")

# --- Process Test 1 (GARCH World) ---
pred_df_garch = pd.DataFrame(oos_predictions_garch).set_index("Date").dropna()

# --- Process Test 2 (Raw World) ---
pred_df_raw = pd.DataFrame(oos_predictions_raw).set_index("Date").dropna()
pred_df_raw = pred_df_raw.join(
    pred_df_garch['Actual_Return'],
    how='inner'
)
pred_df_raw = pred_df_raw.dropna()

# --- Process Test 3 (Volatility World) ---
pred_df_vol = pd.DataFrame(oos_predictions_vol).set_index("Date")
pred_df_vol.replace([np.inf, -np.inf], np.nan, inplace=True)
pred_df_vol.dropna(inplace=True)

# --- ROBUST ALIGNMENT BLOCK ---
common_idx = pred_df_garch.index.intersection(pred_df_raw.index).intersection(pred_df_vol.index)
print(f"  Aligning all results to {len(common_idx)} common dates.")
pred_df_garch = pred_df_garch.loc[common_idx]
pred_df_raw = pred_df_raw.loc[common_idx]
pred_df_vol = pred_df_vol.loc[common_idx]
# --- END ALIGNMENT ---

# 1a. Calculate MSE (Test 1)
mse_bench_g = mean_squared_error(pred_df_garch['Actual_Shock'], pred_df_garch['Pred_Shock_Bench_G'])
mse_chall_g = mean_squared_error(pred_df_garch['Actual_Shock'], pred_df_garch['Pred_Shock_Chall_G'])
# 2a. Calculate Sharpe Ratios (Test 1)
pred_df_garch['Position_Bench_G'] = np.where(pred_df_garch['Pred_Shock_Bench_G'] > TRADING_THRESHOLD, 1,
                              np.where(pred_df_garch['Pred_Shock_Bench_G'] < TRADING_THRESHOLD, -1, 0))
pred_df_garch['Position_Chall_G'] = np.where(pred_df_garch['Pred_Shock_Chall_G'] > TRADING_THRESHOLD, 1,
                              np.where(pred_df_garch['Pred_Shock_Chall_G'] < TRADING_THRESHOLD, -1, 0))
pred_df_garch['Return_Bench_G'] = pred_df_garch['Position_Bench_G'] * pred_df_garch['Actual_Return']
pred_df_garch['Return_Chall_G'] = pred_df_garch['Position_Chall_G'] * pred_df_garch['Actual_Return']
sharpe_bench_g = calculate_sharpe_ratio(pred_df_garch['Return_Bench_G'])
sharpe_chall_g = calculate_sharpe_ratio(pred_df_garch['Return_Chall_G'])

# 1b. Calculate MSE (Test 2)
mse_bench_r = mean_squared_error(pred_df_raw['Actual_Raw'], pred_df_raw['Pred_Raw_Bench'])
mse_chall_r = mean_squared_error(pred_df_raw['Actual_Raw'], pred_df_raw['Pred_Raw_Chall'])
# 2b. Calculate Sharpe Ratios (Test 2)
pred_df_raw['Position_Bench_R'] = np.where(pred_df_raw['Pred_Raw_Bench'] > 0, 1, -1)
pred_df_raw['Position_Chall_R'] = np.where(pred_df_raw['Pred_Raw_Chall'] > 0, 1, -1)
pred_df_raw['Return_Bench_R'] = pred_df_raw['Position_Bench_R'] * pred_df_raw['Actual_Return']
pred_df_raw['Return_Chall_R'] = pred_df_raw['Position_Chall_R'] * pred_df_raw['Actual_Return']
sharpe_bench_r = calculate_sharpe_ratio(pred_df_raw['Return_Bench_R'])
sharpe_chall_r = calculate_sharpe_ratio(pred_df_raw['Return_Chall_R'])

# 1c. Calculate MSE (Test 3)
finite_mask = np.isfinite(pred_df_vol['Pred_Vol_Bench']) & np.isfinite(pred_df_vol['Pred_Vol_Chall'])
pred_df_vol_finite = pred_df_vol.loc[finite_mask]

if pred_df_vol_finite.empty:
    print("  ERROR: No finite 'VOL' predictions were made. Cannot calculate MSE.")
    mse_bench_v = np.nan
    mse_chall_v = np.nan
else:
    mse_bench_v = mean_squared_error(pred_df_vol_finite['Actual_Vol'], pred_df_vol_finite['Pred_Vol_Bench'])
    mse_chall_v = mean_squared_error(pred_df_vol_finite['Actual_Vol'], pred_df_vol_finite['Pred_Vol_Chall'])
# 2c. Calculate Sharpe Ratios (Test 3)
pred_df_vol['Position_Bench_V'] = np.where(pred_df_vol['Pred_Vol_Bench'] > pred_df_vol['Last_Actual_Vol'], 1, -1)
pred_df_vol['Position_Chall_V'] = np.where(pred_df_vol['Pred_Vol_Chall'] > pred_df_vol['Last_Actual_Vol'], 1, -1)
pred_df_vol['Return_Bench_V'] = pred_df_vol['Position_Bench_V'] * pred_df_vol['Actual_Return']
pred_df_vol['Return_Chall_V'] = pred_df_vol['Position_Chall_V'] * pred_df_vol['Actual_Return']
sharpe_bench_v = calculate_sharpe_ratio(pred_df_vol['Return_Bench_V'])
sharpe_chall_v = calculate_sharpe_ratio(pred_df_vol['Return_Chall_V'])


# --- Combine All Results ---
results_garch = {
    "Test": "GARCH", "Source": source_key, "Target": target_key,
    "MSE Benchmark": mse_bench_g, "MSE Challenger": mse_chall_g,
    "Sharpe Benchmark": sharpe_bench_g, "Sharpe Challenger": sharpe_chall_g,
    "OOS_Days": len(pred_df_garch) # Length is now from aligned df
}
ml_results.append(results_garch)

results_raw = {
    "Test": "RAW", "Source": source_key, "Target": target_key,
    "MSE Benchmark": mse_bench_r, "MSE Challenger": mse_chall_r,
    "Sharpe Benchmark": sharpe_bench_r, "Sharpe Challenger": sharpe_chall_r,
    "OOS_Days": len(pred_df_raw) # Length is now from aligned df
}
ml_results.append(results_raw)

results_vol = {
    "Test": "VOL", "Source": source_key, "Target": target_key,
    "MSE Benchmark": mse_bench_v, "MSE Challenger": mse_chall_v,
    "Sharpe Benchmark": sharpe_bench_v, "Sharpe Challenger": sharpe_chall_v,
    "OOS_Days": len(pred_df_vol) # Length is now from aligned df
}
ml_results.append(results_vol)


# --- Dynamic Filename ---
output_slug = f"{source_key}_to_{target_key}"

# Save detailed predictions
preds_g_name = f"XG_preds_GARCH_{output_slug}_vol.csv"
preds_g_path = OUTPUT_DIR / preds_g_name
pred_df_garch.to_csv(preds_g_path)

preds_r_name = f"XG_preds_RAW_{output_slug}_vol.csv"
preds_r_path = OUTPUT_DIR / preds_r_name
pred_df_raw.to_csv(preds_r_path)

preds_v_name = f"XG_preds_VOL_{output_slug}_vol.csv"
preds_v_path = OUTPUT_DIR / preds_v_name
pred_df_vol.to_csv(preds_v_path)


# --- Final Output ---
print("\n" + "=" * 50)
print(f"Final OOS ML (XGBoost) Backtest Results")
print(f"{source_key} -> {target_key} (Lags = {specific_lags})")
print("=" * 50)

results_table = pd.DataFrame(ml_results)

column_order = [
    "Test", "Source", "Target", "OOS_Days",
    "MSE Benchmark", "MSE Challenger",
    "Sharpe Benchmark", "Sharpe Challenger"
]
column_order = [col for col in column_order if col in results_table.columns]
results_table = results_table[column_order]
print(results_table.to_string(float_format="%.6f"))

results_name = f"XG_results_{output_slug}_vol.csv"
results_path = OUTPUT_DIR / results_name
results_table.to_csv(results_path, index=False)
print(f"\nSummary results saved to '{results_path}'")
print(f"Test 1 (GARCH) predictions saved to '{preds_g_path}'")
print(f"Test 2 (RAW) predictions saved to '{preds_r_path}'")
print(f"Test 3 (VOL) predictions saved to '{preds_v_path}'")