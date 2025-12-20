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
from scipy import stats

"""
Trains two XGBoost model:
- Benchmark model that only uses historical data from the target factor
- Challenger model that uses data from the target (crypto) and source (stablecoin) factor

Trains the models on daily data from start of 2020 to end of 2023 (4 years)
Backtests on the year of 2024

Uses Diebold Mariano test to assess performance between models in terms of MSE and Directional Accuracy
"""
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
WIN_LIMITS = [0.01, 0.01]
RANDOM_STATE = 123
VOLATILITY = "Delta_LogGK"

# Define the single pair to test
SOURCE_FACTOR = "Stable_Volatility"
TARGET_FACTOR = "Crypto_Volatility"

XGB_PARAMS = { 
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'random_state': RANDOM_STATE,
    'n_jobs': -1 
}

# ===================================================================
# Statistical Tests (DM & PT)
# ===================================================================

def diebold_mariano_test(real, pred1, pred2, h=1, metric='mse'):
    """
    Calculates the Diebold-Mariano test statistic for predictive accuracy.
    
    Args:
        metric (str): 'mse' for Mean Squared Error, 'direction' for Zero-One Loss.
        
    Interpretation:
        d = Loss(Model1) - Loss(Model2)
        Negative stat -> Model 1 has lower loss (Better).
        Positive stat -> Model 2 has lower loss (Better).
    """
    real = np.array(real)
    pred1 = np.array(pred1)
    pred2 = np.array(pred2)
    
    if metric == 'mse':
        e1 = (real - pred1)**2
        e2 = (real - pred2)**2
    elif metric == 'direction':
        # Zero-One Loss: 1 if prediction sign != real sign, 0 otherwise
        # Matches PT test logic: > 0 is positive, else non-positive
        real_sign = (real > 0).astype(int)
        p1_sign = (pred1 > 0).astype(int)
        p2_sign = (pred2 > 0).astype(int)
        
        e1 = (real_sign != p1_sign).astype(int) # Loss 1
        e2 = (real_sign != p2_sign).astype(int) # Loss 2
    else:
        raise ValueError("Unknown metric. Use 'mse' or 'direction'")
    
    d = e1 - e2  # Loss differential
    mean_d = np.mean(d)
    T = len(d)
    
    # Autocovariance at lag 0
    gamma0 = np.var(d)
    
    if gamma0 == 0:
        return np.nan, np.nan
    
    # Simple DM implementation for h=1:
    dm_stat = mean_d / np.sqrt(gamma0 / T)
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat))) # Two-tailed
    
    return dm_stat, p_value

def pesaran_timmermann_test(real, pred):
    """
    Calculates the Pesaran-Timmermann (PT) test statistic for directional accuracy.
    H0: Independence between actual and predicted signs.
    """
    real = np.array(real)
    pred = np.array(pred)
    
    # Binary Direction: 1 if > 0, 0 otherwise (or based on changes)
    y = (real > 0).astype(int)
    y_hat = (pred > 0).astype(int)
    
    T = len(y)
    P = np.mean(y == y_hat) # Observed accuracy (Proportion correctly predicted)
    
    py = np.mean(y)      # Proportion of actual positives
    py_hat = np.mean(y_hat) # Proportion of predicted positives
    
    # Expected accuracy under independence (P_star)
    P_star = py * py_hat + (1 - py) * (1 - py_hat)
    
    # Variance of P (Standard Error)
    var_P = (P_star * (1 - P_star)) / T
    
    # Avoid division by zero
    if var_P == 0:
        return np.nan, np.nan, P
        
    pt_stat = (P - P_star) / np.sqrt(var_P)
    p_value = 2 * (1 - stats.norm.cdf(abs(pt_stat))) # Two-tailed
    
    return pt_stat, p_value, P # Return Stat, p-val, and Accuracy %

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
        # print(f"  GARCH fit warning: {e}")
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
    """
    Calculates PCA factor avoiding look-ahead bias.
    Fits params on history (start to yesterday), applies to today.
    """
    current_date = pd.to_datetime(current_date)
    yesterday = current_date - pd.Timedelta(days=1)
    
    # 1. Gather Raw Data
    df_list = [data_dict[coin][var] for coin in coins if coin in data_dict and var in data_dict[coin].columns]
    if not df_list: return pd.Series(dtype=float, name=factor_name)
    
    # Raw Data Matrix
    raw_df = pd.concat(df_list, axis=1, keys=coins, join='inner').dropna()
    
    # 2. Strict Train/Test Split
    train_data = raw_df.loc[START_DATE:yesterday]
    test_data = raw_df.loc[[current_date]] # Keep as DataFrame (1 row)
    
    if train_data.empty: return pd.Series(dtype=float, name=factor_name)

    # 3. Winsorize (Manual calculation to avoid leakage)
    lower_limit = train_data.quantile(WIN_LIMITS[0])
    upper_limit = train_data.quantile(1 - WIN_LIMITS[1])
    
    train_data = train_data.clip(lower=lower_limit, upper=upper_limit, axis=1)
    test_data = test_data.clip(lower=lower_limit, upper=upper_limit, axis=1)

    # 4. Standard Scaler
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_data) 
    
    if not test_data.empty:
        test_scaled = scaler.transform(test_data)
    else:
        test_scaled = np.empty((0, train_data.shape[1]))

    # 5. PCA
    pca = PCA(n_components=1)
    train_factor = pca.fit_transform(train_scaled) 
    
    if not test_data.empty:
        test_factor = pca.transform(test_scaled)
    else:
        test_factor = np.array([])

    # 6. Reconstruct Series
    full_index = train_data.index.union(test_data.index)
    full_values = np.concatenate([train_factor.ravel(), test_factor.ravel()])
    
    return pd.Series(full_values, index=full_index, name=factor_name)

def generate_factors_window(data_dict, current_end_date):
    """Generates all factors for the window ending at current_end_date."""
    f = {}
    f["Stable_Volume"] = calculate_pca_for_window(stablecoins, "LogVolChange", data_dict, "PC1_Stable_Volume", current_end_date)
    f["Stable_Volatility"] = calculate_pca_for_window(stablecoins, VOLATILITY, data_dict, "PC1_Stable_Volatility", current_end_date)
    f["Stable_Returns"] = calculate_pca_for_window(stablecoins, "Log Returns", data_dict, "PC1_Stable_Returns", current_end_date)
    f["Crypto_Returns"] = calculate_pca_for_window(cryptos, "Log Returns", data_dict, "PC1_Crypto_Returns", current_end_date)
    f["Crypto_Volatility"] = calculate_pca_for_window(cryptos, VOLATILITY, data_dict, "PC1_Crypto_Volatility", current_end_date)
    f["Crypto_Volume"] = calculate_pca_for_window(cryptos, "LogVolChange", data_dict, "PC1_Crypto_Volume", current_end_date)
    return f

# ===================================================================
# ML Feature Creation Functions
# ===================================================================

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

# --- Initial Factors for Tuning ---
# We compute factors up to TRAIN_END_DATE for the initial hyperparameter tuning
print("Calculating Initial PCA Factors for Hyperparameter Tuning...")
initial_factors = generate_factors_window(coin_data, TRAIN_END_DATE)

print("-" * 50)
print(f"Running Out-of-Sample ML Backtests (EXPANDING PCA)")
print(f"Source: {SOURCE_FACTOR}, Target: {TARGET_FACTOR}, Lags: {SPECIFIC_LAGS}")
print("-" * 50)

ml_results = []
source_key = SOURCE_FACTOR
target_key = TARGET_FACTOR
specific_lags = SPECIFIC_LAGS

# Extract Initial Series
y_series_init = initial_factors[target_key].dropna()
x_series_init = initial_factors[source_key].dropna()
common_idx_init = y_series_init.index.intersection(x_series_init.index)
y_series_init = y_series_init.loc[common_idx_init]
x_series_init = x_series_init.loc[common_idx_init]

# Check Data Sufficiency
if len(y_series_init) < 50:
    print(f"  Skipping test: Not enough data.")
    exit()

print(f"  Finding initial ARMA orders for GARCH filtering...")
fixed_arma_order_y = select_best_arma(y_series_init, max_order=MAX_ARMA_ORDER)
fixed_arma_order_x = select_best_arma(x_series_init, max_order=MAX_ARMA_ORDER)

# Initial GARCH Fit (already winsorized inside generate_factors_window)
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

# Feature Creation for Tuning
X_train_initial_bench_g, y_train_initial_bench_g, X_train_initial_chall_g, y_train_initial_chall_g = create_garch_features(
    u_target_series=u_target_full, u_vol_series=u_vol_full, specific_lags=specific_lags,   
    v_resid_series=v_resid_full, v_vol_series=v_vol_full        
)
X_train_initial_bench_v, y_train_initial_bench_v, X_train_initial_chall_v, y_train_initial_chall_v = create_vol_on_garch_features(
    y_vol_series=u_vol_full, u_shock_series=u_target_full, v_shock_series=v_resid_full, v_vol_series=v_vol_full, specific_lags=specific_lags
)

if X_train_initial_bench_g.empty or X_train_initial_chall_g.empty:
    print("  Skipping tuning: Not enough initial data.")
    exit()

# Tune Hyperparameters
tscv = TimeSeriesSplit(n_splits=5)

print("  Tuning Benchmark GARCH Model...")
grid_search_benchmark_garch = GridSearchCV(xgb.XGBRegressor(**XGB_PARAMS), PARAM_GRID, scoring='neg_mean_squared_error', cv=tscv, n_jobs=-1)
grid_search_benchmark_garch.fit(X_train_initial_bench_g, y_train_initial_bench_g)
best_params_benchmark_garch = {**XGB_PARAMS, **grid_search_benchmark_garch.best_params_}

print("  Tuning Challenger GARCH Model...")
grid_search_challenger_garch = GridSearchCV(xgb.XGBRegressor(**XGB_PARAMS), PARAM_GRID, scoring='neg_mean_squared_error', cv=tscv, n_jobs=-1)
grid_search_challenger_garch.fit(X_train_initial_chall_g, y_train_initial_bench_g)
best_params_challenger_garch = {**XGB_PARAMS, **grid_search_challenger_garch.best_params_}

print("  Tuning Benchmark VOL Model...")
grid_search_benchmark_vol = GridSearchCV(xgb.XGBRegressor(**XGB_PARAMS), PARAM_GRID, scoring='neg_mean_squared_error', cv=tscv, n_jobs=-1)
grid_search_benchmark_vol.fit(X_train_initial_bench_v, y_train_initial_bench_v)
best_params_benchmark_vol = {**XGB_PARAMS, **grid_search_benchmark_vol.best_params_}

print("  Tuning Challenger VOL Model...")
grid_search_challenger_vol = GridSearchCV(xgb.XGBRegressor(**XGB_PARAMS), PARAM_GRID, scoring='neg_mean_squared_error', cv=tscv, n_jobs=-1)
grid_search_challenger_vol.fit(X_train_initial_chall_v, y_train_initial_bench_v)
best_params_challenger_vol = {**XGB_PARAMS, **grid_search_challenger_vol.best_params_}


# Feature Importance Plots
try:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))
    # Garch Plot
    model_chall_garch = grid_search_challenger_garch.best_estimator_
    imp_g = pd.DataFrame({'Feature': model_chall_garch.feature_names_in_, 'Importance': model_chall_garch.feature_importances_}).sort_values('Importance')
    ax1.barh(imp_g['Feature'], imp_g['Importance'], color='skyblue')
    ax1.set_title(f'Test 1: Challenger GARCH\n{SOURCE_FACTOR} -> {TARGET_FACTOR}')
    # Vol Plot
    model_chall_vol = grid_search_challenger_vol.best_estimator_
    imp_v = pd.DataFrame({'Feature': model_chall_vol.feature_names_in_, 'Importance': model_chall_vol.feature_importances_}).sort_values('Importance')
    ax2.barh(imp_v['Feature'], imp_v['Importance'], color='salmon')
    ax2.set_title(f'Test 2: Challenger VOL\n{SOURCE_FACTOR} -> {TARGET_FACTOR}')
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"gain_{SOURCE_FACTOR}_to_{TARGET_FACTOR}.png")
    plt.close(fig)
except Exception: pass


oos_predictions_garch = []
oos_predictions_vol = []

# Define Test Dates
full_dates = coin_data[cryptos[0]].index
test_dates = full_dates[(full_dates > TRAIN_END_DATE) & (full_dates <= FULL_END_DATE)]

print(f"  Running EXPANDING PCA and Expanding Window Forecast for {len(test_dates)} points...")

for current_date in tqdm(test_dates):
    yesterday = current_date - pd.Timedelta(days=1)
    
    # --- CRITICAL: RE-RUN PCA ON EXPANDING WINDOW ---
    current_factors_full = generate_factors_window(coin_data, current_date)
    
    y_series_now = current_factors_full[target_key].dropna()
    x_series_now = current_factors_full[source_key].dropna()
    
    # 1. Define Training Data (Up to Yesterday)
    train_y = y_series_now.loc[:yesterday]
    train_x = x_series_now.loc[:yesterday]
    
    if current_date not in y_series_now.index: continue
    actual_y_value = y_series_now.loc[current_date]

    # 2. Re-fit GARCH on Expanding Window
    model_y = fit_best_garch(train_y, p=MAX_GARCH_ORDER, q=MAX_GARCH_ORDER,
                             mean_p=fixed_arma_order_y[0], mean_q=fixed_arma_order_y[1], dist=GARCH_DIST)
    model_x = fit_best_garch(train_x, p=MAX_GARCH_ORDER, q=MAX_GARCH_ORDER,
                             mean_p=fixed_arma_order_x[0], mean_q=fixed_arma_order_x[1], dist=GARCH_DIST)
    
    if model_y is None or model_x is None: continue

    # 3. Forecast Next Step Parameters (GARCH)
    mu_next, vol_next, nu_next, _ = get_oos_forecast_params(model_y, np.nan) 
    
    # 4. Transform Histories (PIT)
    u_target_exp = transform_to_uniform(model_y).clip(1e-6, 1-1e-6)
    u_vol_exp = get_conditional_volatility(model_y)
    v_resid_exp = transform_to_uniform(model_x).clip(1e-6, 1-1e-6)
    v_vol_exp = get_conditional_volatility(model_x)
    
    u_target_exp.replace([np.inf, -np.inf], np.nan, inplace=True)
    v_resid_exp.replace([np.inf, -np.inf], np.nan, inplace=True)

    # ---------------- TEST 1: GARCH PREDICTION ----------------
    X_bench_g, y_bench_g, X_chall_g, y_chall_g = create_garch_features(
        u_target_exp, u_vol_exp, specific_lags, v_resid_exp, v_vol_exp
    )
    
    if not X_bench_g.empty and not X_chall_g.empty:
        # Fit XGBoost on history
        idx_g = X_bench_g.index.intersection(X_chall_g.index)
        model_bench_g = xgb.XGBRegressor(**best_params_benchmark_garch).fit(X_bench_g.loc[idx_g], y_bench_g.loc[idx_g])
        model_chall_g = xgb.XGBRegressor(**best_params_challenger_garch).fit(X_chall_g.loc[idx_g], y_chall_g.loc[idx_g])
        
        # Create Feature Vector for T+1 (Append dummies and shift)
        u_target_temp = pd.concat([u_target_exp, pd.Series([0], index=[current_date])])
        u_vol_temp = pd.concat([u_vol_exp, pd.Series([vol_next], index=[current_date])])
        v_resid_temp = pd.concat([v_resid_exp, pd.Series([0], index=[current_date])])
        v_vol_temp = pd.concat([v_vol_exp, pd.Series([0], index=[current_date])]) 
        
        X_b_next, _, X_c_next, _ = create_garch_features(u_target_temp, u_vol_temp, specific_lags, v_resid_temp, v_vol_temp)
        
        # Predict Uniform Shock
        pred_bench_u = model_bench_g.predict(X_b_next.iloc[[-1]])[0]
        pred_chall_u = model_chall_g.predict(X_c_next.iloc[[-1]])[0]
        
        # Invert PIT (Uniform -> Std -> Raw)
        dist_cls = model_y.model.distribution
        d_params = [model_y.params[n] for n in dist_cls.parameter_names()]
        std_bench = dist_cls.ppf(np.clip(pred_bench_u, 1e-6, 1-1e-6), d_params)
        std_chall = dist_cls.ppf(np.clip(pred_chall_u, 1e-6, 1-1e-6), d_params)
        
        final_pred_bench_g = mu_next + vol_next * std_bench
        final_pred_chall_g = mu_next + vol_next * std_chall
        
        oos_predictions_garch.append({
            'Date': current_date,
            'Actual': actual_y_value,
            'Pred_Bench': final_pred_bench_g,
            'Pred_Chall': final_pred_chall_g
        })

    # ---------------- TEST 2: VOL PREDICTION ----------------
    X_bench_v, y_bench_v, X_chall_v, y_chall_v = create_vol_on_garch_features(
        u_vol_exp, u_target_exp, v_resid_exp, v_vol_exp, specific_lags
    )

    if not X_bench_v.empty and not X_chall_v.empty:
        idx_v = X_bench_v.index.intersection(X_chall_v.index)
        model_bench_v = xgb.XGBRegressor(**best_params_benchmark_vol).fit(X_bench_v.loc[idx_v], y_bench_v.loc[idx_v])
        model_chall_v = xgb.XGBRegressor(**best_params_challenger_vol).fit(X_chall_v.loc[idx_v], y_chall_v.loc[idx_v])
        
        # Create Feature Vector
        u_vol_temp = pd.concat([u_vol_exp, pd.Series([vol_next], index=[current_date])])
        u_target_temp = pd.concat([u_target_exp, pd.Series([0], index=[current_date])])
        v_resid_temp = pd.concat([v_resid_exp, pd.Series([0], index=[current_date])])
        v_vol_temp = pd.concat([v_vol_exp, pd.Series([0], index=[current_date])])
        
        X_b_next_v, _, X_c_next_v, _ = create_vol_on_garch_features(u_vol_temp, u_target_temp, v_resid_temp, v_vol_temp, specific_lags)
        
        pred_bench_v = model_bench_v.predict(X_b_next_v.iloc[[-1]])[0]
        pred_chall_v = model_chall_v.predict(X_c_next_v.iloc[[-1]])[0]
        
        oos_predictions_vol.append({
            'Date': current_date,
            'Pred_Bench_Vol': pred_bench_v,
            'Pred_Chall_Vol': pred_chall_v,
            'GARCH_Vol_Proxy': vol_next
        })

print("\n" + "="*50)
print("RESULTS SUMMARY")
print("="*50)

summary_rows = []

# --- 1. GARCH Results ---
if oos_predictions_garch:
    df_res_g = pd.DataFrame(oos_predictions_garch).set_index('Date')
    mse_b_g = mean_squared_error(df_res_g['Actual'], df_res_g['Pred_Bench'])
    mse_c_g = mean_squared_error(df_res_g['Actual'], df_res_g['Pred_Chall'])
    
    # 1. Diebold-Mariano Test (MSE)
    dm_stat_g, dm_p_g = diebold_mariano_test(
        df_res_g['Actual'], df_res_g['Pred_Bench'], df_res_g['Pred_Chall'], metric='mse'
    )
    
    # 2. Pesaran-Timmermann Test (Directional Accuracy)
    pt_stat_b, pt_p_b, acc_b = pesaran_timmermann_test(df_res_g['Actual'], df_res_g['Pred_Bench'])
    pt_stat_c, pt_p_c, acc_c = pesaran_timmermann_test(df_res_g['Actual'], df_res_g['Pred_Chall'])

    # 3. Diebold-Mariano Test (Directional / Zero-One Loss)
    # Tests if (Challenger Loss - Benchmark Loss) is significant
    dm_stat_dir, dm_p_dir = diebold_mariano_test(
        df_res_g['Actual'], df_res_g['Pred_Bench'], df_res_g['Pred_Chall'], metric='direction'
    )
    
    print(f"\n[TEST 1: GARCH Model (Shock Prediction)]")
    print(f"  MSE Benchmark:  {mse_b_g:.6f}")
    print(f"  MSE Challenger: {mse_c_g:.6f}")
    print(f"  DM Stat (MSE): {dm_stat_g:.4f} (p-val: {dm_p_g:.4f})")
    print("-" * 30)
    print(f"  Benchmark Directional Acc: {acc_b:.2%} (PT Stat: {pt_stat_b:.2f})")
    print(f"  Challenger Directional Acc: {acc_c:.2%} (PT Stat: {pt_stat_c:.2f})")
    print(f"  DM Stat (Directional): {dm_stat_dir:.4f} (p-val: {dm_p_dir:.4f})")
    
    df_res_g.to_csv(OUTPUT_DIR / f"OOS_Res_GARCH_{SOURCE_FACTOR}_to_{TARGET_FACTOR}.csv")

    summary_rows.append({
        'Test': 'GARCH',
        'Source': SOURCE_FACTOR,
        'Target': TARGET_FACTOR,
        'OOS_Days': len(df_res_g),
        'MSE_Benchmark': mse_b_g,
        'MSE_Challenger': mse_c_g,
        'DM_MSE_Stat': dm_stat_g,
        'DM_MSE_P_Value': dm_p_g,
        'Benchmark_Directional_Acc': acc_b,
        'Challenger_Directional_Acc': acc_c,
        'DM_Directional_Stat': dm_stat_dir,
        'DM_Directional_P_Value': dm_p_dir
    })

# --- 2. VOL Results ---
if oos_predictions_vol:
    df_res_v = pd.DataFrame(oos_predictions_vol).set_index('Date').clip(lower=0)
    
    mse_b_v = mean_squared_error(df_res_v['GARCH_Vol_Proxy'], df_res_v['Pred_Bench_Vol'])
    mse_c_v = mean_squared_error(df_res_v['GARCH_Vol_Proxy'], df_res_v['Pred_Chall_Vol'])
    dm_stat_v, dm_p_v = diebold_mariano_test(
        df_res_v['GARCH_Vol_Proxy'], df_res_v['Pred_Bench_Vol'], df_res_v['Pred_Chall_Vol'], metric='mse'
    )

    print(f"\n[TEST 2: VOL Model (Volatility Prediction)]")
    print(f"  MSE Benchmark:  {mse_b_v:.6f}")
    print(f"  MSE Challenger: {mse_c_v:.6f}")
    print(f"  DM Stat (MSE): {dm_stat_v:.4f} (p-val: {dm_p_v:.4f})")
    
    df_res_v.to_csv(OUTPUT_DIR / f"OOS_Res_VOL_{SOURCE_FACTOR}_to_{TARGET_FACTOR}.csv")

    summary_rows.append({
        'Test': 'VOL',
        'Source': SOURCE_FACTOR,
        'Target': TARGET_FACTOR,
        'OOS_Days': len(df_res_v),
        'MSE_Benchmark': mse_b_v,
        'MSE_Challenger': mse_c_v,
        'DM_MSE_Stat': dm_stat_v,
        'DM_MSE_P_Value': dm_p_v,
        'Benchmark_Directional_Acc': np.nan, 
        'Challenger_Directional_Acc': np.nan,
        'DM_Directional_Stat': np.nan,
        'DM_Directional_P_Value': np.nan
    })

# --- 3. Save Final Summary Table ---
if summary_rows:
    summary_df = pd.DataFrame(summary_rows)
    
    cols = [
        'Test', 'Source', 'Target', 'OOS_Days', 
        'MSE_Benchmark', 'MSE_Challenger', 
        'DM_MSE_Stat', 'DM_MSE_P_Value', 
        'Benchmark_Directional_Acc', 
        'Challenger_Directional_Acc',
        'DM_Directional_Stat', 'DM_Directional_P_Value'
    ]
    summary_df = summary_df[cols]
    
    summary_filename = OUTPUT_DIR / f"Final_Summary_{SOURCE_FACTOR}_to_{TARGET_FACTOR}.csv"
    summary_df.to_csv(summary_filename, index=False)
    print(f"\nFinal summary table saved to: {summary_filename}")
    print(summary_df)

print("\nDone.")