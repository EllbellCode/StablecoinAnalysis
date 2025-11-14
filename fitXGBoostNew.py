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

warnings.filterwarnings('ignore')

# --- OOS Hyperparameter Tuning: Define Grid (Smaller Version) ---
PARAM_GRID = {
    'n_estimators': [50, 100],  #Number of trees              
    'learning_rate': [0.05, 0.1],  #Learning Rate           
    'max_depth': [3, 4, 5],  #Max depth of each tree                   
    'subsample': [0.7, 0.8], # The fraction of rows randomly selected from our dataset for each tree                  
    'colsample_bytree': [0.7, 0.8], # The fraction of columns randomly selected from our dataset for each tree              
    'reg_alpha': [0, 0.05], #L1 Regularisation                  
    'reg_lambda': [0.5, 1.5] #L2 Regularisation                 
}

# --- Constants / Settings ---
DATA_DIR = Path("Data/Verified")
OUTPUT_DIR = Path("Results/ML")
PLOT_DIR = Path("Plots/ML")
START_DATE = '2020-01-01'
TRAIN_END_DATE = '2024-01-01'
FULL_END_DATE = '2025-01-01'
MAX_ARMA_ORDER = 1
MAX_GARCH_ORDER = 1
GARCH_DIST = 'skewt' #should always be skewt
GARCH_MODEL = 'EGARCH' #should always be egarch
SPECIFIC_LAGS = [1, 7, 30]
DAYS_NUMBER = 366
#
# Define the single pair to test
SOURCE_FACTOR = "Crypto_Volatility"
TARGET_FACTOR = "Stable_Volatility"
# --- *** END NEW SETTINGS *** ---


XGB_PARAMS = { # Basic XGBoost parameters
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'random_state': 123,
    'n_jobs': -1 # Use all CPU cores
}

# --- SHARPE RATIO SETTINGS ---
# --- Test 1 (GARCH World) ---
TRADING_THRESHOLD_LONG = 0.6 # Go long if predicted shock > 0.6
TRADING_THRESHOLD_SHORT = 0.4 # Go short if predicted shock < 0.4
# --- Test 2 (Raw World) ---
# For Test 2, we use sign-based logic (Long if > 0, Short if < 0)
#
RISK_FREE_RATE = 0.0 # Assume risk-free rate is 0

# ===================================================================
# Helper Functions (GARCH, PCA, etc. - Mostly unchanged)
# ===================================================================

def get_oos_forecast_params(fitted_model, actual_value):
    """
    Calculates 1-step-ahead forecast parameters (mu, sigma, nu)
    and the OOS uniform shock (U). Needed to get the actual shock U_c,t.
    """
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
        return None

def get_conditional_volatility(model_result):
    """Extracts conditional volatility (sigma^2) from a fitted GARCH model."""
    if model_result is None: return pd.Series(dtype=float)
    return model_result.conditional_volatility


def transform_to_uniform(model_result):
    """
    Extracts standardized residuals from a fitted GARCH model
    and transforms them to uniform [0,1] using the model's fitted distribution.
    """
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

# --- PCA function (Unchanged) ---
def create_pca_factor(coins, var, data_dict, factor_name, train_end_date):
    df_list = [data_dict[coin][var] for coin in coins if coin in data_dict and var in data_dict[coin].columns]
    if not df_list: return pd.Series(dtype=float, name=factor_name)
    data_matrix = pd.concat(df_list, axis=1, keys=coins, join='inner').dropna()
    if data_matrix.empty: return pd.Series(dtype=float, name=factor_name)
    train_matrix = data_matrix.loc[START_DATE:train_end_date]
    test_matrix = data_matrix.loc[train_end_date:].iloc[1:]
    if train_matrix.empty or test_matrix.empty: return pd.Series(dtype=float, name=factor_name)
    scaler = StandardScaler(); train_scaled = scaler.fit_transform(train_matrix)
    pca = PCA(n_components=1); train_factor_scaled = pca.fit_transform(train_scaled)
    print(f"Created Factor '{factor_name}'. PCA Explained Variance: {pca.explained_variance_ratio_[0]:.2%}")
    loadings_dict = {coin: loading for coin, loading in zip(train_matrix.columns, pca.components_[0])}
    print(f"  Loadings: {loadings_dict}\n")
    test_scaled = scaler.transform(test_matrix)
    test_factor_scaled = pca.transform(test_scaled)
    train_series = pd.Series(train_factor_scaled.ravel(), index=train_matrix.index, name=factor_name)
    test_series = pd.Series(test_factor_scaled.ravel(), index=test_matrix.index, name=factor_name)
    return pd.concat([train_series, test_series])

# ===================================================================
# *** ML Feature Creation Functions ***
# ===================================================================

def create_garch_features(u_target_series, u_vol_series, specific_lags, v_resid_series=None, v_vol_series=None):
    """
    Creates lagged features for TEST 1 (GARCH World).
    This builds features for BENCHMARK_GARCH and CHALLENGER_GARCH models.
    """
    df = pd.DataFrame({'u_target': u_target_series})
    df['u_vol'] = u_vol_series # Add target's volatility
    
    # Add lagged target shock features
    for i in specific_lags:
        df[f'u_target_lag_{i}'] = df['u_target'].shift(i)
    # Add lagged target volatility features
    for i in specific_lags:
        df[f'u_vol_lag_{i}'] = df['u_vol'].shift(i)
        
    # Add lagged source features if provided (for challenger model)
    if v_resid_series is not None and v_vol_series is not None:
        df['v_resid'] = v_resid_series
        df['v_vol'] = v_vol_series
        # Add lagged source shock features
        for i in specific_lags:
            df[f'v_resid_lag_{i}'] = df['v_resid'].shift(i)
        # Add lagged source volatility features
        for i in specific_lags:
            df[f'v_vol_lag_{i}'] = df['v_vol'].shift(i)
            
        # Add MA features
        df['v_resid_ma_7_lag_1'] = df['v_resid'].rolling(window=7).mean().shift(1)
        df['v_resid_ma_30_lag_1'] = df['v_resid'].rolling(window=30).mean().shift(1)
        df['v_vol_ma_7_lag_1'] = df['v_vol'].rolling(window=7).mean().shift(1)
        df['v_vol_ma_30_lag_1'] = df['v_vol'].rolling(window=30).mean().shift(1)
        
        # Drop the current (non-lagged) source values
        df = df.drop(columns=['v_resid', 'v_vol']) 

    # Drop the current (non-lagged) target volatility
    df = df.drop(columns='u_vol')
    
    # --- SPLIT BENCHMARK AND CHALLENGER ---
    bench_cols = [col for col in df.columns if col.startswith('u_') and col != 'u_target']
    
    # Create Benchmark DataFrame
    df_bench = df[['u_target'] + bench_cols].dropna()
    y_bench = df_bench['u_target']
    X_bench = df_bench.drop(columns='u_target')
    
    # Create Challenger GARCH DataFrame (if source was provided)
    if v_resid_series is not None:
        df_chall_garch = df.dropna() # Drop all NaNs (from MAs, etc.)
        y_chall_garch = df_chall_garch['u_target']
        X_chall_garch = df_chall_garch.drop(columns='u_target')
    else:
        X_chall_garch, y_chall_garch = (None, None)

    return X_bench, y_bench, X_chall_garch, y_chall_garch


# --- *** REMOVED create_raw_on_garch_features function *** ---


# --- *** NEW FUNCTION for TEST 2 *** ---
def create_raw_on_raw_features(y_raw_series, x_raw_series, specific_lags):
    """
    Creates lagged features for TEST 2 (RAW World).
    This builds features for BENCHMARK_RAW and CHALLENGER_RAW models.
    """
    df = pd.DataFrame({'raw_target': y_raw_series})
    df['raw_source'] = x_raw_series
    
    # --- Add features for BENCHMARK_RAW ---
    # Add lagged target raw features
    for i in specific_lags:
        df[f'raw_target_lag_{i}'] = df['raw_target'].shift(i)
    # Add MA features on the raw target
    df['raw_target_ma_7_lag_1'] = df['raw_target'].rolling(window=7).mean().shift(1)
    df['raw_target_ma_30_lag_1'] = df['raw_target'].rolling(window=30).mean().shift(1)

    # --- Add features for CHALLENGER_RAW ---
    # Add lagged source raw features
    for i in specific_lags:
        df[f'raw_source_lag_{i}'] = df['raw_source'].shift(i)
    # Add MA features on the raw source
    df['raw_source_ma_7_lag_1'] = df['raw_source'].rolling(window=7).mean().shift(1)
    df['raw_source_ma_30_lag_1'] = df['raw_source'].rolling(window=30).mean().shift(1)

    # Drop the current (non-lagged) values
    df = df.drop(columns=['raw_source'])

    # --- SPLIT BENCHMARK and CHALLENGER ---
    bench_cols = [col for col in df.columns if col.startswith('raw_target_') and col != 'raw_target']
    
    # Create Benchmark_Raw DataFrame
    df_bench_raw = df[['raw_target'] + bench_cols].dropna()
    y_bench_raw = df_bench_raw['raw_target']
    X_bench_raw = df_bench_raw.drop(columns='raw_target')
    
    # Create Challenger_Raw DataFrame
    df_chall_raw = df.dropna() # Drop all NaNs (from MAs, etc.)
    y_chall_raw = df_chall_raw['raw_target']
    X_chall_raw = df_chall_raw.drop(columns='raw_target')

    return X_bench_raw, y_bench_raw, X_chall_raw, y_chall_raw
# --- *** END NEW FUNCTION *** ---


def calculate_sharpe_ratio(returns, risk_free_rate=RISK_FREE_RATE):
    """Calculates annualized Sharpe ratio."""
    excess_returns = returns - risk_free_rate / DAYS_NUMBER
    mean_er = excess_returns.mean()
    std_er = excess_returns.std()
    if std_er == 0: return 0.0 # Avoid division by zero
    sharpe = (mean_er / std_er) * np.sqrt(DAYS_NUMBER) # Annualize
    return sharpe

# ===================================================================
# Main ML Backtest Logic
# ===================================================================

# --- Make sure output directories exist ---
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

print("Creating PCA Factors (Train/Test Split aware)...")
factors = {}
# --- Factors ---
factors["Stable_Volume"] = create_pca_factor(stablecoins, "LogVolChange", coin_data, "PC1_Stable_Volume", TRAIN_END_DATE)
factors["Stable_Volatility"] = create_pca_factor(stablecoins, "Delta_LogRV", coin_data, "PC1_Stable_Volatility", TRAIN_END_DATE)
factors["Crypto_Returns"] = create_pca_factor(cryptos, "Log Returns", coin_data, "PC1_Crypto_Returns", TRAIN_END_DATE)
factors["Crypto_Volatility"] = create_pca_factor(cryptos, "Delta_LogRV", coin_data, "PC1_Crypto_Volatility", TRAIN_END_DATE)


print("-" * 50)
print(f"Running Out-of-Sample ML Backtests")
print(f"Test 1: GARCH World (Predicting Shocks)")
print(f"Test 2: Raw World (Predicting Raw Values)")
print(f"Training up to {TRAIN_END_DATE}, Testing on {TRAIN_END_DATE} to {FULL_END_DATE}")
print(f"Source: {SOURCE_FACTOR}, Target: {TARGET_FACTOR}, Lags: {SPECIFIC_LAGS}")
print("-" * 50)

# --- Define Keys ---
ml_results = []
source_key = SOURCE_FACTOR
target_key = TARGET_FACTOR
specific_lags = SPECIFIC_LAGS
max_lag_needed = max(specific_lags)

print(f"\n======================================================")
print(f"  RUNNING SINGLE BACKTEST: {source_key} -> {target_key}")
print(f"======================================================\n")


# --- Get all factor data ---
y_series_all = factors[target_key].dropna() # Target (e.g., Crypto_Volatility)
x_series_all = factors[source_key].dropna() # Source (e.g., Stable_Volatility)
common_idx = y_series_all.index.intersection(x_series_all.index)
y_series_all = y_series_all.loc[common_idx] # This is the RAW target series
x_series_all = x_series_all.loc[common_idx] # This is the RAW source series

# --- Determine initial training set and test dates ---
initial_train_y_raw = y_series_all.loc[:TRAIN_END_DATE]
initial_train_x_raw = x_series_all.loc[:TRAIN_END_DATE]
test_dates = y_series_all.loc[TRAIN_END_DATE:].iloc[1:].index

# --- Check for sufficient data ---
min_obs = max(max_lag_needed, 30) + 50 # Use 30 for MA(30)
if len(initial_train_y_raw) < min_obs or len(test_dates) < 50:
    print(f"  Skipping test: Not enough data for lags {specific_lags} and MA features.")
    print(f"  (Need {min_obs} training obs, have {len(initial_train_y_raw)})")
    exit() # Exit the script

# --- Get fixed ARMA orders based *only* on initial training data ---
print(f"  Finding initial ARMA orders for GARCH filtering...")
fixed_arma_order_y = select_best_arma(initial_train_y_raw, max_order=MAX_ARMA_ORDER)
fixed_arma_order_x = select_best_arma(initial_train_x_raw, max_order=MAX_ARMA_ORDER)

print("  Fitting initial GARCH models to get full residual & vol history for tuning...")
# Fit Y model on initial training data
initial_target_garch_fit = fit_best_garch(initial_train_y_raw, p=MAX_GARCH_ORDER, q=MAX_GARCH_ORDER,
                                          mean_p=fixed_arma_order_y[0], mean_q=fixed_arma_order_y[1],
                                          dist=GARCH_DIST)
# Fit X model on initial training data
initial_source_garch_fit = fit_best_garch(initial_train_x_raw, p=MAX_GARCH_ORDER, q=MAX_GARCH_ORDER,
                                          mean_p=fixed_arma_order_x[0], mean_q=fixed_arma_order_x[1],
                                          dist=GARCH_DIST)

if initial_target_garch_fit is None or initial_source_garch_fit is None:
    print("  Skipping test: Failed to fit initial GARCH models.")
    exit() # Exit the script

# --- DEFINE the variables for features/target (Test 1) ---
u_target_full = transform_to_uniform(initial_target_garch_fit).clip(1e-6, 1-1e-6)
u_vol_full = get_conditional_volatility(initial_target_garch_fit)
v_resid_full = transform_to_uniform(initial_source_garch_fit).clip(1e-6, 1-1e-6)
v_vol_full = get_conditional_volatility(initial_source_garch_fit)


# *** Create feature sets for tuning (Test 1) ***
X_train_initial_bench_g, y_train_initial_bench_g, X_train_initial_chall_g, y_train_initial_chall_g = create_garch_features(
    u_target_series=u_target_full, 
    u_vol_series=u_vol_full,     
    specific_lags=specific_lags,   
    v_resid_series=v_resid_full,   
    v_vol_series=v_vol_full        
)

# *** Create feature sets for tuning (Test 2) ***
X_train_initial_bench_r, y_train_initial_bench_r, X_train_initial_chall_r, y_train_initial_chall_r = create_raw_on_raw_features(
    y_raw_series=initial_train_y_raw,
    x_raw_series=initial_train_x_raw,
    specific_lags=specific_lags
)


if X_train_initial_bench_g.empty or X_train_initial_chall_g.empty or X_train_initial_bench_r.empty or X_train_initial_chall_r.empty:
    print("  Skipping tuning: Not enough initial data for features.")
    best_params_benchmark_garch = XGB_PARAMS
    best_params_challenger_garch = XGB_PARAMS
    best_params_benchmark_raw = XGB_PARAMS
    best_params_challenger_raw = XGB_PARAMS
    # We need to create dummy grid_search objects for the plot to fail gracefully
    class DummyEstimator:
        def __init__(self, params): self.best_estimator_ = xgb.XGBRegressor(**params).fit(np.array([[0,1],[1,0]]), np.array([0,1]))
    grid_search_challenger_garch = DummyEstimator(XGB_PARAMS)
    grid_search_challenger_raw = DummyEstimator(XGB_PARAMS)

else:
    # --- Align Test 1 initial training sets ---
    common_initial_idx_garch = X_train_initial_bench_g.index.intersection(
        X_train_initial_chall_g.index
    )
    y_train_initial_g = y_train_initial_bench_g.loc[common_initial_idx_garch]
    X_train_initial_bench_g = X_train_initial_bench_g.loc[common_initial_idx_garch]
    X_train_initial_chall_g = X_train_initial_chall_g.loc[common_initial_idx_garch]
    
    # --- Align Test 2 initial training sets ---
    common_initial_idx_raw = X_train_initial_bench_r.index.intersection(
        X_train_initial_chall_r.index
    )
    y_train_initial_r = y_train_initial_bench_r.loc[common_initial_idx_raw]
    X_train_initial_bench_r = X_train_initial_bench_r.loc[common_initial_idx_raw]
    X_train_initial_chall_r = X_train_initial_chall_r.loc[common_initial_idx_raw]


    # TimeSeriesSplit for cross-validation
    tscv = TimeSeriesSplit(n_splits=5)

    # --- Tune Test 1 (GARCH World) ---
    print("    Tuning Test 1: Benchmark_GARCH...")
    xgb_benchmark_garch_tune = xgb.XGBRegressor(**XGB_PARAMS)
    grid_search_benchmark_garch = GridSearchCV(
        estimator=xgb_benchmark_garch_tune, param_grid=PARAM_GRID,
        scoring='neg_mean_squared_error', cv=tscv, n_jobs=-1, verbose=1
    )
    grid_search_benchmark_garch.fit(X_train_initial_bench_g, y_train_initial_g)
    best_params_benchmark_garch = {**XGB_PARAMS, **grid_search_benchmark_garch.best_params_}
    print(f"    Best Benchmark_GARCH Params: {grid_search_benchmark_garch.best_params_}")

    print("    Tuning Test 1: Challenger_GARCH...")
    xgb_challenger_garch_tune = xgb.XGBRegressor(**XGB_PARAMS)
    grid_search_challenger_garch = GridSearchCV(
        estimator=xgb_challenger_garch_tune, param_grid=PARAM_GRID,
        scoring='neg_mean_squared_error', cv=tscv, n_jobs=-1, verbose=1
    )
    grid_search_challenger_garch.fit(X_train_initial_chall_g, y_train_initial_g)
    best_params_challenger_garch = {**XGB_PARAMS, **grid_search_challenger_garch.best_params_}
    print(f"    Best Challenger_GARCH Params: {grid_search_challenger_garch.best_params_}")

    # --- Tune Test 2 (Raw World) ---
    print("    Tuning Test 2: Benchmark_Raw...")
    xgb_benchmark_raw_tune = xgb.XGBRegressor(**XGB_PARAMS)
    grid_search_benchmark_raw = GridSearchCV(
        estimator=xgb_benchmark_raw_tune, param_grid=PARAM_GRID,
        scoring='neg_mean_squared_error', cv=tscv, n_jobs=-1, verbose=1
    )
    grid_search_benchmark_raw.fit(X_train_initial_bench_r, y_train_initial_r)
    best_params_benchmark_raw = {**XGB_PARAMS, **grid_search_benchmark_raw.best_params_}
    print(f"    Best Benchmark_Raw Params: {grid_search_benchmark_raw.best_params_}")

    print("    Tuning Test 2: Challenger_Raw...")
    xgb_challenger_raw_tune = xgb.XGBRegressor(**XGB_PARAMS)
    grid_search_challenger_raw = GridSearchCV(
        estimator=xgb_challenger_raw_tune, param_grid=PARAM_GRID,
        scoring='neg_mean_squared_error', cv=tscv, n_jobs=-1, verbose=1
    )
    grid_search_challenger_raw.fit(X_train_initial_chall_r, y_train_initial_r)
    best_params_challenger_raw = {**XGB_PARAMS, **grid_search_challenger_raw.best_params_}
    print(f"    Best Challenger_Raw Params: {grid_search_challenger_raw.best_params_}")


# --- START: FEATURE IMPORTANCE PLOT ---
# --- This plot compares the Test 1 Challenger (GARCH) vs Test 2 Challenger (Raw) ---
print("\n" + "="*50)
print("Generating Feature Importance Plots (Test 1 Challenger vs Test 2 Challenger)...")

try:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10)) # 1 row, 2 columns

    # --- Plot 1: Challenger GARCH (from Test 1) ---
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

    # --- Plot 2: Challenger RAW (from Test 2) ---
    model_chall_raw = grid_search_challenger_raw.best_estimator_
    feature_names_raw = model_chall_raw.feature_names_in_
    importances_raw = model_chall_raw.feature_importances_
    
    importance_df_raw = pd.DataFrame({
        'Feature': feature_names_raw,
        'Importance': importances_raw
    }).sort_values(by='Importance', ascending=True)

    ax2.barh(importance_df_raw['Feature'], importance_df_raw['Importance'], color='salmon')
    ax2.set_title(f'Test 2: Challenger RAW (Predicts Raw Values)\n{SOURCE_FACTOR} -> {TARGET_FACTOR}', fontsize=16)
    ax2.set_xlabel('Feature Importance (Gain)', fontsize=12)
    ax2.set_ylabel('Feature', fontsize=12)
    ax2.grid(axis='x', linestyle='--', alpha=0.7)
    
    # --- Save the combined plot ---
    plt.tight_layout()
    chall_plot_filename = f"gain_{SOURCE_FACTOR}_to_{TARGET_FACTOR}.png"
    output_path = PLOT_DIR / chall_plot_filename
    
    plt.savefig(output_path)
    print(f"Saved Challenger importance plot to {output_path}")
    plt.close(fig) # Close the figure

except Exception as e:
    print(f"Could not generate feature importance plot. Error: {e}")

print("="*50 + "\n")
# --- END: FEATURE IMPORTANCE PLOT ---


# --- >>> END OF TUNING SECTION <<< ---

# --- Store OOS Predictions ---
oos_predictions_garch = [] # For Test 1
oos_predictions_raw = []   # For Test 2

print(f"  Running expanding window forecast for {len(test_dates)} test points...")
for current_date in tqdm(test_dates):
    yesterday = current_date - pd.Timedelta(days=1)
    current_train_y_raw = y_series_all.loc[:yesterday] # Raw Target series
    current_train_x_raw = x_series_all.loc[:yesterday] # Raw Source series
    actual_y_raw = y_series_all.loc[current_date] # Raw Target for P&L

    # 1. Fit GARCH models (for Test 1)
    bench_model_fit = fit_best_garch(current_train_y_raw, p=MAX_GARCH_ORDER, q=MAX_GARCH_ORDER,
                                     mean_p=fixed_arma_order_y[0], mean_q=fixed_arma_order_y[1],
                                     dist=GARCH_DIST)
    x_model_fit = fit_best_garch(current_train_x_raw, p=MAX_GARCH_ORDER, q=MAX_GARCH_ORDER,
                                 mean_p=fixed_arma_order_x[0], mean_q=fixed_arma_order_x[1],
                                 dist=GARCH_DIST)
    if bench_model_fit is None or x_model_fit is None: continue
        
    # 2. Create all GARCH-filtered series up to yesterday (for Test 1)
    u_target_is_full = transform_to_uniform(bench_model_fit).clip(1e-6, 1-1e-6)
    u_vol_is_full = get_conditional_volatility(bench_model_fit)
    v_resid_is_full = transform_to_uniform(x_model_fit).clip(1e-6, 1-1e-6)
    v_vol_is_full = get_conditional_volatility(x_model_fit)

    if u_target_is_full.empty: continue

    # 3. Create Lagged Feature Sets for Training (Test 1)
    X_train_bench_g, y_train_bench_g, X_train_chall_g, y_train_chall_g = create_garch_features(
        u_target_is_full, u_vol_is_full, specific_lags, 
        v_resid_is_full, v_vol_is_full
    )
    
    # 4. Create Lagged Feature Sets for Training (Test 2)
    X_train_bench_r, y_train_bench_r, X_train_chall_r, y_train_chall_r = create_raw_on_raw_features(
        current_train_y_raw, current_train_x_raw, specific_lags
    )

    # Check if any feature sets are empty
    if X_train_bench_g.empty or X_train_chall_g.empty or X_train_bench_r.empty or X_train_chall_r.empty:
        continue

    # --- Align Test 1 Training Data ---
    common_train_idx_g = X_train_bench_g.index.intersection(
        X_train_chall_g.index
    )
    if common_train_idx_g.empty: continue
    
    y_train_g = y_train_bench_g.loc[common_train_idx_g]
    X_train_bench_g = X_train_bench_g.loc[common_train_idx_g]
    X_train_chall_g = X_train_chall_g.loc[common_train_idx_g]

    # --- Align Test 2 Training Data ---
    common_train_idx_r = X_train_bench_r.index.intersection(
        X_train_chall_r.index
    )
    if common_train_idx_r.empty: continue

    y_train_r = y_train_bench_r.loc[common_train_idx_r]
    X_train_bench_r = X_train_bench_r.loc[common_train_idx_r]
    X_train_chall_r = X_train_chall_r.loc[common_train_idx_r]

    # 5. Train XGBoost Models (Test 1)
    model_bench_garch = xgb.XGBRegressor(**best_params_benchmark_garch)
    model_chall_garch = xgb.XGBRegressor(**best_params_challenger_garch)
    
    model_bench_garch.fit(X_train_bench_g, y_train_g)
    model_chall_garch.fit(X_train_chall_g, y_train_g)
    
    # 6. Train XGBoost Models (Test 2)
    model_bench_raw = xgb.XGBRegressor(**best_params_benchmark_raw)
    model_chall_raw = xgb.XGBRegressor(**best_params_challenger_raw)
    
    model_bench_raw.fit(X_train_bench_r, y_train_r)
    model_chall_raw.fit(X_train_chall_r, y_train_r)

    # 7. Create Features for Today's Prediction (using data up to t-1)
    
    # Check if history is long enough
    if len(u_target_is_full) < max_lag_needed or len(v_resid_is_full) < 30 or len(current_train_y_raw) < 30: # 30 for MAs
        continue 
        
    try:
        # --- Test 1: GARCH World Prediction Features ---
        u_resid_lags = np.array([u_target_is_full.iloc[-i] for i in specific_lags])
        u_vol_lags = np.array([u_vol_is_full.iloc[-i] for i in specific_lags])
        features_today_bench_g = np.concatenate([u_resid_lags, u_vol_lags])
        bench_g_cols = [f'u_target_lag_{i}' for i in specific_lags] + [f'u_vol_lag_{i}' for i in specific_lags]
        X_pred_bench_g = pd.DataFrame(features_today_bench_g.reshape(1, -1), columns=bench_g_cols)

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
        
        chall_garch_v_cols = [f'v_resid_lag_{i}' for i in specific_lags] + \
                             [f'v_vol_lag_{i}' for i in specific_lags] + \
                             ['v_resid_ma_7_lag_1', 'v_resid_ma_30_lag_1', 'v_vol_ma_7_lag_1', 'v_vol_ma_30_lag_1']
        chall_garch_cols = bench_g_cols + chall_garch_v_cols
        X_pred_chall_garch = pd.DataFrame(features_today_garch, columns=chall_garch_cols)

        #--- Test 2: RAW World Prediction Features ---
        raw_target_lags = np.array([current_train_y_raw.iloc[-i] for i in specific_lags])
        raw_target_ma_7 = current_train_y_raw.rolling(window=7).mean().iloc[-1]
        raw_target_ma_30 = current_train_y_raw.rolling(window=30).mean().iloc[-1]
        
        # --- FIX START ---
        # Create the 1D array of benchmark features first
        features_today_bench_r_1D = np.concatenate([
            raw_target_lags,
            np.array([raw_target_ma_7, raw_target_ma_30])
        ])
        
        # Create the 2D version *only* for the DataFrame
        features_today_bench_r_2D = features_today_bench_r_1D.reshape(1, -1)
        # --- FIX END ---
        
        bench_r_cols = [f'raw_target_lag_{i}' for i in specific_lags] + \
                       ['raw_target_ma_7_lag_1', 'raw_target_ma_30_lag_1']
        
        # Use the 2D version for the benchmark prediction
        X_pred_bench_r = pd.DataFrame(features_today_bench_r_2D, columns=bench_r_cols)

        # Features for Challenger_Raw
        v_raw_lags = np.array([current_train_x_raw.iloc[-i] for i in specific_lags])
        v_raw_ma_7 = current_train_x_raw.rolling(window=7).mean().iloc[-1]
        v_raw_ma_30 = current_train_x_raw.rolling(window=30).mean().iloc[-1]
        
        # This is the 1D array of new features
        features_today_raw_v = np.concatenate([
            v_raw_lags,
            np.array([v_raw_ma_7, v_raw_ma_30])
        ])
        
        # Now concatenate the two 1D arrays
        features_today_chall_r = np.concatenate([features_today_bench_r_1D, features_today_raw_v]).reshape(1, -1)
        
        # --- FIX IS HERE ---
        # The column names must match the training function: 'raw_source_'
        chall_raw_v_cols = [f'raw_source_lag_{i}' for i in specific_lags] + \
                           ['raw_source_ma_7_lag_1', 'raw_source_ma_30_lag_1']
        # --- END FIX ---

        chall_r_cols = bench_r_cols + chall_raw_v_cols
        
        if np.isnan(features_today_chall_r).any() or np.isnan(features_today_bench_r_1D).any(): 
            continue
        X_pred_chall_r = pd.DataFrame(features_today_chall_r, columns=chall_r_cols)

    except IndexError:
        continue # Not enough history for the specific lags

    # 8. Make 1-Step-Ahead Predictions (Test 1)
    pred_bench_g = model_bench_garch.predict(X_pred_bench_g)[0]
    pred_chall_g = model_chall_garch.predict(X_pred_chall_garch)[0]
    
    # 9. Make 1-Step-Ahead Predictions (Test 2)
    pred_bench_r = model_bench_raw.predict(X_pred_bench_r)[0]
    pred_chall_r = model_chall_raw.predict(X_pred_chall_r)[0] # This line was erroring

    # 10. Get Actual Shock for Today (for Test 1 MSE)
    _, _, _, actual_u_c_t = get_oos_forecast_params(bench_model_fit, actual_y_raw)

    if not np.isfinite(actual_u_c_t): continue

    # 11. Store results for Test 1
    oos_predictions_garch.append({
        "Date": current_date,
        "Actual_Shock": actual_u_c_t,
        "Pred_Shock_Bench_G": pred_bench_g,
        "Pred_Shock_Chall_G": pred_chall_g,
        "Actual_Return": actual_y_raw # Store the raw return for Sharpe calculation
    })
    
    # 12. Store results for Test 2
    oos_predictions_raw.append({
        "Date": current_date,
        "Actual_Raw": actual_y_raw,
        "Pred_Raw_Bench": pred_bench_r,
        "Pred_Raw_Chall": pred_chall_r
    })

# --- Evaluate OOS Predictions (after loop) ---
if len(oos_predictions_garch) < 20 or len(oos_predictions_raw) < 20:
    print("  Skipping Evaluation: Not enough valid OOS forecasts generated.")
    exit()

print("  OOS loop finished. Evaluating models...")

# --- Process Test 1 (GARCH World) ---
pred_df_garch = pd.DataFrame(oos_predictions_garch).set_index("Date").dropna()

# 1a. Calculate MSE (Test 1)
mse_bench_g = mean_squared_error(pred_df_garch['Actual_Shock'], pred_df_garch['Pred_Shock_Bench_G'])
mse_chall_g = mean_squared_error(pred_df_garch['Actual_Shock'], pred_df_garch['Pred_Shock_Chall_G'])

# 2a. Calculate Sharpe Ratios (Test 1 - Threshold Logic)
pred_df_garch['Position_Bench_G'] = np.where(pred_df_garch['Pred_Shock_Bench_G'] > TRADING_THRESHOLD_LONG, 1,
                              np.where(pred_df_garch['Pred_Shock_Bench_G'] < TRADING_THRESHOLD_SHORT, -1, 0))
pred_df_garch['Position_Chall_G'] = np.where(pred_df_garch['Pred_Shock_Chall_G'] > TRADING_THRESHOLD_LONG, 1,
                              np.where(pred_df_garch['Pred_Shock_Chall_G'] < TRADING_THRESHOLD_SHORT, -1, 0))

pred_df_garch['Return_Bench_G'] = pred_df_garch['Position_Bench_G'] * pred_df_garch['Actual_Return']
pred_df_garch['Return_Chall_G'] = pred_df_garch['Position_Chall_G'] * pred_df_garch['Actual_Return']

sharpe_bench_g = calculate_sharpe_ratio(pred_df_garch['Return_Bench_G'])
sharpe_chall_g = calculate_sharpe_ratio(pred_df_garch['Return_Chall_G'])


# --- Process Test 2 (Raw World) ---
pred_df_raw = pd.DataFrame(oos_predictions_raw).set_index("Date").dropna()
# Merge in Actual_Return from pred_df_garch for Sharpe calculation
pred_df_raw = pred_df_raw.join(
    pred_df_garch['Actual_Return'],
    how='inner'      # only keep common dates
)
pred_df_raw = pred_df_raw.dropna() # Ensure alignment

# 1b. Calculate MSE (Test 2)
mse_bench_r = mean_squared_error(pred_df_raw['Actual_Raw'], pred_df_raw['Pred_Raw_Bench'])
mse_chall_r = mean_squared_error(pred_df_raw['Actual_Raw'], pred_df_raw['Pred_Raw_Chall'])

# 2b. Calculate Sharpe Ratios (Test 2 - Sign-Based Logic)
pred_df_raw['Position_Bench_R'] = np.where(pred_df_raw['Pred_Raw_Bench'] > 0, 1, -1)
pred_df_raw['Position_Chall_R'] = np.where(pred_df_raw['Pred_Raw_Chall'] > 0, 1, -1)

pred_df_raw['Return_Bench_R'] = pred_df_raw['Position_Bench_R'] * pred_df_raw['Actual_Return']
pred_df_raw['Return_Chall_R'] = pred_df_raw['Position_Chall_R'] * pred_df_raw['Actual_Return']

sharpe_bench_r = calculate_sharpe_ratio(pred_df_raw['Return_Bench_R'])
sharpe_chall_r = calculate_sharpe_ratio(pred_df_raw['Return_Chall_R'])


# --- Combine All Results into two-row format ---
results_garch = {
    "Test": "GARCH",
    "Source": source_key,
    "Target": target_key,
    "MSE Benchmark": mse_bench_g,
    "MSE Challenger": mse_chall_g,
    "Sharpe Benchmark": sharpe_bench_g,
    "Sharpe Challenger": sharpe_chall_g,
    "OOS_Days": len(pred_df_garch)
}
ml_results.append(results_garch)

results_raw = {
    "Test": "RAW",
    "Source": source_key,
    "Target": target_key,
    "MSE Benchmark": mse_bench_r,
    "MSE Challenger": mse_chall_r,
    "Sharpe Benchmark": sharpe_bench_r,
    "Sharpe Challenger": sharpe_chall_r,
    "OOS_Days": len(pred_df_raw)
}
ml_results.append(results_raw)

# --- Dynamic Filename ---
output_slug = f"{source_key}_to_{target_key}"

# Save detailed predictions
preds_g_name = f"XG_preds_GARCH_{output_slug}.csv"
preds_g_path = OUTPUT_DIR / preds_g_name
pred_df_garch.to_csv(preds_g_path)

preds_r_name = f"XG_preds_RAW_{output_slug}.csv"
preds_r_path = OUTPUT_DIR / preds_r_name
pred_df_raw.to_csv(preds_r_path)


# --- Final Output ---
print("\n" + "=" * 50)
print(f"Final OOS ML (XGBoost) Backtest Results")
print(f"{source_key} -> {target_key} (Lags = {specific_lags})")
print("=" * 50)

results_table = pd.DataFrame(ml_results)

# --- Set column order based on user request ---
column_order = [
    "Test", "Source", "Target", "OOS_Days",
    "MSE Benchmark", "MSE Challenger",
    "Sharpe Benchmark", "Sharpe Challenger"
]
# Filter for only columns that exist
column_order = [col for col in column_order if col in results_table.columns]
results_table = results_table[column_order]
print(results_table.to_string(float_format="%.6f"))

# --- MODIFIED: Save new simplified results table ---
results_name = f"XG_results_{output_slug}.csv"
results_path = OUTPUT_DIR / results_name
results_table.to_csv(results_path, index=False)
print(f"\nSummary results saved to '{results_path}'")
print(f"Test 1 (GARCH) predictions saved to '{preds_g_path}'")
print(f"Test 2 (RAW) predictions saved to '{preds_r_path}'")