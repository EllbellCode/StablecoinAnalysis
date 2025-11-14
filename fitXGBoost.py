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

warnings.filterwarnings('ignore')

# --- OOS Hyperparameter Tuning: Define Grid (Smaller Version) ---
PARAM_GRID = {
    'n_estimators': [50],                
    'learning_rate': [0.05],             
    'max_depth': [3],                      
    'subsample': [0.7, 0.8],                  
    'colsample_bytree': [0.7, 0.8],           
    'reg_alpha': [0, 0.05],                   
    'reg_lambda': [0.5, 1.5]                  
}

# We will overwrite n_estimators, learning_rate etc. with the best found params

# --- Constants / Settings ---
DATA_DIR = Path("Data/Verified")
START_DATE = '2020-01-01'
TRAIN_END_DATE = '2024-01-01'
FULL_END_DATE = '2025-01-01'
STATIONARY_VOL = "Delta_LogRV"

# GARCH Settings for Filtering
MAX_ARMA_ORDER = 1
MAX_GARCH_ORDER = 1
GARCH_DIST = 'skewt'

# --- ML SETTINGS ---
N_LAGS = 10 # Number of past lags to use as features
XGB_PARAMS = { # Basic XGBoost parameters (can be tuned further)
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'random_state': 123,
    'n_jobs': -1 # Use all CPU cores
}

# --- SHARPE RATIO SETTINGS ---
TRADING_THRESHOLD_LONG = 0.6 # Go long if predicted shock > 0.6
TRADING_THRESHOLD_SHORT = 0.4 # Go short if predicted shock < 0.4
RISK_FREE_RATE = 0.0 # Assume risk-free rate is 0

# ===================================================================
# Helper Functions (GARCH, PCA, etc. - Mostly unchanged)
# ===================================================================

def get_oos_forecast_params(fitted_model, actual_value):
    """
    Calculates 1-step-ahead forecast parameters (mu, sigma, nu)
    and the OOS uniform shock (U). Needed to get the actual shock U_c,t.
    """
    # 1. Get the 1-step-ahead forecast
    forecast = fitted_model.forecast(horizon=1, reindex=False)
    mean_forecast = forecast.mean.iloc[0, 0]
    var_forecast = forecast.variance.iloc[0, 0]
    scale_forecast = np.sqrt(var_forecast)

    try:
        # 1. Calculate the standardized shock: z_t = (y_t - mu_t) / sigma_t
        std_shock = (actual_value - mean_forecast) / scale_forecast # <-- DEFINITION ADDED HERE

        dist = fitted_model.model.distribution 
        all_params = fitted_model.params

        # Get the names of the distribution parameters (e.g., ['nu', 'lambda'])
        dist_param_names = dist.parameter_names()

        # Extract *only* those parameters from the full model results
        dist_params = [all_params[name] for name in dist_param_names]

        # 3. Apply the CDF of the fitted distribution to the standardized shock.
        # Pass the correct, filtered list of parameters
        uniform_transform = dist.cdf(std_shock, parameters=dist_params)

        # Get nu just for returning it (if it exists)
        nu = all_params.get('nu', np.inf)

        return mean_forecast, scale_forecast, nu, uniform_transform

    except Exception as e:
        # Return NaNs on failure
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
        # Using GARCH for stability
        am = arch_model(series, vol='GARCH', p=p, q=q,
                        mean=mean_model, lags=ar_lags,
                        dist=dist)
        res = am.fit(update_freq=0, disp='off', options={'maxiter': 200})
        return res
    except Exception as e:
        return None


def get_standardized_residuals(model_result):
    """Extracts standardized residuals from a fitted GARCH model."""
    if model_result is None: return pd.Series(dtype=float)
    return model_result.std_resid

# *** NEW FUNCTION ***
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
    
    # Get standardized residuals (z_t)
    std_resid = model_result.std_resid.dropna()
    if std_resid.empty:
        return pd.Series(dtype=float)
    
    # Get the fitted distribution (e.g., 'skewt') and its parameters
    dist = model_result.model.distribution 
    all_params = model_result.params
    
    # Get the names of the distribution parameters (e.g., ['nu', 'lambda'])
    dist_param_names = dist.parameter_names()
        
    # Extract *only* those parameters from the full model results
    dist_params = [all_params[name] for name in dist_param_names]
    
    # Apply the CDF to the residuals
    # Pass the correct, filtered list of parameters
    uniform_shocks = dist.cdf(std_resid, parameters=dist_params)
    
    # Return as a Series with the correct index
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
# *** MODIFIED Helper Function for ML Backtest ***
# ===================================================================

def create_lagged_features(u_target_series, u_vol_series, v_resid_series=None, v_vol_series=None, n_lags=N_LAGS):
    """
    Creates lagged features for ML model.
    Target y_t is the current shock U_c,t (u_target_series).
    Features X_t are lagged shocks and lagged volatilities.
    
    Benchmark Model: Lags of u_target_series, Lags of u_vol_series
    Challenger Model: Lags of u_target_series, Lags of u_vol_series,
                      Lags of v_resid_series, Lags of v_vol_series
                      
    Returns feature DataFrame X and target Series y.
    """
    df = pd.DataFrame({'u_target': u_target_series})
    df['u_vol'] = u_vol_series # Add target's volatility
    
    # --- FIX: Create columns in grouped order to match prediction logic ---
    # Add lagged target shock features
    for i in range(1, n_lags + 1):
        df[f'u_target_lag_{i}'] = df['u_target'].shift(i)
    # Add lagged target volatility features
    for i in range(1, n_lags + 1):
        df[f'u_vol_lag_{i}'] = df['u_vol'].shift(i)
        
    # Add lagged source features if provided (for challenger model)
    if v_resid_series is not None and v_vol_series is not None:
        df['v_resid'] = v_resid_series
        df['v_vol'] = v_vol_series
        # Add lagged source shock features
        for i in range(1, n_lags + 1):
            df[f'v_resid_lag_{i}'] = df['v_resid'].shift(i)
        # Add lagged source volatility features
        for i in range(1, n_lags + 1):
            df[f'v_vol_lag_{i}'] = df['v_vol'].shift(i)
        # Drop the current (non-lagged) source values
        df = df.drop(columns=['v_resid', 'v_vol']) 
    # --- END FIX ---

    # Drop the current (non-lagged) target volatility
    df = df.drop(columns='u_vol')
    
    # Drop rows with NaNs created by lagging
    df.dropna(inplace=True)
    
    # Separate features (X) and target (y)
    y = df['u_target']
    X = df.drop(columns='u_target')
    
    return X, y

def calculate_sharpe_ratio(returns, risk_free_rate=RISK_FREE_RATE):
    """Calculates annualized Sharpe ratio."""
    excess_returns = returns - risk_free_rate / 252 # Daily risk-free rate
    mean_er = excess_returns.mean()
    std_er = excess_returns.std()
    if std_er == 0: return 0.0 # Avoid division by zero
    sharpe = (mean_er / std_er) * np.sqrt(252) # Annualize
    return sharpe

# ===================================================================
# Main ML Backtest Logic
# ===================================================================

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
# --- Factors --- (Assumed correct creation)
factors["Stable_Volume"] = create_pca_factor(stablecoins, "LogVolChange", coin_data, "PC1_Stable_Volume", TRAIN_END_DATE)
factors["Stable_Volatility"] = create_pca_factor(stablecoins, STATIONARY_VOL, coin_data, "PC1_Stable_Volatility", TRAIN_END_DATE)
factors["Crypto_Returns"] = create_pca_factor(cryptos, "Log Returns", coin_data, "PC1_Crypto_Returns", TRAIN_END_DATE)
factors["Crypto_Volatility"] = create_pca_factor(cryptos, STATIONARY_VOL, coin_data, "PC1_Crypto_Volatility", TRAIN_END_DATE)


print("-" * 50)
print(f"Running Out-of-Sample Copula-ML Backtests (with Volatility Features)")
print(f"Training up to {TRAIN_END_DATE}, Testing on {TRAIN_END_DATE} to {FULL_END_DATE}")
print("-" * 50)

tests_to_run = [
    ("Stable_Volume", "Crypto_Returns"),
    ("Stable_Volume", "Crypto_Volatility"),
    ("Stable_Volatility", "Crypto_Returns"),
    ("Stable_Volatility", "Crypto_Volatility"),
]

ml_results = []

for source_key, target_key in tests_to_run:
    print(f"\n===== ML OOS TEST: {source_key} (lagged) -> {target_key} (t) =====")

    # --- Get all factor data ---
    y_series_all = factors[target_key].dropna()
    x_series_all = factors[source_key].dropna()
    common_idx = y_series_all.index.intersection(x_series_all.index)
    y_series_all = y_series_all.loc[common_idx]
    x_series_all = x_series_all.loc[common_idx]
    
    # --- Determine initial training set and test dates ---
    initial_train_y = y_series_all.loc[:TRAIN_END_DATE]
    initial_train_x = x_series_all.loc[:TRAIN_END_DATE]
    test_dates = y_series_all.loc[TRAIN_END_DATE:].iloc[1:].index
    

    if len(initial_train_y) < (N_LAGS + 50) or len(test_dates) < 50:
        print("  Skipping test: Not enough initial training or test data.")
        continue

    # --- Get fixed ARMA orders based *only* on initial training data ---
    print(f"  Finding initial ARMA orders for GARCH filtering...")
    fixed_arma_order_y = select_best_arma(initial_train_y, max_order=MAX_ARMA_ORDER)
    fixed_arma_order_x = select_best_arma(initial_train_x, max_order=MAX_ARMA_ORDER)

    # --- >>> MODIFIED: FIT INITIAL GARCH MODELS TO GET FULL HISTORY <<< ---
    print("  Fitting initial GARCH models to get full residual & vol history for tuning...")
    
    # Fit Y model on initial training data
    initial_bench_fit = fit_best_garch(initial_train_y, p=MAX_GARCH_ORDER, q=MAX_GARCH_ORDER,
                                       mean_p=fixed_arma_order_y[0], mean_q=fixed_arma_order_y[1],
                                       dist=GARCH_DIST)
    # Fit X model on initial training data
    initial_x_fit = fit_best_garch(initial_train_x, p=MAX_GARCH_ORDER, q=MAX_GARCH_ORDER,
                                   mean_p=fixed_arma_order_x[0], mean_q=fixed_arma_order_x[1],
                                   dist=GARCH_DIST)

    if initial_bench_fit is None or initial_x_fit is None:
        print("  Skipping test: Failed to fit initial GARCH models.")
        continue

    # Extract and Transform full history of residuals
    initial_x_resid = get_standardized_residuals(initial_x_fit)
    
    # *** NEW: Extract volatility history ***
    initial_bench_vol = get_conditional_volatility(initial_bench_fit)
    initial_x_vol = get_conditional_volatility(initial_x_fit)

    # --- DEFINE the variables for features/target ---
    # Target (y) is the uniform shock of the target series
    u_target_full = transform_to_uniform(initial_bench_fit)
    v_resid_full = transform_to_uniform(initial_x_fit)
    
    u_target_full = u_target_full.clip(1e-6, 1-1e-6)
    v_resid_full = v_resid_full.clip(1e-6, 1-1e-6)
    
    
    # Feature 2: Conditional volatility of the target series
    u_vol_full = pd.Series(initial_bench_vol, index=initial_bench_vol.index, name='u_vol')
    
    # Feature 3: Conditional volatility of the source series
    v_vol_full = pd.Series(initial_x_vol, index=initial_x_vol.index, name='v_vol')


    # *** MODIFIED: Call new create_lagged_features function ***
    X_train_initial_chall, y_train_initial_chall = create_lagged_features(
        u_target_full, # Pass the target shock series (y)
        u_vol_full,    # Pass the target volatility series
        v_resid_full,  # Pass the source shock series
        v_vol_full,    # Pass the source volatility series
        n_lags=N_LAGS
    )
    X_train_initial_bench, y_train_initial_bench = create_lagged_features(
        u_target_full, # Pass the target shock series (y)
        u_vol_full,    # Pass the target volatility series
        None,          # No source features for benchmark
        None,
        n_lags=N_LAGS
    )

    if X_train_initial_bench.empty or X_train_initial_chall.empty:
        print("  Skipping tuning: Not enough initial data for features.")
        best_params_challenger = XGB_PARAMS
        best_params_benchmark = XGB_PARAMS
    else:
        # Align the initial training sets before tuning
        common_initial_idx = X_train_initial_chall.index.intersection(X_train_initial_bench.index)
        X_train_initial_chall = X_train_initial_chall.loc[common_initial_idx]
        X_train_initial_bench = X_train_initial_bench.loc[common_initial_idx]
        # Use one common target series after alignment
        y_train_initial = y_train_initial_chall.loc[common_initial_idx] # Or y_train_initial_bench, they should be identical on common_idx

        # TimeSeriesSplit for cross-validation
        tscv = TimeSeriesSplit(n_splits=5)

        # --- Tune Challenger Model ---
        print("    Tuning Challenger model...")
        xgb_challenger_tune = xgb.XGBRegressor(**XGB_PARAMS)
        grid_search_challenger = GridSearchCV(
            estimator=xgb_challenger_tune,
            param_grid=PARAM_GRID,
            scoring='neg_mean_squared_error',
            cv=tscv,
            n_jobs=-1,
            verbose=1
        )
        grid_search_challenger.fit(X_train_initial_chall, y_train_initial)
        best_params_challenger = {**XGB_PARAMS, **grid_search_challenger.best_params_}
        print(f"    Best Challenger Params: {grid_search_challenger.best_params_}")

        # --- Tune Benchmark Model ---
        print("    Tuning Benchmark model...")
        xgb_benchmark_tune = xgb.XGBRegressor(**XGB_PARAMS)
        grid_search_benchmark = GridSearchCV(
            estimator=xgb_benchmark_tune,
            param_grid=PARAM_GRID,
            scoring='neg_mean_squared_error',
            cv=tscv,
            n_jobs=-1,
            verbose=1
        )
        grid_search_benchmark.fit(X_train_initial_bench, y_train_initial)
        best_params_benchmark = {**XGB_PARAMS, **grid_search_benchmark.best_params_}
        print(f"    Best Benchmark Params: {grid_search_benchmark.best_params_}")

    # --- >>> END OF TUNING SECTION <<< ---
    
    # --- Store OOS Predictions ---
    oos_predictions = []

    print(f"  Running expanding window forecast for {len(test_dates)} test points...")
    for current_date in tqdm(test_dates):
        yesterday = current_date - pd.Timedelta(days=1)
        current_train_y_raw = y_series_all.loc[:yesterday]
        current_train_x_raw = x_series_all.loc[:yesterday]
        actual_y_raw = y_series_all.loc[current_date] # Need raw for P&L

        # 1. Fit GARCH models to get residuals UP TO YESTERDAY
        bench_model_fit = fit_best_garch(current_train_y_raw, p=MAX_GARCH_ORDER, q=MAX_GARCH_ORDER,
                                         mean_p=fixed_arma_order_y[0], mean_q=fixed_arma_order_y[1],
                                         dist=GARCH_DIST)
        x_model_fit = fit_best_garch(current_train_x_raw, p=MAX_GARCH_ORDER, q=MAX_GARCH_ORDER,
                                     mean_p=fixed_arma_order_x[0], mean_q=fixed_arma_order_x[1],
                                     dist=GARCH_DIST)
        if bench_model_fit is None or x_model_fit is None: continue
            
        # 2. Extract and Transform Residuals (Uniform Shocks) UP TO YESTERDAY
        
        x_resid_is = get_standardized_residuals(x_model_fit)
        
        # *** NEW: Extract volatility ***
        bench_vol_is = get_conditional_volatility(bench_model_fit)
        x_vol_is = get_conditional_volatility(x_model_fit)
        
        # Create the four series needed for feature generation
        
        u_target_is_full = transform_to_uniform(bench_model_fit)
        v_resid_is_full = transform_to_uniform(x_model_fit)
        
        u_target_is_full = u_target_is_full.clip(1e-6, 1-1e-6)
        v_resid_is_full = v_resid_is_full.clip(1e-6, 1-1e-6)
        
        u_vol_is_full = pd.Series(bench_vol_is, index=bench_vol_is.dropna().index, name='u_vol')
        v_vol_is_full = pd.Series(x_vol_is, index=x_vol_is.dropna().index, name='v_vol')


        if u_target_is_full.empty or v_resid_is_full.empty or u_vol_is_full.empty or v_vol_is_full.empty: continue

        # 3. Create Lagged Feature Sets for Training
        # *** MODIFIED: Call new create_lagged_features function ***
        X_challenger, y_challenger = create_lagged_features(
            u_target_is_full, u_vol_is_full, 
            v_resid_is_full, v_vol_is_full, 
            n_lags=N_LAGS
        )
        X_benchmark, y_benchmark = create_lagged_features(
            u_target_is_full, u_vol_is_full, 
            None, None, 
            n_lags=N_LAGS
        )

        if X_benchmark.empty or X_challenger.empty: continue

        # Ensure targets are aligned (they should be)
        common_train_idx = X_benchmark.index.intersection(X_challenger.index)
        y_train = y_benchmark.loc[common_train_idx]
        X_train_bench = X_benchmark.loc[common_train_idx]
        X_train_chall = X_challenger.loc[common_train_idx]

        if y_train.empty: continue

        # 4. Train XGBoost Models
        model_bench = xgb.XGBRegressor(**best_params_benchmark)
        model_chall = xgb.XGBRegressor(**best_params_challenger)
        
        model_bench.fit(X_train_bench, y_train)
        model_chall.fit(X_train_chall, y_train)

        # 5. Create Features for Today's Prediction (using data up to t-1)
        # *** MODIFIED: Get lags for all 4 feature series ***
        u_resid_lags = u_target_is_full.iloc[-N_LAGS:].values
        u_vol_lags = u_vol_is_full.iloc[-N_LAGS:].values
        v_resid_lags = v_resid_is_full.iloc[-N_LAGS:].values
        v_vol_lags = v_vol_is_full.iloc[-N_LAGS:].values

        if len(u_resid_lags) < N_LAGS or len(u_vol_lags) < N_LAGS or len(v_resid_lags) < N_LAGS or len(v_vol_lags) < N_LAGS:
            continue # Not enough history to create features

        # Concatenate in the correct order, reversing to get [t-1, t-2, ...]
        features_today_bench = np.concatenate([u_resid_lags[::-1], u_vol_lags[::-1]]).reshape(1, -1)
        features_today_chall = np.concatenate([
            u_resid_lags[::-1], u_vol_lags[::-1], 
            v_resid_lags[::-1], v_vol_lags[::-1]
        ]).reshape(1, -1)

        # Create DataFrame with correct column names for prediction
        bench_resid_cols = [f'u_target_lag_{i}' for i in range(1, N_LAGS + 1)]
        bench_vol_cols = [f'u_vol_lag_{i}' for i in range(1, N_LAGS + 1)]
        bench_cols = bench_resid_cols + bench_vol_cols
        
        chall_v_resid_cols = [f'v_resid_lag_{i}' for i in range(1, N_LAGS + 1)]
        chall_v_vol_cols = [f'v_vol_lag_{i}' for i in range(1, N_LAGS + 1)]
        chall_cols = bench_cols + chall_v_resid_cols + chall_v_vol_cols
        
        X_pred_bench = pd.DataFrame(features_today_bench, columns=bench_cols)
        X_pred_chall = pd.DataFrame(features_today_chall, columns=chall_cols)

        # 6. Make 1-Step-Ahead Predictions
        pred_bench = model_bench.predict(X_pred_bench)[0]
        pred_chall = model_chall.predict(X_pred_chall)[0]

        # 7. Get Actual Shock for Today (for MSE calculation)
        _, _, _, actual_u_c_t = get_oos_forecast_params(bench_model_fit, actual_y_raw)

        if not np.isfinite(actual_u_c_t): continue

        # 8. Store results for this day
        oos_predictions.append({
            "Date": current_date,
            "Actual_Shock": actual_u_c_t,
            "Pred_Shock_Bench": pred_bench,
            "Pred_Shock_Chall": pred_chall,
            "Actual_Return": actual_y_raw # Store the raw return for Sharpe calculation
        })

    # --- Evaluate OOS Predictions (after loop) ---
    if len(oos_predictions) < 20:
        print("  Skipping Evaluation: Not enough valid OOS forecasts generated.")
        continue

    print("  OOS loop finished. Evaluating ML models...")
    pred_df = pd.DataFrame(oos_predictions).set_index("Date").dropna()

    # 1. Calculate MSE
    mse_bench = mean_squared_error(pred_df['Actual_Shock'], pred_df['Pred_Shock_Bench'])
    mse_chall = mean_squared_error(pred_df['Actual_Shock'], pred_df['Pred_Shock_Chall'])

    # 2. Calculate Sharpe Ratios
    pred_df['Position_Bench'] = np.where(pred_df['Pred_Shock_Bench'] > TRADING_THRESHOLD_LONG, 1,
                                  np.where(pred_df['Pred_Shock_Bench'] < TRADING_THRESHOLD_SHORT, -1, 0))
    pred_df['Position_Chall'] = np.where(pred_df['Pred_Shock_Chall'] > TRADING_THRESHOLD_LONG, 1,
                                  np.where(pred_df['Pred_Shock_Chall'] < TRADING_THRESHOLD_SHORT, -1, 0))

    pred_df['Return_Bench'] = pred_df['Position_Bench'] * pred_df['Actual_Return']
    pred_df['Return_Chall'] = pred_df['Position_Chall'] * pred_df['Actual_Return']

    sharpe_bench = calculate_sharpe_ratio(pred_df['Return_Bench'])
    sharpe_chall = calculate_sharpe_ratio(pred_df['Return_Chall'])

    ml_results.append({
        "Source_Factor": source_key,
        "Target_Factor": target_key,
        "OOS_Days": len(pred_df),
        "MSE_Benchmark": mse_bench,
        "MSE_Challenger": mse_chall,
        "Sharpe_Benchmark": sharpe_bench,
        "Sharpe_Challenger": sharpe_chall,
    })
    
    # Save detailed predictions
    pred_df.to_csv(f"ml_oos_predictions_{source_key}_to_{target_key}.csv")


# --- Final Output ---
print("\n" + "=" * 50)
print("Final OOS Copula-ML (XGBoost) Backtest Results (with Vol Features)")
print("=" * 50)
results_table = pd.DataFrame(ml_results)
print(results_table.to_string(float_format="%.6f"))

# Save results
results_table.to_csv("copula_ml_oos_results_with_vol.csv", index=False)
print("\nSummary results saved to 'copula_ml_oos_results_with_vol.csv'")