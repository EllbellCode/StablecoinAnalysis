import pandas as pd
import numpy as np
from pathlib import Path
from arch import arch_model
from statsmodels.tsa.arima.model import ARIMA
import xgboost as xgb
import warnings
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# --- Configuration ---
DATA_DIR = Path("Data/Verified")
OUTPUT_DIR = Path("Results/ML/Eigen_Portfolio")
PLOT_DIR = Path("Plots/ML/Eigen_Portfolio")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# Assets
PORTFOLIO_ASSETS = ["BNB", "BTC", "ETH", "XRP"] 
STABLECOINS = ["DAI", "USDC", "USDT"]

# Backtest Settings
START_DATE = '2020-01-01'
TRAIN_END_DATE = '2024-01-01'
FULL_END_DATE = '2025-01-01'
TARGET_VOL_ANNUAL = 0.50 
MAX_LEVERAGE = 2        

# Model Settings (Same as fitXGBoostPCA.py)
GARCH_DIST = 'skewt'
GARCH_MODEL = 'EGARCH' 
SPECIFIC_LAGS = [1, 7, 30]
VOLATILITY_VAR = "Delta_LogGK"
XGB_PARAMS = { 
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'random_state': 123,
    'n_jobs': -1 
}
PARAM_GRID = {'n_estimators': [50, 100], 'learning_rate': [0.05, 0.1], 'max_depth': [3, 4]}

# ===================================================================
# 1. Helper Functions (Adapted from fitXGBoostPCA.py)
# ===================================================================

def fit_best_garch(series, dist=GARCH_DIST):
    # Simplified ARIMA selection + GARCH fit
    try:
        # Fixed AR(1) for speed in expanding window, or use select_best_arma logic
        am = arch_model(series, vol=GARCH_MODEL, p=1, q=1, mean='AR', lags=1, dist=dist)
        res = am.fit(update_freq=0, disp='off', options={'maxiter': 100})
        return res
    except: return None

def transform_to_uniform(model_result):
    if model_result is None: return pd.Series(dtype=float)
    std_resid = model_result.std_resid.dropna()
    dist = model_result.model.distribution 
    params = [model_result.params[n] for n in dist.parameter_names()]
    return pd.Series(dist.cdf(std_resid, parameters=params), index=std_resid.index)

def get_conditional_volatility(model_result):
    if model_result is None: return pd.Series(dtype=float)
    return model_result.conditional_volatility

# --- NEW FUNCTION: Extract PCA Weights ---
def calculate_pca_factor_series(coins, var, data_dict, current_date):
    current_date = pd.to_datetime(current_date)
    yesterday = current_date - pd.Timedelta(days=1)
    
    df_list = [data_dict[c][var] for c in coins]
    raw_df = pd.concat(df_list, axis=1, keys=coins, join='inner').dropna()
    
    train_data = raw_df.loc[:yesterday]
    test_data = raw_df.loc[[current_date]] 
    
    if train_data.empty: return pd.Series(dtype=float)
    
    # --- FIX: Explicitly select quantiles to avoid unpacking error ---
    quantiles = train_data.quantile([0.01, 0.99])
    lower = quantiles.loc[0.01]
    upper = quantiles.loc[0.99]
    # -----------------------------------------------------------------

    train_data = train_data.clip(lower=lower, upper=upper, axis=1)
    test_data = test_data.clip(lower=lower, upper=upper, axis=1) 

    scaler = StandardScaler()
    pca = PCA(n_components=1)
    
    train_scaled = scaler.fit_transform(train_data)
    train_factor = pca.fit_transform(train_scaled)
    
    if not test_data.empty:
        test_scaled = scaler.transform(test_data)
        test_factor = pca.transform(test_scaled)
        full_val = np.concatenate([train_factor.ravel(), test_factor.ravel()])
        full_idx = train_data.index.union(test_data.index)
    else:
        full_val = train_factor.ravel()
        full_idx = train_data.index
        
    return pd.Series(full_val, index=full_idx)

def get_pca_portfolio_weights(coins, var, data_dict, current_date):
    current_date = pd.to_datetime(current_date)
    yesterday = current_date - pd.Timedelta(days=1)
    
    df_list = [data_dict[c][var] for c in coins]
    raw_df = pd.concat(df_list, axis=1, keys=coins, join='inner').dropna()
    train_data = raw_df.loc[:yesterday] 
    
    if train_data.empty: return None

    # --- FIX: Explicitly select quantiles ---
    quantiles = train_data.quantile([0.01, 0.99])
    lower = quantiles.loc[0.01]
    upper = quantiles.loc[0.99]
    # ----------------------------------------
    
    train_data = train_data.clip(lower=lower, upper=upper, axis=1)

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_data)
    
    pca = PCA(n_components=1)
    pca.fit(train_scaled)
    
    loadings = pca.components_[0] 
    asset_vols = np.sqrt(scaler.var_)
    
    raw_weights = loadings / asset_vols
    abs_weights = np.abs(raw_weights)
    final_weights = abs_weights / np.sum(abs_weights)
    
    weight_dict = dict(zip(coins, final_weights))
    return weight_dict

# --- Feature Creation (Same as fitXGBoostPCA.py) ---
def create_garch_features(u_target, u_vol, v_resid, v_vol, lags):
    df = pd.DataFrame({'u_target': u_target})
    df['u_vol'] = u_vol
    
    # Lags for Target
    for i in lags:
        df[f'u_target_lag_{i}'] = df['u_target'].shift(i)
        df[f'u_vol_lag_{i}'] = df['u_vol'].shift(i)
        
    # Source Factors (Stablecoins)
    df['v_resid'] = v_resid
    df['v_vol'] = v_vol
    for i in lags:
        df[f'v_resid_lag_{i}'] = df['v_resid'].shift(i)
        df[f'v_vol_lag_{i}'] = df['v_vol'].shift(i)
        
    df = df.drop(columns=['u_vol', 'v_resid', 'v_vol']) # Keep lags only + target
    return df.dropna()

# ===================================================================
# 2. Main Logic
# ===================================================================

print("1. Loading Data...")
coin_data = {}
all_coins = STABLECOINS + PORTFOLIO_ASSETS
for file in DATA_DIR.glob("*.csv"):
    coin_name = file.stem.replace("Verif_", "")
    if coin_name in all_coins:
        df = pd.read_csv(file, parse_dates=['Date']).set_index('Date').sort_index()
        coin_data[coin_name] = df[df.index >= START_DATE]

# We need a proxy "Target" to train the model history. 
# We will use the PCA Factor of Crypto Returns as the "Synthetic Asset" we model.
# In the trading loop, we will construct the portfolio to match this synthetic asset.

print("2. Initial Tuning (Using PC1 Factors)...")
limit_date = pd.to_datetime(TRAIN_END_DATE)

# Calculate Initial Factors
target_pca = calculate_pca_factor_series(PORTFOLIO_ASSETS, "Log Returns", coin_data, limit_date)
stable_pca_vol = calculate_pca_factor_series(STABLECOINS, VOLATILITY_VAR, coin_data, limit_date)
stable_pca_volume = calculate_pca_factor_series(STABLECOINS, "LogVolChange", coin_data, limit_date)

# GARCH Fits for Tuning
# Target (Crypto PC1)
target_model = fit_best_garch(target_pca)
u_target = transform_to_uniform(target_model).clip(1e-6, 1-1e-6)
u_vol = get_conditional_volatility(target_model)

# Source (Stable Vol - used as 'v' in your original script logic)
# Note: In fitXGBoostPCA, 'v' was the source factor. 
# We'll use Stable Volume as the primary source 'v' as per your findings.
source_model = fit_best_garch(stable_pca_volume) 
v_resid = transform_to_uniform(source_model).clip(1e-6, 1-1e-6)
v_vol = get_conditional_volatility(source_model)

# Create Features
common_idx = u_target.index.intersection(v_resid.index)
df_features = create_garch_features(u_target.loc[common_idx], u_vol.loc[common_idx], v_resid.loc[common_idx], v_vol.loc[common_idx], SPECIFIC_LAGS)

# Tune XGBoost (Predicting Uniform Shock 'u_target')
tscv = TimeSeriesSplit(n_splits=3)
gs = GridSearchCV(xgb.XGBRegressor(**XGB_PARAMS), PARAM_GRID, scoring='neg_mean_squared_error', cv=tscv, n_jobs=-1)
# Benchmark: Predict u_target using only u_target lags (drop v cols)
# Challenger: Predict u_target using all cols
X_chall = df_features.drop(columns='u_target')
y_chall = df_features['u_target']
gs.fit(X_chall, y_chall)
best_params = {**XGB_PARAMS, **gs.best_params_}
print(f"   Best Params: {gs.best_params_}")

# ===================================================================
# 3. Expanding Window Backtest (CORRECTED SCALING)
# ===================================================================
print(f"3. Running Backtest ({TRAIN_END_DATE} to {FULL_END_DATE})...")

# 1. Setup Test Dates based on available Data (Fixing the empty date bug)
full_idx = coin_data[PORTFOLIO_ASSETS[0]].index
test_dates = full_idx[(full_idx > TRAIN_END_DATE) & (full_idx <= FULL_END_DATE)]
results = []
target_vol_daily = TARGET_VOL_ANNUAL / np.sqrt(365)

# 2. Initialize a history of Portfolio Returns (for the Portfolio GARCH)
# We need to construct the history of the Eigen-Portfolio up to the start date
# A simple approximation is an equal-weight or pre-calculated eigen-weight history
# For expanding window, we can build it cumulatively.
print("   Pre-calculating Eigen-Portfolio history for scaling...")
hist_dates = full_idx[full_idx <= TRAIN_END_DATE]
hist_returns = []
for d in hist_dates:
    # Use Equal weight for history initialization to be safe/fast, 
    # or calculate dynamic weights (slower). 
    # Let's use Equal Weight for the 'warmup' period.
    rets = [coin_data[c].loc[d, 'Log Returns'] for c in PORTFOLIO_ASSETS]
    hist_returns.append(np.mean(rets))

portfolio_return_history = pd.Series(hist_returns, index=hist_dates)

for current_date in tqdm(test_dates):
    yesterday = current_date - pd.Timedelta(days=1)
    
    # --- A. UPDATE FACTORS & PCA MODEL (The "Signal") ---
    target_series = calculate_pca_factor_series(PORTFOLIO_ASSETS, "Log Returns", coin_data, current_date)
    source_series = calculate_pca_factor_series(STABLECOINS, "LogVolChange", coin_data, current_date)
    
    # Fit GARCH on PCA Factors (to feed XGBoost)
    model_target_pca = fit_best_garch(target_series.loc[:yesterday])
    model_source_pca = fit_best_garch(source_series.loc[:yesterday])
    
    if model_target_pca is None or model_source_pca is None: continue
    
    # --- B. PREDICT SHOCK (Using XGBoost) ---
    # 1. Transform PCA history to Uniform Features
    u_t = transform_to_uniform(model_target_pca).clip(1e-6, 1-1e-6)
    u_v = get_conditional_volatility(model_target_pca)
    v_r = transform_to_uniform(model_source_pca).clip(1e-6, 1-1e-6)
    v_v = get_conditional_volatility(model_source_pca)
    
    common_idx = u_t.index.intersection(v_r.index)
    df_train = create_garch_features(u_t.loc[common_idx], u_v.loc[common_idx], v_r.loc[common_idx], v_v.loc[common_idx], SPECIFIC_LAGS)
    
    if df_train.empty: continue
    
    model_xgb = xgb.XGBRegressor(**best_params)
    model_xgb.fit(df_train.drop(columns='u_target'), df_train['u_target'])
    
    # 2. Create "Next Day" Features
    fc_target = model_target_pca.forecast(horizon=1, reindex=False)
    vol_next_pca = np.sqrt(fc_target.variance.iloc[0, 0])
    
    u_t_next = pd.concat([u_t, pd.Series([0.5], index=[current_date])])
    u_v_next = pd.concat([u_v, pd.Series([vol_next_pca], index=[current_date])])
    v_r_next = pd.concat([v_r, pd.Series([0.5], index=[current_date])])
    v_v_next = pd.concat([v_v, pd.Series([0], index=[current_date])])
    
    df_next = create_garch_features(u_t_next, u_v_next, v_r_next, v_v_next, SPECIFIC_LAGS)
    if current_date not in df_next.index: continue
    
    # 3. Predict The Uniform Shock (0 to 1)
    pred_u = model_xgb.predict(df_next.loc[[current_date]].drop(columns='u_target'))[0]
    
    # --- C. SCALING: GET PORTFOLIO VOLATILITY ---
    # 1. Calculate Yesterday's Realized Portfolio Return (using yesterday's weights)
    #    (For simplicity in this loop, we append the actual calculated return at the end of the loop)
    #    We fit GARCH on the *history* of the portfolio returns.
    
    model_portfolio_garch = fit_best_garch(portfolio_return_history)
    if model_portfolio_garch is None:
        # Fallback if fit fails
        vol_next_port = portfolio_return_history.rolling(30).std().iloc[-1]
    else:
        vol_next_port = np.sqrt(model_portfolio_garch.forecast(horizon=1, reindex=False).variance.iloc[0,0])

    # --- D. MAP SHOCK TO PORTFOLIO ---
    # We use the Portfolio GARCH's distribution to map the predicted Uniform Shock (u)
    # to a Realized Return Shock (z) in portfolio units.
    
    if model_portfolio_garch is not None:
        dist_port = model_portfolio_garch.model.distribution
        params_port = [model_portfolio_garch.params[n] for n in dist_port.parameter_names()]
        # Inverse CDF: Map u (e.g., 0.05) -> z (e.g., -1.65 sigmas)
        pred_z_score = dist_port.ppf(np.clip(pred_u, 1e-6, 1-1e-6), params_port)
    else:
        # Fallback Normal PPF
        from scipy.stats import norm
        pred_z_score = norm.ppf(np.clip(pred_u, 1e-6, 1-1e-6))

    # --- E. TRADING LOGIC ---
    # Vol Forecast = Baseline Vol * Stress Factor
    # If the model predicts a shock > 1 sigma (either direction), we assume higher risk.
    # We take max(1, abs(z)) to ensure we don't artificially lower risk below baseline GARCH
    risk_multiplier = max(1.0, abs(pred_z_score))
    
    final_vol_forecast = vol_next_port * risk_multiplier
    
    # Calculate Weights for TOMORROW (Trading Day)
    weights = get_pca_portfolio_weights(PORTFOLIO_ASSETS, "Log Returns", coin_data, current_date)
    if weights is None: weights = {c: 0.25 for c in PORTFOLIO_ASSETS}

    # Calculate Actual Return for Today (to append to history)
    # Note: We trade at 'current_date' based on signal. We observe return at 'current_date'.
    port_ret = 0
    for coin, w in weights.items():
        ret = coin_data[coin].loc[current_date, 'Log Returns']
        port_ret += w * ret
    
    # Update History for next GARCH fit
    portfolio_return_history.loc[current_date] = port_ret
    
    # Position Sizing
    w_trade = min(target_vol_daily / final_vol_forecast, MAX_LEVERAGE)
    
    results.append({
        'Date': current_date,
        'Actual_Port_Ret': port_ret,
        'Pred_Uniform': pred_u,
        'Pred_Z': pred_z_score,
        'Baseline_Vol': vol_next_port,
        'Final_Vol': final_vol_forecast,
        'Weight_Trade': w_trade,
        'Strat_Ret': w_trade * port_ret
    })

# --- OUTPUT RESULTS ---
if not results:
    print("No results.")
else:
    df_res = pd.DataFrame(results).set_index('Date')
    df_res['Cum_Bench'] = (1 + df_res['Actual_Port_Ret']).cumprod()
    df_res['Cum_Strat'] = (1 + df_res['Strat_Ret']).cumprod()
    
    def get_metrics(s):
        ann_ret = s.mean() * 365
        ann_vol = s.std() * np.sqrt(365)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
        return f"Ret: {ann_ret:.1%}, Vol: {ann_vol:.1%}, Sharpe: {sharpe:.2f}"

    print("\n" + "="*50)
    print("EIGEN-PORTFOLIO BACKTEST (CORRECTED)")
    print("="*50)
    print(f"Buy & Hold (Eigen-Weighted): {get_metrics(df_res['Actual_Port_Ret'])}")
    print(f"Model Strategy:              {get_metrics(df_res['Strat_Ret'])}")
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(df_res['Cum_Bench'], label='Buy & Hold', color='gray', alpha=0.5)
    plt.plot(df_res['Cum_Strat'], label='Model Strategy', color='green', linewidth=1.5)
    plt.title("Eigen-Portfolio: Corrected Volatility Scaling")
    plt.legend()
    plt.savefig(PLOT_DIR / "Corrected_Strategy.png")