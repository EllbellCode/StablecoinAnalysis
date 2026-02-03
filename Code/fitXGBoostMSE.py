"""
Trains two XGBoost model:
- Benchmark model that only uses historical data from the target factor
- Challenger model that uses data from the target (crypto) and source (stablecoin) factor

Trains the models on daily data from start of 2020 to end of 2023 (4 years)
Backtests on the year of 2024

Uses Diebold Mariano test to assess performance between models in terms of MSE and Directional Accuracy
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
from scipy import stats

warnings.filterwarnings('ignore')

# Config
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

# Define the single pair to test
SOURCE_FACTOR = "Crypto_Volume"
TARGET_FACTOR = "Stable_Upside"

XGB_PARAMS = { 
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'random_state': RANDOM_STATE,
    'n_jobs': -1 
}

def diebold_mariano_test(real, pred_bench, pred_chall, h=1, metric='mse', nested=True):
    
    T = len(real)
    e1 = (real - pred_bench)
    e2 = (real - pred_chall)

    if metric == 'mse':
        if nested:
            # Clark-West
            d = e1**2 - (e2**2 - (pred_bench - pred_chall)**2)
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
    
    #HLN
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

def get_oos_forecast_params(fitted_model, actual_value):
    
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

    if series.empty: 
        return None
    
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
    """Generates all factors for the window ending at current_end_date."""
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
print(f"Running Out-of-Sample ML Backtests ({mode_str} PCA, Window: {WINDOW_SIZE})")
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
    feature_names = X_train_initial_chall_g.columns
    
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=True)

    plt.figure(figsize=(12, 8))
    colors = ['#2ecc71' if 'source' in col else '#3498db' for col in feature_importance_df['Feature']]
    
    plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color=colors)
    plt.xlabel('XGBoost Feature Importance (Gain)')
    plt.title(f'Feature Importance: {SOURCE_FACTOR} predicting {TARGET_FACTOR}\n(Blue: Target Lags | Green: Source/Stable Features)')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()

    plot_path = PLOT_DIR / f"Feature_Importance_{SOURCE_FACTOR}_to_{TARGET_FACTOR}.png"
    plt.savefig(plot_path)
    print(f"  Feature importance plot saved to: {plot_path}")

except Exception as e:
    print(f"  Warning: Could not generate importance plot: {e}")


oos_predictions_garch = []
oos_predictions_vol = []

full_dates = coin_data[cryptos[0]].index
test_dates = full_dates[(full_dates > TRAIN_END_DATE) & (full_dates <= FULL_END_DATE)]

print(f"  Running EXPANDING PCA and Expanding Window Forecast for {len(test_dates)} points...")

for current_date in tqdm(test_dates):
    yesterday = current_date - pd.Timedelta(days=1)
    
    current_factors_full = generate_factors_window(coin_data, current_date)
    
    y_series_now = current_factors_full[target_key].dropna()
    x_series_now = current_factors_full[source_key].dropna()
    
    train_y = y_series_now.loc[:yesterday]
    train_x = x_series_now.loc[:yesterday]
    
    if current_date not in y_series_now.index: continue
    actual_y_value = y_series_now.loc[current_date]

    model_y = fit_best_garch(train_y, p=MAX_GARCH_ORDER, q=MAX_GARCH_ORDER,
                             mean_p=fixed_arma_order_y[0], mean_q=fixed_arma_order_y[1], dist=GARCH_DIST)
    model_x = fit_best_garch(train_x, p=MAX_GARCH_ORDER, q=MAX_GARCH_ORDER,
                             mean_p=fixed_arma_order_x[0], mean_q=fixed_arma_order_x[1], dist=GARCH_DIST)
    
    if model_y is None or model_x is None: continue

    mu_next, vol_next, nu_next, _ = get_oos_forecast_params(model_y, np.nan) 
    
    u_target_exp = transform_to_uniform(model_y).clip(1e-6, 1-1e-6)
    u_vol_exp = get_conditional_volatility(model_y)
    v_resid_exp = transform_to_uniform(model_x).clip(1e-6, 1-1e-6)
    v_vol_exp = get_conditional_volatility(model_x)
    
    u_target_exp.replace([np.inf, -np.inf], np.nan, inplace=True)
    v_resid_exp.replace([np.inf, -np.inf], np.nan, inplace=True)

    X_bench_g, y_bench_g, X_chall_g, y_chall_g = create_garch_features(
        u_target_exp, u_vol_exp, specific_lags, v_resid_exp, v_vol_exp
    )
    
    if not X_bench_g.empty and not X_chall_g.empty:
    
        idx_g = X_bench_g.index.intersection(X_chall_g.index)
        model_bench_g = xgb.XGBRegressor(**best_params_benchmark_garch).fit(X_bench_g.loc[idx_g], y_bench_g.loc[idx_g])
        model_chall_g = xgb.XGBRegressor(**best_params_challenger_garch).fit(X_chall_g.loc[idx_g], y_chall_g.loc[idx_g])
        
        u_target_temp = pd.concat([u_target_exp, pd.Series([0], index=[current_date])])
        u_vol_temp = pd.concat([u_vol_exp, pd.Series([vol_next], index=[current_date])])
        v_resid_temp = pd.concat([v_resid_exp, pd.Series([0], index=[current_date])])
        v_vol_temp = pd.concat([v_vol_exp, pd.Series([0], index=[current_date])]) 
        
        X_b_next, _, X_c_next, _ = create_garch_features(u_target_temp, u_vol_temp, specific_lags, v_resid_temp, v_vol_temp)
        
        pred_bench_u = model_bench_g.predict(X_b_next.iloc[[-1]])[0]
        pred_chall_u = model_chall_g.predict(X_c_next.iloc[[-1]])[0]
        
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

print("\n" + "="*50)
print("RESULTS SUMMARY")
print("="*50)

summary_rows = []

if oos_predictions_garch:
    df_res_g = pd.DataFrame(oos_predictions_garch).set_index('Date')
    mse_b_g = mean_squared_error(df_res_g['Actual'], df_res_g['Pred_Bench'])
    mse_c_g = mean_squared_error(df_res_g['Actual'], df_res_g['Pred_Chall'])
    
    dm_stat_g, dm_p_g = diebold_mariano_test(
        df_res_g['Actual'], df_res_g['Pred_Bench'], df_res_g['Pred_Chall'], metric='mse'
    )
    
    pt_stat_b, pt_p_b, acc_b = pesaran_timmermann_test(df_res_g['Actual'], df_res_g['Pred_Bench'])
    pt_stat_c, pt_p_c, acc_c = pesaran_timmermann_test(df_res_g['Actual'], df_res_g['Pred_Chall'])

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