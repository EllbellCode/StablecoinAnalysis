import pandas as pd
import numpy as np
from pathlib import Path
from arch import arch_model
from statsmodels.tsa.arima.model import ARIMA
from scipy.stats import t, norm, distributions
from scipy.stats import multivariate_t
from scipy.optimize import minimize
import warnings
from tqdm import tqdm # Now very important for the OOS loop
import statsmodels.api as sm

# --- Added imports for PCA ---
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

warnings.filterwarnings('ignore')

# --- OOS CHANGE: Updated Constants / Settings ---
DATA_DIR = Path("Data/Verified")
START_DATE = '2020-01-01'
TRAIN_END_DATE = '2024-01-01' # End date for initial training
FULL_END_DATE = '2025-01-01' # End of OOS test data
STATIONARY_VOL = "Delta_LogRV" # Use the confirmed stationary volatility

# GARCH Settings
MAX_ARMA_ORDER = 3 # Max p, q for ARMA mean model
MAX_GARCH_ORDER = 1 # Max p, q for GARCH variance model (usually 1,1 is enough)
GARCH_DIST = 't' # Distribution assumption for GARCH errors

# Copula Settings
COPULA_TYPES = ['gaussian', 'student-t'] # Copulas to test

# AG Test Settings
HAC_LAGS = None # Lags for Newey-West in AG test (None lets it choose)

# ===================================================================
# Helper Functions
# ===================================================================

# --- OOS CHANGE: This function is no longer needed for OOS testing ---
# def calculate_t_loglik_series(model_result):
#     ...

# --- OOS CHANGE: NEW helper function to get 1-step-ahead forecast LL ---
def get_oos_loglik_and_uniform(fitted_model, actual_value):
    """
    Calculates the 1-step-ahead log-likelihood and uniform transform (U)
    for an actual outcome, given a fitted GARCH model.
    """
    # 1. Get the 1-step-ahead forecast
    forecast = fitted_model.forecast(horizon=1, reindex=False)
    
    mean_forecast = forecast.mean.iloc[0, 0]
    var_forecast = forecast.variance.iloc[0, 0]
    scale_forecast = np.sqrt(var_forecast)
    
    # 2. Get distribution parameters from the fitted model
    dist_name = fitted_model.model.distribution.name
    
    # 3. Calculate log-likelihood and CDF (uniform)
    try:
        if dist_name == 't' or dist_name == "Standardized Student's t":
            nu = fitted_model.params['nu']
            log_lik = t.logpdf(actual_value, df=nu, loc=mean_forecast, scale=scale_forecast)
            uniform_transform = t.cdf(actual_value, df=nu, loc=mean_forecast, scale=scale_forecast)
        elif dist_name == 'norm':
            log_lik = norm.logpdf(actual_value, loc=mean_forecast, scale=scale_forecast)
            uniform_transform = norm.cdf(actual_value, loc=mean_forecast, scale=scale_forecast)
        else:
            # Fallback for other distributions (e.g., GED) - requires more params
            print(f"Warning: Untested distribution {dist_name}. Defaulting to norm.")
            log_lik = norm.logpdf(actual_value, loc=mean_forecast, scale=scale_forecast)
            uniform_transform = norm.cdf(actual_value, loc=mean_forecast, scale=scale_forecast)
            
        return log_lik, uniform_transform
        
    except Exception as e:
        # print(f"  Warning in loglik calculation: {e}. Returning NaN.")
        return np.nan, np.nan


def select_best_arma(series, max_order=MAX_ARMA_ORDER):
    """
    Selects the best ARMA(p,q) order based on AIC.
    Returns (best_p, best_q).
    """
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
    # print(f"    Best ARMA order selected: {best_order} (AIC: {best_aic:.2f})")
    return best_order


def fit_best_garch(series, p, q, mean_p, mean_q, dist=GARCH_DIST):
    """
    Fits the specified ARMA(mean_p, mean_q)-GARCH(p,q) model.
    """
    series = series.dropna()
    if series.empty: return None
    
    # --- OOS CHANGE: Added robustness for mean_p = 0 ---
    if mean_p == 0:
        mean_model = 'Constant'
        ar_lags = None
    else:
        mean_model = 'AR'
        ar_lags = mean_p
    # --- End of change ---

    try:
        am = arch_model(series, vol='EGARCH', p=p, q=q,
                        mean=mean_model, lags=ar_lags, # Use the robust mean_model
                        dist=dist)
        res = am.fit(update_freq=0, disp='off')
        return res
    except Exception as e:
        # print(f"    Error fitting GARCH: {e}")
        return None


def get_standardized_residuals(model_result):
    """
    Extracts standardized residuals from a fitted GARCH model.
    """
    if model_result is None: return pd.Series(dtype=float)
    return model_result.std_resid


def transform_to_uniform(residuals, dist_params):
    """
    Transforms residuals to uniform [0,1] using the specified distribution's CDF.
    """
    residuals = residuals.dropna()
    if residuals.empty: return np.array([])

    dist_name = dist_params.get('dist', 't')
    if dist_name == 't' or dist_name == "Standardized Student's t":
        nu = dist_params.get('nu', 4) # Default or get from fit
        return distributions.t.cdf(residuals, df=nu)
    elif dist_name == 'norm':
         return distributions.norm.cdf(residuals)
    else:
        from statsmodels.distributions.empirical_distribution import ECDF
        ecdf = ECDF(residuals)
        return ecdf(residuals)


def gaussian_copula_logpdf(u, v, rho):
    """Calculates log PDF for Gaussian copula."""
    if not -1 < rho < 1: return -np.inf
    
    # Clip inputs for numerical stability at boundaries
    u = np.clip(u, 1e-6, 1 - 1e-6)
    v = np.clip(v, 1e-6, 1 - 1e-6)
    
    z_u = norm.ppf(u)
    z_v = norm.ppf(v)

    term1 = -0.5 * np.log(1 - rho**2)
    term2 = - (rho**2 * (z_u**2 + z_v**2) - 2 * rho * z_u * z_v) / (2 * (1 - rho**2))
    
    return term1 + term2

def t_copula_logpdf(u, v, rho, nu):
    """Calculates log PDF for Student-t copula."""
    if not -1 < rho < 1 or nu <= 2: return -np.inf
    
    # Clip inputs for numerical stability at boundaries
    u = np.clip(u, 1e-6, 1 - 1e-6)
    v = np.clip(v, 1e-6, 1 - 1e-6)
    
    t_u = t.ppf(u, df=nu)
    t_v = t.ppf(v, df=nu)

    data_matrix = np.column_stack([t_u, t_v])
    
    try:
        log_num = multivariate_t.logpdf(data_matrix, shape=[[1, rho], [rho, 1]], df=nu)
    except Exception as e:
        # print(f"  Warning in mvt_logpdf: {e}")
        return -np.inf # Return -inf on numerical error
        
    log_den1 = t.logpdf(t_u, df=nu)
    log_den2 = t.logpdf(t_v, df=nu)

    return log_num - log_den1 - log_den2


def fit_copula(u, v, copula_type='student-t'):
    """
    Fits the specified copula type using Maximum Likelihood Estimation.
    """
    u = np.clip(u, 1e-6, 1 - 1e-6)
    v = np.clip(v, 1e-6, 1 - 1e-6)

    if copula_type == 'gaussian':
        def neg_log_lik(params):
            rho = params[0]
            loglik = gaussian_copula_logpdf(u, v, rho)
            return -np.sum(loglik[np.isfinite(loglik)])

        initial_rho = np.corrcoef(norm.ppf(u), norm.ppf(v))[0, 1]
        bounds = [(-0.999, 0.999)]
        result = minimize(neg_log_lik, [initial_rho], method='L-BFGS-B', bounds=bounds)
        
        if result.success and -0.999 < result.x[0] < 0.999:
            rho_mle = result.x[0]
            mll = -result.fun
            params = {'rho': rho_mle, 'type': 'gaussian'}
            return params, mll
        else:
            return None, -np.inf

    elif copula_type == 'student-t':
        def neg_log_lik(params):
            rho, nu = params
            loglik = t_copula_logpdf(u, v, rho, nu)
            return -np.sum(loglik[np.isfinite(loglik)])

        initial_rho = np.corrcoef(t.ppf(u, df=4), t.ppf(v, df=4))[0, 1]
        initial_nu = 4.0
        bounds = [(-0.999, 0.999), (2.1, 50)] 

        result = minimize(neg_log_lik, [initial_rho, initial_nu], method='L-BFGS-B', bounds=bounds)
        if result.success:
            rho_mle, nu_mle = result.x
            mll = -result.fun
            params = {'rho': rho_mle, 'nu': nu_mle, 'type': 'student-t'}
            return params, mll
        else:
            return None, -np.inf
    else:
        raise ValueError(f"Unsupported copula type: {copula_type}")

def select_best_copula(u, v, copula_types=COPULA_TYPES):
    """Selects the best copula based on AIC."""
    best_aic = np.inf
    best_params = None
    best_mll = -np.inf

    for c_type in copula_types:
        # print(f"    Trying Copula Type: {c_type}")
        params, mll = fit_copula(u, v, copula_type=c_type)
        if params is not None:
            k = len(params) -1 
            aic = 2 * k - 2 * mll
            # print(f"      MLL={mll:.2f}, AIC={aic:.2f}")
            if aic < best_aic:
                best_aic = aic
                best_params = params
                best_mll = mll

    # if best_params:
    #     print(f"    Best Copula Selected: {best_params.get('type','N/A')} (AIC: {best_aic:.2f})")
    return best_params, best_mll


def calculate_copula_loglik_series(u, v, copula_params):
    """Calculates the time series of log likelihoods for a fitted copula."""
    copula_type = copula_params.get('type')
    if copula_type == 'gaussian':
        rho = copula_params['rho']
        return gaussian_copula_logpdf(u, v, rho)
    elif copula_type == 'student-t':
        rho = copula_params['rho']
        nu = copula_params['nu']
        return t_copula_logpdf(u, v, rho, nu)
    else:
        raise ValueError("Unsupported copula type")


def amisano_giacomini_test(loglik_diff_series, hac_lags=None):
    """
    Performs the Amisano-Giacomini test for comparing predictive likelihoods.
    """
    d = loglik_diff_series.dropna()
    T = len(d)
    if T < 2: return np.nan, np.nan
    
    X = sm.add_constant(np.ones(T))
    try:
        ols_fit = sm.OLS(d, X).fit()
    except Exception as e:
        print(f"  Error in AG Test OLS fit: {e}")
        return np.nan, np.nan
    
    if hac_lags is None:
        hac_lags = int(np.floor(T**(1/4))) 

    try:
        hac_cov = ols_fit.get_robustcov_results(cov_type='HAC', maxlags=hac_lags)
        ag_stat = hac_cov.tvalues[0]
        p_value = hac_cov.pvalues[0]
        return ag_stat, p_value
    except Exception as e:
        print(f"  Error calculating HAC variance in AG Test: {e}")
        return np.nan, np.nan



def create_pca_factor(coins, var, data_dict, factor_name, train_end_date):
    """
    Builds a data matrix, fits PCA on training data, and transforms
    both train and test data to prevent data leakage.
    Returns one combined pd.Series for the full period.
    """
    df_list = [data_dict[coin][var] for coin in coins if coin in data_dict and var in data_dict[coin].columns]
    if not df_list:
        print(f"Warning: No data found for variable '{var}' in coins {coins}.")
        return pd.Series(dtype=float, name=factor_name)

    data_matrix = pd.concat(df_list, axis=1, keys=coins, join='inner')
    data_matrix.dropna(inplace=True) 

    if data_matrix.empty:
         print(f"Warning: No overlapping data found for variable '{var}' in coins {coins}.")
         return pd.Series(dtype=float, name=factor_name)

    # 1. Split data into train and test matrices
    train_matrix = data_matrix.loc[START_DATE:train_end_date]
    test_matrix = data_matrix.loc[train_end_date:].iloc[1:] #iloc[1:] to avoid overlap

    if train_matrix.empty or test_matrix.empty:
        print(f"Warning: Not enough data to split train/test for {factor_name}.")
        return pd.Series(dtype=float, name=factor_name)

    # 2. Fit scaler and PCA *only* on training data
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_matrix)
    
    pca = PCA(n_components=1)
    train_factor_scaled = pca.fit_transform(train_scaled)
    
    print(f"Created Factor '{factor_name}'. PCA Explained Variance: {pca.explained_variance_ratio_[0]:.2%}")
    loadings_dict = {coin: loading for coin, loading in zip(train_matrix.columns, pca.components_[0])}
    print(f"  Loadings: {loadings_dict}\n")

    # 3. Transform test data using the *fitted* objects
    test_scaled = scaler.transform(test_matrix)
    test_factor_scaled = pca.transform(test_scaled)
    
    # 4. Combine and return as a single Series
    train_series = pd.Series(train_factor_scaled.ravel(), index=train_matrix.index, name=factor_name)
    test_series = pd.Series(test_factor_scaled.ravel(), index=test_matrix.index, name=factor_name)
    
    return pd.concat([train_series, test_series])

# ===================================================================
# Main Script Logic
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
        # --- OOS CHANGE: Load *all* data up to the full end date ---
        df = df[(df['Date'] >= START_DATE) & (df['Date'] <= FULL_END_DATE)]
        coin_data[coin_name] = df.set_index('Date')

print("Creating PCA Factors (Train/Test Split aware)...")
factors = {}
factors["Stable_Volume"] = create_pca_factor(
    stablecoins, "LogVolChange", coin_data, "PC1_Stable_Volume", TRAIN_END_DATE
)
factors["Stable_Volatility"] = create_pca_factor(
    stablecoins, STATIONARY_VOL, coin_data, "PC1_Stable_Volatility", TRAIN_END_DATE
)
factors["Crypto_Returns"] = create_pca_factor(
    cryptos, "Log Returns", coin_data, "PC1_Crypto_Returns", TRAIN_END_DATE
)
factors["Crypto_Volatility"] = create_pca_factor(
    cryptos, STATIONARY_VOL, coin_data, "PC1_Crypto_Volatility", TRAIN_END_DATE
)

print("-" * 50)
print(f"Running LAGGED Out-of-Sample Copula-GARCH Backtests (Predictive Test)")
print(f"Training up to {TRAIN_END_DATE}, Testing on {TRAIN_END_DATE} to {FULL_END_DATE}")
print("-" * 50)

tests_to_run = [
    ("Stable_Volume", "Crypto_Returns"),
    ("Stable_Volume", "Crypto_Volatility"),
    ("Stable_Volatility", "Crypto_Returns"),
    ("Stable_Volatility", "Crypto_Volatility"),
]

ag_results = []

for source_key, target_key in tests_to_run:
    print(f"\n===== OOS LAGGED TEST: {source_key} (t-1) -> {target_key} (t) =====")

    # --- Get all factor data ---
    y_series_all = factors[target_key].dropna()
    x_series_all = factors[source_key].dropna()
    
    # Align both series to their common dates
    common_idx = y_series_all.index.intersection(x_series_all.index)
    y_series_all = y_series_all.loc[common_idx]
    x_series_all = x_series_all.loc[common_idx]
    
    # --- Determine initial training set and test dates ---
    initial_train_y = y_series_all.loc[:TRAIN_END_DATE]
    initial_train_x = x_series_all.loc[:TRAIN_END_DATE]
    
    test_dates = y_series_all.loc[TRAIN_END_DATE:].iloc[1:].index

    if len(initial_train_y) < 200 or len(test_dates) < 50:
        print("  Skipping test: Not enough initial training or test data.")
        continue

    # --- Get fixed ARMA orders based *only* on initial training data ---
    print(f"  Finding initial ARMA orders on data up to {TRAIN_END_DATE}...")
    fixed_arma_order_y = select_best_arma(initial_train_y, max_order=MAX_ARMA_ORDER)
    fixed_arma_order_x = select_best_arma(initial_train_x, max_order=MAX_ARMA_ORDER)
    print(f"  Fixed ARMA for {target_key}: {fixed_arma_order_y}")
    print(f"  Fixed ARMA for {source_key}: {fixed_arma_order_x}")
    
    # --- This is the main OOS forecasting loop ---
    oos_ll_bench_list = []
    oos_ll_copula_part_list = []
    oos_dates_list = []

    print(f"  Running expanding window forecast for {len(test_dates)} test points...")
    for current_date in tqdm(test_dates):
        # 1. Define current training data (all data up to yesterday, t-1)
        yesterday = current_date - pd.Timedelta(days=1)
        current_train_y = y_series_all.loc[:yesterday]
        current_train_x = x_series_all.loc[:yesterday] # X data also up to t-1
        
        # Get the actual outcome for today (t)
        actual_y = y_series_all.loc[current_date]

        # 2. Fit GARCH models on *current* training data (up to t-1)
        bench_model_fit = fit_best_garch(current_train_y, p=MAX_GARCH_ORDER, q=MAX_GARCH_ORDER,
                                         mean_p=fixed_arma_order_y[0], mean_q=fixed_arma_order_y[1],
                                         dist=GARCH_DIST)
        
        x_model_fit = fit_best_garch(current_train_x, p=MAX_GARCH_ORDER, q=MAX_GARCH_ORDER,
                                     mean_p=fixed_arma_order_x[0], mean_q=fixed_arma_order_x[1],
                                     dist=GARCH_DIST)
        
        if bench_model_fit is None or x_model_fit is None:
            continue # Skip this day if a model fails to fit
            
        # 3. Get OOS log-likelihood and uniform for *today* (t)
        ll_bench, u_c_oos = get_oos_loglik_and_uniform(bench_model_fit, actual_y)
        
        if not np.isfinite(ll_bench) or not np.isfinite(u_c_oos):
            continue 

        # 4. Get IN-SAMPLE shocks (up to t-1)
        bench_resid_is = get_standardized_residuals(bench_model_fit)
        x_resid_is = get_standardized_residuals(x_model_fit)
        
        bench_dist_params = {'dist': GARCH_DIST, 'nu': bench_model_fit.params['nu']}
        x_dist_params = {'dist': GARCH_DIST, 'nu': x_model_fit.params['nu']}
        
        # Transform full in-sample residual series to uniforms
        u_target_is_full = transform_to_uniform(bench_resid_is, bench_dist_params)
        v_source_is_full = transform_to_uniform(x_resid_is, x_dist_params)
        
        u_target_is_full = pd.Series(u_target_is_full, index=bench_resid_is.dropna().index)
        v_source_is_full = pd.Series(v_source_is_full, index=x_resid_is.dropna().index)

        if u_target_is_full.empty or v_source_is_full.empty:
            continue

        # --- LAGGED CHANGE: Create lagged in-sample data for copula fitting ---
        v_source_is_lagged = v_source_is_full.shift(1) # V_s(t-1)
        
        # Align U_c(t) with V_s(t-1)
        resid_df_is_lagged = pd.concat([u_target_is_full, v_source_is_lagged], axis=1, join='inner').dropna()
        resid_df_is_lagged.columns = ['u_c', 'v_s_lag1']
        
        if len(resid_df_is_lagged) < 50: # Need some data to fit a copula
            continue 

        u_target_is = resid_df_is_lagged['u_c']
        v_source_is = resid_df_is_lagged['v_s_lag1']
        
        # 5. Fit Copula on LAGGED (t vs t-1) in-sample data
        best_copula_params, copula_mll_is = select_best_copula(u_target_is, v_source_is, copula_types=COPULA_TYPES)

        if best_copula_params is None:
            continue # Skip if copula fails

        # 6. Calculate OOS copula log-likelihood part for *today*
        # We need U_c(t) (which is u_c_oos)
        # We need V_s(t-1) (which is the *last* value of the in-sample V_s series)
        v_s_yesterday = v_source_is_full.iloc[-1]
        
        if not np.isfinite(v_s_yesterday):
            continue
            
        ll_copula_part = calculate_copula_loglik_series(u_c_oos, v_s_yesterday, best_copula_params)
        
        if not np.isfinite(ll_copula_part):
            continue

        # 7. Store the daily OOS results
        oos_ll_bench_list.append(ll_bench)
        oos_ll_copula_part_list.append(ll_copula_part)
        oos_dates_list.append(current_date)

    # --- AG test is now run *after* the loop ---
    if len(oos_dates_list) < 20:
        print("  Skipping AG test: Not enough valid OOS forecasts generated.")
        continue
        
    print("  OOS loop finished. Running Amisano-Giacomini Test...")
    
    ll_bench_series = pd.Series(oos_ll_bench_list, index=oos_dates_list)
    ll_copula_part_series = pd.Series(oos_ll_copula_part_list, index=oos_dates_list)
    
    # The difference series IS just the copula part
    # d_t = (LL_Benchmark_t + LL_Copula_t) - LL_Benchmark_t = LL_Copula_t
    loglik_diff_series = ll_copula_part_series

    ag_stat, p_value = amisano_giacomini_test(loglik_diff_series, hac_lags=HAC_LAGS)
    print(f"  --> AG Statistic = {ag_stat:.4f}, p-value = {p_value:.4f}")

    ll_series_df = pd.DataFrame({
        'LL_Benchmark': ll_bench_series,
        'LL_Copula_Part': ll_copula_part_series,
        'LL_Difference': loglik_diff_series
    })
    ll_filename = f"ag_test_series_{source_key}_to_{target_key}.csv"
    ll_series_df.to_csv(ll_filename)
    print(f"  Log-likelihood series saved to {ll_filename}")
    
    ag_results.append({
        "Source_Factor": source_key,
        "Target_Factor": target_key,
        "OOS_Days": len(loglik_diff_series),
        "Mean_OOS_LL_Bench": ll_bench_series.mean(),
        "Mean_OOS_LL_Copula_Part": loglik_diff_series.mean(),
        "AG_Statistic": ag_stat,
        "AG_p_value": p_value
    })

# --- Final Output ---
print("\n" + "=" * 50)
print("Final LAGGED OUT-OF-SAMPLE Amisano-Giacomini Test Results")
print("=" * 50)
results_df = pd.DataFrame(ag_results)
print(results_df.to_string())

# Save results
results_df.to_csv("Results\GARCH\copula_garch_pca_OOS_results.csv", index=False)
print("Summary results saved to 'Results\GARCH\copula_garch_pca_OOS_results.csv'")