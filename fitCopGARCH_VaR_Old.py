import pandas as pd
import numpy as np
from pathlib import Path
from arch import arch_model
from statsmodels.tsa.arima.model import ARIMA
from scipy.stats import t, norm, distributions, chi2
from scipy.stats import multivariate_t
from scipy.optimize import minimize, brentq
import warnings
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# --- Constants / Settings ---
DATA_DIR = Path("Data/Verified")
START_DATE = '2020-01-01'
TRAIN_END_DATE = '2024-01-01'
FULL_END_DATE = '2025-01-01'
STATIONARY_VOL = "Delta_LogRV"

# GARCH Settings
MAX_ARMA_ORDER = 1
MAX_GARCH_ORDER = 1
GARCH_DIST = 't'

# Copula Settings
COPULA_TYPES = ['gaussian', 'student-t']

# --- VaR BACKTEST SETTINGS ---
VAR_LEVEL = 0.05 # VaR

# ===================================================================
# NEW Helper Functions for VaR Backtest
# ===================================================================

def plot_uniform_diagnostics(u_series, v_series, date_label):
    """
    Plots histograms of the transformed residuals to check for Uniformity.
    Pauses execution until the plot window is closed.
    """
    plt.figure(figsize=(12, 5))
    
    # Plot U (Target)
    plt.subplot(1, 2, 1)
    plt.hist(u_series, bins=30, color='skyblue', edgecolor='black', density=True)
    plt.axhline(y=1.0, color='r', linestyle='--', linewidth=2, label="Ideal Uniform")
    plt.title(f"Target Residuals (U) - {date_label}\nShould be flat (Uniform[0,1])")
    plt.xlabel("Transformed Value")
    plt.ylabel("Density")
    plt.legend()

    # Plot V (Source Lagged)
    plt.subplot(1, 2, 2)
    plt.hist(v_series, bins=30, color='lightgreen', edgecolor='black', density=True)
    plt.axhline(y=1.0, color='r', linestyle='--', linewidth=2, label="Ideal Uniform")
    plt.title(f"Source Residuals (V_lag) - {date_label}\nShould be flat (Uniform[0,1])")
    plt.xlabel("Transformed Value")
    plt.legend()

    plt.tight_layout()
    print(f"\n[DIAGNOSTIC] Displaying Uniformity Check for {date_label}. Close plot to continue...")
    plt.show()

def get_conditional_copula_quantile(v, copula_params, alpha=VAR_LEVEL):
    """
    Calculates the conditional alpha-quantile of U given V=v.
    This solves for u* such that C(u* | v) = alpha.
    Uses Brent's method (brentq) to find the root.
    """
    u_min, u_max = 1e-9, 1.0 - 1e-9 # Bounds for the quantile
    copula_type = copula_params.get('type')

    try:
        if copula_type == 'gaussian':
            rho = copula_params['rho']
            v_norm = norm.ppf(v)
            # Define the function whose root we want: C(u|v) - alpha = 0
            def func_to_solve(u):
                u_norm = norm.ppf(u)
                conditional_quantile = norm.cdf((u_norm - rho * v_norm) / np.sqrt(1 - rho**2))
                return conditional_quantile - alpha
            
        elif copula_type == 'student-t':
            rho = copula_params['rho']
            nu = copula_params['nu']
            v_t = t.ppf(v, df=nu)
            
            # Define the function whose root we want: C(u|v) - alpha = 0
            def func_to_solve(u):
                u_t = t.ppf(u, df=nu)
                num = (u_t - rho * v_t)
                den = np.sqrt(((nu + v_t**2) * (1 - rho**2)) / (nu + 1))
                conditional_quantile = t.cdf(num / den, df=nu + 1)
                return conditional_quantile - alpha
        else:
            # Fallback: assume independence
            return alpha 

        # Find the root using brentq
        u_star = brentq(func_to_solve, u_min, u_max)
        return u_star
        
    except (ValueError, RuntimeError) as e:
        # If root finding fails (e.g., bad bounds), return naive quantile
        # print(f"Warning: Conditional quantile solver failed ({e}). Using naive quantile.")
        return alpha

def get_oos_forecast_params(fitted_model, actual_value):
    """
    Calculates 1-step-ahead forecast parameters (mu, sigma, nu)
    and the OOS uniform shock (U).
    """
    # 1. Get the 1-step-ahead forecast
    forecast = fitted_model.forecast(horizon=1, reindex=False)
    mean_forecast = forecast.mean.iloc[0, 0]
    var_forecast = forecast.variance.iloc[0, 0]
    scale_forecast = np.sqrt(var_forecast)
    
    # 2. Get distribution parameters
    dist_name = fitted_model.model.distribution.name
    
    try:
        if dist_name == 't' or dist_name == "Standardized Student's t":
            nu = fitted_model.params['nu']
            uniform_transform = t.cdf(actual_value, df=nu, loc=mean_forecast, scale=scale_forecast)
        elif dist_name == 'norm':
            nu = np.inf # For norm, nu is infinite
            uniform_transform = norm.cdf(actual_value, loc=mean_forecast, scale=scale_forecast)
        else:
            nu = np.inf
            uniform_transform = norm.cdf(actual_value, loc=mean_forecast, scale=scale_forecast)
            
        return mean_forecast, scale_forecast, nu, uniform_transform
        
    except Exception as e:
        return np.nan, np.nan, np.nan, np.nan


def kupiec_pof_test(actual_series, var_series, var_level):
    """
    Performs the Kupiec Proportion of Failures (POF) test.
    H0: The observed failure rate is equal to the target VaR level.
    Returns (LR_statistic, p_value, observed_failure_rate)
    """
    breaches = actual_series < var_series
    
    T = len(breaches)
    N = breaches.sum() # Number of failures (breaches)
    
    if T == 0: return np.nan, np.nan, np.nan
    
    p = var_level
    pi_hat = N / T # Observed failure rate
    
    if N == 0 or N == T:
        # Handle edge cases where optimizer fails
        return np.nan, np.nan, pi_hat
        
    # Likelihood Ratio (LR) test statistic
    log_L_null = (T - N) * np.log(1 - p) + N * np.log(p)
    log_L_alt = (T - N) * np.log(1 - pi_hat) + N * np.log(pi_hat)
    
    LR_stat = -2 * (log_L_null - log_L_alt)
    
    # p-value from chi-squared distribution with 1 degree of freedom
    p_value = 1.0 - chi2.cdf(LR_stat, 1)
    
    return LR_stat, p_value, pi_hat


# ===================================================================
# GARCH, Copula, and PCA Helper Functions (Unchanged from your script)
# ===================================================================

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
        # --- ROBUSTNESS CHANGE: Using 'GARCH' instead of 'EGARCH' ---
        am = arch_model(series, vol='EGARCH', p=p, q=q,
                        mean=mean_model, lags=ar_lags,
                        dist=dist)
        # Increase maxiter for better convergence
        res = am.fit(update_freq=0, disp='off', options={'maxiter': 200})
        return res
    except Exception as e:
        # print(f"    Error fitting GARCH: {e}")
        return None


def get_standardized_residuals(model_result):
    """Extracts standardized residuals from a fitted GARCH model."""
    if model_result is None: return pd.Series(dtype=float)
    return model_result.std_resid


def transform_to_uniform(residuals, dist_params):
    """Transforms residuals to uniform [0,1] using the specified distribution's CDF."""
    residuals = residuals.dropna()
    if residuals.empty: return np.array([])
    dist_name = dist_params.get('dist', 't')
    
    if dist_name == 't' or dist_name == "Standardized Student's t":
        nu = dist_params.get('nu', 4)
        return distributions.t.cdf(residuals, df=nu)
    elif dist_name == 'norm':
         return distributions.norm.cdf(residuals)
    else:
        from statsmodels.distributions.empirical_distribution import ECDF
        ecdf = ECDF(residuals)
        return ecdf(residuals)


# --- Copula functions (gaussian_copula_logpdf, t_copula_logpdf, fit_copula, etc.) ---
# (Assume these are present and correct as in the previous file)
# ... [Omitted for brevity, but they must be included in your final script] ...
def gaussian_copula_logpdf(u, v, rho):
    if not -1 < rho < 1: return -np.inf
    u = np.clip(u, 1e-6, 1 - 1e-6); v = np.clip(v, 1e-6, 1 - 1e-6)
    z_u = norm.ppf(u); z_v = norm.ppf(v)
    term1 = -0.5 * np.log(1 - rho**2)
    term2 = - (rho**2 * (z_u**2 + z_v**2) - 2 * rho * z_u * z_v) / (2 * (1 - rho**2))
    return term1 + term2

def t_copula_logpdf(u, v, rho, nu):
    if not -1 < rho < 1 or nu <= 2: return -np.inf
    u = np.clip(u, 1e-6, 1 - 1e-6); v = np.clip(v, 1e-6, 1 - 1e-6)
    t_u = t.ppf(u, df=nu); t_v = t.ppf(v, df=nu)
    data_matrix = np.column_stack([t_u, t_v])
    try:
        log_num = multivariate_t.logpdf(data_matrix, shape=[[1, rho], [rho, 1]], df=nu)
    except Exception as e:
        return -np.inf
    log_den1 = t.logpdf(t_u, df=nu); log_den2 = t.logpdf(t_v, df=nu)
    return log_num - log_den1 - log_den2

def fit_copula(u, v, copula_type='student-t'):
    u = np.clip(u, 1e-6, 1 - 1e-6); v = np.clip(v, 1e-6, 1 - 1e-6)
    if copula_type == 'gaussian':
        def neg_log_lik(params):
            rho = params[0]
            loglik = gaussian_copula_logpdf(u, v, rho)
            return -np.sum(loglik[np.isfinite(loglik)])
        initial_rho = np.corrcoef(norm.ppf(u), norm.ppf(v))[0, 1]
        bounds = [(-0.999, 0.999)]
        result = minimize(neg_log_lik, [initial_rho], method='L-BFGS-B', bounds=bounds)
        if result.success and -0.999 < result.x[0] < 0.999:
            return {'rho': result.x[0], 'type': 'gaussian'}, -result.fun
        else: return None, -np.inf
    elif copula_type == 'student-t':
        def neg_log_lik(params):
            rho, nu = params
            loglik = t_copula_logpdf(u, v, rho, nu)
            return -np.sum(loglik[np.isfinite(loglik)])
        initial_rho = np.corrcoef(t.ppf(u, df=4), t.ppf(v, df=4))[0, 1]
        initial_nu = 4.0; bounds = [(-0.999, 0.999), (2.1, 50)] 
        result = minimize(neg_log_lik, [initial_rho, initial_nu], method='L-BFGS-B', bounds=bounds)
        if result.success:
            return {'rho': result.x[0], 'nu': result.x[1], 'type': 'student-t'}, -result.fun
        else: return None, -np.inf
    else: raise ValueError(f"Unsupported copula type: {copula_type}")

def select_best_copula(u, v, copula_types=COPULA_TYPES):
    best_aic = np.inf; best_params = None; best_mll = -np.inf
    for c_type in copula_types:
        params, mll = fit_copula(u, v, copula_type=c_type)
        if params is not None:
            k = len(params) -1; aic = 2 * k - 2 * mll
            if aic < best_aic:
                best_aic, best_params, best_mll = aic, params, mll
    return best_params, best_mll


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
# Main VaR Backtest Logic
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
# --- Factors --- (Omitted for brevity, but they are created here)
factors["Stable_Volatility"] = create_pca_factor(stablecoins, STATIONARY_VOL, coin_data, "PC1_Stable_Volatility", TRAIN_END_DATE)
factors["Crypto_Returns"] = create_pca_factor(cryptos, "Log Returns", coin_data, "PC1_Crypto_Returns", TRAIN_END_DATE)
factors["Crypto_Volatility"] = create_pca_factor(cryptos, STATIONARY_VOL, coin_data, "PC1_Crypto_Volatility", TRAIN_END_DATE)


print("-" * 50)
print(f"Running LAGGED Out-of-Sample VaR Backtest at {VAR_LEVEL*100}% Level")
print("-" * 50)

# --- OOS VaR: Focusing only on the significant relationships ---
# tests_to_run = [
#     ("Stable_Volatility", "Crypto_Returns"),
#     ("Stable_Volatility", "Crypto_Volatility"),
# ]

tests_to_run = [("Stable_Volatility", "Crypto_Volatility")]
backtest_results = []

for source_key, target_key in tests_to_run:
    print(f"\n===== VaR Backtest: {source_key} (t-1) -> {target_key} (t) =====")

    y_series_all = factors[target_key].dropna()
    x_series_all = factors[source_key].dropna()
    common_idx = y_series_all.index.intersection(x_series_all.index)
    y_series_all = y_series_all.loc[common_idx]
    x_series_all = x_series_all.loc[common_idx]
    
    initial_train_y = y_series_all.loc[:TRAIN_END_DATE]
    initial_train_x = x_series_all.loc[:TRAIN_END_DATE]
    test_dates = y_series_all.loc[TRAIN_END_DATE:].iloc[1:].index

    if len(initial_train_y) < 200 or len(test_dates) < 50:
        print("  Skipping test: Not enough initial training or test data.")
        continue

    print(f"  Finding initial ARMA orders on data up to {TRAIN_END_DATE}...")
    fixed_arma_order_y = select_best_arma(initial_train_y, max_order=MAX_ARMA_ORDER)
    fixed_arma_order_x = select_best_arma(initial_train_x, max_order=MAX_ARMA_ORDER)
    print(f"  Fixed ARMA for {target_key}: {fixed_arma_order_y}")
    print(f"  Fixed ARMA for {source_key}: {fixed_arma_order_x}")
    
    # --- OOS VaR: Store results in a list ---
    oos_results_list = []
    DIAGNOSTIC_SHOWN = False

    print(f"  Running expanding window VaR forecast for {len(test_dates)} test points...")
    for current_date in tqdm(test_dates):
        yesterday = current_date - pd.Timedelta(days=1)
        current_train_y = y_series_all.loc[:yesterday]
        current_train_x = x_series_all.loc[:yesterday]
        actual_y = y_series_all.loc[current_date]

        # 1. Fit GARCH models
        bench_model_fit = fit_best_garch(current_train_y, p=MAX_GARCH_ORDER, q=MAX_GARCH_ORDER,
                                         mean_p=fixed_arma_order_y[0], mean_q=fixed_arma_order_y[1],
                                         dist=GARCH_DIST)
        x_model_fit = fit_best_garch(current_train_x, p=MAX_GARCH_ORDER, q=MAX_GARCH_ORDER,
                                     mean_p=fixed_arma_order_x[0], mean_q=fixed_arma_order_x[1],
                                     dist=GARCH_DIST)
        if bench_model_fit is None or x_model_fit is None: continue
            
        # 2. Get OOS forecast params for Y(t) and U-shock for X(t-1)
        # We pass `actual_y` but only use the U-shock for the *copula* model
        # The benchmark model only needs the forecasted params
        f_mean, f_scale, f_nu, u_c_oos = get_oos_forecast_params(bench_model_fit, actual_y)
        
        if not np.isfinite(f_mean): continue 

        # 3. Calculate Benchmark VaR
        bench_quantile = t.ppf(VAR_LEVEL, df=f_nu)
        var_benchmark = f_mean + f_scale * bench_quantile

        # 4. Get V(t-1) for Copula
        # Get the *last* residual from the X model fit
        x_resid_is = get_standardized_residuals(x_model_fit)
        x_dist_params = {'dist': GARCH_DIST, 'nu': x_model_fit.params['nu']}
        v_source_is_full = transform_to_uniform(x_resid_is, x_dist_params)
        v_s_yesterday = v_source_is_full[-1] # This is V_s(t-1)
        
        if not np.isfinite(v_s_yesterday): continue

        # 5. Fit Copula on LAGGED (t vs t-1) in-sample data
        bench_resid_is = get_standardized_residuals(bench_model_fit)
        bench_dist_params = {'dist': GARCH_DIST, 'nu': bench_model_fit.params['nu']}
        u_target_is_full = transform_to_uniform(bench_resid_is, bench_dist_params)
        
        u_target_is_full = pd.Series(u_target_is_full, index=bench_resid_is.dropna().index)
        v_source_is_full = pd.Series(v_source_is_full, index=x_resid_is.dropna().index)

        v_source_is_lagged = v_source_is_full.shift(1)
        resid_df_is_lagged = pd.concat([u_target_is_full, v_source_is_lagged], axis=1, join='inner').dropna()
        resid_df_is_lagged.columns = ['u_c', 'v_s_lag1']

        if not DIAGNOSTIC_SHOWN and len(resid_df_is_lagged) > 100:
            plot_uniform_diagnostics(resid_df_is_lagged['u_c'], resid_df_is_lagged['v_s_lag1'], str(current_date.date()))
            DIAGNOSTIC_SHOWN = True
        
        if len(resid_df_is_lagged) < 50: continue 

        best_copula_params, _ = select_best_copula(resid_df_is_lagged['u_c'], resid_df_is_lagged['v_s_lag1'], copula_types=COPULA_TYPES)
        if best_copula_params is None: continue

        # 6. Calculate Copula VaR
        # Find the 5% quantile of U_c(t) *given* V_s(t-1)
        u_star_quantile = get_conditional_copula_quantile(v_s_yesterday, best_copula_params, alpha=VAR_LEVEL)
        
        # Map that uniform-quantile back to the Y-forecast distribution
        copula_quantile = t.ppf(u_star_quantile, df=f_nu)
        var_copula = f_mean + f_scale * copula_quantile

        # 7. Store results
        oos_results_list.append({
            "Date": current_date,
            "Actual_Y": actual_y,
            "VaR_Benchmark": var_benchmark,
            "VaR_Copula": var_copula
        })

    # --- OOS VaR Backtest (after the loop) ---
    if len(oos_results_list) < 20:
        print("  Skipping Backtest: Not enough valid OOS forecasts generated.")
        continue
        
    print("  OOS loop finished. Running VaR Backtest...")
    
    results_df = pd.DataFrame(oos_results_list).set_index("Date")
    
    # Calculate breaches
    results_df['Breach_Benchmark'] = results_df['Actual_Y'] < results_df['VaR_Benchmark']
    results_df['Breach_Copula'] = results_df['Actual_Y'] < results_df['VaR_Copula']
    
    # Run Kupiec Tests
    lr_bench, p_bench, rate_bench = kupiec_pof_test(results_df['Actual_Y'], results_df['VaR_Benchmark'], VAR_LEVEL)
    lr_copula, p_copula, rate_copula = kupiec_pof_test(results_df['Actual_Y'], results_df['VaR_Copula'], VAR_LEVEL)

    backtest_results.append({
        "Source_Factor": source_key,
        "Target_Factor": target_key,
        "OOS_Days": len(results_df),
        "Expected_Breach_%": VAR_LEVEL * 100,
        "Benchmark_Breach_%": rate_bench * 100,
        "Benchmark_Kupiec_p": p_bench,
        "Copula_Breach_%": rate_copula * 100,
        "Copula_Kupiec_p": p_copula,
    })
    
    # Save the detailed VaR series for this test
    results_df.to_csv(f"var_backtest_series_{source_key}_to_{target_key}_{VAR_LEVEL}.csv")

# --- Final Output ---
print("\n" + "=" * 50)
print(f"Final OOS VaR Backtest Results ({VAR_LEVEL*100}%)")
print("=" * 50)
results_table = pd.DataFrame(backtest_results)
print(results_table.to_string())

# Save results
results_table.to_csv("Results/GARCH/VaR_OOS/var_backtest_summary_results.csv", index=False)
print("Summary results saved to 'Results/GARCH/VaR_OOS/var_backtest_summary_results.csv'")