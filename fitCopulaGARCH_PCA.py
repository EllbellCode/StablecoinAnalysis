import pandas as pd
import numpy as np
from pathlib import Path
from arch import arch_model
from statsmodels.tsa.arima.model import ARIMA
from scipy.stats import t, norm, distributions
from scipy.special import loggamma
from scipy.stats import multivariate_t, kstest
from scipy.optimize import minimize
import warnings
import statsmodels.api as sm
from arch.univariate import SkewStudent, Normal, StudentsT, GeneralizedError
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

warnings.filterwarnings('ignore')

# --- Constants / Settings ---
DATA_DIR = Path("Data/Verified")
START_DATE = '2020-01-01'
END_DATE = '2023-12-31' # End date for training/in-sample
STATIONARY_VOL = "Delta_LogRV" # Use the confirmed stationary volatility

# GARCH Settings
MAX_ARMA_ORDER = 3 # Max p, q for ARMA mean model
MAX_GARCH_ORDER = 1 # Max p, q for GARCH variance model (usually 1,1 is enough)
GARCH_DIST = 't' # Distribution assumption for GARCH errors

# Copula Settings
COPULA_TYPES = ['gaussian', 't', 'clayton', 'gumbel']

# AG Test Settings
HAC_LAGS = None # Lags for Newey-West in AG test (None lets it choose)

# ===================================================================
# Existing Helper Functions (Assume these are correctly implemented)
# ===================================================================

def check_uniformity(transformed_data, name):
    # Tests if the data is different from a Uniform(0,1) distribution
    ks_stat, p_value = kstest(transformed_data, 'uniform')
    print(f"  Uniformity Test for {name} (KS-Test):")
    print(f"    KS Stat: {ks_stat:.4f}, p-value: {p_value:.4e}")
    if p_value < 0.05:
        print("    -> FAIL: Data is NOT Uniform (Marginal model is bad)")
    else:
        print("    -> PASS: Data is Uniform (Marginal model is good)")

def clayton_copula_logpdf(u, v, theta):
    """Log PDF for Clayton Copula (Lower Tail Dependence). Theta > 0."""
    if theta <= 0: return -np.inf
    # Avoid log(0) by clipping
    u = np.clip(u, 1e-10, 1-1e-10)
    v = np.clip(v, 1e-10, 1-1e-10)
    term1 = np.log(1 + theta)
    term2 = -(1 + theta) * (np.log(u) + np.log(v))
    term3 = -(1 / theta + 2) * np.log(np.maximum(u**(-theta) + v**(-theta) - 1, 1e-10))
    return term1 + term2 + term3

def gumbel_copula_logpdf_robust(u, v, theta):
    """Robust Log PDF for Gumbel Copula (Upper Tail Dependence). Theta >= 1."""
    if theta < 1.0: return -np.inf
    u = np.clip(u, 1e-10, 1-1e-10)
    v = np.clip(v, 1e-10, 1-1e-10)
    
    x = -np.log(u)
    y = -np.log(v)
    
    log_x = np.log(x)
    log_y = np.log(y)
    
    log_t = np.logaddexp(theta * log_x, theta * log_y)
    
    t_inv_theta = np.exp(log_t / theta)

    term1 = -t_inv_theta
    term2 = (theta - 1) * (log_x + log_y)  # equivalent to log( (x*y)^(theta-1) )
    
    term3 = np.log(np.maximum(t_inv_theta + theta - 1, 1e-10))

    term4 = (1.0/theta - 2) * log_t
    
    term5 = x + y

    return term1 + term2 + term3 + term4 + term5

def calculate_garch_loglik_series(model_result, dist_name='skewt'):
    """
    Robustly calculates the log-likelihood series using arch's internal methods.
    """
    resid = model_result.resid
    sigma2 = model_result.conditional_volatility ** 2
    params = model_result.params
    
    if dist_name == 'skewt':
        dist = SkewStudent()
        eta = params.get('nu', params.get('eta', 8.0))
        lmbda = params.get('lambda', 0.0)
        dist_params = [eta, lmbda]
    elif dist_name == 't':
        dist = StudentsT()
        nu = params.get('nu', params.get('eta', 8.0))
        dist_params = [nu]

    elif dist_name == 'ged':
        dist = GeneralizedError()
        dist_params = [params.get('nu', 2.0)]

    elif dist_name == 'norm':
        dist = Normal()
        dist_params = []
    else:
        raise ValueError(f"Unsupported dist for loglik: {dist_name}")
        
    # Calculate LL for every point
    ll_values = dist.loglikelihood(dist_params, resid, sigma2)
    return pd.Series(ll_values, index=resid.index).dropna()

def select_best_arma(series, max_order=MAX_ARMA_ORDER):
    """
    Selects the best ARMA(p,q) order based on AIC.
    Returns (best_p, best_q).
    (Assumes this function exists and works correctly)
    """
    # Placeholder implementation - replace with your actual function
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
    print(f"    Best ARMA order selected: {best_order} (AIC: {best_aic:.2f})")
    return best_order


def fit_best_garch(series, p, q, mean_p, mean_q, dist=GARCH_DIST):
    """
    Fits the specified ARMA(mean_p, mean_q)-GARCH(p,q) model.
    Returns the fitted model result.
    (Assumes this function exists and works correctly)
    """
    # Placeholder implementation - replace with your actual function
    series = series.dropna()
    if series.empty: return None

    try:
        am = arch_model(series, vol='EGARCH', p=p, q=q,
                        mean='AR', lags=mean_p, # Specify ARMA orders here
                        dist=dist)
        res = am.fit(update_freq=0, disp='off')
        return res
    except Exception as e:
        print(f"    Error fitting GARCH: {e}")
        return None


def get_standardized_residuals(model_result):
    """
    Extracts standardized residuals from a fitted GARCH model.
    Returns a pandas Series.
    (Assumes this function exists and works correctly)
    """
    # Placeholder implementation - replace with your actual function
    if model_result is None: return pd.Series(dtype=float)
    return model_result.std_resid


def transform_to_uniform(residuals, dist_params):
    """
    Transforms residuals to uniform [0,1] using the specified distribution's CDF.
    Returns a numpy array.
    (Assumes this function exists and works correctly)
    """

    residuals = residuals.dropna()
    if residuals.empty: return np.array([])

    dist_name = dist_params.get('dist', 't')

    if dist_name == 'skewt':

        dist = SkewStudent()
        eta = dist_params.get('nu', dist_params.get('eta', 8.0))
        lmbda = dist_params.get('lambda', 0.0)
    
        return dist.cdf(residuals.values, [eta, lmbda])
    
    elif dist_name == 't':
        nu = dist_params.get('nu', 4) # Default or get from fit
        return distributions.t.cdf(residuals, df=nu)
    
    elif dist_name == 'ged':
        dist = GeneralizedError()
        return dist.cdf(residuals.values, [dist_params['nu']])
    
    elif dist_name == 'norm':
         return distributions.norm.cdf(residuals)
    
    else:
        # Fallback to Empirical CDF (ECDF) if distribution unknown
        from statsmodels.distributions.empirical_distribution import ECDF
        ecdf = ECDF(residuals)
        return ecdf(residuals)

def get_params(model_fit, dist_type):
    params = model_fit.params
    d_params = {'dist': dist_type}
    if dist_type == 'skewt':
        
        d_params['eta'] = params.get('nu', params.get('eta', 8.0))
        d_params['lambda'] = params.get('lambda', 0.0)
    elif dist_type == 't':
        d_params['nu'] = params.get('nu', 8.0)
    elif dist_type == 'ged':
        d_params['nu'] = params.get('nu', 2.0)
    return d_params

def gaussian_copula_logpdf(u, v, rho):
    """Calculates log PDF for Gaussian copula."""
    if not -1 < rho < 1: return -np.inf
    term1 = -0.5 * np.log(1 - rho**2)
    z_u = norm.ppf(u)
    z_v = norm.ppf(v)
    term2 = - (rho**2 * (z_u**2 + z_v**2) - 2 * rho * z_u * z_v) / (2 * (1 - rho**2))
    
    return term1 + term2

def t_copula_logpdf(u, v, rho, nu):
    """Calculates log PDF for Student-t copula."""
    if not -1 < rho < 1 or nu <= 2: return -np.inf
    t_u = t.ppf(u, df=nu)
    t_v = t.ppf(v, df=nu)

    data_matrix = np.column_stack([t_u, t_v])
    log_num = multivariate_t.logpdf(data_matrix, shape=[[1, rho], [rho, 1]], df=nu)
    log_den1 = t.logpdf(t_u, df=nu)
    log_den2 = t.logpdf(t_v, df=nu)

    return log_num - log_den1 - log_den2


def fit_copula(u, v, copula_type='t'):
    """
    Fits the specified copula type using Maximum Likelihood Estimation.
    Returns parameters and max log-likelihood.
    (Assumes this function exists and works correctly - simplified here)
    """
    u = np.clip(u, 1e-6, 1 - 1e-6)
    v = np.clip(v, 1e-6, 1 - 1e-6)
    data = np.vstack([u, v]).T

    if copula_type == 'gaussian':
        def neg_log_lik(params):
            rho = params[0]
            if not -0.999 < rho < 0.999: return np.inf
            loglik = gaussian_copula_logpdf(u, v, rho)
            return -np.sum(loglik)

        initial_rho = np.corrcoef(norm.ppf(u), norm.ppf(v))[0, 1]
        bounds = [(-0.999, 0.999)]
        result = minimize(neg_log_lik, [initial_rho], method='L-BFGS-B', bounds=bounds)
        if result.success:
            rho_mle = result.x[0]
            mll = -result.fun
            params = {'rho': rho_mle, 'type': 'gaussian'}
            print(f"    Fitted Gaussian Copula: rho={rho_mle:.4f}")
            return params, mll
        else:
            print(f"    Gaussian Copula fitting failed: {result.message}")
            return None, -np.inf

    elif copula_type == 't':
        def neg_log_lik(params):
            rho, nu = params
            if not -0.999 < rho < 0.999 or nu <= 2.1: return np.inf
            loglik = t_copula_logpdf(u, v, rho, nu)
            return -np.sum(loglik)

        # Good initial guesses are important
        initial_rho = np.corrcoef(t.ppf(u, df=4), t.ppf(v, df=4))[0, 1] # Use robust correlation
        initial_nu = 4.0
        bounds = [(-0.999, 0.999), (2.1, 50)] # Bound nu for stability

        result = minimize(neg_log_lik, [initial_rho, initial_nu], method='L-BFGS-B', bounds=bounds)
        if result.success:
            rho_mle, nu_mle = result.x
            mll = -result.fun
            params = {'rho': rho_mle, 'nu': nu_mle, 'type': 't'}
            print(f"    Fitted Student-t Copula: rho={rho_mle:.4f}, nu={nu_mle:.2f}")
            return params, mll
        else:
            print(f"    Student-t Copula fitting failed: {result.message}")
            return None, -np.inf

    elif copula_type == 'clayton':
        res = minimize(lambda p: -np.sum(clayton_copula_logpdf(u, v, p[0])), 
                       x0=[1.0], # Initial guess
                       method='L-BFGS-B', 
                       bounds=[(0.01, 20.0)]) # Theta must be > 0
        if res.success:
             print(f"    Fitted Clayton: theta={res.x[0]:.4f}")
             return {'theta': res.x[0], 'type': 'clayton'}, -res.fun
        else: return None, -np.inf

    elif copula_type == 'gumbel':
        res = minimize(lambda p: -np.sum(gumbel_copula_logpdf_robust(u, v, p[0])), 
                       x0=[1.1], # Initial guess must be >= 1
                       method='L-BFGS-B', 
                       bounds=[(1.001, 20.0)]) # Theta must be >= 1
        if res.success:
             print(f"    Fitted Gumbel: theta={res.x[0]:.4f}")
             return {'theta': res.x[0], 'type': 'gumbel'}, -res.fun
        else: return None, -np.inf

    else:
        raise ValueError(f"Unsupported copula type: {copula_type}")

def select_best_copula(u, v, copula_types=COPULA_TYPES):
    """Selects the best copula based on AIC."""
    best_aic = np.inf
    best_params = None
    best_mll = -np.inf

    for c_type in copula_types:
        print(f"    Trying Copula Type: {c_type}")
        params, mll = fit_copula(u, v, copula_type=c_type)
        if params is not None:
            k = len(params) -1 # Number of parameters (-1 for 'type' key)
            aic = 2 * k - 2 * mll
            print(f"      MLL={mll:.2f}, AIC={aic:.2f}")
            if aic < best_aic:
                best_aic = aic
                best_params = params
                best_mll = mll

    print(f"    Best Copula Selected: {best_params.get('type','N/A')} (AIC: {best_aic:.2f})")
    return best_params, best_mll


def calculate_copula_loglik_series(u, v, copula_params):
    """Calculates the time series of log likelihoods for a fitted copula."""
    u = np.clip(u, 1e-6, 1 - 1e-6)
    v = np.clip(v, 1e-6, 1 - 1e-6)

    copula_type = copula_params.get('type')
    if copula_type == 'gaussian':
        rho = copula_params['rho']
        return gaussian_copula_logpdf(u, v, rho)
    elif copula_type == 't':
        rho = copula_params['rho']
        nu = copula_params['nu']
        return t_copula_logpdf(u, v, rho, nu)
    elif copula_type == 'clayton':
        return clayton_copula_logpdf(u, v, copula_params['theta'])
    elif copula_type == 'gumbel':
        return gumbel_copula_logpdf_robust(u, v, copula_params['theta'])
    else:
        raise ValueError("Unsupported copula type")


def amisano_giacomini_test(loglik_diff_series, hac_lags=None):
    """
    Performs the Amisano-Giacomini test for comparing predictive likelihoods.
    loglik_diff_series = loglik_copula_model - loglik_benchmark_model
    Returns (test_statistic, p_value)
    """
    d = loglik_diff_series.dropna()
    T = len(d)
    if T < 2: return np.nan, np.nan

    # The AG test is a t-test on the mean of the loss-differential series (d),
    # using a HAC standard error. We get this from a constant-only OLS model.
    
    # 1. Set up the OLS model: d = constant
    X = sm.add_constant(np.ones(T))
    
    # 2. Fit the OLS model
    try:
        ols_fit = sm.OLS(d, X).fit()
    except Exception as e:
        print(f"  Error in AG Test OLS fit: {e}")
        return np.nan, np.nan
    
    # 3. Get the mean (which is the 'const' parameter)
    mean_d = ols_fit.params[0]
    
    # 4. Set default HAC lags if not provided (a common rule of thumb)
    if hac_lags is None:
        hac_lags = int(np.floor(T**(1/4))) # Using T^(1/4) rule

    try:
        # 5. Get the HAC-robust covariance matrix for the parameters
        hac_cov = ols_fit.get_robustcov_results(
            cov_type='HAC', 
            maxlags=hac_lags
        )
        
        # 6. The AG statistic is the t-statistic of the 'const' parameter
        ag_stat = hac_cov.tvalues[0]
        
        # 7. Get the two-sided p-value
        p_value = hac_cov.pvalues[0]

        return ag_stat, p_value

    except Exception as e:
        print(f"  Error calculating HAC variance in AG Test: {e}")
        return np.nan, np.nan


# ===================================================================
# NEW PCA Factor Creation Function (Adapted from testCGC_PCA.py)
# ===================================================================
def create_pca_factor(coins, var, data_dict, factor_name):
    """
    Builds a data matrix, standardizes, and runs PCA.
    Returns a pd.Series (the PC1 factor) aligned by date.
    """
    df_list = [data_dict[coin][var] for coin in coins if coin in data_dict and var in data_dict[coin].columns]
    if not df_list:
        print(f"Warning: No data found for variable '{var}' in coins {coins}.")
        return pd.Series(dtype=float, name=factor_name)

    # Use outer join first to get all dates, then inner on concat? Or just inner.
    # Inner join is safer to ensure all data exists for a given date.
    data_matrix = pd.concat(df_list, axis=1, keys=coins, join='inner')
    data_matrix.dropna(inplace=True) # Drop rows where any coin is missing data

    if data_matrix.empty:
         print(f"Warning: No overlapping data found for variable '{var}' in coins {coins}.")
         return pd.Series(dtype=float, name=factor_name)

    # Standardize
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_matrix)

    # PCA
    pca = PCA(n_components=1)
    factor = pca.fit_transform(data_scaled)

    print(f"Created Factor '{factor_name}'. PCA Explained Variance: {pca.explained_variance_ratio_[0]:.2%}")
    # Print component loadings (the "drill-down")
    loadings_dict = {coin: loading for coin, loading in zip(data_matrix.columns, pca.components_[0])}
    print(f"  Loadings: {loadings_dict}\n")

    return pd.Series(factor.ravel(), index=data_matrix.index, name=factor_name)


# ===================================================================
# Main Script Logic
# ===================================================================

print("Loading and preparing data...")
# Load all coin data
coin_data = {}
stablecoins = ["DAI", "USDC", "USDT"]
cryptos = ["BNB", "BTC", "ETH", "XRP"]
all_coins = stablecoins + cryptos

for file in DATA_DIR.glob("*.csv"):
    coin_name = file.stem.replace("Verif_", "")
    if coin_name in all_coins:
        df = pd.read_csv(file, parse_dates=['Date']).sort_values("Date")
        df = df[(df['Date'] >= START_DATE) & (df['Date'] <= END_DATE)]
        coin_data[coin_name] = df.set_index('Date')

print("Creating PCA Factors...")
factors = {}
factors["Stable_Volume"] = create_pca_factor(
    stablecoins, "LogVolChange", coin_data, "PC1_Stable_Volume"
)
factors["Stable_Volatility"] = create_pca_factor(
    stablecoins, STATIONARY_VOL, coin_data, "PC1_Stable_Volatility"
)
factors["Crypto_Returns"] = create_pca_factor(
    cryptos, "Log Returns", coin_data, "PC1_Crypto_Returns"
)
factors["Crypto_Volatility"] = create_pca_factor(
    cryptos, STATIONARY_VOL, coin_data, "PC1_Crypto_Volatility"
)

print("-" * 50)
print("Running Copula-GARCH Backtests for 4 Factor Pairs")
print("-" * 50)

# Define the 4 tests (Source Key, Target Key)
tests_to_run = [
    ("Stable_Volume", "Crypto_Returns"),
    ("Stable_Volume", "Crypto_Volatility"),
    ("Stable_Volatility", "Crypto_Returns"),
    ("Stable_Volatility", "Crypto_Volatility"),
]

ag_results = []

for source_key, target_key in tests_to_run:
    print(f"\n===== TESTING: {source_key} -> {target_key} =====")

    y_series = factors[target_key] # Target series
    x_series = factors[source_key] # Predictor series

    x_series = x_series.shift(1) # Lag Stablecoin Data

    # Align data (important!) - use inner join to ensure common dates
    aligned_data = pd.concat([y_series, x_series], axis=1, join='inner').dropna()
    y = aligned_data[y_series.name]
    x = aligned_data[x_series.name]

    if len(y) < 50: # Need enough data for GARCH
        print("  Skipping test: Not enough overlapping data points.")
        continue

    # --- 1. Fit Benchmark Model (ARMA-GARCH on Target Y) ---
    print(f"  Fitting Benchmark ARMA-GARCH for {target_key}...")
    arma_order_y = select_best_arma(y, max_order=MAX_ARMA_ORDER)
    bench_model_fit = fit_best_garch(y, p=MAX_GARCH_ORDER, q=MAX_GARCH_ORDER,
                                     mean_p=arma_order_y[0], mean_q=arma_order_y[1],
                                     dist=GARCH_DIST)
    
    if bench_model_fit is None:
        print("  Failed to fit benchmark model. Skipping test.")
        continue

    # 1. Define the SERIES for the AG test
    bench_loglik_series = calculate_garch_loglik_series(bench_model_fit, dist_name=GARCH_DIST)

    # 2. Define the FLOAT for printing
    bench_loglik = bench_model_fit.loglikelihood 

    bench_resid = get_standardized_residuals(bench_model_fit)
    bench_dist_params = get_params(bench_model_fit, GARCH_DIST)
    bench_uniform = transform_to_uniform(bench_resid, bench_dist_params)
    
    # 3. Fix the print statement to use the FLOAT variable
    print(f"  Benchmark LogLikelihood: {bench_loglik:.2f}")

    # --- 2. Fit ARMA-GARCH for Predictor X (Needed for its residuals) ---
    print(f"  Fitting ARMA-GARCH for predictor {source_key}...")
    arma_order_x = select_best_arma(x, max_order=MAX_ARMA_ORDER)
    x_model_fit = fit_best_garch(x, p=MAX_GARCH_ORDER, q=MAX_GARCH_ORDER,
                                 mean_p=arma_order_x[0], mean_q=arma_order_x[1],
                                 dist=GARCH_DIST)
    if x_model_fit is None:
        print("  Failed to fit predictor model. Skipping test.")
        continue
    x_resid = get_standardized_residuals(x_model_fit)
    x_dist_params = get_params(x_model_fit, GARCH_DIST)
    x_uniform = transform_to_uniform(x_resid, x_dist_params)

    # --- 3. Fit Copula to Residuals ---
    print(f"  Fitting Copula between residuals of {source_key} and {target_key}...")
    # Ensure residuals are aligned before fitting copula

    # 1. Combine the two residual series into a single DataFrame
    resid_df = pd.concat([bench_resid, x_resid], axis=1, keys=['bench', 'x'])

    # 2. Drop any row where *either* residual is NaN
    resid_df.dropna(inplace=True)

    if len(resid_df) < 2:
        print("  Not enough aligned residuals for copula fitting. Skipping test.")
        continue

    # 3. Now that they are perfectly aligned and NaN-free, transform them.
    u_target = transform_to_uniform(resid_df['bench'], bench_dist_params)
    v_source = transform_to_uniform(resid_df['x'], x_dist_params)

    print("    Verifying Marginal Model Adequacy (KS-Test)...")
    ks_p_target = check_uniformity(u_target, f"Target ({target_key})")
    ks_p_source = check_uniformity(v_source, f"Source ({source_key})")
    
    best_copula_params, copula_mll = select_best_copula(u_target, v_source, copula_types=COPULA_TYPES)

    if best_copula_params is None:
        print("  Failed to fit any copula. Skipping test.")
        continue

    # --- 4. Calculate Log Likelihoods for AG Test ---

    # Copula model loglik series = Benchmark LL + Copula LL
    # We need to align the series again before calculation
    common_index_lik = bench_loglik_series.index.intersection(resid_df.index)

    u_series = pd.Series(u_target, index=resid_df.index)
    v_series = pd.Series(v_source, index=resid_df.index)

    u_target_aligned = u_series.loc[common_index_lik].values
    v_source_aligned = v_series.loc[common_index_lik].values
    
    #u_target_aligned = transform_to_uniform(bench_resid.loc[common_index_lik], bench_dist_params)
    #v_source_aligned = transform_to_uniform(x_resid.loc[common_index_lik], x_dist_params)

    copula_loglik_series = calculate_copula_loglik_series(u_target_aligned, v_source_aligned, best_copula_params)
    
    # Ensure copula_loglik_series is a pandas Series with the correct index
    copula_loglik_series = pd.Series(copula_loglik_series, index=common_index_lik)

    # Combine benchmark likelihoods and copula likelihoods
    # Align them first
    combined_lik = pd.concat([bench_loglik_series, copula_loglik_series], axis=1, join='inner')
    combined_lik.columns = ['bench_ll', 'copula_ll_part']

    # Full Copula model LL = Benchmark LL (Target Margin) + Copula LL Part
    copula_full_loglik_series = combined_lik['bench_ll'] + combined_lik['copula_ll_part']
    
    # Log Likelihood Difference Series d_t = LL_Copula - LL_Benchmark
    loglik_diff_series = copula_full_loglik_series - combined_lik['bench_ll'] # This simplifies to just copula_ll_part

    # --- 5. Run Amisano-Giacomini Test ---
    print("  Running Amisano-Giacomini Test...")
    ag_stat, p_value = amisano_giacomini_test(loglik_diff_series, hac_lags=HAC_LAGS)
    print(f"  --> AG Statistic = {ag_stat:.4f}, p-value = {p_value:.4f}")

    ag_results.append({
        "Source_Factor": source_key,
        "Target_Factor": target_key,
        "Benchmark_LL": bench_loglik,
        "Copula_Type": best_copula_params.get('type', 'N/A'),
        "Copula_Params": {k:v for k,v in best_copula_params.items() if k!='type'},
        "Copula_Part_LL": copula_mll, # LogLikelihood of JUST the copula part
        "AG_Statistic": ag_stat,
        "AG_p_value": p_value
    })

# --- 6. Final Output ---
print("\n" + "=" * 50)
print("Final Amisano-Giacomini Test Results")
print("=" * 50)
results_df = pd.DataFrame(ag_results)
print(results_df.to_string())

# Save results
results_df.to_csv("Results\GARCH\copula_garch_pca_results.csv", index=False)
print("\nResults saved to 'Results\GARCH\copula_garch_pca_results.csv'")