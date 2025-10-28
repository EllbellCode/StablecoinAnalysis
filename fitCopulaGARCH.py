import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from arch import arch_model
import pyvinecopulib as pvc
import warnings

warnings.filterwarnings("ignore")

# =============================================================================
# --- Helper Functions ---
# =============================================================================

def plot_results(results_df, crypto_col, stable_col):
    plt.figure(figsize=(15, 7))
    plt.plot(results_df['Actual'], 'k-', label='Actual Log Return', alpha=0.5, linewidth=1)
    plt.plot(results_df['Benchmark_VaR'], 'b-', label='Benchmark (EGARCH) 5% VaR', linewidth=1.5)
    plt.plot(results_df['Challenger_VaR'], 'r--', label=f'Challenger (Copula-EGARCH with {stable_col}) 5% VaR', linewidth=1.5)
    bench_breaches = results_df[results_df['Actual'] < results_df['Benchmark_VaR']]
    chall_breaches = results_df[results_df['Actual'] < results_df['Challenger_VaR']]
    plt.scatter(bench_breaches.index, bench_breaches['Actual'], color='blue', s=40, label=f'Benchmark Breach ({len(bench_breaches)})', zorder=5)
    plt.scatter(chall_breaches.index, chall_breaches['Actual'], color='red', s=40, marker='x', label=f'Challenger Breach ({len(chall_breaches)})', zorder=5)
    plt.title(f'VaR Backtest: {crypto_col} (1-Year Test Set)', fontsize=16)
    plt.ylabel('Log Return')
    plt.legend(loc='lower left')
    plt.grid(True, alpha=0.3)
    plt.ylim(results_df[['Actual', 'Benchmark_VaR', 'Challenger_VaR']].min().min() - 0.01,
             results_df[['Actual', 'Benchmark_VaR', 'Challenger_VaR']].max().max() + 0.01)
    plt.tight_layout()
    plt.show()

def print_statistics(results_df, n_test_effective, alpha=0.05):
    bench_breaches = (results_df['Actual'] < results_df['Benchmark_VaR']).sum()
    chall_breaches = (results_df['Actual'] < results_df['Challenger_VaR']).sum()
    bench_pct = bench_breaches / n_test_effective
    chall_pct = chall_breaches / n_test_effective
    expected_pct = alpha
    print("\n--- Backtest Statistics ---")
    print(f"Effective Test Days (after removing NaNs): {n_test_effective}")
    print(f"Expected Breach Rate: {expected_pct:.1%} ({n_test_effective * expected_pct:.1f} days)")
    print("-" * 30)
    print(f"Benchmark (EGARCH):")
    print(f"  Breaches: {bench_breaches} days")
    print(f"  Breach Rate: {bench_pct:.2%}")
    print("-" * 30)
    print(f"Challenger (Copula-EGARCH):")
    print(f"  Breaches: {chall_breaches} days")
    print(f"  Breach Rate: {chall_pct:.2%}")
    print("-" * 30)
    if np.abs(bench_pct - expected_pct) <= np.abs(chall_pct - expected_pct):
        print("Winner (closer to expected rate): Benchmark")
    else:
        print("Winner (closer to expected rate): Challenger")

def fit_egarch_skewt(data, series_name=""):
    """
    Fits an EGARCH(1,1) model with a Skewed Student's t distribution.
    """
    model = arch_model(data, vol='EGARCH', p=1, o=1, q=1, dist='skewt')
    res = model.fit(disp='off', show_warning=False)

    df_param_name = 'eta'
    skew_param_name = 'lambda'

    std_resid = res.std_resid
    forecast = res.forecast(horizon=1, reindex=False)

    forecast_params = {
        'mu': forecast.mean.iloc[-1, 0],
        'sigma': np.sqrt(forecast.variance.iloc[-1, 0]),
        'df': res.params[df_param_name],
        'skew': res.params[skew_param_name]
    }

    return res, std_resid, forecast_params

fit_egarch_skewt.df_param_name = 'eta'
fit_egarch_skewt.skew_param_name = 'lambda'


def get_pit(std_resid):
    """
    Performs the Probability Integral Transform (PIT) on standardized residuals
    using a non-parametric (empirical) CDF to get U(0,1) data.
    """
    u_data = (pd.Series(std_resid).rank().values - 0.5) / len(std_resid)
    return np.clip(u_data, 1e-6, 1 - 1e-6)

def fit_copula(u1, u2):
    """
    Fits a bivariate copula, allowing .select() to choose the best family
    from a predefined set based on BIC.
    """
    test_families = [
        pvc.BicopFamily.indep,
        pvc.BicopFamily.gaussian,
        pvc.BicopFamily.student,
        pvc.BicopFamily.clayton,
        pvc.BicopFamily.gumbel,
        pvc.BicopFamily.frank
    ]

    controls = pvc.FitControlsBicop(
        family_set=test_families,
        selection_criterion='bic'
    )
    
    data_np = np.column_stack((u1, u2))

    copula = pvc.Bicop()
    copula.select(data=data_np, controls=controls)
        
    return copula


def run_backtest(df_full, crypto_col, stable_col, n_train, n_test):
    """
    Runs the main "horse race" backtest using EGARCH-SkewT marginals.
    """
    actuals = []
    benchmark_vaR_5pct = []
    challenger_vaR_5pct = []

    benchmark_sigmas = []
    benchmark_dfs = []
    benchmark_skews = []

    print(f"--- Running Backtest ---")
    print(f"Target: {crypto_col}")
    print(f"Predictor: {stable_col}")
    print(f"Training size: {n_train}, Test size: {n_test}\n")

    for i in tqdm(range(n_test)):
        train_start = 0
        train_end = n_train + i
        test_idx = n_train + i

        actual_outcome = df_full[crypto_col].iloc[test_idx]
        actuals.append(actual_outcome)

        crypto_train = df_full[crypto_col].iloc[train_start:train_end] * 100
        stable_train = df_full[stable_col].iloc[train_start:train_end] * 100

        # --- A. BENCHMARK FORECAST (EGARCH-SkewT) ---
        bench_res, _, bench_fc_params = fit_egarch_skewt(crypto_train, f"Benchmark {crypto_col} @ step {i}")

        bench_df_val = bench_fc_params.get('df', np.nan)
        bench_sigma_val = bench_fc_params.get('sigma', np.nan)
        bench_mu_val = bench_fc_params.get('mu', np.nan)
        bench_skew_val = bench_fc_params.get('skew', np.nan)

        bench_dist = bench_res.model.distribution
        bench_dist_params = [bench_df_val, bench_skew_val]
        z_quantile = bench_dist.ppf(0.05, bench_dist_params)
        #bench_vaR = (z_quantile * bench_sigma_val + bench_mu_val) / 100 # Scale back
        bench_vaR = bench_mu_val / 100

        benchmark_vaR_5pct.append(bench_vaR)
        benchmark_sigmas.append(bench_sigma_val / 100.0 if np.isfinite(bench_sigma_val) else np.nan)
        benchmark_dfs.append(bench_df_val)
        benchmark_skews.append(bench_skew_val)


        # --- B. CHALLENGER FORECAST (Copula-EGARCH-SkewT) ---
        
        # 1. Fit marginals
        crypto_res, crypto_std_resid, crypto_fc_params = fit_egarch_skewt(crypto_train, f"Challenger {crypto_col} @ step {i}")
        stable_res, stable_std_resid, stable_fc_params = fit_egarch_skewt(stable_train, f"Challenger {stable_col} @ step {i}")

        # 2. Get PIT data
        u_crypto_full = get_pit(crypto_std_resid)
        u_stable_full = get_pit(stable_std_resid)

        # 3. Create lagged datasets for the copula
        u_crypto_for_copula = u_crypto_full[1:]
        u_stable_for_copula = u_stable_full[:-1]

        # 4. Fit copula on the *lagged* data
        copula = fit_copula(u_crypto_for_copula, u_stable_for_copula)

        # 5. Generate Conditional Forecast
        z_stable_last = stable_std_resid[-1]
        stable_df = stable_fc_params.get('df', np.nan)
        stable_skew = stable_fc_params.get('skew', np.nan)
        
        u_stable_cond = u_stable_full[-1]
        u_stable_cond_clipped = np.clip(u_stable_cond, 1e-6, 1 - 1e-6)

        # --- Simulation using hinv1 (with fix for indep copula) ---
        n_sim = 5000
        v_samples = np.random.rand(n_sim) # These are V ~ U(0,1)

        if copula.family == pvc.BicopFamily.indep:
            # For an independence copula, U_1 | U_2 = V.
            u_crypto_sim = v_samples
        else:
            # For all other (dependent) copulas, hinv1 works correctly.
            hinv1_input = np.column_stack((v_samples, np.full(n_sim, u_stable_cond_clipped)))
            hinv1_input = np.asfortranarray(hinv1_input)
            u_crypto_sim = copula.hinv1(hinv1_input)

        u_crypto_sim = np.clip(u_crypto_sim, 1e-6, 1-1e-6)
        # --- End Simulation ---

        # Get crypto forecast params
        crypto_df = crypto_fc_params.get('df', np.nan)
        crypto_skew = crypto_fc_params.get('skew', np.nan)
        fc_sigma = crypto_fc_params.get('sigma', np.nan)
        fc_mu = crypto_fc_params.get('mu', np.nan)
        
        # Instantiate SkewStudent for crypto PPF
        crypto_dist = crypto_res.model.distribution
        crypto_dist_params = [crypto_df, crypto_skew]
        z_crypto_sim = crypto_dist.ppf(u_crypto_sim, crypto_dist_params)

        # Final transformation and VaR
        returns_sim = (z_crypto_sim * fc_sigma + fc_mu) / 100 # Scale back
        challenger_vaR = np.percentile(returns_sim, 50)

        challenger_vaR_5pct.append(challenger_vaR)


    # --- C. Package results ---
    results_df = pd.DataFrame({
        'Actual': actuals,
        'Benchmark_VaR': benchmark_vaR_5pct,
        'Challenger_VaR': challenger_vaR_5pct,
        'Benchmark_Sigma': benchmark_sigmas,
        'Benchmark_DF': benchmark_dfs,
        'Benchmark_Skew': benchmark_skews
    }, index=df_full.index[n_train:n_train + n_test])

    initial_rows = len(results_df)
    results_df.dropna(subset=['Benchmark_VaR', 'Challenger_VaR'], inplace=True)
    final_rows = len(results_df)
    print(f"Removed {initial_rows - final_rows} rows due to NaN forecasts.")

    return results_df


# --- Main Execution Block ---
if __name__ == "__main__":
    # --- Load Data ---
    df_btc = pd.read_csv(r"Data\Verified\Verif_BTC.csv", index_col='Date', parse_dates=True)
    df_usdt = pd.read_csv(r"Data\Verified\Verif_USDT.csv", index_col='Date', parse_dates=True)

    # --- Prepare Data ---
    crypto_target_col = 'Log Returns'
    stable_predictor_col = 'Log RV'
    df_btc_subset = df_btc[[crypto_target_col]].rename(columns={crypto_target_col: 'BTC_LogRet'})
    df_usdt_subset = df_usdt[[stable_predictor_col]].rename(columns={stable_predictor_col: 'USDT_LogRV'})
    df_merged = pd.merge(df_btc_subset, df_usdt_subset, left_index=True, right_index=True, how='inner')
    df_merged.dropna(subset=['BTC_LogRet', 'USDT_LogRV'], inplace=True)
    
    print(f"Data loaded and merged. Shape: {df_merged.shape}")
    print(f"Date range: {df_merged.index.min()} to {df_merged.index.max()}")

    # --- Define Backtest Parameters ---
    total_obs = len(df_merged)
    n_test = 366 
    n_train = total_obs - n_test

    # --- Run the Backtest ---
    results = run_backtest(df_merged, 'BTC_LogRet', 'USDT_LogRV', n_train, n_test)

    # --- Show Results ---
    n_test_effective = len(results)
    
    print_statistics(results, n_test_effective, alpha=0.05)
    plot_results(results, 'BTC Log Returns', 'USDT LogRV') # VaR plot

    # --- Plot Benchmark Parameters (includes Skew) ---
    plt.figure(figsize=(15, 10))

    # Plot Sigma
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(results.index, results['Benchmark_Sigma'], label='Benchmark Fcst Sigma', color='purple')
    ax1_twin = ax1.twinx()
    ax1_twin.plot(results.index, results['Actual'], label='Actual Ret (RHS)', color='gray', alpha=0.4, lw=0.8)
    ax1.set_ylabel('Volatility (Std Dev)')
    ax1_twin.set_ylabel('Log Return')
    ax1.set_title('Benchmark Model Parameters (EGARCH-SkewT) during Backtest')
    ax1.legend(loc='upper left'); ax1_twin.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Plot DF ('eta')
    ax2 = plt.subplot(3, 1, 2, sharex=ax1)
    ax2.plot(results.index, results['Benchmark_DF'], label='Benchmark Fitted DF (eta)', color='green')
    ax2.axhline(y=2, color='red', linestyle='--', lw=1, label='DF=2 (Min Req)') # Label updated
    ax2.axhline(y=4, color='orange', linestyle='--', lw=1, label='DF=4')
    ax2.set_ylabel('Degrees of Freedom (eta)')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    df_q99 = results['Benchmark_DF'].quantile(0.99)
    df_top_limit = max(10, df_q99 * 1.1 if np.isfinite(df_q99) else 50)
    ax2.set_ylim(bottom=1.5, top=df_top_limit) # Min possible is > 2

    # Plot Skew ('lambda')
    ax3 = plt.subplot(3, 1, 3, sharex=ax1)
    ax3.plot(results.index, results['Benchmark_Skew'], label='Benchmark Fitted Skew (lambda)', color='blue')
    ax3.axhline(y=0, color='black', linestyle=':', lw=1, label='Symmetric (lambda=0)')
    ax3.set_ylabel('Skew Parameter (lambda)')
    ax3.set_xlabel('Date')
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3)
    skew_abs_max = results['Benchmark_Skew'].abs().max()
    if np.isfinite(skew_abs_max):
            limit = max(1.1 * skew_abs_max, 0.1)
            ax3.set_ylim(-limit, limit)

    plt.tight_layout()
    plt.show()