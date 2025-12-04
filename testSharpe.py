import pandas as pd
from pathlib import Path
import warnings
import numpy as np
import sys
import os
from scipy.stats import norm

# --- Import core bootstrap components ---
# This is more fundamental than SharpeRatioTest and should exist in older versions
try:
    from arch.bootstrap import CircularBlockBootstrap
except ImportError:
    print("--- ENVIRONMENT PROBLEM DETEECTED ---", file=sys.stderr)
    print(f"Error: Your 'arch' library is missing 'CircularBlockBootstrap'.", file=sys.stderr)
    print("This is a core component. Your 'arch' installation may be corrupted or very old.", file=sys.stderr)
    print("\nPlease try running this command to fix it:\n", file=sys.stderr)
    print(f'"{sys.executable}" -m pip install --upgrade arch\n', file=sys.stderr)
    sys.exit(1)


warnings.filterwarnings('ignore')

# --- Configuration ---
RESULTS_DIR = Path("Results/ML/Winsor")
N_BOOTSTRAP_REPS = 1000 # Number of bootstrap replications

def calculate_sharpe(returns):
    """
    Calculates the annualized Sharpe ratio.
    Helper function for the bootstrap.
    """
    # Assuming daily data and 0 risk-free rate
    # We use population std (ddof=0) for consistency
    mean_ret = np.mean(returns)
    std_ret = np.std(returns, ddof=0)
    
    if std_ret == 0:
        # If no variance, return 0 or inf based on mean
        return np.inf if mean_ret > 0 else 0.0
        
    # We annualize here, assuming DAYS_NUMBER = 366 from main script
    # This is consistent with your main script's calculation
    return (mean_ret / std_ret) * np.sqrt(366)

def sharpe_diff_stat(x):
    """
    This is the function applied to each bootstrap sample.
    It calculates the difference in Sharpe Ratios.
    x is an (N, 2) array of [challenger_returns, benchmark_returns]
    """
    chall_returns = x[:, 0]
    bench_returns = x[:, 1]
    
    sharpe_chall = calculate_sharpe(chall_returns)
    sharpe_bench = calculate_sharpe(bench_returns)
    
    return sharpe_chall - sharpe_bench

def run_sharpe_tests():
    """
    Finds all prediction CSVs, runs a manual Ledoit & Wolf test
    using Circular Block Bootstrap, and prints a summary table.
    """
    print(f"Scanning for prediction files in: {RESULTS_DIR.resolve()}\n")
    
    pred_files = list(RESULTS_DIR.glob("XG_preds_*.csv"))
    
    if not pred_files:
        print("No prediction files (XG_preds_*.csv) found.")
        return

    results_list = []

    for file in pred_files:
        try:
            df = pd.read_csv(file).dropna()
            
            if df.empty or len(df) < 30:
                print(f"Skipping {file.name}: File is empty or has insufficient data.")
                continue

            test_type = "Unknown"
            returns_bench_series = None
            returns_chall_series = None

            # --- Identify test type and get data ---
            if "GARCH" in file.name:
                test_type = "GARCH"
                required_cols = ['Return_Bench_G', 'Return_Chall_G']
                if not all(c in df.columns for c in required_cols):
                    print(f"Skipping {file.name}: Missing required GARCH return columns.")
                    continue
                returns_bench_series = df['Return_Bench_G']
                returns_chall_series = df['Return_Chall_G']

            elif "RAW" in file.name:
                test_type = "RAW"
                required_cols = ['Return_Bench_R', 'Return_Chall_R']
                if not all(c in df.columns for c in required_cols):
                    print(f"Skipping {file.name}: Missing required RAW return columns.")
                    continue
                returns_bench_series = df['Return_Bench_R']
                returns_chall_series = df['Return_Chall_R']

            elif "VOL" in file.name:
                test_type = "VOL"
                required_cols = ['Return_Bench_V', 'Return_Chall_V']
                if not all(c in df.columns for c in required_cols):
                    print(f"Skipping {file.name}: Missing required VOL return columns.")
                    continue
                returns_bench_series = df['Return_Bench_V']
                returns_chall_series = df['Return_Chall_V']
            
            else:
                print(f"Skipping {file.name}: Could not determine test type (GARCH or RAW).")
                continue
                
            # --- Start Manual Ledoit & Wolf Test ---
            
            # 1. Calculate observed statistics
            obs_sharpe_chall = calculate_sharpe(returns_chall_series)
            obs_sharpe_bench = calculate_sharpe(returns_bench_series)
            obs_stat_diff = obs_sharpe_chall - obs_sharpe_bench

            # 2. Prepare data for bootstrap
            # Stack returns into an (N, 2) array to preserve correlations
            stacked_returns = np.column_stack([returns_chall_series, returns_bench_series])
            
            # 3. Determine optimal block size (Ledoit & Wolf recommendation)
            # This is a common rule of thumb for CBB
            N = len(stacked_returns)
            block_size = int(N**(1/3)) + 1
            
            # 4. Set up and run the bootstrap
            # This simulates the sampling distribution of the Sharpe Ratio difference
            
            # --- FIX: Call signature based on provided documentation ---
            bs = CircularBlockBootstrap(block_size,         # block_size (int)
                                        stacked_returns,    # *args
                                        seed=123)            # **kwargs (seed)
            
            # Apply our statistic function to all bootstrap samples
            # Pass reps to the .apply() method
            bootstrapped_diffs = bs.apply(sharpe_diff_stat,
                                          reps=N_BOOTSTRAP_REPS)
            
            # 5. Calculate bootstrap standard error and p-value
            # Filter out any potential NaNs/Infs from bootstrap
            valid_diffs = bootstrapped_diffs[np.isfinite(bootstrapped_diffs)]
            if len(valid_diffs) < N_BOOTSTRAP_REPS * 0.9:
                print(f"Skipping {file.name}: Bootstrap produced too many invalid results.")
                continue

            se_boot = np.std(valid_diffs) # Standard error of the difference
            
            if se_boot == 0:
                # No variance in bootstrap results
                p_value = 0.0 if obs_stat_diff > 0 else 1.0
                test_stat = np.inf if obs_stat_diff > 0 else -np.inf
            else:
                # 6. Construct the t-statistic and p-value
                # H0: True_Diff = 0
                # H1: True_Diff > 0
                
                # --- BUG FIX HERE ---
                # The t-stat tests the null hypothesis that the difference is 0.
                # The old formula was: (obs_stat_diff - np.mean(valid_diffs)) / se_boot
                # This was incorrectly centering the statistic around its (non-zero) mean.
                # The correct formula is:
                test_stat = obs_stat_diff / se_boot
                
                # Get the one-sided p-value for H1: Sharpe(Challenger) > Sharpe(Benchmark)
                p_value = norm.sf(test_stat) # sf = 1 - cdf

            # Check for significance
            is_significant_and_better = (p_value < 0.05)

            results_list.append({
                "File": file.name,
                "Test_Type": test_type,
                "OOS_Days": len(df),
                "SR_Test_Statistic": test_stat,
                "p_value (one-sided)": p_value,
                "Challenger_Signif_Better (p < 0.05)": "Yes" if is_significant_and_better else "No"
            })

        except Exception as e:
            if "inputs do not contain finite values" in str(e):
                print(f"Skipping {file.name}: Contains non-finite (NaN/Inf) return values.")
            else:
                print(f"Error processing {file.name}: {e}")

    # --- Print the final results table ---
    if results_list:
        print(f"--- Manual Sharpe Ratio Test (CBB, {N_BOOTSTRAP_REPS} Reps) ---")
        results_df = pd.DataFrame(results_list).sort_values(by="File")
        
        # Pretty-print the table
        print(results_df.to_string(index=False, float_format="%.6f"))
    else:
        print("No valid results were processed.")

# Call the function directly to run the tests when the script is executed
run_sharpe_tests()