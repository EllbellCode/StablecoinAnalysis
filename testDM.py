import pandas as pd
from pathlib import Path
import warnings
from dieboldmariano import dm_test

warnings.filterwarnings('ignore')

# --- Configuration ---
# This path is based on the OUTPUT_DIR in your fitXGBoostNew.py script
RESULTS_DIR = Path("Results/ML/Winsor")

def run_dm_tests():
    """
    Finds all prediction CSVs in the RESULTS_DIR, runs Diebold-Mariano tests
    on them, and prints a summary table.
    """
    print(f"Scanning for prediction files in: {RESULTS_DIR.resolve()}\n")
    
    # Find all prediction CSV files
    pred_files = list(RESULTS_DIR.glob("XG_preds_*.csv"))
    
    if not pred_files:
        print("No prediction files (XG_preds_*.csv) found.")
        print("Please run the main 'fitXGBoostNew.py' script first to generate results.")
        return

    results_list = []

    for file in pred_files:
        try:
            df = pd.read_csv(file).dropna()
            
            if df.empty:
                print(f"Skipping {file.name}: File is empty or contains only NaNs.")
                continue

            test_type = "Unknown"
            # This library takes actuals and predictions
            actual_vals = None 
            pred_bench = None
            pred_chall = None

            # --- Identify test type and get data ---
            
            if "GARCH" in file.name:
                test_type = "GARCH"
                required_cols = ['Actual_Shock', 'Pred_Shock_Bench_G', 'Pred_Shock_Chall_G']
                if not all(c in df.columns for c in required_cols):
                    print(f"Skipping {file.name}: Missing required GARCH columns.")
                    continue
                
                actual_vals = df['Actual_Shock']
                pred_bench = df['Pred_Shock_Bench_G'] # P1
                pred_chall = df['Pred_Shock_Chall_G'] # P2

            elif "RAW" in file.name:
                test_type = "RAW"
                required_cols = ['Actual_Raw', 'Pred_Raw_Bench', 'Pred_Raw_Chall']
                if not all(c in df.columns for c in required_cols):
                    print(f"Skipping {file.name}: Missing required RAW columns.")
                    continue

                actual_vals = df['Actual_Raw']
                pred_bench = df['Pred_Raw_Bench']
                pred_chall = df['Pred_Raw_Chall']

            elif "VOL" in file.name:
                test_type = "VOL"
                required_cols = ['Actual_Vol', 'Pred_Vol_Bench', 'Pred_Vol_Chall']
                if not all(c in df.columns for c in required_cols):
                    print(f"Skipping {file.name}: Missing required RAW columns.")
                    continue
                
                actual_vals = df['Actual_Vol']
                pred_bench = df['Pred_Vol_Bench'] # P1
                pred_chall = df['Pred_Vol_Chall'] # P2
            
            else:
                print(f"Skipping {file.name}: Could not determine test type (GARCH or RAW).")
                continue

            # --- Run the Diebold-Mariano Test ---
            # H0: Benchmark and Challenger have equal forecast accuracy.
            # H1 (one_sided=True): Challenger (P2) has at least as much accuracy as Benchmark (P1)
            
            dm_stat, lib_pvalue = dm_test(
                actual_vals,         # V (The actual values)
                pred_bench,          # P1 (Benchmark predictions)
                pred_chall,          # P2 (Challenger predictions)
                h=1,                 # 1-step-ahead forecast
                one_sided=True       
            )
            
            # --- LOGIC FIX START ---
            # The library calculates the p-value from the left tail (P(Z < dm_stat)).
            # For our H1 ("Challenger is better"), we need the right tail (P(Z > dm_stat)).
            # Therefore, the correct p-value is 1.0 - the p-value from the library.
            
            correct_pvalue = 1.0 - lib_pvalue
            
            # With one_sided=True, a low p-value directly indicates P2 is better.
            is_significant_and_better = (correct_pvalue < 0.05)
            # --- LOGIC FIX END ---

            results_list.append({
                "File": file.name,
                "Test_Type": test_type,
                "OOS_Days": len(df),
                "DM_Statistic": dm_stat,
                "p_value (one-sided)": correct_pvalue, # <-- Use the corrected p-value
                "Challenger_Signif_Better (p < 0.05)": "Yes" if is_significant_and_better else "No"
            })

        except Exception as e:
            print(f"Error processing {file.name}: {e}")

    # --- Print the final results table ---
    if results_list:
        print("--- Diebold-Mariano Test Results (Forecast Accuracy) ---")
        results_df = pd.DataFrame(results_list).sort_values(by="File")
        
        # Pretty-print the table
        print(results_df.to_string(index=False, float_format="%.6f"))
    else:
        print("No valid results were processed.")

# Call the function directly to run the tests when the script is executed
run_dm_tests()