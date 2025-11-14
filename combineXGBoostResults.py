import pandas as pd
from pathlib import Path

# --- Configuration ---
# Set this to the same OUTPUT_DIR from your main script
RESULTS_DIR = Path("Results/ML")

# This is the prefix your main script uses for its summary files
FILE_PATTERN = "XG_results_*.csv"

# --- Output Filenames ---
GARCH_OUTPUT_FILE = RESULTS_DIR / "Combined_GARCH.csv"
RAW_OUTPUT_FILE = RESULTS_DIR / "Combined_RAW.csv"

# --- Main Script ---
def combine_results():
    """
    Finds all individual XGBoost result files, combines them,
    and splits them into two separate GARCH and RAW summary tables.
    """
    print(f"Looking for result files in: {RESULTS_DIR}")
    
    result_files = list(RESULTS_DIR.glob(FILE_PATTERN))
    
    if not result_files:
        print(f"Error: No result files found matching '{FILE_PATTERN}' in {RESULTS_DIR}")
        print("Please make sure you've run the 'fitXGBoostNew.py' script first.")
        return

    print(f"Found {len(result_files)} result files. Combining...")

    # Read all individual result files into a list of DataFrames
    all_dfs = []
    for f in result_files:
        try:
            df = pd.read_csv(f)
            all_dfs.append(df)
        except pd.errors.EmptyDataError:
            print(f"Warning: Skipping empty file: {f.name}")
        except Exception as e:
            print(f"Error reading {f.name}: {e}")

    if not all_dfs:
        print("Error: No valid result dataframes to combine.")
        return

    # Combine all DataFrames into one master table
    master_df = pd.concat(all_dfs, ignore_index=True)

    # Split the master table into GARCH and RAW results
    garch_results_df = master_df[master_df['Test'] == 'GARCH'].copy()
    raw_results_df = master_df[master_df['Test'] == 'RAW'].copy()

    # Sort the tables for better readability
    garch_results_df.sort_values(by=['Source', 'Target'], inplace=True)
    raw_results_df.sort_values(by=['Source', 'Target'], inplace=True)
    
    # --- Save and Print GARCH Results ---
    if not garch_results_df.empty:
        print("\n" + "=" * 50)
        print("Combined GARCH Test Results")
        print("=" * 50)
        print(garch_results_df.to_string(float_format="%.6f"))
        
        # Save to CSV
        garch_results_df.to_csv(GARCH_OUTPUT_FILE, index=False)
        print(f"\nSuccessfully saved GARCH results to: {GARCH_OUTPUT_FILE}")
    else:
        print("\nNo GARCH results found to save.")

    # --- Save and Print RAW Results ---
    if not raw_results_df.empty:
        print("\n" + "=" * 50)
        print("Combined RAW Test Results")
        print("=" * 50)
        print(raw_results_df.to_string(float_format="%.6f"))
        
        # Save to CSV
        raw_results_df.to_csv(RAW_OUTPUT_FILE, index=False)
        print(f"\nSuccessfully saved RAW results to: {RAW_OUTPUT_FILE}")
    else:
        print("\nNo RAW results found to save.")

if __name__ == "__main__":
    # Ensure the output directory exists before trying to save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    combine_results()