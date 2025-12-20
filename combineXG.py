import pandas as pd
from pathlib import Path

# --- Settings ---
RESULTS_DIR = Path("Results/ML/XG")
OUTPUT_FILE = RESULTS_DIR / "Unified_XGBoost_Summary.csv"
SIGNIFICANCE_THRESHOLD = 0.05

def categorize_result(row):
    """
    Determines if the result is a Significant Win, Loss, or Neutral
    based on P-Value and MSE comparison.
    """
    if row['DM_MSE_P_Value'] < SIGNIFICANCE_THRESHOLD:
        if row['MSE_Challenger'] < row['MSE_Benchmark']:
            return "✅ SIGNIFICANT WIN"
        else:
            return "❌ SIGNIFICANT LOSS" # Benchmark was significantly better
    else:
        # If P > 0.05, we check if there was at least a nominal improvement
        if row['MSE_Challenger'] < row['MSE_Benchmark']:
            return "Insignificant Improvement"
        else:
            return "Insignificant Loss"

def main():
    if not RESULTS_DIR.exists():
        print(f"Directory {RESULTS_DIR} does not exist.")
        return

    # 1. Find all matching summary files
    all_files = list(RESULTS_DIR.glob("Final_Summary_*.csv"))
    
    if not all_files:
        print("No summary files found.")
        return

    print(f"Found {len(all_files)} files. Merging...")

    # 2. Read and Combine
    df_list = []
    for file in all_files:
        try:
            temp_df = pd.read_csv(file)
            df_list.append(temp_df)
        except Exception as e:
            print(f"Error reading {file.name}: {e}")

    if not df_list:
        return

    combined_df = pd.concat(df_list, ignore_index=True)

    # 3. Add 'Clarity' Metrics
    
    # Calculate % Reduction in Error (Higher is better)
    combined_df['MSE_Reduction_Pct'] = (
        (combined_df['MSE_Benchmark'] - combined_df['MSE_Challenger']) 
        / combined_df['MSE_Benchmark']
    ) * 100

    # Calculate Directional Accuracy Difference (Positive is better)
    if 'Benchmark_Directional_Acc' in combined_df.columns and 'Challenger_Directional_Acc' in combined_df.columns:
        combined_df['Dir_Acc_pct'] = (combined_df['Challenger_Directional_Acc']/combined_df['Benchmark_Directional_Acc'] - 1) * 100
    else:
        combined_df['Dir_Acc_pct'] = 0.0

    # Add text conclusion column
    combined_df['Conclusion'] = combined_df.apply(categorize_result, axis=1)

    # 4. Reorder Columns for Readability
    # Putting Key Metrics up front
    cols = [
        'Test', 'Source', 'Target', 'OOS_Days', 'Conclusion',
        'MSE_Reduction_Pct', #'Dir_Acc_pct',
        'DM_MSE_P_Value', #'DM_Stat', 
        #'MSE_Benchmark', 'MSE_Challenger',
        'Dir_Acc_pct',
        #'Benchmark_Directional_Acc', 'Challenger_Directional_Acc',
        'DM_Directional_P_Value'
    ]
    
    # Ensure all cols exist
    existing_cols = [c for c in cols if c in combined_df.columns]
    combined_df = combined_df[existing_cols]

    # 5. Sort: Significant Wins first, then by highest MSE Reduction
    combined_df.sort_values(
        by=['Conclusion', 'MSE_Reduction_Pct'], 
        ascending=[True, False], 
        inplace=True
    )

    # 6. Save
    combined_df.to_csv(OUTPUT_FILE, index=False)
    
    print("-" * 30)
    print(f"Saved Unified Summary to: {OUTPUT_FILE}")
    print("-" * 30)
    
    # --- TABLE 1: Significant MSE Wins ---
    print("Significant MSE Results (p < 0.05):")
    # Filter for Significant Wins
    sig_wins = combined_df[combined_df['Conclusion'] == "✅ SIGNIFICANT WIN"]
    if not sig_wins.empty:
        print(sig_wins[['Test', 'Source', 'Target', 'MSE_Reduction_Pct']])
    else:
        print("No significant MSE wins found.")

    # --- TABLE 2: Positive Directional Accuracy ---
    print("\n" + "-" * 30)
    print("Improved Directional Accuracy Results (> 0 diff):")
    
    if 'Dir_Acc_pct' in combined_df.columns:
        # Filter for rows where Challenger Acc > Benchmark Acc
        da_wins = combined_df[combined_df['Dir_Acc_pct'] > 0].copy()
        
        if not da_wins.empty:
            # Sort by the highest improvement
            da_wins.sort_values(by='Dir_Acc_pct', ascending=False, inplace=True)
            
            # Print relevant columns
            print(da_wins[['Test', 'Source', 'Target', 'Dir_Acc_pct']])
        else:
            print("No improvements in Directional Accuracy found.")

if __name__ == "__main__":
    main()