"""
Script for aggregating XGBoost MSE backtest results
"""

import pandas as pd
from pathlib import Path



#Settings
RESULTS_DIR = Path("Results/MSE")
OUTPUT_FILE = RESULTS_DIR / "Unified_XGBoost_Summary.csv"
SIGNIFICANCE_THRESHOLD = 0.05

#Determines Win or Loss
def categorize_result(row):

    if row['DM_MSE_P_Value'] < SIGNIFICANCE_THRESHOLD:
        if row['MSE_Challenger'] < row['MSE_Benchmark']:
            return "SIGNIFICANT WIN"
        else:
            return "SIGNIFICANT LOSS"
    else:
        
        if row['MSE_Challenger'] < row['MSE_Benchmark']:
            return "Insignificant Improvement"
        else:
            return "Insignificant Loss"

def main():
    if not RESULTS_DIR.exists():
        print(f"Directory {RESULTS_DIR} does not exist.")
        return

    all_files = list(RESULTS_DIR.glob("Final_Summary_*.csv"))
    
    if not all_files:
        print("No summary files found.")
        return

    print(f"Found {len(all_files)} files. Merging...")

    #Load data
    df_list = []
    for file in all_files:
        try:
            temp_df = pd.read_csv(file)
            df_list.append(temp_df)
        except Exception as e:
            print(f"Error reading {file.name}: {e}")

    if not df_list:
        return

    # combine and calculate MSE reduction
    combined_df = pd.concat(df_list, ignore_index=True)
    combined_df['MSE_Reduction_Pct'] = (
        (combined_df['MSE_Benchmark'] - combined_df['MSE_Challenger']) 
        / combined_df['MSE_Benchmark']
    ) * 100

    
    if 'Benchmark_Directional_Acc' in combined_df.columns and 'Challenger_Directional_Acc' in combined_df.columns:
        combined_df['Dir_Acc_pct'] = (combined_df['Challenger_Directional_Acc']/combined_df['Benchmark_Directional_Acc'] - 1) * 100
    else:
        combined_df['Dir_Acc_pct'] = 0.0

    
    combined_df['Conclusion'] = combined_df.apply(categorize_result, axis=1)

    
    cols = [
        'Test', 'Source', 'Target', 'OOS_Days', 'Conclusion',
        'MSE_Reduction_Pct', #'Dir_Acc_pct',
        'DM_MSE_P_Value', #'DM_Stat', 
        #'MSE_Benchmark', 'MSE_Challenger',
        'Dir_Acc_pct',
        #'Benchmark_Directional_Acc', 'Challenger_Directional_Acc',
        'DM_Directional_P_Value'
    ]
    

    existing_cols = [c for c in cols if c in combined_df.columns]
    combined_df = combined_df[existing_cols]
    combined_df = combined_df[combined_df['Test'] == 'GARCH']


    combined_df.sort_values(
        by=['Conclusion', 'MSE_Reduction_Pct'], 
        ascending=[True, False], 
        inplace=True
    )

    combined_df.to_csv(OUTPUT_FILE, index=False)
    
    print("-" * 30)
    print(f"Saved Unified Summary to: {OUTPUT_FILE}")
    print("-" * 30)
    
    print("Significant MSE Results (p < 0.05):")

    sig_wins = combined_df[combined_df['Conclusion'] == "SIGNIFICANT WIN"]
    if not sig_wins.empty:
        print(sig_wins[['Test', 'Source', 'Target', 'MSE_Reduction_Pct']])
    else:
        print("No significant MSE wins found.")

    print("\n" + "-" * 30)
    print("Improved Directional Accuracy Results (> 0 diff):")
    
    if 'Dir_Acc_pct' in combined_df.columns:
        
        da_wins = combined_df[combined_df['Dir_Acc_pct'] > 0].copy()
        
        if not da_wins.empty:
            
            da_wins.sort_values(by='Dir_Acc_pct', ascending=False, inplace=True)
            
           
            print(da_wins[['Test', 'Source', 'Target', 'Dir_Acc_pct']])
        else:
            print("No improvements in Directional Accuracy found.")

if __name__ == "__main__":
    main()