import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
import re 

# --- 1. Setup ---
input_dir = Path("Results/GrangerCopula")
output_dir = Path("Plots/Granger")
output_dir.mkdir(parents=True, exist_ok=True) 

# Find all NULLS files (including those with _Win)
nulls_files = list(input_dir.glob("GC_Nulls_*.csv"))

if not nulls_files:
    print(f"Error: No files found matching 'GC_Nulls_*.csv' in {input_dir.resolve()}")
else:
    print(f"Found {len(nulls_files)} nulls files to process.")

# --- 2. Outer Loop (Iterate over files) ---
for nulls_file_path in nulls_files:
    
    # --- A. Find Corresponding RESULTS file ---
    # This simple replace works for both standard and winsorized files
    # e.g., GC_Nulls_StableCrypto_7_Win.csv -> GC_Results_StableCrypto_7_Win.csv
    results_file_name = nulls_file_path.name.replace('GC_Nulls_', 'GC_Results_')
    results_file_path = input_dir / results_file_name
    
    if not results_file_path.exists():
        print(f"  Warning: No results file found for {nulls_file_path.name}. Skipping.")
        continue
        
    try:
        results_df = pd.read_csv(results_file_path)
        results_df['Source'] = results_df['Source'].str.strip()
        results_df['Target'] = results_df['Target'].str.strip()
        results_df.set_index(['Source', 'Target'], inplace=True)
    except Exception as e:
        print(f"  CRITICAL ERROR: Failed to load results file {results_file_path.name}. Error: {e}. Skipping.")
        continue

    # --- B. Load the NULLS file ---
    try:
        nulls_df = pd.read_csv(nulls_file_path)
        nulls_df['Source'] = nulls_df['Source'].str.strip()
        nulls_df['Target'] = nulls_df['Target'].str.strip()
    except Exception as e:
        print(f"  Error reading {nulls_file_path.name}: {e}. Skipping.")
        continue

    # --- UPDATED REGEX for Lag and Winsorization ---
    # Matches: _7.csv OR _7_Win.csv OR _7_win.csv
    lag_match = re.search(r'_(\d+)(?:_([Ww]in))?\.csv$', nulls_file_path.name)
    
    if lag_match:
        lag = lag_match.group(1)
        is_winsorized = bool(lag_match.group(2)) # True if group 2 exists (Win/win)
    else:
        lag = "Unknown"
        is_winsorized = False
    
    win_status = "Winsorized" if is_winsorized else "Raw"
    
    print(f"\n--- Processing {nulls_file_path.name} (Lag {lag}, {win_status}) ---")

    # --- 3. Inner Loop ---
    for _, nulls_row in nulls_df.iterrows():
        source_name = nulls_row.get('Source')
        target_name = nulls_row.get('Target')
        null_dist_str = nulls_row.get('Nulls')
        
        if not all([source_name, target_name, null_dist_str]):
            continue

        # --- D. Look up Stats ---
        try:
            stats_row = results_df.loc[(source_name, target_name)]
            gc_test_stat = stats_row.get('GC')
            p_value = stats_row.get('p-value')
        except KeyError:
            print(f"  ---> WARNING: No results found for ({source_name}, {target_name}). Plotting hist only.")
            gc_test_stat = None
            p_value = None

        # --- E. Parse Null Distribution ---
        try:
            null_dist = np.array(json.loads(null_dist_str))
            null_dist = null_dist[~np.isnan(null_dist)]
        except Exception as e:
            continue

        if null_dist.size == 0: continue
        
        # --- F. Create Plot ---
        plt.figure(figsize=(12, 7))
        sns.histplot(null_dist, bins=30, kde=True, color='teal' if is_winsorized else 'skyblue', 
                     label=f'Null Dist ({win_status})')
        
        # --- G. Add Stats ---
        if gc_test_stat is not None and p_value is not None:
            plt.axvline(gc_test_stat, color='red', linestyle='--', linewidth=2, 
                        label=f'Observed GC: {gc_test_stat:.4f}')
            
            stats_text = f'p-value = {p_value:.4f}\n({win_status})'
            plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=12,
                     verticalalignment='top', horizontalalignment='right',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
        
        title = f'Copula GC Null Distribution: {source_name} -> {target_name}\n(Lag {lag} | {win_status})'
        plt.title(title, fontsize=16)
        plt.xlabel('Copula Granger Causality Statistic', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)
        
        # --- Save Plot with Suffix ---
        filename = f"Null_{source_name}_to_{target_name}_lag_{lag}.png"
        output_path = output_dir / filename 
        
        try:
            plt.savefig(output_path)
            print(f"  Saved plot to {output_path}")
        except Exception as e:
            print(f"  Error saving plot {output_path}: {e}")
            
        plt.close()

print("\nAll plots saved successfully.")