"""
Plots the synthetic CGC distribution using the 200 synthetic CGC statistics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
import re 

input_dir = Path("Results/GrangerCopula/")
output_dir = Path("Plots/GrangerCopula/")
output_dir.mkdir(parents=True, exist_ok=True) 

nulls_files = list(input_dir.glob("GC_Nulls_*.csv"))

if not nulls_files:
    print(f"Error: No files found matching 'GC_Nulls_*.csv' in {input_dir.resolve()}")
else:
    print(f"Found {len(nulls_files)} nulls files to process.")

for nulls_file_path in nulls_files:
    
    results_file_name = nulls_file_path.name.replace('GC_Nulls_', 'GC_Results_')
    results_file_path = input_dir / results_file_name
    
    if not results_file_path.exists():
        print(f"  Warning: No results file found for {nulls_file_path.name}. Skipping.")
        continue
        
    try:
        results_df = pd.read_csv(results_file_path)
        if 'Source' in results_df.columns: results_df['Source'] = results_df['Source'].str.strip()
        if 'Target' in results_df.columns: results_df['Target'] = results_df['Target'].str.strip()
        results_df.set_index(['Source', 'Target'], inplace=True)
    except Exception as e:
        print(f"  CRITICAL ERROR: Failed to load results file {results_file_path.name}. Error: {e}. Skipping.")
        continue

    try:
        nulls_df = pd.read_csv(nulls_file_path)
        if 'Source' in nulls_df.columns: nulls_df['Source'] = nulls_df['Source'].str.strip()
        if 'Target' in nulls_df.columns: nulls_df['Target'] = nulls_df['Target'].str.strip()
    except Exception as e:
        print(f"  Error reading {nulls_file_path.name}: {e}. Skipping.")
        continue

    scope_match = re.search(r'_([A-Za-z0-9]+)\.csv$', nulls_file_path.name)
    
    if scope_match:
        scope = scope_match.group(1)
    else:
        scope = "Unknown"
    
    print(f"\n--- Processing {nulls_file_path.name} (Scope: {scope}) ---")

    for _, nulls_row in nulls_df.iterrows():
        source_name = nulls_row.get('Source')
        target_name = nulls_row.get('Target')
        null_dist_str = nulls_row.get('Nulls')
        
        if not all([source_name, target_name, null_dist_str]):
            continue

        try:
            stats_row = results_df.loc[(source_name, target_name)]
            
            if isinstance(stats_row, pd.DataFrame):
                stats_row = stats_row.iloc[0]
                
            gc_test_stat = stats_row.get('GC')
            p_value = stats_row.get('p-value')
        except KeyError:
            print(f"  ---> WARNING: No results found for ({source_name}, {target_name}). Plotting hist only.")
            gc_test_stat = None
            p_value = None

        try:
            null_dist = np.array(json.loads(null_dist_str))
            null_dist = null_dist[~np.isnan(null_dist)]
        except Exception as e:
            continue

        if null_dist.size == 0: continue
        
        plt.figure(figsize=(12, 7))
        ax = plt.gca()
        
        sns.histplot(null_dist, bins=30, kde=True, color='midnightblue', alpha=0.85)

        if len(ax.lines) > 0:
    
            ax.lines[0].set_color('red')
            ax.lines[0].set_linewidth(3)
        
        
        if gc_test_stat is not None and p_value is not None:
            
            plt.axvline(gc_test_stat, color='red', linestyle='--', linewidth=2.5, 
                        label=f'Observed GC: {gc_test_stat:.4f}')
        
        plt.xlabel('Copula Granger Causality Statistic', fontsize=24)
        plt.ylabel('Frequency', fontsize=24)
        plt.tick_params(axis='both', which='major', labelsize=24)
        
        plt.legend(loc='upper right', fontsize=24, framealpha=1.0, facecolor='white')
        
        filename = f"Null_{source_name}_to_{target_name}_{scope}.png"
        output_path = output_dir / filename 
        
        try:
            plt.savefig(output_path)
        except Exception as e:
            print(f"  Error saving plot {output_path}: {e}")
            
        plt.close()

print("\nAll plots processed successfully.")