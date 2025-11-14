import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
import re # Import regular expressions to parse filenames

# --- 1. Setup ---
# Define the directory to search for input files (current directory)
input_dir = Path("Results/Granger")

# Define the output directory
output_dir = Path("Plots/Granger")
output_dir.mkdir(parents=True, exist_ok=True) # Use parents=True to create nested dirs

# Use glob to find all NULLS files
nulls_files = list(input_dir.glob("GC_Nulls_*.csv"))

if not nulls_files:
    print(f"Error: No files found matching 'GC_Nulls_*.csv' in {input_dir.resolve()}")
else:
    print(f"Found {len(nulls_files)} nulls files to process.")

# --- 2. Outer Loop (Iterate over files) ---
for nulls_file_path in nulls_files:
    
    # --- A. Find and Load Corresponding RESULTS file ---
    results_file_name = nulls_file_path.name.replace('GC_Nulls_', 'GC_Results_')
    results_file_path = input_dir / results_file_name
    
    if not results_file_path.exists():
        print(f"  Warning: No results file found for {nulls_file_path.name}. Skipping.")
        continue
        
    try:
        # Load the results and set a (Source, Target) index for easy lookup
        results_df = pd.read_csv(results_file_path)
        
        # --- FIX: Strip whitespace from key columns ---
        # This handles hidden spaces like " Crypto_Returns"
        results_df['Source'] = results_df['Source'].str.strip()
        results_df['Target'] = results_df['Target'].str.strip()
        # --- END FIX ---
        
        results_df.set_index(['Source', 'Target'], inplace=True)
    except Exception as e:
        print(f"  CRITICAL ERROR: Failed to load or set index for results file {results_file_path.name}. Error: {e}. Skipping file.")
        continue

    # --- B. Load the NULLS file ---
    try:
        nulls_df = pd.read_csv(nulls_file_path)
        
        # --- FIX: Strip whitespace from key columns ---
        nulls_df['Source'] = nulls_df['Source'].str.strip()
        nulls_df['Target'] = nulls_df['Target'].str.strip()
        # --- END FIX ---
        
    except Exception as e:
        print(f"  Error reading {nulls_file_path.name}: {e}. Skipping.")
        continue

    # Try to get the lag from the filename
    lag_match = re.search(r'_(\d+)\.csv$', nulls_file_path.name)
    lag = lag_match.group(1) if lag_match else "Unknown"
    
    print(f"\n--- Processing {nulls_file_path.name} (Lag {lag}) ---")
    
    # === NEW DEBUGGING STEP ===
    # This will show you all the keys available in the results file
    print(f"  Available keys in {results_file_path.name}:")
    print(f"  {list(results_df.index)}")
    # ==========================


    # --- 3. Inner Loop (Iterate over rows in the file) ---
    for _, nulls_row in nulls_df.iterrows():
        
        # --- C. Data Extraction (Already stripped from DF) ---
        source_name = nulls_row.get('Source')
        target_name = nulls_row.get('Target')
        null_dist_str = nulls_row.get('Nulls') # Use 'Nulls' column
        
        if not all([source_name, target_name, null_dist_str]):
            print(f"  Skipping row with missing data (Source, Target, or Nulls).")
            continue
            
        print(f"\n  Looking for key: ({source_name}, {target_name})") # Debug print

        # --- D. Look up Stats from RESULTS file ---
        try:
            stats_row = results_df.loc[(source_name, target_name)]
            
            # === THE FIX: Look for 'GC' not 'GC_test' ===
            gc_test_stat = stats_row.get('GC') 
            # ============================================
            
            p_value = stats_row.get('p-value') # The 'p-value' column
            
            if gc_test_stat is None or p_value is None:
                raise KeyError("'GC' or 'p-value' missing from results file.")
                
        except KeyError:
            print(f"  ---> WARNING: No results found for this key. Plotting histogram only.")
            gc_test_stat = None
            p_value = None
        except Exception as e:
            print(f"  Error looking up stats: {e}. Plotting histogram only.")
            gc_test_stat = None
            p_value = None

        # --- E. Load Null Distribution ---
        try:
            null_dist = np.array(json.loads(null_dist_str))
            null_dist = null_dist[~np.isnan(null_dist)]
        except Exception as e:
            print(f"  Could not parse null distribution for {source_name} -> {target_name}. Skipping. Error: {e}")
            continue

        if null_dist.size == 0:
            print(f"  No valid null distribution data for {source_name} -> {target_name}. Skipping.")
            continue
        
        # --- F. Create Plot ---
        plt.figure(figsize=(12, 7))
        sns.histplot(null_dist, bins=30, kde=True, color='skyblue', label='Null Distribution (Bootstrap)')
        
        # --- G. Add Stats to Plot (if found) ---
        if gc_test_stat is not None and p_value is not None:
            
            # Plot the observed test statistic
            plt.axvline(
                gc_test_stat, 
                color='red', 
                linestyle='--', 
                linewidth=2, 
                label=f'Observed GC Statistic ({gc_test_stat:.4f})'
            )
            
            # Add a text box for the p-value
            plt.text(
                0.95, 0.95, 
                f'p-value = {p_value:.4f}', 
                transform=plt.gca().transAxes, 
                fontsize=14,
                verticalalignment='top', 
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.5)
            )
        
        # --- Titles and Labels ---
        title = f'Copula GC Null Distribution: {source_name} -> {target_name} (Lag {lag})'
        plt.title(title, fontsize=16)
        plt.xlabel('Copula Granger Causality Statistic', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)
        
        # --- Save Plot ---
        filename = f"Null_{source_name}_to_{target_name}_lag_{lag}.png"
        output_path = output_dir / filename 
        
        try:
            plt.savefig(output_path)
            print(f"  Saved plot to {output_path}")
        except Exception as e:
            print(f"  Error saving plot {output_path}: {e}")
            
        plt.close()

print("\nAll plots saved successfully.")