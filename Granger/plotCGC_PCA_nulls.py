import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path

# --- 1. Setup ---
# Define the input file from your PCA-based test
results_file = "grangerCopula_PCA_Nulls.csv"
output_dir = Path("Plots/Plots_CGC_PCA_Nulls")
output_dir.mkdir(exist_ok=True)

try:
    df = pd.read_csv(results_file)
except FileNotFoundError:
    print(f"Error: {results_file} not found.")
    print("Please run the 'testCGC_PCA.py' script first.")
    exit()

# --- 2. Plotting Loop ---
# Iterate over each row in your results (each of the 4 tests)
for _, row in df.iterrows():
    
    # Get the factor names for labeling
    source_name = row['Source_Factor']
    target_name = row['Target_Factor']
    
    # Get the test statistics
    gc_test_stat = row['GC_test']
    
    # Load the null distribution from the JSON string
    try:
        null_dist = np.array(json.loads(row['null_dist']))
        # Remove any NaNs from bootstrap failures if they exist
        null_dist = null_dist[~np.isnan(null_dist)]
    except Exception as e:
        print(f"Could not parse null distribution for {source_name} -> {target_name}. Skipping. Error: {e}")
        continue

    if null_dist.size == 0:
        print(f"No valid null distribution data for {source_name} -> {target_name}. Skipping.")
        continue

    # Calculate p-value
    p_value = np.mean(null_dist >= gc_test_stat)
    
    # --- Create Plot ---
    plt.figure(figsize=(12, 7))
    sns.histplot(null_dist, bins=30, kde=True, color='skyblue', label='Null Distribution (Bootstrap)')
    
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
    title = f'Copula GC Null Distribution: {source_name} -> {target_name}'
    plt.title(title, fontsize=16)
    plt.xlabel('Copula Granger Causality Statistic', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # --- Save Plot ---
    # Create a safe filename
    filename = f"Null_{source_name}_to_{target_name}.png"
    output_path = output_dir / filename
    
    try:
        plt.savefig(output_path)
        print(f"Saved plot to {output_path}")
    except Exception as e:
        print(f"Error saving plot {output_path}: {e}")
        
    plt.close()

print("\nAll plots saved successfully.")