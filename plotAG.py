import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# --- SETTINGS ---
# These MUST match the pairs you tested
tests_to_run = [
    ("Stable_Volume", "Crypto_Returns"),
    ("Stable_Volume", "Crypto_Volatility"),
    ("Stable_Volatility", "Crypto_Returns"),
    ("Stable_Volatility", "Crypto_Volatility"),
]

print("--- Generating Cumulative Log-Likelihood Plots ---")

for source_key, target_key in tests_to_run:
    
    # 1. Define the input file name
    filename = f"ag_test_series_{source_key}_to_{target_key}.csv"
    
    # 2. Check if the file exists
    if not Path(filename).exists():
        print(f"Warning: File not found, skipping plot for: {filename}")
        continue
        
    print(f"Plotting {filename}...")
    
    # 3. Load the data
    ll_df = pd.read_csv(filename, index_col=0, parse_dates=True)
    
    # 4. Calculate the cumulative log-likelihood difference
    # We use cumsum() to see the trend of outperformance
    ll_df['Cumulative_LL_Difference'] = ll_df['LL_Difference'].cumsum()
    
    # 5. Create the plot
    plt.figure(figsize=(15, 7))
    
    plt.plot(ll_df.index, ll_df['Cumulative_LL_Difference'], 
             label='Cumulative Log-Likelihood (Challenger - Benchmark)', 
             color='blue', linewidth=2)
    
    # Add a horizontal line at 0 for reference
    plt.axhline(0, color='red', linestyle='--', linewidth=1, label='No Difference (Benchmark = Challenger)')

    plt.title(f'Cumulative Predictive Likelihood: {source_key} (t-1) -> {target_key} (t)')
    plt.ylabel('Cumulative Log-Likelihood Score (Higher is better)')
    plt.xlabel('Date (2024)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    
    # 6. Save the plot to a new file
    plot_filename = f"ag_test_plot_{source_key}_to_{target_key}.png"
    plt.savefig(f"Plots/{plot_filename}")
    plt.close() # Close the plot to save memory
    print