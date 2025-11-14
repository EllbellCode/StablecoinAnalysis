import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import warnings
import re

warnings.filterwarnings('ignore')

# Directory where fitter saved the CSVs
DATA_DIR = Path("Results/GARCH/VaR")
PLOT_DIR = Path("Plots/VaR")
PLOT_DIR.mkdir(parents=True, exist_ok=True)

print("--- Generating VaR Backtest Plots for ALL Combinations ---")

# 1. Find all relevant files automatically
plot_files = list(DATA_DIR.glob("VaR_backtest_*_to_*_ALPHA*_MA*.csv"))

if not plot_files:
    print(f"No plotting files found in {DATA_DIR}. Run fitARXGARCH.py first.")
    exit()

print(f"Found {len(plot_files)} backtest files. Starting plot generation...")

for filepath in plot_files:
    filename = filepath.name
    
    # 2. Extract metadata from filename using Regex for robust parsing
    # Expected format: VaR_backtest_{SOURCE}_to_{TARGET}_ALPHA{ALPHA}_MA{MA}.csv
    match = re.match(r"VaR_backtest_(.+)_to_(.+)_ALPHA(.+)_MA(.+)\.csv", filename)
    if not match:
        print(f"Skipping likely invalid file: {filename}")
        continue
        
    source, target, alpha_str, ma_str = match.groups()
    alpha_val = float(alpha_str)
    
    print(f"Plotting: {target} | Src: {source} (MA{ma_str}) | VaR: {alpha_val}")

    # 3. Load Data
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    if df.empty: continue

    # 4. Prepare Breaches
    bench_breaches = df[df['Breach_Benchmark']]
    chall_breaches = df[df['Breach_Challenger']]

    # 5. Calculate sensible Y-axis limits (1st/99th percentiles + buffer)
    q_low = df['Actual_Y'].quantile(0.01)
    q_high = df['Actual_Y'].quantile(0.99)
    padding = (q_high - q_low) * 0.25
    # If it's a flat line, give it some minimal padding
    if padding == 0: padding = abs(q_high) * 0.1 if q_high != 0 else 0.1
    
    # 6. Create Plot
    plt.figure(figsize=(14, 7))
    
    # Actuals
    plt.plot(df.index, df['Actual_Y'], label=f'Actual {target}', 
             color='gray', alpha=0.4, linewidth=1, zorder=1)
    
    # Benchmark
    plt.plot(df.index, df['VaR_Benchmark'], 
             label=f'Benchmark GARCH ({len(bench_breaches)} breaches)', 
             color='blue', linestyle='--', linewidth=1.5, zorder=2)
    plt.scatter(bench_breaches.index, bench_breaches['Actual_Y'], 
                color='blue', marker='x', s=50, zorder=4, label='_nolegend_')

    # Challenger
    plt.plot(df.index, df['VaR_Challenger'], 
             label=f'ARX-GARCH MA({ma_str}) ({len(chall_breaches)} breaches)', 
             color='darkgreen', linewidth=2, zorder=3)
    plt.scatter(chall_breaches.index, chall_breaches['Actual_Y'], 
                color='lime', edgecolors='darkgreen', marker='o', s=60, 
                zorder=5, label=f'Challenger Breach')

    # Formatting
    plt.title(f'{float(alpha_str)*100:.0f}% VaR: {target} modeled by {source} (MA{ma_str})', fontsize=12)
    plt.ylabel(target)
    plt.ylim(q_low - padding, q_high + padding)
    plt.legend(loc='upper left', framealpha=0.9)
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()

    # 7. Save
    out_name = f"Plot_{target}_by_{source}_MA{ma_str}.png"
    plt.savefig(PLOT_DIR / out_name, dpi=100)
    plt.close()

print(f"\n--- All plots saved to {PLOT_DIR} ---")