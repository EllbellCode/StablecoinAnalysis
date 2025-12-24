import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick

DIR = 'Results/Thesis_Chapter_5/'
OUTPUT_FILE = DIR +'PnL_Over_Time.png'
INITIAL_CAPITAL = 10000

# Set plotting style for better aesthetics
sns.set_theme(style="whitegrid")

def plot_pnl(file_path=DIR + 'Backtest_Results.csv'):
    # 1. Load Data
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: Could not find {file_path}")
        return

    # Ensure Date parsing
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
    
    print(f"Plotting for Initial Capital: ${INITIAL_CAPITAL:,.2f}")

    # 3. Calculate Account Balance (Wealth Index * Initial Capital)
    # The backtest saves Log Returns -> exp(cumsum) gives the multiplier (e.g., 1.5x)
    # We multiply that by your starting cash.
    cumulative_df = INITIAL_CAPITAL * np.exp(df.cumsum())

    # 4. Setup Plotting Grid (2x2 for the 4 Vol Targets)
    vol_targets = ['0.2', '0.3', '0.4', '0.5']
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f'Strategy Equity Curve (Start: ${INITIAL_CAPITAL:,.0f})', fontsize=16, y=0.98)

    # 5. Iterate and Plot
    for i, ax in enumerate(axes.flatten()):
        if i < len(vol_targets):
            vt = vol_targets[i]
            
            # --- Plot Lines ---
            # Buy & Hold (Gray, dashed background reference)
            ax.plot(cumulative_df.index, cumulative_df['BuyHold'], 
                    label='Buy & Hold', color='gray', alpha=0.3, linestyle='--')
            
            # Strategies
            ax.plot(cumulative_df.index, cumulative_df[f'Benchmark_{vt}'], 
                    label='Benchmark', color='#1f77b4', linewidth=1.5)
            
            ax.plot(cumulative_df.index, cumulative_df[f'Naive_{vt}'], 
                    label='Naive', color='#ff7f0e', linewidth=1.5, linestyle=':')
            
            ax.plot(cumulative_df.index, cumulative_df[f'Challenger_{vt}'], 
                    label='Challenger (XGB)', color='#2ca02c', linewidth=2.5)
            
            # --- Formatting ---
            ax.set_title(f'Volatility Target: {float(vt)*100:.0f}%', fontsize=12, fontweight='bold')
            ax.set_ylabel('Account Value ($)')
            
            # Format Y-axis as Dollars
            fmt = '${x:,.0f}'
            tick = mtick.StrMethodFormatter(fmt)
            ax.yaxis.set_major_formatter(tick)
            
            ax.legend(loc='upper left', fontsize=9, frameon=True)
            ax.grid(True, linestyle='--', alpha=0.5)

            # Add final value annotation for Challenger
            final_val = cumulative_df[f'Challenger_{vt}'].iloc[-1]
            total_return = (final_val - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
            
            ax.annotate(f'${final_val:,.0f}\n(+{total_return:.1f}%)', 
                        xy=(cumulative_df.index[-1], final_val),
                        xytext=(5, 0), textcoords='offset points',
                        color='#2ca02c', fontweight='bold', fontsize=10)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust for suptitle
    
    # Save and Show
    plt.savefig(OUTPUT_FILE, dpi=300)
    print(f"Plot saved to {OUTPUT_FILE}")
    plt.show()

if __name__ == "__main__":
    plot_pnl()