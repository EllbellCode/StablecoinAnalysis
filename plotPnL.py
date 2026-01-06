"""
Plots the PNL of our volatility targeting model over time
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick

DIR = 'Results/VolTargeting/'
OUTPUT_FILE = DIR + 'PnL_Over_Time.png'
SELECTED_VOL_TARGET = '0.5'
BASE_FONT_SIZE = 30

sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.size': BASE_FONT_SIZE,
    'axes.labelsize': BASE_FONT_SIZE,
    'axes.titlesize': BASE_FONT_SIZE,
    'xtick.labelsize': BASE_FONT_SIZE,
    'ytick.labelsize': BASE_FONT_SIZE,
    'legend.fontsize': BASE_FONT_SIZE,
    'figure.titlesize': BASE_FONT_SIZE
})

def plot_pnl(file_path=DIR + 'Backtest_Results.csv'):
   
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: Could not find {file_path}")
        return

    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
    
    print(f"Plotting Percentage Returns for Vol Target: {SELECTED_VOL_TARGET}")

    cumulative_df = (np.exp(df.cumsum()) - 1) * 100

    fig, ax = plt.subplots(figsize=(16, 10))
    
    vt = SELECTED_VOL_TARGET
    challenger_col = f'Challenger_{vt}'
    
    ax.plot(cumulative_df.index, cumulative_df['BuyHold'], 
            label='Buy & Hold', color='red', 
            linewidth=2.0, linestyle='-', zorder=1)
    
    ax.plot(cumulative_df.index, cumulative_df[challenger_col], 
            label='Challenger', color='midnightblue', 
            linewidth=3.0, zorder=2)
    
    ax.set_ylabel('Cumulative Return (%)')
    ax.set_xlabel('Date') 
    
    fmt = '%.0f%%'
    tick = mtick.FormatStrFormatter(fmt)
    ax.yaxis.set_major_formatter(tick)
    
    ax.legend(loc='upper left', frameon=True)
    ax.grid(True, linestyle='--', alpha=0.5)

    final_val = cumulative_df[challenger_col].iloc[-1]

    plt.tight_layout()
    
    plt.savefig(OUTPUT_FILE, dpi=300)
    print(f"Plot saved to {OUTPUT_FILE}")
    plt.show()

if __name__ == "__main__":
    plot_pnl()