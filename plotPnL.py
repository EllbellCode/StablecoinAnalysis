import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import warnings

# --- Configuration ---
# These paths should be the same as in your fitXGBoostNew.py script
RESULTS_DIR = Path("Results/ML/Winsor")
PLOT_DIR = Path("Plots/ML/PnL/Winsor")

# Ensure the plot directory exists
PLOT_DIR.mkdir(parents=True, exist_ok=True)
warnings.filterwarnings('ignore')

def plot_pnl_curves():
    """
    Finds all prediction CSVs in the RESULTS_DIR, calculates
    cumulative PnL (equity curves), and saves a plot for each.
    """
    print(f"Scanning for prediction files in: {RESULTS_DIR.resolve()}\n")
    
    # Find all prediction CSV files
    pred_files = list(RESULTS_DIR.glob("XG_preds_*.csv"))
    
    if not pred_files:
        print("No prediction files (XG_preds_*.csv) found.")
        print(f"Please check that files exist in {RESULTS_DIR}")
        return

    num_plots_generated = 0

    for file in pred_files:
        try:
            df = pd.read_csv(file, parse_dates=['Date'], index_col='Date')
            
            if df.empty:
                print(f"Skipping {file.name}: File is empty.")
                continue

            bench_col = None
            chall_col = None

            # --- Identify which return columns to use ---
            # Added logic for VOL models here
            if "GARCH" in file.name and 'Return_Bench_G' in df.columns:
                bench_col = 'Return_Bench_G'
                chall_col = 'Return_Chall_G'
                model_type = "GARCH (Shock Prediction)"
            elif "RAW" in file.name and 'Return_Bench_R' in df.columns:
                bench_col = 'Return_Bench_R'
                chall_col = 'Return_Chall_R'
                model_type = "RAW (Directional Prediction)"
            elif "VOL" in file.name and 'Return_Bench_V' in df.columns:
                bench_col = 'Return_Bench_V'
                chall_col = 'Return_Chall_V'
                model_type = "VOL (Volatility Expansion)"
            else:
                print(f"Skipping {file.name}: Could not identify model type (GARCH/RAW/VOL) or columns missing.")
                continue

            # --- Calculate Cumulative PnL (Equity Curves) ---
            df['Equity_Bench'] = df[bench_col].cumsum()
            df['Equity_Chall'] = df[chall_col].cumsum()

            # --- Create the Plot ---
            plt.figure(figsize=(14, 8))
            
            plt.plot(df.index, df['Equity_Bench'], label='Benchmark', color='blue', linewidth=1.5)
            plt.plot(df.index, df['Equity_Chall'], label='Challenger', color='red', linestyle='--', linewidth=2)
            
            # --- Fill between ---
            plt.fill_between(df.index, df['Equity_Bench'], df['Equity_Chall'], 
                             where=df['Equity_Chall'] > df['Equity_Bench'], 
                             color='green', alpha=0.1, interpolate=True, label='Challenger Outperformance')
                             
            plt.fill_between(df.index, df['Equity_Bench'], df['Equity_Chall'], 
                             where=df['Equity_Chall'] < df['Equity_Bench'], 
                             color='red', alpha=0.1, interpolate=True, label='Benchmark Outperformance')

            # --- Add Titles and Labels ---
            # Clean up the title for better readability
            clean_name = file.stem.replace('XG_preds_', '').replace('_', ' ')
            title = f"Cumulative P&L: {clean_name}\nType: {model_type}"
            
            plt.title(title, fontsize=16)
            plt.xlabel('Date', fontsize=12)
            plt.ylabel('Cumulative Return (PnL)', fontsize=12)
            plt.legend(loc='upper left', fontsize=10)
            plt.grid(True, linestyle=':', alpha=0.6)
            plt.tight_layout()

            # --- Save the Plot ---
            plot_filename = f"PnL_{file.stem}.png"
            save_path = PLOT_DIR / plot_filename
            
            plt.savefig(save_path, dpi=150)
            plt.close() 
            
            print(f"  > Saved PnL plot to {save_path}")
            num_plots_generated += 1

        except Exception as e:
            print(f"Error processing {file.name}: {e}")

    print(f"\nDone. Generated {num_plots_generated} PnL plots in {PLOT_DIR.resolve()}")

if __name__ == "__main__":
    plot_pnl_curves()