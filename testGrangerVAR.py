"""
Granger Causality Test Implemented withing a Vector Autorregression (VAR) framwork

In a standard Linear Regression like OLS, we would have to 
assume a unidirectional outcome (Crypto causes stable or vice versa)
However, the relationship is likely bidirectional

VAR assumes everything causes everything (endogenous), allowing us to 
capture the bidirectional relationship

We use Garman Klass for Crypto Volatility and a 30 day RV for Stable Volatility

GK allows us to capture the intraday activity of Cryptos

Stablecoins are mean reverting (pegged at 1USD) - most of the intraday depegging is
just noise but GK sees this and thinks a huge depeg risk has ocurred
By using a 30 day RV, it helps us smooth over this noise so one noisy event does
not blow up our results
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from pathlib import Path
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.api import VAR
import warnings
from statsmodels.tsa.stattools import grangercausalitytests

warnings.filterwarnings('ignore')

# ===================================================================
# CONFIGURATION
# ===================================================================
DATA_DIR = Path("Data/Verified")
OUTPUT_DIR = Path("Results/GrangerVAR")
PLOT_DIR = Path("Plots/GrangerVAR")
START_DATE = '2020-01-01'
END_DATE = '2024-01-01'
CRYPTO_VOL = "Delta_LogGK"
STABLE_VOL = "Delta_LogGK"
MAXLAGS = 1
WINSOR_QUANTILE = 0.01 
MIN_PCA_WINDOW = 60 
IRF_PERIODS = 20 

# ===================================================================
# Helper Functions
# ===================================================================

def check_stationarity(series):
    try:
        result = adfuller(series.dropna())
        return result[1] < 0.05
    except:
        return False

def get_expanding_pca(df, min_periods=30):
    """
    Calculates the 1st Principal Component using an Expanding Window.
    """
    n_samples, n_features = df.shape
    pc_series = np.full(n_samples, np.nan)
    prev_components = None
    
    print(f"  -> Generating Expanding PCA for {n_features} vars...")
    
    for t in range(min_periods, n_samples):
        window_data = df.iloc[:t+1].values
        scaler = StandardScaler()
        scaled_window = scaler.fit_transform(window_data)
        
        pca = PCA(n_components=1)
        pca.fit(scaled_window)
        
        current_components = pca.components_[0]
        
        # Sign Consistency Check
        sign_multiplier = 1.0
        if prev_components is not None:
            if np.dot(prev_components, current_components) < 0:
                sign_multiplier = -1.0
        
        current_obs_scaled = scaler.transform(window_data[-1].reshape(1, -1))
        pc_value = pca.transform(current_obs_scaled)[0, 0]
        
        pc_series[t] = pc_value * sign_multiplier
        prev_components = current_components * sign_multiplier

    return pd.Series(pc_series, index=df.index)

def load_and_process_factors(data_dir, start_date, end_date, winsor_limit):
    print("Loading and processing data...")
    coin_data = {}
    if not data_dir.exists(): raise FileNotFoundError(f"Data directory not found: {data_dir}")

    for file in data_dir.glob("*.csv"):
        if (c := file.stem.replace("Verif_", "")) in ["DAI", "USDC", "USDT", "BNB", "BTC", "ETH", "XRP"]:
            df = pd.read_csv(file, parse_dates=['Date']).sort_values("Date")
            coin_data[c] = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)].set_index('Date')

    def get_fac(coins, var, name):
        df_list = [coin_data[c][var] for c in coins if c in coin_data]
        if not df_list: return pd.Series(dtype=float, name=name)
        
        df = pd.concat(df_list, axis=1, keys=coins, join='inner').dropna()
        for col in df.columns: 
            df[col] = winsorize(df[col], limits=[winsor_limit, winsor_limit])
        
        pca_series = get_expanding_pca(df, min_periods=MIN_PCA_WINDOW)
        pca_series.name = name
        return pca_series

    factors = {
        "Stable_Volume": get_fac(["DAI", "USDC", "USDT"], "LogVolChange", "Stable_Volume"),
        "Stable_Volatility": get_fac(["DAI", "USDC", "USDT"], STABLE_VOL, "Stable_Volatility"),
        "Stable_Returns": get_fac(["DAI", "USDC", "USDT"], "Log Returns", "Stable_Returns"),
        "Crypto_Volume": get_fac(["BNB", "BTC", "ETH", "XRP"], "LogVolChange", "Crypto_Volume"),
        "Crypto_Volatility": get_fac(["BNB", "BTC", "ETH", "XRP"], CRYPTO_VOL, "Crypto_Volatility"),
        "Crypto_Returns": get_fac(["BNB", "BTC", "ETH", "XRP"], "Log Returns", "Crypto_Returns"),
    }
    return factors

def save_significant_irfs(var_results, causality_df, output_dir):
    """
    Generates and saves IRF plots for significant relationships.
    Forces X-axis to be integers.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n--- Generating IRF Plots for Significant Pairs ---")
    
    sig_results = causality_df[causality_df['p_value'] < 0.05]
    
    if sig_results.empty:
        print("No significant relationships found to plot.")
        return

    irf = var_results.irf(IRF_PERIODS)
    
    for index, row in sig_results.iterrows():
        source = row['Source']
        target = row['Target']
        p_val = row['p_value']
        
        print(f"Plotting: {source} -> {target}")
        
        # Generate plot
        fig = irf.plot(impulse=source, response=target, orth=True, signif=0.05)
        
        # FORCE INTEGER TICKS ON X-AXIS
        # The fig returned by statsmodels may contain multiple subplots, iterate all axes
        for ax in fig.axes:
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.set_xlabel("Days after shock")
        
        plt.title(f"Impulse: {source} -> Response: {target}\n(p-value: {p_val:.4f})")
        plt.tight_layout()
        
        filename = f"IRF_{source}_to_{target}.png"
        plt.savefig(output_dir / filename, dpi=300)
        plt.close(fig)

def run_system_var(factors_dict):
    print("\n--- Building System VAR Model ---")
    
    # 1. Consolidate Data
    df = pd.concat(factors_dict.values(), axis=1).dropna()
    
    # Reorder columns (Crypto First for Cholesky)
    desired_order = [
        'Crypto_Volume', 'Crypto_Volatility', 'Crypto_Returns',
        'Stable_Volume', 'Stable_Volatility', 'Stable_Returns'
    ]
    final_order = [c for c in desired_order if c in df.columns]
    df = df[final_order]
    
    print(f"Data Shape: {df.shape}")

    # 2. Fit VAR
    model = VAR(df)
    lag_results = model.select_order(maxlags=MAXLAGS)
    optimal_lag = lag_results.aic
    print(f"Optimal Lag (AIC): {optimal_lag}")
    
    var_results = model.fit(optimal_lag)
    
    # 3. Run Causality Tests (FILTERED)
    results_list = []
    targets = df.columns.tolist()
    
    print(f"\n{'Causing Variable':<20} | {'Target Variable':<20} | {'p-value':<8} | {'Result'}")
    print("-" * 80)

    for target in targets:
        # Identify group of target
        target_group = target.split('_')[0]  # "Stable" or "Crypto"
        
        potential_causes = [c for c in targets if c != target]
        
        for cause in potential_causes:
            # Identify group of cause
            cause_group = cause.split('_')[0] # "Stable" or "Crypto"
            
            # FILTER: SKIP if they belong to the same group
            # We only want Stable->Crypto or Crypto->Stable
            if target_group == cause_group:
                continue
                
            try:
                test_result = var_results.test_causality(caused=target, causing=cause)
                p_val = test_result.pvalue
                sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.1 else ""
                
                print(f"{cause:<20} | {target:<20} | {p_val:<8.4f} | {sig}")
                
                results_list.append({
                    "Source": cause,
                    "Target": target,
                    "Lag": optimal_lag,
                    "p_value": p_val,
                    "Significant": sig
                })
            except Exception as e:
                print(f"Error testing {cause} -> {target}: {e}")

    return var_results, pd.DataFrame(results_list)

def plot_fevd(var_results, output_dir):
    """
    Manually plots FEVD with custom colors and percentage labels.
    Generates one plot per variable.
    MODIFIED: Legend now includes the long-run contribution percentage.
    """
    # 1. Get FEVD data
    n_periods = 20
    fevd = var_results.fevd(n_periods)
    
    # .decomp shape is (n_vars, n_periods, n_vars) -> (Target, Time, Source)
    decomp_data = fevd.decomp 
    names = var_results.names
    n_vars = len(names)
    
    # 2. Define a distinct color palette
    colors = [plt.cm.tab10(k) for k in range(n_vars)]
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n--- Generating Enhanced FEVD Plots ---")
    
    # 3. Loop through each Target variable to create its own plot
    for i, target_name in enumerate(names):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Access the i-th target, all periods, all sources
        data_for_target = decomp_data[i, :, :] 
        
        # 'bottom' tracks where the next bar segment should start (for stacking)
        bottom = np.zeros(n_periods)
        
        # Loop through each Source to plot bars
        for j, source_name in enumerate(names):
            values = data_for_target[:, j] # Shape: (20,)
            
            # --- NEW LOGIC: Get final period value for Legend ---
            final_value = values[-1] 
            label_text = f"{source_name} ({final_value:.1%})"
            # ----------------------------------------------------

            # Plot the bar segment
            ax.bar(range(n_periods), values, bottom=bottom, label=label_text, 
                   color=colors[j], width=0.85, edgecolor='white', linewidth=0.5)
            
            # Add Percentage Labels inside bars (if segment > 5%)
            for t in range(n_periods):
                height = values[t]
                if height > 0.05: 
                    y_pos = bottom[t] + height / 2
                    ax.text(t, y_pos, f"{height:.0%}", ha='center', va='center', 
                            fontsize=8, color='white', fontweight='bold')
            
            # Update bottom for the next segment
            bottom += values
        
        # Formatting
        ax.set_title(f"Variance Decomposition: What drives {target_name}?", fontsize=14, pad=15)
        ax.set_ylabel("Variance Explained (0-1)", fontsize=12)
        ax.set_xlabel("Periods after Shock (Days)", fontsize=12)
        ax.set_ylim(0, 1.05)
        ax.set_xlim(-0.5, n_periods - 0.5)
        
        # Force integer ticks on x-axis
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        # Move legend outside
        # We use the new 'label_text' automatically here
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), 
                  ncol=3, frameon=False, title="Long-Run Contribution")
        
        plt.tight_layout()
        
        # Save
        filename = f"FEVD_{target_name}_Colorful.png"
        save_loc = output_dir / filename
        plt.savefig(save_loc, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {filename}")


# ===================================================================
# Main Execution
# ===================================================================
if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    factors = load_and_process_factors(DATA_DIR, START_DATE, END_DATE, WINSOR_QUANTILE)
    
    var_model, df_results = run_system_var(factors)
    
    save_path = OUTPUT_DIR / "VAR_Cross_Asset_Causality.csv"
    df_results.to_csv(save_path, index=False)
    
    save_significant_irfs(var_model, df_results, PLOT_DIR)
    plot_fevd(var_model, PLOT_DIR)
    
    print("-" * 90)
    print(f"Analysis Complete.")
    print(f"1. Data Table: {save_path.resolve()}")
    print(f"2. IRF Plots:  {PLOT_DIR.resolve()}")

