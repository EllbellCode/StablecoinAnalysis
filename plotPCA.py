import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
from scipy.stats import skew, kurtosis
from scipy.stats.mstats import winsorize

# Set plot style for better visuals
sns.set_theme(style="whitegrid")

# ===================================================================
# Configuration
# ===================================================================
DATA_DIR = Path("Data/Verified")
OUTPUT_DIR = Path("Plots/PCA") 

START_DATE = '2020-01-01'
TRAIN_END_DATE = '2024-01-01' 
STATIONARY_VOL = "Delta_LogRV"

STABLECOINS = ["DAI", "USDC", "USDT"]
CRYPTOS = ["BNB", "BTC", "ETH", "XRP"]
ALL_COINS = STABLECOINS + CRYPTOS

WINSORIZE_QUANTILE = 0.01 #set to 0 to remove winsorizing
# ===================================================================
# Helper Functions
# ===================================================================

def load_data(data_dir, all_coins, start_date):
    """Loads data for all specified coins."""
    print("Loading data...")
    data_dict = {}
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found at: {data_dir.absolute()}")
        
    for file in data_dir.glob("*.csv"):
        coin_name = file.stem.replace("Verif_", "")
        if coin_name in all_coins:
            df = pd.read_csv(file, parse_dates=['Date']).sort_values("Date")
            df = df[df['Date'] >= start_date]
            data_dict[coin_name] = df.set_index('Date')
    print(f"Loaded {len(data_dict)} coins.")
    return data_dict

def horn_parallel_analysis(n_observations, n_variables, n_iterations=100, percentile=95):
    """
    Performs Horn's Parallel Analysis to determine a threshold for significant eigenvalues.
    """
    print(f"Running Horn's Parallel Analysis (Iter={n_iterations}, P={percentile}%)...")
    
    # Store eigenvalues from random data
    random_eigenvalues = np.zeros((n_iterations, n_variables))
    
    for i in range(n_iterations):
        # 1. Generate random, uncorrelated data
        random_data = np.random.normal(size=(n_observations, n_variables))
        
        # 2. Standardize it (just as we do with real data)
        scaler = StandardScaler()
        random_scaled = scaler.fit_transform(random_data)
        
        # 3. Run PCA
        pca_random = PCA(n_components=n_variables)
        pca_random.fit(random_scaled)
        
        # 4. Store the eigenvalues
        random_eigenvalues[i, :] = pca_random.explained_variance_
        
    # Get the 95th percentile eigenvalue for each component
    threshold_eigenvalues = np.percentile(random_eigenvalues, percentile, axis=0)
    
    return threshold_eigenvalues

# === MODIFIED FUNCTION ===
def run_pca_analysis(coins, var, data_dict, train_end_date, win_limits=(WINSORIZE_QUANTILE, WINSORIZE_QUANTILE)):
    """
    Extracts training data, WINSORIZES it, standardizes it, fits full PCA, 
    runs Horn's analysis, and returns PCA results.
    
    win_limits: Tuple (lower, upper) percentage to cut. 
                e.g., (0.01, 0.01) replaces the bottom 1% and top 1%.
    """
    df_list = [data_dict[coin][var] for coin in coins if coin in data_dict and var in data_dict[coin].columns]
    if not df_list:
        print(f"Warning: No data found for variable '{var}' in coins {coins}.")
        return None, None, None, None, 0

    data_matrix = pd.concat(df_list, axis=1, keys=coins, join='inner')
    data_matrix.dropna(inplace=True) 

    # STRICT TRAIN SPLIT
    train_matrix = data_matrix.loc[START_DATE:train_end_date].copy()

    if train_matrix.empty:
        print(f"Warning: No training data found between {START_DATE} and {train_end_date} for {var}.")
        return None, None, None, None, 0

    # --- NEW: WINSORIZATION STEP ---
    print(f"Winsorizing data (Limits: {win_limits}) before PCA...")
    # Winsorize column by column to preserve dataframe structure
    for col in train_matrix.columns:
        train_matrix[col] = winsorize(train_matrix[col], limits=win_limits)
    # -------------------------------

    print(f"Fitting PCA on {len(train_matrix)} observation days (Train End: {train_end_date})")

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_matrix)
    
    n_components = train_scaled.shape[1]
    
    # --- Run Horn's Analysis ---
    n_obs, n_vars = train_scaled.shape
    horn_thresholds = horn_parallel_analysis(n_obs, n_vars)
    # ---------------------------
    
    pca = PCA(n_components=n_components)
    
    # Fit and transform to get the actual PC series
    pca_data = pca.fit_transform(train_scaled)
    
    # --- Compare eigenvalues ---
    real_eigenvalues = pca.explained_variance_
    n_significant_pcs = np.sum(real_eigenvalues > horn_thresholds)
    
    # Create a pandas Series, preserving the index
    pc1_series_with_dates = pd.Series(pca_data[:, 0], 
                                      index=train_matrix.index, 
                                      name="PC1")
    
    return pca, train_matrix.columns, pc1_series_with_dates, horn_thresholds, n_significant_pcs

# === END MODIFIED FUNCTION ===


def plot_scree(pca, horn_thresholds, n_significant_pcs, factor_name, coin_list, output_dir):
    """
    Plots and saves combined Scree (Eigenvalues vs. Horn's) and Cumulative Variance plot.
    """
    if pca is None: return

    n_components = len(pca.explained_variance_)
    ind = np.arange(1, n_components + 1)
    
    # --- Get Real Eigenvalues and Variance Ratios ---
    real_eigenvalues = pca.explained_variance_
    var_ratio = pca.explained_variance_ratio_
    cum_var_ratio = np.cumsum(var_ratio)

    plt.figure(figsize=(10, 6))
    ax1 = plt.gca()
    
    # --- Plot 1: Real Eigenvalues (Bars) ---
    bars = ax1.bar(ind, real_eigenvalues, color='royalblue', alpha=0.7, label='Actual Eigenvalues')
    
    # === ADDED: Annotation loop for percentage labels ===
    for i, bar in enumerate(bars):
        height = bar.get_height()
        # Use the variance ratio for the label
        label_text = f'{var_ratio[i]:.1%}'
        ax1.annotate(label_text, 
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3),  # 3 points vertical offset
                     textcoords="offset points",
                     ha='center', va='bottom', fontsize=9)
    # ====================================================

    # --- Plot 2: Horn's Threshold (Line) ---
    ax1.plot(ind, horn_thresholds, color='red', marker='o', linestyle='--', 
             linewidth=2, label="Horn's 95th Percentile (Noise)")
    
    ax1.set_xlabel('Principal Component')
    ax1.set_ylabel('Eigenvalue (Variance)', color='royalblue')
    ax1.set_xticks(ind)
    
    # --- Plot 3: Cumulative Variance (on secondary axis) ---
    ax2 = ax1.twinx()
    ax2.plot(ind, cum_var_ratio, color='darkorange', marker='s', 
             linewidth=2, label='Cumulative Variance')
    ax2.set_ylabel('Cumulative Variance Ratio', color='darkorange')
    ax2.set_ylim(0, 1.05)

    plt.title(f'PCA Scree Plot & Horn\'s Analysis: {factor_name}\n(Training Data: {START_DATE} to {TRAIN_END_DATE})')
    
    # --- Combine legends ---
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    
    # === MODIFIED: Changed legend location ===
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
    # =========================================
    
    plt.tight_layout()

    safe_name = factor_name.replace(":", "").replace(" ", "_")
    plt.savefig(output_dir / f"Scree_{safe_name}.png", dpi=300)
    plt.close()

    print(f"--- Summary for {factor_name} ---")
    print(f"Horn's Analysis suggests {n_significant_pcs} significant component(s).")
    print("-" * 55)
    print(f"{'Component':<12} | {'Actual Eigenvalue':<18} | {'Horns Threshold':<18} | {'Significant':<12}")
    print("-" * 55)
    
    for i in range(n_components):
        is_sig = "YES" if real_eigenvalues[i] > horn_thresholds[i] else "No"
        print(f"{'PC' + str(i+1):<12} | {real_eigenvalues[i]:<18.4f} | {horn_thresholds[i]:<18.4f} | {is_sig:<12}")
    print("-" * 55)


def plot_pc1_distribution(pc1_series, factor_name, output_dir):
    """Plots the histogram and KDE of the PC1 time series with stats."""
    if pc1_series is None or len(pc1_series) == 0: return

    plt.figure(figsize=(8, 6))
    
    # Plot Histogram with KDE
    sns.histplot(pc1_series, kde=True, stat='density', color='teal', 
                 edgecolor='white', alpha=0.6, line_kws={'linewidth': 2})

    # Calculate Stats
    mu_val = np.mean(pc1_series)
    std_val = np.std(pc1_series)
    skew_val = skew(pc1_series)
    kurt_val = kurtosis(pc1_series) # Fisher kurtosis (normal = 0)

    # Add stats box
    stats_text = (f"Mean: {mu_val:.2f}\n"
                  f"Std Dev: {std_val:.2f}\n"
                  f"Skewness: {skew_val:.2f}\n"
                  f"Ex. Kurtosis: {kurt_val:.2f}")
    
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
    plt.gca().text(0.03, 0.97, stats_text, transform=plt.gca().transAxes, 
                   fontsize=10, verticalalignment='top', bbox=props)

    plt.title(f'Distribution of PC1 Factor: {factor_name}\n(Training Data)')
    plt.xlabel('PC1 Value (Standardized Units)')
    plt.ylabel('Density')
    plt.tight_layout()

    safe_name = factor_name.replace(":", "").replace(" ", "_")
    save_path = output_dir / f"Dist_PC1_{safe_name}.png"
    plt.savefig(save_path, dpi=300)
    print(f"Distribution plot saved to: {save_path}")
    plt.close()


# === NEW FUNCTION ===
def plot_pc1_over_time(pc1_series, factor_name, output_dir):
    """Plots the PC1 factor as a time series."""
    if pc1_series is None or pc1_series.empty:
        print(f"Skipping time series plot for {factor_name}: No data.")
        return

    plt.figure(figsize=(12, 6))
    
    pc1_series.plot(linewidth=1.5, color='navy', alpha=0.8)
    
    plt.title(f'PC1 Time Series: {factor_name}\n(Training Data: {START_DATE} to {TRAIN_END_DATE})')
    plt.xlabel('Date')
    plt.ylabel('PC1 Value (Standardized Units)')
    
    # Add a horizontal line at 0 for reference
    plt.axhline(0, color='red', linestyle='--', linewidth=0.8, label='Zero Line')
    plt.legend()
    
    plt.tight_layout()
    
    safe_name = factor_name.replace(":", "").replace(" ", "_")
    save_path = output_dir / f"TimeSeries_PC1_{safe_name}.png"
    plt.savefig(save_path, dpi=300)
    print(f"Time series plot saved to: {save_path}")
    plt.close()

# === END NEW FUNCTION ===


# ===================================================================
# Main Execution
# ===================================================================

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        coin_data = load_data(DATA_DIR, ALL_COINS, START_DATE)
    except Exception as e:
        print(f"Error loading data: {e}")
        exit()

    analysis_groups = [
        ("Stablecoins: Volume", STABLECOINS, "LogVolChange"),
        ("Stablecoins: Volatility", STABLECOINS, STATIONARY_VOL),
        ("Stablecoins: Returns", STABLECOINS, "Log Returns"),
        ("Cryptos: Returns", CRYPTOS, "Log Returns"),
        ("Cryptos: Volatility", CRYPTOS, STATIONARY_VOL),
        ("Cryptos: Volume", CRYPTOS, "LogVolChange")
    ]

    print("\nStarting PCA Analysis (Training Data Only)...\n")
    
    # === ADDED: List to store summary results ===
    summary_results = []
    
    for title, coins, var_name in analysis_groups:
        print(f"\nProcessing: {title}")
        
        pca_result, coin_cols, pc1_series, horn_thresholds, n_sig_pcs = run_pca_analysis(
            coins, var_name, coin_data, TRAIN_END_DATE
        )
        
        if pca_result is not None:
            plot_scree(pca_result, horn_thresholds, n_sig_pcs, title, coin_cols, OUTPUT_DIR)
            plot_pc1_distribution(pc1_series, title, OUTPUT_DIR)
            
            # === ADDED: Call to the new plotting function ===
            plot_pc1_over_time(pc1_series, title, OUTPUT_DIR)
            # ===============================================
            
            # === ADDED: Calculate and store summary data ===
            # Sum variance of *only* the significant components
            total_var_explained = np.sum(pca_result.explained_variance_ratio_[:n_sig_pcs])
            
            summary_results.append({
                "Variable": title,
                "Number of significant PCs": n_sig_pcs,
                "Total Variance Explained": total_var_explained
            })
            
        else:
            print(f"Skipping {title} due to lack of data.")
            # === ADDED: Store placeholder for skipped data ===
            summary_results.append({
                "Variable": title,
                "Number of significant PCs": 0,
                "Total Variance Explained": 0.0
            })
            
    print(f"\nAnalysis Complete. Plots saved to {OUTPUT_DIR.absolute()}")

    # === ADDED: Create and print the final summary table ===
    summary_df = pd.DataFrame(summary_results)
    
    print("\n" + "="*70)
    print("           PCA Summary Table (Training Data)")
    print("="*70)
    
    # Format the dataframe for clean printing
    print(summary_df.to_string(
        index=False,
        formatters={'Total Variance Explained': '{:,.2%}'.format}
    ))
    print("="*70)