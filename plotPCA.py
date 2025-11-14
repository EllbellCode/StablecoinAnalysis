import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
from scipy.stats import skew, kurtosis

# Set plot style for better visuals
sns.set_theme(style="whitegrid")

# ===================================================================
# Configuration
# ===================================================================
DATA_DIR = Path("Data/Verified")
OUTPUT_DIR = Path("Plots/Plots_PCA") 

START_DATE = '2020-01-01'
TRAIN_END_DATE = '2024-01-01' 
STATIONARY_VOL = "Delta_LogRV"

STABLECOINS = ["DAI", "USDC", "USDT"]
CRYPTOS = ["BNB", "BTC", "ETH", "XRP"]
ALL_COINS = STABLECOINS + CRYPTOS

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

def run_pca_analysis(coins, var, data_dict, train_end_date):
    """
    Extracts training data, standardizes it, fits full PCA, and returns PC1 series.
    """
    df_list = [data_dict[coin][var] for coin in coins if coin in data_dict and var in data_dict[coin].columns]
    if not df_list:
        print(f"Warning: No data found for variable '{var}' in coins {coins}.")
        return None, None, None

    data_matrix = pd.concat(df_list, axis=1, keys=coins, join='inner')
    data_matrix.dropna(inplace=True) 

    # STRICT TRAIN SPLIT
    train_matrix = data_matrix.loc[START_DATE:train_end_date]

    if train_matrix.empty:
        print(f"Warning: No training data found between {START_DATE} and {train_end_date} for {var}.")
        return None, None, None

    print(f"Fitting PCA on {len(train_matrix)} observation days (Train End: {train_end_date})")

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_matrix)
    
    n_components = train_scaled.shape[1]
    pca = PCA(n_components=n_components)
    
    # Fit and transform to get the actual PC series
    pca_data = pca.fit_transform(train_scaled)
    
    # Extract just PC1 (column 0)
    pc1_series = pca_data[:, 0]
    
    return pca, train_matrix.columns, pc1_series

def plot_scree(pca, factor_name, coin_list, output_dir):
    """Plots and saves combined Scree and Cumulative Variance plot."""
    if pca is None: return

    n_components = len(pca.explained_variance_ratio_)
    ind = np.arange(1, n_components + 1)
    var_ratio = pca.explained_variance_ratio_
    cum_var_ratio = np.cumsum(var_ratio)

    plt.figure(figsize=(10, 6))
    ax1 = plt.gca()
    bars = ax1.bar(ind, var_ratio, color='royalblue', alpha=0.7, label='Individual Variance')
    ax1.set_xlabel('Principal Component')
    ax1.set_ylabel('Explained Variance Ratio', color='royalblue')
    ax1.set_ylim(0, 1.05)
    ax1.set_xticks(ind)

    for bar in bars:
        height = bar.get_height()
        ax1.annotate(f'{height:.1%}', xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

    ax2 = ax1.twinx()
    ax2.plot(ind, cum_var_ratio, color='darkorange', marker='o', linewidth=2, label='Cumulative Variance')
    ax2.set_ylabel('Cumulative Variance Ratio', color='darkorange')
    ax2.set_ylim(0, 1.05)

    plt.title(f'PCA Scree Plot: {factor_name}\nTraining Data ({START_DATE} to {TRAIN_END_DATE})')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')
    plt.tight_layout()

    safe_name = factor_name.replace(":", "").replace(" ", "_")
    plt.savefig(output_dir / f"Scree_{safe_name}.png", dpi=300)
    plt.close()

    print(f"--- Summary for {factor_name} ---")
    for i, (var, cum_var) in enumerate(zip(var_ratio, cum_var_ratio)):
        print(f"PC{i+1}: {var:.2%} (Cum: {cum_var:.2%})")
    print("-" * 40)

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
        ("Cryptos: Returns", CRYPTOS, "Log Returns"),
        ("Cryptos: Volatility", CRYPTOS, STATIONARY_VOL),
    ]

    print("\nStarting PCA Analysis (Training Data Only)...\n")
    
    for title, coins, var_name in analysis_groups:
        print(f"Processing: {title}")
        # Now returns PC1 series as well
        pca_result, coin_cols, pc1_series = run_pca_analysis(coins, var_name, coin_data, TRAIN_END_DATE)
        
        if pca_result is not None:
            plot_scree(pca_result, title, coin_cols, OUTPUT_DIR)
            plot_pc1_distribution(pc1_series, title, OUTPUT_DIR)
        else:
            print(f"Skipping {title} due to lack of data.")
            
    print(f"\nAnalysis Complete. Plots saved to {OUTPUT_DIR.absolute()}")