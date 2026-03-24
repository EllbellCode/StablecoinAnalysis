import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
from scipy.stats import skew, kurtosis
from scipy.stats.mstats import winsorize

sns.set_theme(style="white")

DATA_DIR = Path("Data/Verified")
OUTPUT_DIR = Path("Plots/PCA")
RESULTS_DIR = Path("Results/PCA")

START_DATE = '2020-01-01'
TRAIN_END_DATE = '2024-01-01' 
STATIONARY_VOL = "RS"

STABLECOINS = ["DAI", "USDC", "USDT"]
CRYPTOS = ["BNB", "BTC", "ETH", "XRP"]
ALL_COINS = STABLECOINS + CRYPTOS

WINSORIZE_QUANTILE = 0.01 

def load_data(data_dir, all_coins, start_date):
    
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
    
    print(f"Running Horn's Parallel Analysis (Iter={n_iterations}, P={percentile}%)...")
    
    random_eigenvalues = np.zeros((n_iterations, n_variables))
    
    for i in range(n_iterations):
        random_data = np.random.normal(size=(n_observations, n_variables))
        scaler = StandardScaler()
        random_scaled = scaler.fit_transform(random_data)
        pca_random = PCA(n_components=n_variables)
        pca_random.fit(random_scaled)
        random_eigenvalues[i, :] = pca_random.explained_variance_
        
    threshold_eigenvalues = np.percentile(random_eigenvalues, percentile, axis=0)
    return threshold_eigenvalues

def run_pca_analysis(coins, var, data_dict, train_end_date, win_limits=(WINSORIZE_QUANTILE, WINSORIZE_QUANTILE)):
    
    df_list = [data_dict[coin][var] for coin in coins if coin in data_dict and var in data_dict[coin].columns]
    if not df_list:
        print(f"Warning: No data found for variable '{var}' in coins {coins}.")
        return None, None, None, None, 0

    data_matrix = pd.concat(df_list, axis=1, keys=coins, join='inner')
    data_matrix.dropna(inplace=True) 

    train_matrix = data_matrix.loc[START_DATE:train_end_date].copy()

    if train_matrix.empty:
        print(f"Warning: No training data found between {START_DATE} and {train_end_date} for {var}.")
        return None, None, None, None, 0

    print(f"Winsorizing data (Limits: {win_limits}) before PCA...")
    for col in train_matrix.columns:
        train_matrix[col] = winsorize(train_matrix[col], limits=win_limits)

    print(f"Fitting PCA on {len(train_matrix)} observation days (Train End: {train_end_date})")

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_matrix)
    
    n_components = train_scaled.shape[1]
    n_obs, n_vars = train_scaled.shape
    horn_thresholds = horn_parallel_analysis(n_obs, n_vars)
    
    pca = PCA(n_components=n_components)
    pca_data = pca.fit_transform(train_scaled)
    
    real_eigenvalues = pca.explained_variance_
    n_significant_pcs = np.sum(real_eigenvalues > horn_thresholds)
    
    pc1_series_with_dates = pd.Series(pca_data[:, 0], 
                                      index=train_matrix.index, 
                                      name="PC1")
    
    return pca, train_matrix.columns, pc1_series_with_dates, horn_thresholds, n_significant_pcs

def plot_scree(pca, horn_thresholds, n_significant_pcs, factor_name, coin_list, output_dir):
    
    if pca is None: return

    n_components = len(pca.explained_variance_)
    ind = np.arange(1, n_components + 1)
    real_eigenvalues = pca.explained_variance_
    var_ratio = pca.explained_variance_ratio_
    
    plt.figure(figsize=(10, 6))
    ax1 = plt.gca()
    
    bars = ax1.bar(ind, real_eigenvalues, color='midnightblue', alpha=0.85, label='Actual Eigenvalues')

    max_height = max(real_eigenvalues)
    
    ax1.set_ylim(0, max_height * 1.15)
    
    for i, bar in enumerate(bars):
        height = bar.get_height()
        label_text = f'{var_ratio[i]:.1%}'
        ax1.annotate(label_text, 
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), 
                     textcoords="offset points",
                     ha='center', va='bottom', fontsize=20, color='black')

    ax1.plot(ind, horn_thresholds, color='red', marker='o', linestyle='--', 
             linewidth=2, label="Horn's 95th Percentile")
    
    ax1.set_xlabel('Principal Component', color='black', fontsize=20)
    ax1.set_ylabel('Eigenvalue (Variance)', color='black', fontsize = 20)
    ax1.set_xticks(ind)
    ax1.tick_params(axis='x', colors='black', labelsize=20)
    ax1.tick_params(axis='y', colors='black', labelsize=20)
    
    ax1.legend(loc='upper right', fontsize=20) 
    
    plt.tight_layout()

    safe_name = factor_name.replace(":", "").replace(" ", "_")
    plt.savefig(output_dir / f"Scree_{safe_name}.png", dpi=300)
    plt.close()

    print(f"--- Summary for {factor_name} ---")
    print(f"Horn's Analysis suggests {n_significant_pcs} significant component(s).")
    print("-" * 55)

def plot_pc1_distribution(pc1_series, factor_name, output_dir):
    if pc1_series is None or len(pc1_series) == 0: return

    plt.figure(figsize=(8, 6))
    sns.histplot(pc1_series, kde=True, stat='density', color='teal', 
                 edgecolor='white', alpha=0.6, line_kws={'linewidth': 2})

    mu_val = np.mean(pc1_series)
    std_val = np.std(pc1_series)
    skew_val = skew(pc1_series)
    kurt_val = kurtosis(pc1_series)

    stats_text = (f"Mean: {mu_val:.2f}\n"
                  f"Std Dev: {std_val:.2f}\n"
                  f"Skewness: {skew_val:.2f}\n"
                  f"Ex. Kurtosis: {kurt_val:.2f}")
    
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
    plt.gca().text(0.03, 0.97, stats_text, transform=plt.gca().transAxes, 
                   fontsize=20, verticalalignment='top', bbox=props)
    
    plt.xlabel('PC1 Value (Standardized Units)', color='black')
    plt.ylabel('Density', color='black')
    plt.tick_params(axis='x', colors='black')
    plt.tick_params(axis='y', colors='black')
    
    plt.tight_layout()

    safe_name = factor_name.replace(":", "").replace(" ", "_")
    save_path = output_dir / f"Dist_PC1_{safe_name}.png"
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_pc1_over_time(pc1_series, factor_name, output_dir):
    if pc1_series is None or pc1_series.empty:
        return

    plt.figure(figsize=(12, 6))
    pc1_series.plot(linewidth=1.5, color='navy', alpha=0.8)
    
    plt.xlabel('Date', color='black')
    plt.ylabel('PC1 Value (Standardized Units)', color='black')
    plt.tick_params(axis='x', colors='black')
    plt.tick_params(axis='y', colors='black')
    
    plt.axhline(0, color='red', linestyle='--', linewidth=0.8, label='Zero Line')
    plt.legend()
    plt.tight_layout()
    
    safe_name = factor_name.replace(":", "").replace(" ", "_")
    save_path = output_dir / f"TimeSeries_PC1_{safe_name}.png"
    plt.savefig(save_path, dpi=300)
    plt.close()


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        coin_data = load_data(DATA_DIR, ALL_COINS, START_DATE)
    except Exception as e:
        print(f"Error loading data: {e}")
        exit()

    analysis_groups = [
        ("Stablecoins: Volume", STABLECOINS, "LogVolChange"),
        ("Stablecoins: Volatility", STABLECOINS, STATIONARY_VOL),
        ("Stablecoins: Upside", STABLECOINS, "Upside_Vol"),
        ("Stablecoins: Downside", STABLECOINS, "Downside_Vol"),
        ("Cryptos: Volume", CRYPTOS, "LogVolChange"),
        ("Cryptos: Volatility", CRYPTOS, STATIONARY_VOL),
        ("Cryptos: Upside", CRYPTOS, "Upside_Vol"),
        ("Cryptos: Downside", CRYPTOS, "Downside_Vol")
    ]

    print("\nStarting PCA Analysis (Training Data Only)...\n")
    
    table_summary_rows = []
    
    all_loadings_list = []
    
    for title, coins, var_name in analysis_groups:
        print(f"\nProcessing: {title}")
        
        pca_result, coin_cols, pc1_series, horn_thresholds, n_sig_pcs = run_pca_analysis(
            coins, var_name, coin_data, TRAIN_END_DATE
        )
        
        if pca_result is not None:
           
            plot_scree(pca_result, horn_thresholds, n_sig_pcs, title, coin_cols, OUTPUT_DIR)
            plot_pc1_distribution(pc1_series, title, OUTPUT_DIR)
            plot_pc1_over_time(pc1_series, title, OUTPUT_DIR)
            
            eigenvalues = pca_result.explained_variance_
            
            row_data = {"Factor": title}
            
            for i in range(4):
                pc_num = i + 1
                if i < len(eigenvalues):
                   
                    eig_val = eigenvalues[i]
                    crit_val = horn_thresholds[i]
                    row_data[f"PC{pc_num} Eigen Value"] = f"{eig_val:.2f}"
                    row_data[f"Horn Critical value for PC{pc_num}"] = f"{crit_val:.2f}"

                else:
                   
                    row_data[f"PC{pc_num} Eigen Value"] = "-"
                    row_data[f"Horn Critical value for PC{pc_num}"] = "-"
            
            pc1_var_explained = pca_result.explained_variance_ratio_[0]
            row_data["Variation Explained percentage"] = f"{pc1_var_explained:.1%}"
            
            table_summary_rows.append(row_data)
           
            pc1_vector = pca_result.components_[0]
            group_loadings = pd.DataFrame({
                'Group': title,
                'Coin': coin_cols,
                'PC1_Loading': pc1_vector
            })
            all_loadings_list.append(group_loadings)
            
        else:
            print(f"Skipping {title} due to lack of data.")
            
    print(f"\nAnalysis Complete. Plots saved to {OUTPUT_DIR.absolute()}")

    if table_summary_rows:
        
        columns_order = [
            "Factor",
            "PC1 Eigen Value", "Horn Critical value for PC1",
            "PC2 Eigen Value", "Horn Critical value for PC2",
            "PC3 Eigen Value", "Horn Critical value for PC3",
            "PC4 Eigen Value", "Horn Critical value for PC4",
            "Variation Explained percentage"
        ]
        
        summary_table_df = pd.DataFrame(table_summary_rows)
        summary_table_df = summary_table_df[columns_order]
        summary_path = RESULTS_DIR / "pca_summary_table.csv"
        summary_table_df.to_csv(summary_path, index=False)
        
        print("\n" + "="*80)
        print("           PCA Summary Table (Saved to CSV)")
        print("="*80)
        print(summary_table_df.to_string(index=False))
        print(f"\nSummary table saved to: {summary_path}")

    if all_loadings_list:
        final_loadings_df = pd.concat(all_loadings_list, ignore_index=True)
        final_loadings_df.to_csv(RESULTS_DIR / "PCA_Loadings_Raw.csv", index=False)

        print("\n" + "="*80)
        print("           Generating Combined Loadings Matrix")
        print("="*80)

        row_map = {
            "Stablecoins: Volume": "Returns / Volume",
            "Stablecoins: Volatility": "Volatility",
            "Stablecoins: Upside": "Upside Volatility",
            "Stablecoins: Downside": "Downside Volatility",
            
            "Cryptos: Returns": "Returns / Volume",
            "Cryptos: Volatility": "Volatility",
            "Cryptos: Upside": "Upside Volatility",
            "Cryptos: Downside": "Downside Volatility"
        }
        
        final_loadings_df['RowLabel'] = final_loadings_df['Group'].map(row_map)
        
        matrix_combined = final_loadings_df.pivot(index='RowLabel', columns='Coin', values='PC1_Loading')

        desired_row_order = [
            "Returns / Volume", 
            "Volatility", 
            "Upside Volatility", 
            "Downside Volatility"
        ]
    
        existing_rows = [r for r in desired_row_order if r in matrix_combined.index]
        matrix_combined = matrix_combined.reindex(existing_rows)
 
        ordered_cols = []
        for c in STABLECOINS + CRYPTOS:
            if c in matrix_combined.columns:
                ordered_cols.append(c)
        
        matrix_combined = matrix_combined[ordered_cols]

        print("\nCombined Matrix Preview:")
        print(matrix_combined.round(4))
        
        save_path = RESULTS_DIR / "PCA_Loadings_Matrix_Combined.csv"
        matrix_combined.to_csv(save_path)
        print(f"Combined matrix saved to: {save_path}")
        print("="*80)