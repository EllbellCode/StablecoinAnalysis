import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from statsmodels.tsa.api import VAR
import warnings

warnings.filterwarnings('ignore')

# ===================================================================
# CONFIGURATION
# ===================================================================
DATA_DIR = Path("Data/Verified")
OUTPUT_DIR = Path("Results/AsymmetryTest")

START_DATE = '2020-01-01'
END_DATE = '2024-01-01'

# CONFIGURATION: Select your specific volatility measures here
CRYPTO_VOL_COLUMN = "Delta_LogGK"   # e.g., "Delta_LogRV", "GarmanKlass_Vol"
STABLE_VOL_COLUMN = "Delta_LogRV"   # e.g., "Parkinson_Vol"

MAXLAGS = 5
WINSOR_QUANTILE = 0.01 
MIN_PCA_WINDOW = 60 

# ===================================================================
# Helper Functions
# ===================================================================

def get_expanding_pca(df, min_periods=30):
    """
    Calculates the 1st Principal Component using an Expanding Window.
    """
    n_samples, n_features = df.shape
    pc_series = np.full(n_samples, np.nan)
    prev_components = None
    
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

def load_factors_with_specific_vol(data_dir, start_date, end_date, winsor_limit, crypto_vol_col, stable_vol_col):
    print(f"  -> Loading Factors...")
    print(f"     Crypto Vol: {crypto_vol_col}")
    print(f"     Stable Vol: {stable_vol_col}")
    
    coin_data = {}
    if not data_dir.exists(): raise FileNotFoundError(f"Data directory not found: {data_dir}")

    for file in data_dir.glob("*.csv"):
        if (c := file.stem.replace("Verif_", "")) in ["DAI", "USDC", "USDT", "BNB", "BTC", "ETH", "XRP"]:
            df = pd.read_csv(file, parse_dates=['Date']).sort_values("Date")
            coin_data[c] = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)].set_index('Date')

    def get_fac(coins, var, name):
        # Validation: Check if column exists for at least one coin
        valid_coins = [c for c in coins if c in coin_data and var in coin_data[c].columns]
        if not valid_coins:
            print(f"     [WARNING] Column '{var}' not found for {name}. Returning Empty.")
            return pd.Series(dtype=float, name=name)

        df_list = [coin_data[c][var] for c in valid_coins]
        df = pd.concat(df_list, axis=1, keys=valid_coins, join='inner').dropna()
        for col in df.columns: 
            df[col] = winsorize(df[col], limits=[winsor_limit, winsor_limit])
        
        pca_series = get_expanding_pca(df, min_periods=MIN_PCA_WINDOW)
        pca_series.name = name
        return pca_series

    # 1. Get Base Factors
    factors = {
        "Crypto_Returns": get_fac(["BNB", "BTC", "ETH", "XRP"], "Log Returns", "Crypto_Returns"),
        "Stable_Returns": get_fac(["DAI", "USDC", "USDT"], "Log Returns", "Stable_Returns"),
        
        # USE SPECIFIC CONFIGURATION HERE
        "Crypto_Volatility": get_fac(["BNB", "BTC", "ETH", "XRP"], crypto_vol_col, "Crypto_Volatility"),
        "Stable_Volatility": get_fac(["DAI", "USDC", "USDT"], stable_vol_col, "Stable_Volatility"),
        
        "Crypto_Volume": get_fac(["BNB", "BTC", "ETH", "XRP"], "LogVolChange", "Crypto_Volume"),
        "Stable_Volume": get_fac(["DAI", "USDC", "USDT"], "LogVolChange", "Stable_Volume"),
    }
    
    # 2. CREATE ASYMMETRIC FACTORS
    if not factors["Stable_Returns"].empty:
        sr = factors["Stable_Returns"]
        factors["Stable_Pos"] = sr.apply(lambda x: x if x > 0 else 0)
        factors["Stable_Pos"].name = "Stable_Pos"
        factors["Stable_Neg"] = sr.apply(lambda x: x if x < 0 else 0)
        factors["Stable_Neg"].name = "Stable_Neg"
    
    return factors

def run_asymmetric_causality(factors_dict):
    """
    Runs VAR for Crypto Targets against Stable Pos/Neg/Vol/Vol.
    """
    results_summary = []
    
    # Define System Variables
    var_columns = [
        "Crypto_Returns", "Crypto_Volatility", "Crypto_Volume",
        "Stable_Pos", "Stable_Neg", 
        "Stable_Volatility", "Stable_Volume"
    ]
    
    existing_cols = [c for c in var_columns if c in factors_dict and not factors_dict[c].empty]
    df = pd.concat([factors_dict[c] for c in existing_cols], axis=1).dropna()
    
    print(f"  -> Fitting System VAR (Shape: {df.shape})...")
    
    try:
        model = VAR(df)
        lag_results = model.select_order(maxlags=MAXLAGS)
        optimal_lag = lag_results.aic if lag_results.aic > 0 else 1
        print(f"     Optimal Lag: {optimal_lag}")
        
        var_results = model.fit(optimal_lag)
        
        # Targets: All Crypto Variables
        targets = ["Crypto_Returns", "Crypto_Volatility", "Crypto_Volume"]
        
        # Sources: All Stable Variables (Broken into components)
        sources = ["Stable_Pos", "Stable_Neg", "Stable_Volatility", "Stable_Volume"]
        
        for target in targets:
            if target not in df.columns: continue
            
            for source in sources:
                if source not in df.columns: continue
                
                try:
                    test_res = var_results.test_causality(caused=target, causing=source)
                    p_val = test_res.pvalue
                    sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.1 else ""
                    
                    results_summary.append({
                        "Source": source,
                        "Target": target,
                        "Lag": optimal_lag,
                        "p_value": p_val,
                        "Significance": sig
                    })
                    print(f"     {source:<17} -> {target:<20} | p={p_val:.4f} {sig}")
                    
                except Exception as e:
                    print(f"     [Error] {source} -> {target}: {e}")
                    
    except Exception as e:
        print(f"  [Critical Error] Failed to fit VAR: {e}")
        
    return results_summary

# ===================================================================
# Main Execution Loop
# ===================================================================
if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("Starting Asymmetry Analysis...")
    print("=" * 80)
    
    # 1. Load Factors (Single Pass)
    factors = load_factors_with_specific_vol(
        DATA_DIR, START_DATE, END_DATE, WINSOR_QUANTILE, 
        CRYPTO_VOL_COLUMN, STABLE_VOL_COLUMN
    )
    
    # 2. Run Analysis
    all_results = run_asymmetric_causality(factors)
        
    # 3. Save Results
    if all_results:
        final_df = pd.DataFrame(all_results)
        save_path = OUTPUT_DIR / "Asymmetry_Results_Summary.csv"
        final_df.to_csv(save_path, index=False)
        
        print("\n" + "=" * 80)
        print(f"Analysis Complete. Results saved to:\n{save_path.resolve()}")
        print("=" * 80)
        print(final_df)
    else:
        print("No results generated. Check data.")