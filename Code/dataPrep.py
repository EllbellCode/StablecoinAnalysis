"""
Script for preparing our data for the analysis.

Converts our daily OHLC data into all the metrics/features used in the research
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Convert suffixes into numerical values
def convertVol(volume_str):
    if pd.isna(volume_str):
        return None

    suffixes = {'k': 1e3, 'K': 1e3, 'm': 1e6, 'M': 1e6, 'b': 1e9, 'B': 1e9}
    if isinstance(volume_str, str) and volume_str[-1] in suffixes:
        return float(volume_str[:-1]) * suffixes[volume_str[-1]]

    return float(volume_str)

# Standardise data using the above conversion function
def standardise(data_dir):
    data_dir = Path(data_dir)
    output_dir = data_dir / 'Verified'
    output_dir.mkdir(exist_ok=True)

    for file in data_dir.glob("*.csv"):
        print(f"Processing {file.name}")
        df = pd.read_csv(file)

        
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors='coerce')

        if 'Vol.' in df.columns:
            df.rename(columns={'Vol.': 'Volume'}, inplace=True)

        for col in ['Close', 'Open', 'High', 'Low']:
            if col in df.columns:
                # Remove commas
                df[col] = df[col].astype(str).str.replace(',', '', regex=False)
                df[col] = pd.to_numeric(df[col], errors='coerce')
                # FORCE POSITIVE: Replace <= 0 with NaN
                df[col] = df[col].where(df[col] > 0, np.nan)

        if 'Volume' in df.columns:

            if df['Volume'].dtype == object:
                df['Volume'] = df['Volume'].apply(convertVol)
            
            df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
            df['Volume'] = df['Volume'].where(df['Volume'] > 0, np.nan)

        out_path = output_dir / f"Verif_{file.name}"
        df.to_csv(out_path, index=False)
        print(f"Saved to {out_path}")

# Check for missing values
def checkMissing(df, start, end):

    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    expected_dates = pd.date_range(start=start, end=end).date
    actual_dates = df['Date'].dt.date.dropna().unique()

    return sorted(set(expected_dates) - set(actual_dates))

# Interpolate missing data (not used)
def interpolate(missing_dict, data_dir):
    
    data_dir = Path(data_dir)

    for file_name, missing_dates in missing_dict.items():
        path = data_dir / file_name
        df = pd.read_csv(path)
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.set_index('Date').sort_index()

        # Reindex to insert missing dates as NaNs
        full_range = pd.date_range(start=df.index.min(), end=df.index.max())
        df = df.reindex(full_range)

        # Forward Fill ONLY (No look-ahead)
        df = df.ffill(limit=5)

        # Reset index
        df = df.reset_index().rename(columns={'index': 'Date'})
        df.to_csv(path, index=False)
        print(f"Interpolated (Forward Fill) complete for {file_name}")

# Calculate returns   
def calcReturns(data_dir):
    data_dir = Path(data_dir)

    for file in data_dir.glob("*.csv"):
        df = pd.read_csv(file)
        
        # 1. Force Standard Sort Order (Ascending: Oldest -> Newest)
        df['Date'] = pd.to_datetime(df['Date'])
        df.sort_values("Date", ascending=True, inplace=True)

        if 'Close' in df.columns:
            # 2. Use pct_change() for robust (Today / Yesterday) - 1
            df['Returns'] = df['Close'].pct_change()
            
            # 3. Log Returns: ln(Today / Yesterday)
            df['Log Returns'] = np.log(df['Close'] / df['Close'].shift(1))
            
            df.to_csv(file, index=False)
            print(f"Returns calculated for {file.name}")

# Calculate volume metric
def volNorm(data_dir):
    data_dir = Path(data_dir)

    for file in data_dir.glob("*.csv"):
        df = pd.read_csv(file)
        df['Date'] = pd.to_datetime(df['Date'])
        df.sort_values("Date", ascending=True, inplace=True)

        df['VolChange'] = df['Volume'].pct_change(fill_method=None)
        
        safe_vol = df['Volume'].where(df['Volume'] > 0, np.nan)
        vol_ratio = safe_vol / safe_vol.shift(1)
        df['LogVolChange'] = np.log(vol_ratio.where(vol_ratio > 0, np.nan))

        df.to_csv(file, index=False)
        print(f"Volume normalised for {file.name}")

# Calculate volatility metrics
def addVolatility(data_dir):
    
    data_dir = Path(data_dir)

    for file in data_dir.glob("*.csv"):
        df = pd.read_csv(file)
        
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df.sort_values("Date", ascending=True, inplace=True)

        o = df['Open'].where(df['Open'] > 0, np.nan)
        h = df['High'].where(df['High'] > 0, np.nan)
        l = df['Low'].where(df['Low'] > 0, np.nan)
        c = df['Close'].where(df['Close'] > 0, np.nan)

        log_o = np.log(o)
        log_h = np.log(h)
        log_l = np.log(l)
        log_c = np.log(c)
        
        # Garman Klass
        gk_term1 = 0.5 * ((log_h - log_l) ** 2)
        gk_term2 = (2 * np.log(2) - 1) * ((log_c - log_o) ** 2)
        df['GarmanKlass'] = np.sqrt((gk_term1 - gk_term2).clip(lower=0))
        df['Delta_LogGK'] = np.log(df['GarmanKlass'].replace(0, np.nan)).diff()

        # Rogers Satchell Upside and Downside
        rs_upside_var = (log_h - log_c) * (log_h - log_o)  
        rs_downside_var = (log_l - log_c) * (log_l - log_o)
        rs_total_var = rs_upside_var + rs_downside_var
        df['RS'] = np.sqrt(rs_total_var.clip(lower=0)).diff()
        df['Upside_Vol'] = np.sqrt(rs_upside_var.clip(lower=0)).diff()
        df['Downside_Vol'] = np.sqrt(rs_downside_var.clip(lower=0)).diff()

        
        df.to_csv(file, index=False)

# Rolling weekly and monthly metrics
def addRollingMetrics(data_dir):
    data_dir = Path(data_dir)
    windows = {'Weekly': 7, 'Monthly': 30}
    
    for file in data_dir.glob("*.csv"):
        df = pd.read_csv(file)
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df.sort_values('Date', ascending=True, inplace=True)
        
        for suffix, window in windows.items():
            
            roll_open = df['Open'].shift(window - 1)  
            roll_close = df['Close']                  
            roll_high = df['High'].rolling(window=window).max() 
            roll_low = df['Low'].rolling(window=window).min()  
            roll_vol = df['Volume'].rolling(window=window).sum()

            log_o = np.log(roll_open.where(roll_open > 0, np.nan))
            log_c = np.log(roll_close.where(roll_close > 0, np.nan))
            log_h = np.log(roll_high.where(roll_high > 0, np.nan))
            log_l = np.log(roll_low.where(roll_low > 0, np.nan))
            
            rs_upside_var = (log_h - log_c) * (log_h - log_o)
            rs_downside_var = (log_l - log_c) * (log_l - log_o)
            rs_total_var = rs_upside_var + rs_downside_var
            
            df[f'Upside_Vol_{suffix}'] = np.sqrt(rs_upside_var.clip(lower=0)).diff()
            df[f'Downside_Vol_{suffix}'] = np.sqrt(rs_downside_var.clip(lower=0)).diff()
            df[f'RS_{suffix}'] = np.sqrt(rs_total_var.clip(lower=0)).diff()

        
            price_ratio = df['Close'] / df['Close'].shift(window)
            df[f'Log Returns_{suffix}'] = np.log(price_ratio.where(price_ratio > 0, np.nan))
            
            safe_vol = roll_vol.where(roll_vol > 0, np.nan)
            df[f'LogVolChange_{suffix}'] = np.log(safe_vol / safe_vol.shift(1))
            
            high_low_ratio = (roll_high / roll_low).where((roll_high > 0) & (roll_low > 0), np.nan)
            close_open_ratio = (roll_close / roll_open).where((roll_close > 0) & (roll_open > 0), np.nan)
            term1 = 0.5 * (np.log(high_low_ratio) ** 2)
            term2 = (2 * np.log(2) - 1) * (np.log(close_open_ratio) ** 2)
            gk_vol = np.sqrt((term1 - term2).clip(lower=0))
            df[f'Delta_LogGK_{suffix}'] = np.log(gk_vol.where(gk_vol > 0, np.nan)).diff()
            
        df.to_csv(file, index=False)
        print(f"Rolling metrics for {file.name}")

def main():

    data_folder = Path("Data")
    verif_folder = data_folder / "Verified"
    start_date = '2020-01-01'
    end_date = '2025-01-01'
    missing_dates_dict = {}

    standardise(data_folder)

    for file in verif_folder.glob("*.csv"):
        df = pd.read_csv(file)

        df = df.rename(columns={'Start': 'Date'})
        df = df.drop('End', axis=1)
        df.to_csv(file, index=False)

        start = start_date
        missing = checkMissing(df, start, end_date)
        if missing:
            missing_dates_dict[file.name] = missing

    print("Missing Dates:", missing_dates_dict)

    interpolate(missing_dates_dict, verif_folder)
    calcReturns(verif_folder)
    volNorm(verif_folder)
    addVolatility(verif_folder)
    addRollingMetrics(verif_folder)

if __name__ == "__main__":
    main()