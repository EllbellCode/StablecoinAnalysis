import pandas as pd
import numpy as np
from pathlib import Path

def convertVol(volume_str):
    if pd.isna(volume_str):
        return None

    suffixes = {'k': 1e3, 'K': 1e3, 'm': 1e6, 'M': 1e6, 'b': 1e9, 'B': 1e9}
    if isinstance(volume_str, str) and volume_str[-1] in suffixes:
        return float(volume_str[:-1]) * suffixes[volume_str[-1]]

    return float(volume_str)

def standardize(data_dir):
    data_dir = Path(data_dir)
    output_dir = data_dir / 'Verified'
    output_dir.mkdir(exist_ok=True)

    for file in data_dir.glob("*.csv"):
        print(f"Processing {file.name}")
        df = pd.read_csv(file)

        # 1. Standardize Date
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors='coerce')

        # 2. Standardize Column Names (Renaming Vol. to Volume immediately)
        if 'Vol.' in df.columns:
            df.rename(columns={'Vol.': 'Volume'}, inplace=True)

        # 3. Clean Numeric Columns (Price)
        for col in ['Close', 'Open', 'High', 'Low']:
            if col in df.columns:
                # Remove commas
                df[col] = df[col].astype(str).str.replace(',', '', regex=False)
                df[col] = pd.to_numeric(df[col], errors='coerce')
                # FORCE POSITIVE: Replace <= 0 with NaN
                df[col] = df[col].where(df[col] > 0, np.nan)

        # 4. Clean Volume (Now runs for ALL files)
        if 'Volume' in df.columns:
            # Handle 'K', 'M', 'B' suffixes if they exist
            if df['Volume'].dtype == object:
                df['Volume'] = df['Volume'].apply(convertVol)
            
            # FORCE POSITIVE: Replace <= 0 with NaN
            df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
            df['Volume'] = df['Volume'].where(df['Volume'] > 0, np.nan)

        # Save
        out_path = output_dir / f"Verif_{file.name}"
        df.to_csv(out_path, index=False)
        print(f"Saved to {out_path}")

def checkMissing(df, start, end):

    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    expected_dates = pd.date_range(start=start, end=end).date
    actual_dates = df['Date'].dt.date.dropna().unique()

    return sorted(set(expected_dates) - set(actual_dates))

def interpolate(missing_dict, data_dir):
    data_dir = Path(data_dir)

    for file_name, missing_dates in missing_dict.items():
        path = data_dir / file_name
        df = pd.read_csv(path)
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

        for miss_date in missing_dates:
            miss_dt = pd.to_datetime(miss_date)
            before = df[df['Date'] == miss_dt - pd.Timedelta(days=1)]
            after = df[df['Date'] == miss_dt + pd.Timedelta(days=1)]

            if not before.empty and not after.empty:
                new_row = {
                    'Date': miss_dt,
                    'Close': (before['Close'].values[0] + after['Close'].values[0]) / 2,
                    'Open': (before['Open'].values[0] + after['Open'].values[0]) / 2,
                    'High': (before['High'].values[0] + after['High'].values[0]) / 2,
                    'Low': (before['Low'].values[0] + after['Low'].values[0]) / 2,
                    'Volume': (before['Volume'].values[0] + after['Volume'].values[0]) / 2
                }
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                print(f"Interpolated: {miss_date} in {file_name}")

        df.sort_values("Date", inplace=True)
        df.to_csv(path, index=False)

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

def volNorm(data_dir):
    data_dir = Path(data_dir)

    for file in data_dir.glob("*.csv"):
        df = pd.read_csv(file)
        df['Date'] = pd.to_datetime(df['Date'])
        df.sort_values("Date", ascending=True, inplace=True)

        # Volume Change
        # FIX: Explicitly set fill_method to None to silence the warning
        df['VolChange'] = df['Volume'].pct_change(fill_method=None)
        
        # Log Change: Protect against 0 and Negatives
        safe_vol = df['Volume'].where(df['Volume'] > 0, np.nan)
        vol_ratio = safe_vol / safe_vol.shift(1)
        df['LogVolChange'] = np.log(vol_ratio.where(vol_ratio > 0, np.nan))

        df.to_csv(file, index=False)
        print(f"Volume normalized for {file.name}")

def addVolatility(data_dir):
    """
    Calculates Garman-Klass, Rogers-Satchell (Split & Total), and Yang-Zhang volatilities.
    Args:
        data_dir: Path to directory containing CSVs.
        window: Rolling window size for Yang-Zhang (default 30 days).
    """
    data_dir = Path(data_dir)

    for file in data_dir.glob("*.csv"):
        df = pd.read_csv(file)
        
        # 1. Standardize and Sort
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df.sort_values("Date", ascending=True, inplace=True)

        # 2. Log Prices (Base for all calculations)
        # We replace <=0 with NaN to avoid -inf errors in logs
        o = df['Open'].where(df['Open'] > 0, np.nan)
        h = df['High'].where(df['High'] > 0, np.nan)
        l = df['Low'].where(df['Low'] > 0, np.nan)
        c = df['Close'].where(df['Close'] > 0, np.nan)

        log_o = np.log(o)
        log_h = np.log(h)
        log_l = np.log(l)
        log_c = np.log(c)
        
        # -------------------------------------------------------
        # A. GARMAN-KLASS (Standard)
        # -------------------------------------------------------
        # 0.5 * (ln(H/L)^2) - (2ln2 - 1) * (ln(C/O)^2)
        gk_term1 = 0.5 * ((log_h - log_l) ** 2)
        gk_term2 = (2 * np.log(2) - 1) * ((log_c - log_o) ** 2)
        
        df['GarmanKlass'] = np.sqrt((gk_term1 - gk_term2).clip(lower=0))
        
        # Log-Diff (Stationary metric for Granger)
        df['Delta_LogGK'] = np.log(df['GarmanKlass'].replace(0, np.nan)).diff()

        # -------------------------------------------------------
        # B. ROGERS-SATCHELL (Decomposed & Total)
        # -------------------------------------------------------
        # RS uses Daily estimators, so no rolling window needed for the base calculation.
        
        # 1. Upside Variance Term: (High - Close) * (High - Open)
        # Measures turbulence contributed by upward extensions
        rs_upside_var = (log_h - log_c) * (log_h - log_o)
        
        # 2. Downside Variance Term: (Low - Close) * (Low - Open)
        # Measures turbulence contributed by downward extensions
        rs_downside_var = (log_l - log_c) * (log_l - log_o)
        
        # 3. Total RS Variance (Standard Textbook Formula)
        rs_total_var = rs_upside_var + rs_downside_var

        # 4. Save Metrics
        # Note: We take sqrt to convert Variance -> Volatility
        df['RS'] = np.sqrt(rs_total_var.clip(lower=0)).diff()
        
        # Split Volatilities (Drift Independent Upside/Downside)
        # We use .diff() here to make them stationary for your Granger test immediately
        df['Upside_Vol'] = np.sqrt(rs_upside_var.clip(lower=0)).diff()
        df['Downside_Vol'] = np.sqrt(rs_downside_var.clip(lower=0)).diff()

        
        df.to_csv(file, index=False)

def addRollingMetrics(data_dir):
    data_dir = Path(data_dir)
    windows = {'Weekly': 7, 'Monthly': 30}
    
    for file in data_dir.glob("*.csv"):
        df = pd.read_csv(file)
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df.sort_values('Date', ascending=True, inplace=True)
        
        for suffix, window in windows.items():
            # 1. Construct the "Big Candle" (Rolling Window)
            roll_open = df['Open'].shift(window - 1)  # Open of the first day in window
            roll_close = df['Close']                  # Close of the last day in window
            roll_high = df['High'].rolling(window=window).max() # Highest High in window
            roll_low = df['Low'].rolling(window=window).min()   # Lowest Low in window
            roll_vol = df['Volume'].rolling(window=window).sum()

            # 2. Safe Logarithms (Replace <=0 with NaN)
            log_o = np.log(roll_open.where(roll_open > 0, np.nan))
            log_c = np.log(roll_close.where(roll_close > 0, np.nan))
            log_h = np.log(roll_high.where(roll_high > 0, np.nan))
            log_l = np.log(roll_low.where(roll_low > 0, np.nan))
            
            # 3. Rogers-Satchell on the Rolling Candle
            # Upside Term: (High - Close) * (High - Open)
            rs_upside_var = (log_h - log_c) * (log_h - log_o)
            
            # Downside Term: (Low - Close) * (Low - Open)
            rs_downside_var = (log_l - log_c) * (log_l - log_o)
            
            # Total Variance
            rs_total_var = rs_upside_var + rs_downside_var
            
            # 4. Save & Make Stationary (Diff)
            # We take Sqrt to get Volatility units, then Diff for stationarity
            
            # Split Metrics
            df[f'Upside_Vol_{suffix}'] = np.sqrt(rs_upside_var.clip(lower=0)).diff()
            df[f'Downside_Vol_{suffix}'] = np.sqrt(rs_downside_var.clip(lower=0)).diff()
            
            # Total Rogers-Satchell (Full Metric)
            df[f'RS_{suffix}'] = np.sqrt(rs_total_var.clip(lower=0)).diff()

            # -----------------------------------------------------------
            # Other Rolling Metrics
            # -----------------------------------------------------------
            
            # Log Returns (Over the full window)
            # Measures "Weekly Return" (Close_t vs Close_{t-7})
            price_ratio = df['Close'] / df['Close'].shift(window)
            df[f'Log Returns_{suffix}'] = np.log(price_ratio.where(price_ratio > 0, np.nan))
            
            # Log Volume Change (Change in Total Volume)
            safe_vol = roll_vol.where(roll_vol > 0, np.nan)
            df[f'LogVolChange_{suffix}'] = np.log(safe_vol / safe_vol.shift(1))
            
            # Rolling Garman-Klass (as a reference/backup)
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

    standardize(data_folder)

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