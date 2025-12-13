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

        # Convert date
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors='coerce')

        # Clean numeric columns
        for col in ['Close', 'Open', 'High', 'Low']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace(',', '', regex=False).astype(float)

        # Handle Volume
        if 'Vol.' in df.columns:
            df['Volume'] = df['Volume'].apply(convertVol)
            df['Volume'] = df['Volume'].fillna(999_999_999_999).round()

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

        if 'Close' in df.columns:
            
            #Backwards order as we sort the date at the end!
            df['Returns'] = df['Close'] / df['Close'].shift(-1) - 1
            df['Log Returns'] = np.log(df['Close'] / df['Close'].shift(-1))
            df.to_csv(file, index=False)

def volNorm(data_dir):

    data_dir = Path(data_dir)

    for file in data_dir.glob("*.csv"):

        df = pd.read_csv(file)

        df['VolChange'] = df['Volume'] / df['Volume'].shift(-1) - 1
        df['LogVolChange'] = np.log(df['Volume'] / df['Volume'].shift(-1))

        df = df.sort_values("Date", ascending=True)
        df.to_csv(file, index=False)

def addVolatility(data_dir):
    data_dir = Path(data_dir)

    for file in data_dir.glob("*.csv"):
        df = pd.read_csv(file)

        # 1. Calculate Garman-Klass Volatility
        # Note: We use clip(lower=0) inside sqrt to handle potential tiny floating point errors returning negative zeros
        term1 = 0.5 * (np.log(df['High'] / df['Low']) ** 2)
        term2 = (2 * np.log(2) - 1) * (np.log(df['Close'] / df['Open']) ** 2)
        
        df['GarmanKlass'] = np.sqrt((term1 - term2).clip(lower=0))

        # 2. Safe Log Calculation
        # Replace 0 with NaN (or a small epsilon like 1e-9) before logging to avoid -inf
        df['LogGarmanKlass'] = np.log(df['GarmanKlass'].replace(0, np.nan))
        df['Delta_LogGK'] = df['LogGarmanKlass'].diff()


        df = df.sort_values("Date", ascending=True)
        df.to_csv(file, index=False)

def addRollingMetrics(data_dir):
    """
    Calculates rolling Weekly (7-day) and Monthly (30-day) candle metrics.
    Adds: Log Returns, Log Volume Change, and Delta Log Garman-Klass for these rolling windows.
    """
    data_dir = Path(data_dir)
    
    # Define windows: Name -> Days
    windows = {
        'Weekly': 7,
        'Monthly': 30
    }
    
    for file in data_dir.glob("*.csv"):
        df = pd.read_csv(file)
        
        # Ensure data is sorted by Date (Ascending) for rolling calculations to look BACKWARDS correctly
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df.sort_values('Date', ascending=True, inplace=True)
        
        for suffix, window in windows.items():
            # --- 1. Construct the Rolling Candle ---
            # Open: The Open price of the start of the window (window-1 days ago)
            roll_open = df['Open'].shift(window - 1)
            # Close: Today's Close
            roll_close = df['Close']
            # High: Max High over the window
            roll_high = df['High'].rolling(window=window).max()
            # Low: Min Low over the window
            roll_low = df['Low'].rolling(window=window).min()
            # Volume: Sum of Volume over the window
            roll_vol = df['Volume'].rolling(window=window).sum()
            
            # --- 2. Calculate Metrics on the Rolling Candle ---
            
            # A. Log Returns (Candle Return)
            # ln(Close / Open) for the aggregated period
            df[f'Log Returns_{suffix}'] = np.log(roll_close / roll_open)
            
            # B. Log Volume Change
            # Change in the rolling volume sum: ln(CurrentRollingVol / PrevRollingVol) -> ln(Vol).diff()
            df[f'LogVolChange_{suffix}'] = np.log(roll_vol.replace(0, np.nan)).diff()
            
            # C. Delta Log Garman-Klass
            # Calculate GK on the rolling High/Low/Open/Close
            term1 = 0.5 * (np.log(roll_high / roll_low) ** 2)
            term2 = (2 * np.log(2) - 1) * (np.log(roll_close / roll_open) ** 2)
            gk_vol = np.sqrt((term1 - term2).clip(lower=0))
            
            # Log and Diff for stationarity
            df[f'Delta_LogGK_{suffix}'] = np.log(gk_vol.replace(0, np.nan)).diff()
            
        df.to_csv(file, index=False)

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
    
    # NEW STEP: Add Rolling Weekly/Monthly Metrics
    addRollingMetrics(verif_folder)

if __name__ == "__main__":
    main()