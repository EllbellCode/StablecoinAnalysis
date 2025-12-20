import pandas as pd

DIR = 'Results/GrangerCopula/'
# 1. Load the three datasets
# Using the naming convention you provided
try:
    df_day = pd.read_csv(DIR + 'GC_Results_StableCrypto_Day.csv')
    df_week = pd.read_csv(DIR + 'GC_Results_StableCrypto_Week.csv')
    df_month = pd.read_csv(DIR + 'GC_Results_StableCrypto_Month.csv')
except FileNotFoundError as e:
    print(f"Error loading files: {e}")
    print("Please make sure the files are in the same directory as this script and end in .csv")
    exit()

# 2. Function to clean and prepare each dataframe
def prepare_data(df, column_name):
    # Select only the columns we need: Source, Target, and p-value
    # We drop 'InputScope' and others as they aren't in your requested output
    df_clean = df[['Source', 'Target', 'p-value']].copy()
    
    # Rename 'p-value' to the specific significance name requested
    df_clean = df_clean.rename(columns={'p-value': column_name})
    
    return df_clean

# 3. Process the data
# Extract the relevant columns and rename them
day_data = prepare_data(df_day, 'day significance')
week_data = prepare_data(df_week, 'week significance')
month_data = prepare_data(df_month, 'month significance')

# 4. Merge the dataframes
# We merge on 'Source' and 'Target'
combined_df = day_data.merge(week_data, on=['Source', 'Target'], how='outer')
combined_df = combined_df.merge(month_data, on=['Source', 'Target'], how='outer')

combined_df.sort_values(by=['Target', 'Source'], inplace=True)

# 5. Save the result
output_filename = 'GC_Results_Combined.csv'
combined_df.to_csv(DIR + output_filename, index=False)

print(f"Success! Combined data saved to '{output_filename}'")
print(combined_df)