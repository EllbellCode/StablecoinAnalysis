"""
Script for aggregating all Copula Granger results into a single table
"""


import pandas as pd

DIR = 'Results/GrangerCopula/'
#Load the datasets
try:
    # df_day = pd.read_csv(DIR + 'GC_Results_StableCrypto_Day.csv')
    # df_week = pd.read_csv(DIR + 'GC_Results_StableCrypto_Week.csv')
    # df_month = pd.read_csv(DIR + 'GC_Results_StableCrypto_Month.csv')
    df_day = pd.read_csv(DIR + 'GC_Results_CryptoStable_Day.csv')
    df_week = pd.read_csv(DIR + 'GC_Results_CryptoStable_Week.csv')
    df_month = pd.read_csv(DIR + 'GC_Results_CryptoStable_Month.csv')
except FileNotFoundError as e:
    print(f"Error loading files: {e}")
    print("Please make sure the files are in the same directory as this script and end in .csv")
    exit()

#Select and rename columns
def prepare_data(df, column_name):
    
    df_clean = df[['Source', 'Target', 'p-value']].copy()
    df_clean = df_clean.rename(columns={'p-value': column_name})
    
    return df_clean

#process data
day_data = prepare_data(df_day, 'day significance')
week_data = prepare_data(df_week, 'week significance')
month_data = prepare_data(df_month, 'month significance')

#merge
combined_df = day_data.merge(week_data, on=['Source', 'Target'], how='outer')
combined_df = combined_df.merge(month_data, on=['Source', 'Target'], how='outer')
combined_df.sort_values(by=['Target', 'Source'], inplace=True)

#save
output_filename = 'GC_Results_Combined.csv'
combined_df.to_csv(DIR + output_filename, index=False)

print(f"Success! Combined data saved to '{output_filename}'")
print(combined_df)