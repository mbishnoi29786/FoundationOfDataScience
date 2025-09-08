# T5. Use .join() function instead of merge() to produce the same result as T4 above.
import pandas as pd
# Read the datasets
df1 = pd.read_csv('./climate_precip.csv')
df2 = pd.read_csv('./climate_temp.csv')
# Set the index to STATION and DATE for both DataFrames
df1.set_index(['STATION', 'DATE'], inplace=True)
df2.set_index(['STATION', 'DATE'], inplace=True)
# Join the DataFrames using .join() method
joined_df = df1.join(df2, how='inner', lsuffix='_precip', rsuffix='_temp')
# Filter for the specific station
joined_df = joined_df.loc['GHCND:USC00045721']
print("Dimensions of joined DataFrame for station GHCND:USC00045721:", joined_df.shape)
