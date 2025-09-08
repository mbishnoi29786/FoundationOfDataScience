# T1. Read both datasets above into two DataFrames and print their respective dimensions (number of rows and columns).
import pandas as pd
# Read the datasets
df1 = pd.read_csv('./climate_precip.csv')
df2 = pd.read_csv('./climate_temp.csv')
# Print the dimensions of the DataFrames
print("Dimensions of climate_precip DataFrame:", df1.shape)
print("Dimensions of climate_temp DataFrame:", df2.shape)

# T2. Print the list of keys (i.e. column names) of each DataFrame using the keys() function.
# print("Keys of climate_precip DataFrame:", df1.keys())
# print("Keys of climate_temp DataFrame:", df2.keys())


# T3. Use the merge() function to obtain a new DataFrame that contains the precipitation and temperature records of station GHCND:USC00045721 only. How many rows and columns do you observe in this new DataFrame?
merged_df = pd.merge(df1, df2, on=['STATION', 'DATE'])
merged_df = merged_df[merged_df['STATION'] == 'GHCND:USC00045721']
print("Dimensions of merged DataFrame for station GHCND:USC00045721:", merged_df.shape)

# T4. Repeat task 3, but specifies explicitly that the join must be performed on the STATION and DATE column only. What difference do you observe? Why?
merged_df_explicit = pd.merge(df1, df2, on=['STATION', 'DATE'], how='inner')
merged_df_explicit = merged_df_explicit[merged_df_explicit['STATION'] == 'GHCND:USC00045721']
print("Dimensions of explicitly merged DataFrame for station GHCND:USC00045721:", merged_df_explicit.shape)

