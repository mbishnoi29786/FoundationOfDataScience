# T6. Use concat() to double the amount of precipitation data. Save the output as double_climate.precip.csv
import pandas as pd
# Read the datasets
df1 = pd.read_csv('./climate_precip.csv')
# Concatenate the DataFrame with itself to double the amount of precipitation data
double_climate_precip = pd.concat([df1, df1], ignore_index=True)
# Save the output to a new CSV file
double_climate_precip.to_csv('./double_climate.precip.csv', index=False)
# Print the dimensions of the joined DataFrame
print("Dimensions of double climate_precip DataFrame:", double_climate_precip.shape)
print("Dimensions of original climate_precip DataFrame:", df1.shape)

# T7. Use concat() to concatenate all precipitation and temperature datasets along their columns. Only keep rows that have values in both datasets. Save the output as climate_precip_temp.csv
# Read the datasets
df2 = pd.read_csv('./climate_temp.csv')
# Concatenate the DataFrames along columns, keeping only rows with values in both datasets
climate_precip_temp = pd.concat([df1, df2], axis=1, join='inner')
# Save the output to a new CSV file
climate_precip_temp.to_csv('./climate_precip_temp.csv', index=False)
# Print the dimensions of the concatenated DataFrame
print("Dimensions of climate_precip_temp DataFrame:", climate_precip_temp.shape)
print("Dimensions of original climate_temp DataFrame:", df2.shape)
