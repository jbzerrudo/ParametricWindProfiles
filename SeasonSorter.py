import pandas as pd

# 1. Load the CSV file (low_memory=False resolves the DtypeWarning)
file_path = r'D:\2026\ParametricWindModel\ibtracs.WP.list.v04r01.csv'
df = pd.read_csv(file_path, low_memory=False, skiprows=[1])

# 2. Convert SEASON to numeric 
# (errors='coerce' ignores the weird units row at the top of IBTrACS)
df['SEASON'] = pd.to_numeric(df['SEASON'], errors='coerce')

# 3. Filter using standard numbers instead of dates
mask = (df['SEASON'] >= 1977) & (df['SEASON'] <= 2024)

# 4. Apply the filter
filtered_df = df[mask]

keep_cols = [
    'SID', 'SEASON', 'NUMBER', 'BASIN', 'SUBBASIN', 'NAME',
    'ISO_TIME', 'NATURE', 'LAT', 'LON',
    'WMO_WIND', 'WMO_PRES',
    'USA_WIND', 'USA_PRES', 'USA_SSHS',
    'USA_RMW',
    'USA_R34_NE', 'USA_R34_SE', 'USA_R34_SW', 'USA_R34_NW',
    'USA_R50_NE', 'USA_R50_SE', 'USA_R50_SW', 'USA_R50_NW',
    'USA_R64_NE', 'USA_R64_SE', 'USA_R64_SW', 'USA_R64_NW'
]

filtered_df = filtered_df[[c for c in keep_cols if c in filtered_df.columns]]

# 5. Save the filtered data to a new CSV
filtered_df.to_csv(r'D:\2026\ParametricWindModel\filtered_ibtracs_v2.csv', index=False)

print(f"Original row count: {len(df)}")
print(f"Filtered row count: {len(filtered_df)}")
