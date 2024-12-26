# First cell - imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Second cell - load data
data = pd.read_csv('raw_data.csv', index_col=0)
data.index = pd.to_datetime(data.index)
print("Data shape:", data.shape)
data.head()

# Third cell - handle missing values
data = data.fillna(method='ffill')
print("Missing values after forward fill:")
print(data.isnull().sum())

# Fourth cell - calculate YOY changes
for column in data.columns:
    if column != 'CSUSHPISA':  # Keep target variable as is
        data[f'{column}_YOY'] = data[column].pct_change(periods=12) * 100

# Remove rows with NaN values
data = data.dropna()
print("Final data shape:", data.shape)

# Fifth cell - save processed data
data.to_csv('processed_data.csv')
print("Data preprocessing complete!")
data.head()