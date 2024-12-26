import pandas as pd
from fredapi import Fred
import os

# Replace with your actual FRED API key
FRED_API_KEY = "YOUR_API_KEY_HERE"

def collect_data():
    # Initialize FRED API
    fred = Fred(api_key=4cd411f227ee2f98ee4dd66d95959067)
    
    # Dictionary of series we want to collect
    series_ids = {
        'CSUSHPISA': 'Home Price Index',
        'MORTGAGE30US': 'Mortgage Rate',
        'MSACSR': 'Housing Supply',
        'PERMIT': 'Building Permits',
        'MEHOINUSA672N': 'Median Household Income',
        'UNRATE': 'Unemployment Rate',
        'CPIAUCSL': 'Consumer Price Index',
        'GDP': 'GDP'
    }
    
    # Create empty DataFrame to store our data
    data = pd.DataFrame()
    
    # Collect each series
    for series_id, series_name in series_ids.items():
        print(f"Collecting {series_name}...")
        series = fred.get_series(series_id)
        data[series_id] = series
    
    # Save the raw data
    data.to_csv('raw_data.csv')
    return data

if __name__ == "__main__":
    data = collect_data()
    print("Data collection complete!")
    