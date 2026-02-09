import pandas as pd
import numpy as np
import datetime

def download_oni_v3():
    print("Downloading ONI data...")
    url = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"
    try:
        # Parse the fixed-width/whitespace-separated file
        df = pd.read_table(url, sep=r'\s+', engine='python')
    except Exception as e:
        print(f"Error downloading ONI data: {e}")
        return

    # Map Season string to the integer "Center" Month
    season_map = {
        'DJF': 1, 'JFM': 2, 'FMA': 3, 'MAM': 4, 
        'AMJ': 5, 'MJJ': 6, 'JJA': 7, 'JAS': 8, 
        'ASO': 9, 'SON': 10, 'OND': 11, 'NDJ': 12
    }
    
    # 1. Rename 'SEAS' to 'month' directly so pd.to_datetime finds it
    df['month'] = df['SEAS'].map(season_map)
    df = df.rename(columns={'YR': 'year'})
    
    # 2. Create the "Center Date" (e.g., Jan 1st for DJF)
    # pd.to_datetime automatically looks for 'year', 'month', 'day' columns
    df['center_date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))
    
    # 3. APPLY PUBLICATION LAG (The Leakage Fix)
    # The 'DJF' value (centered on Jan) is calculated after Feb ends.
    # It is released by NOAA in early March.
    # We add 2 months to the center date to simulate this release delay.
    # Jan 1 (Center) -> March 1 (Available to public)
    df['date_on'] = df['center_date'] + pd.DateOffset(months=2)
    
    # Clean up columns
    oni = df[['date_on', 'ANOM']].rename(columns={'ANOM': 'climate_risk_ONI_index'})
    
    # Ensure sorted by date so ffill works forward in time
    oni = oni.sort_values('date_on')

    # 4. PREVENT INTERPOLATION LEAKAGE
    # Use Forward Fill (ffill) only. 
    # Logic: On Jan 15, we don't know the Feb 1 value yet. 
    # We only know the last value published (likely from Nov/Dec).
    # Linear interpolation would "cheat" by drawing a line to the future Feb 1 value.
    oni = oni.set_index('date_on').resample('D').ffill().reset_index()
    
    # Calculate Momentum (safe now because the index is lagged correctly)
    oni['climate_risk_ONI_momentum'] = oni['climate_risk_ONI_index'].diff(60)
    
    # Handle NaNs at the very start of history
    oni = oni.fillna(0)
    
    # Save
    oni.to_csv('external_oni.csv', index=False)
    print("ONI V3 Created (Leakage Fixed & Syntax Corrected).")

if __name__ == "__main__":
    download_oni_v3()