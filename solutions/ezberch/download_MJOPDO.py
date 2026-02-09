import pandas as pd
import requests
import io
import datetime

def download_external_data():
    headers = {'User-Agent': 'Mozilla/5.0'}
    print("Downloading MJO and PDO data from stable mirrors...")

    # --- 1. MJO (Madden-Julian Oscillation) ---
    # Source: Australian Bureau of Meteorology (The primary stable mirror for RMM)
    mjo_url = "http://www.bom.gov.au/climate/mjo/graphics/rmm.74toRealtime.txt"
    try:
        response = requests.get(mjo_url, headers=headers)
        response.raise_for_status()
        
        # BOM file has a 2-line header. 
        # Columns: year, month, day, RMM1, RMM2, phase, amplitude, source
        mjo_df = pd.read_csv(io.StringIO(response.text), sep=r'\s+', skiprows=2, header=None,
                             names=['year', 'month', 'day', 'RMM1', 'RMM2', 'phase', 'amplitude', 'source'])
        
        mjo_df['date_on'] = pd.to_datetime(mjo_df[['year', 'month', 'day']])
        
        # PUBLICATION LAG: RMM is calculated daily with a ~1 day delay.
        # Adding 2 days ensures zero data leakage.
        mjo_df['date_on'] = mjo_df['date_on'] + pd.DateOffset(days=2)
        
        mjo_out = mjo_df[['date_on', 'phase', 'amplitude']].copy()
        print("✅ MJO data downloaded successfully (BOM mirror).")
    except Exception as e:
        print(f"❌ Error downloading MJO: {e}")
        mjo_out = pd.DataFrame()

    # --- 2. PDO (Pacific Decadal Oscillation) ---
    # Source: NOAA Physical Sciences Laboratory (More stable than the NCEI link)
    pdo_url = "https://psl.noaa.gov/data/correlation/pdo.data"
    try:
        response = requests.get(pdo_url, headers=headers)
        response.raise_for_status()
        
        # PSL .data files have 1 line of header, then rows: Year Jan Feb ... Dec
        lines = response.text.split('\n')
        data_lines = []
        for line in lines[1:]: # Skip the "PDO" header line
            parts = line.split()
            if len(parts) == 13: # Only take rows with Year + 12 months
                data_lines.append(parts)
            if "99.9" in line or "-99.9" in line: # Stop at footer
                break
                
        pdo_raw = pd.DataFrame(data_lines, dtype=float)
        pdo_raw.columns = ['year'] + [str(i) for i in range(1, 13)]
        
        # Melt to long format
        pdo_long = pdo_raw.melt(id_vars='year', var_name='month', value_name='pdo_index')
        pdo_long['date_on'] = pd.to_datetime(pdo_long[['year', 'month']].assign(day=1))
        
        # PUBLICATION LAG: Monthly PDO is released mid-next-month.
        # Adding 45 days to the month-start prevents leakage.
        pdo_long['date_on'] = pdo_long['date_on'] + pd.DateOffset(days=45)
        
        pdo_out = pdo_long[['date_on', 'pdo_index']].sort_values('date_on')
        # Filter out future/placeholder dates (often 99.9 in these files)
        pdo_out = pdo_out[pdo_out['pdo_index'] < 50] 
        print("✅ PDO data downloaded successfully (NOAA PSL archive).")
    except Exception as e:
        print(f"❌ Error downloading PDO: {e}")
        pdo_out = pd.DataFrame()

    # --- 3. MERGE & SAVE ---
    if not mjo_out.empty and not pdo_out.empty:
        combined = pd.merge(mjo_out, pdo_out, on='date_on', how='outer').sort_values('date_on')
        
        # Forward fill ensures daily data availability for the main competition script
        combined = combined.set_index('date_on').resample('D').ffill().reset_index()
        combined = combined.fillna(0)
        
        # Cut off any dates beyond today (to prevent forecasting issues)
        today = pd.Timestamp(datetime.date.today())
        combined = combined[combined['date_on'] <= today]
        
        combined.to_csv('external_indices.csv', index=False)
        print(f"📊 Final file 'external_indices.csv' saved with {len(combined)} rows.")

if __name__ == "__main__":
    download_external_data()