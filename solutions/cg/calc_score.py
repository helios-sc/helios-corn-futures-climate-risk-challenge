import pandas as pd
import numpy as np

def compute_cfcs(df):
    """Compute CFCS score for a dataframe."""
    climate_cols = [c for c in df.columns if c.startswith("climate_risk_")]
    futures_cols = [c for c in df.columns if c.startswith("futures_")]
    
    correlations = []
    
    for country in df['country_name'].unique():
        df_country = df[df['country_name'] == country]
        
        for month in df_country['date_on_month'].unique():
            df_month = df_country[df_country['date_on_month'] == month]
            
            for clim in climate_cols:
                for fut in futures_cols:
                    if df_month[clim].std() > 0 and df_month[fut].std() > 0:
                        corr = df_month[[clim, fut]].corr().iloc[0, 1]
                        correlations.append(corr)
    
    correlations = pd.Series(correlations).dropna()
    abs_corrs = correlations.abs()
    sig_corrs = abs_corrs[abs_corrs >= 0.5]
    
    avg_sig = sig_corrs.mean() if len(sig_corrs) > 0 else 0
    max_corr = abs_corrs.max()
    sig_pct = len(sig_corrs) / len(correlations) * 100 if len(correlations) > 0 else 0
    
    avg_sig_score = min(100, avg_sig * 100)
    max_score = min(100, max_corr * 100)
    
    cfcs = (0.5 * avg_sig_score) + (0.3 * max_score) + (0.2 * sig_pct)
    
    print(f"CFCS: {cfcs:.4f}")
    return cfcs

df = pd.read_csv('submission.csv')
# Must ensure 'date_on' is datetime and 'date_on_month' exists if used
if 'date_on' in df.columns:
    df['date_on'] = pd.to_datetime(df['date_on'])
    df['date_on_month'] = df['date_on'].dt.month

compute_cfcs(df)
