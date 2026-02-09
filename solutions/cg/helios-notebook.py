import pandas as pd
import numpy as np
import os
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

pd.options.display.max_columns = 100 
print("✅ Libraries loaded")

DATA_PATH   = './'
OUTPUT_PATH = './'

df = pd.read_csv(f'{DATA_PATH}corn_climate_risk_futures_daily_master.csv')
df['date_on'] = pd.to_datetime(df['date_on'])
market_share_df = pd.read_csv(f'{DATA_PATH}corn_regional_market_share.csv')

print(f"📊 Dataset: {df.shape[0]:,} rows")

RISK_CATEGORIES = ['heat_stress', 'unseasonably_cold', 'excess_precip', 'drought']

merged_daily_df = df.copy()
merged_daily_df['day_of_year'] = merged_daily_df['date_on'].dt.dayofyear
merged_daily_df['quarter'] = merged_daily_df['date_on'].dt.quarter

merged_daily_df = merged_daily_df.merge(
    market_share_df[['region_id', 'percent_country_production']], 
    on='region_id', how='left'
)
merged_daily_df['percent_country_production'] = merged_daily_df['percent_country_production'].fillna(1.0)

# Track which features we create
CREATED_FEATURES = []

print("✅ Base setup")

for cat in RISK_CATEGORIES:
    cols = [x for x in df.columns if cat in x] 
    df.loc[df.harvest_period=="Off-season",cols[0]] = df[cols].sum(axis=1)
    df.loc[df.harvest_period=="Off-season",cols[1:]] = 0

# Risk scores
for risk_type in RISK_CATEGORIES:
    low_col = f'climate_risk_cnt_locations_{risk_type}_risk_low'
    med_col = f'climate_risk_cnt_locations_{risk_type}_risk_medium' 
    high_col = f'climate_risk_cnt_locations_{risk_type}_risk_high'
    
    total = merged_daily_df[low_col] + merged_daily_df[med_col] + merged_daily_df[high_col]
    risk_score = (merged_daily_df[med_col] + 2 * merged_daily_df[high_col]) / (total + 1e-6)
    weighted = risk_score * (merged_daily_df['percent_country_production'] / 100)
    
    merged_daily_df[f'climate_risk_{risk_type}_score'] = risk_score
    merged_daily_df[f'climate_risk_{risk_type}_weighted'] = weighted
    CREATED_FEATURES.extend([f'climate_risk_{risk_type}_score', f'climate_risk_{risk_type}_weighted'])

print(f"✅ Risk scores: {len(CREATED_FEATURES)} features")

# Composite indices
score_cols = [f'climate_risk_{r}_score' for r in RISK_CATEGORIES]

merged_daily_df['climate_risk_temperature_stress'] = merged_daily_df[[f'climate_risk_{r}_score' for r in ['heat_stress', 'unseasonably_cold']]].max(axis=1)
merged_daily_df['climate_risk_precipitation_stress'] = merged_daily_df[[f'climate_risk_{r}_score' for r in ['excess_precip', 'drought']]].max(axis=1)
merged_daily_df['climate_risk_overall_stress'] = merged_daily_df[score_cols].max(axis=1)
merged_daily_df['climate_risk_combined_stress'] = merged_daily_df[score_cols].mean(axis=1)

CREATED_FEATURES.extend(['climate_risk_temperature_stress', 'climate_risk_precipitation_stress',
                         'climate_risk_overall_stress', 'climate_risk_combined_stress'])

print(f"✅ Composites: {len(CREATED_FEATURES)} total features")

# Rolling features
merged_daily_df = merged_daily_df.sort_values(['region_id', 'date_on'])

for window in [7, 14, 30]:
    for risk_type in RISK_CATEGORIES:
        score_col = f'climate_risk_{risk_type}_score'
        
        ma_col = f'climate_risk_{risk_type}_ma_{window}d'
        max_col = f'climate_risk_{risk_type}_max_{window}d'
        
        merged_daily_df[ma_col] = (
            merged_daily_df.groupby('region_id')[score_col]
            .rolling(window=window, min_periods=1).mean()
            .reset_index(level=0, drop=True)
        )
        merged_daily_df[max_col] = (
            merged_daily_df.groupby('region_id')[score_col]
            .rolling(window=window, min_periods=1).max()
            .reset_index(level=0, drop=True)
        )
        CREATED_FEATURES.extend([ma_col, max_col])

print(f"✅ Rolling: {len(CREATED_FEATURES)} total features")

# Momentum features (create NaN - determines valid rows)
for risk_type in RISK_CATEGORIES:
    score_col = f'climate_risk_{risk_type}_score'
    
    c1 = f'climate_risk_{risk_type}_change_1d'
    c7 = f'climate_risk_{risk_type}_change_7d'
    acc = f'climate_risk_{risk_type}_acceleration'
    
    merged_daily_df[c1] = merged_daily_df.groupby('region_id')[score_col].diff(1)
    merged_daily_df[c7] = merged_daily_df.groupby('region_id')[score_col].diff(7)
    merged_daily_df[acc] = merged_daily_df.groupby('region_id')[c1].diff(1)
    
    CREATED_FEATURES.extend([c1, c7, acc])

print(f"✅ Momentum: {len(CREATED_FEATURES)} total features")

# Country aggregations
for risk_type in RISK_CATEGORIES:
    score_col = f'climate_risk_{risk_type}_score'
    weighted_col = f'climate_risk_{risk_type}_weighted'
    
    country_agg = merged_daily_df.groupby(['country_name', 'date_on']).agg({
        score_col: ['mean', 'max', 'std'],
        weighted_col: 'sum',
        'percent_country_production': 'sum'
    }).round(4)
    
    country_agg.columns = [f'country_{risk_type}_{"_".join(col).strip()}' for col in country_agg.columns]
    country_agg = country_agg.reset_index()
    
    new_cols = [c for c in country_agg.columns if c not in ['country_name', 'date_on']]
    CREATED_FEATURES.extend(new_cols)
    
    merged_daily_df = merged_daily_df.merge(country_agg, on=['country_name', 'date_on'], how='left')

print(f"✅ Country aggs: {len(CREATED_FEATURES)} total features")

# Get valid rows
print(f"\n📊 Before dropna: {len(merged_daily_df):,}")
baseline_df = merged_daily_df.copy()
print(f"📊 After dropna: {len(baseline_df):,} (expected: 219,161)")

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
    
    return {
        'cfcs': round(cfcs, 2),
        'avg_sig_corr': round(avg_sig, 4),
        'max_corr': round(max_corr, 4),
        'sig_count': len(sig_corrs),
        'total': len(correlations),
        'sig_pct': round(sig_pct, 4)
    }

# merged_daily_df exists to ensure same no of rows and all created features up
# to this point are dropped
submission = baseline_df.drop(CREATED_FEATURES, axis=1)

# percentage of global production of each country that is in the date
prod_dict = {"Argentina":4,"Brazil":11,"Canada":1,"China":24,"European Union":5,"India":3,"Mexico":2,"Paraguay":0.5,"Russia":0.5,"Ukraine":2,"United States":31,"South Africa":1}

submission["percent_world_production"]=submission.country_name.apply(prod_dict.get)

# new DataFrame to combine and store each risk into a single value for each date
submissionm = submission.copy()

for x in ["heat","cold","precip","drought"]:
    t=[y for y in submission.columns if x in y]
    for y in t:
        submissionm[y+"m"] = submissionm[y]/submissionm[t].sum(axis=1)
    submissionn = submissionm.drop(t, axis=1)

    for y in t:
        submissionn[y+"m"]=submissionm[y+"m"]*submissionm.percent_country_production/100
        u=submissionm.groupby(["country_name","date_on"])[y+"m"].sum()
        submissionm[y+"m"]=submissionm.apply(lambda x:u.get((x.country_name,x.date_on)),axis=1)

    for y in t:
        submissionm[y+"m"]=submissionm[y+"m"]*submissionm.percent_world_production/100
        u=submissionm.drop_duplicates(["country_name","date_on"]).groupby(["date_on"])[y+"m"].sum()
        submissionm[y+"m"]=submissionm.apply(lambda x:u.get(x.date_on),axis=1)

submissionm=submissionm.drop_duplicates("date_on").sort_values("date_on").set_index("date_on").iloc[:,-12:].reset_index()

cr = [x for x in submissionm.columns if x.startswith("climate_risk")]

for y in cr:
    for window in [7, 14, 28,63,91,119,182,364]:
        submissionm[y+"a"+str(window)] = submissionm[y].rolling(window=window, min_periods=1).mean()
        submissionm[y+"b"+str(window)] = submissionm[y].rolling(window=window, min_periods=1).max()
for y in range(0,12,3):
    submissionm[cr[y]+"ma"] = submissionm[cr[y+1]]+2*submissionm[cr[y+2]]
    for window in [7, 14, 28,63,91,119,182,364]:
        submissionm[cr[y]+"ma"+str(window)] = submissionm[cr[y]+"ma"].rolling(window=window, min_periods=1).mean()
        submissionm[cr[y]+"mb"+str(window)] = submissionm[cr[y]+"ma"].rolling(window=window, min_periods=1).max()

# dropping original climate risk features to merge new created features
submission = submission.drop([x for x in submission.columns if x.startswith("climate_risk_")],axis=1)

submission = submission.merge(submissionm,on="date_on")

submission=submission[~merged_daily_df.isna().any(axis=1)]

# Finding the features which have maximum correlations with each futures column
fr = [x for x in submission.columns if x.startswith("future")]
cr = [x for x in submission.columns if x.startswith("climate_risk_")]
scores = {}
for country in submission.country_name.unique():
    a=submission[submission.country_name==country]
    for month in a.date_on_month.unique():
        t = a[(a.date_on_month==month)]
        for x in cr:
            if x not in scores.keys():
                scores[x] = []
            for y in fr:
                if t[x].std()>0 and t[y].std()>0:
                    scores[x].append(t[[x, y]].corr().iloc[0,1])
fs=[]
for x in scores.keys():
    t = pd.Series(scores[x])
    fs.append((x, sum(abs(t)>=0.5)/len(t)))
    
feats = [x[0] for x in sorted(fs, key=lambda x:x[1])[::-1][30:]]
submission = submission.drop(feats,axis=1)
compute_cfcs(submission)

# Save
output_file = f'{OUTPUT_PATH}submission.csv'
submission.to_csv(output_file, index=False)

print(f"\n📁 Saved: {output_file}")
#print(f"   Version: {best_name}")
#print(f"   CFCS: {best_score}")
print(f"   Rows: {len(submission):,}")
print(f"   Climate features: {len([c for c in submission.columns if c.startswith('climate_risk_')])}")