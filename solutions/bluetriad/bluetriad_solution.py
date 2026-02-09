#!/usr/bin/env python
# coding: utf-8

# # 🌽 Helios Corn Futures Climate Challenge - Feature Engineering + External Data (No Future data at all)
# 
# ---
# 
# ### Building Upon The Notebook from Ergün Tiryaki, also none of the features I add are made using futures or lagged futures, the added data is purely climate related
# 
# This notebook builds upon the [Improved Feature Engineering](https://www.kaggle.com/code/erguntiryaki/improved-feature-engineering) with external climate information and features made from that data. All of the data I have added is publicly available online and can be downloaded from the [Climate Prediction Center](https://www.cpc.ncep.noaa.gov/data/indices/Readme.index.shtml) Website.
# 
# ---
# 
# ### Types of Added Climate Data (All of these might not be included in the final data because of feature selection)
# 
# | Column Name Prefix| Meaning | Description |
# |--------------|-------------|--------------|
# | **CPOLR** | Central Pacific OLR Index | Monthly Central Pacific OLR Index (1991-2020 base period 170°E-140°W,5°S-5°N) |
# | **Romi** | Real-time OLR MJO Index | Projection of 9 day running average OLR anomalies onto the daily spatial EOF patterns of 30-96 day eastward filtered OLR. OLR anomalies are calculated by first subtracting the previous 40 day mean OLR. The running average is tapered as the target date is approached. |
# | **ONI** | Oceanic Niño Index |  The ONI is one measure of the El Niño-Southern Oscillation, and other indices can confirm whether features consistent with a coupled ocean-atmosphere phenomenon accompanied these periods. |
# | **Ninoxx** | Sea Surface Temperatures SST | based on sea surface temperature (SST) anomalies averaged across a given region. |
# | **NAO, AAO and PNA** | North Atlantic Oscillation, Antarctic Oscillation and Pacific-North American | major atmospheric pressure variability patterns that drive climate, weather, and temperature shifts in the Southern and Northern Hemispheres, respectively |
# | **OLR** | Outgoing Long Wave Radiation | Outgoing Longwave Radiation (OLR) is the infrared energy (heat) radiated from the Earth-atmosphere system back into space |
# | **SOI** | Southern Oscillation Index | The Southern Oscillation Index (SOI) measures the large-scale, monthly fluctuation in surface air pressure between Tahiti and Darwin, Australia |
# 
# ---
# There are other columns in the dataset, you can find more information about them on the website.
# 
# ### 💡 Key Strategy: Quality Over Quantity
# 
# **CFCS Formula:**
# ```
# CFCS = (0.5 × Avg_Sig_Corr) + (0.3 × Max_Corr) + (0.2 × Sig_Count%)
# ```
# 
# **Critical Insight:** `Sig_Count% = significant_correlations / total_correlations × 100`
# 
# ⚠️ **Adding weak features HURTS your score** by increasing the denominator without adding significant correlations!
# 
# Adding so many new columns and especially the multiple features generated from them can increase the feature count by a lot so this notebook does feature selection/removal in two steps
# 
# 1. Dropping features with significant correlations less than 400
# 2. Forward Selection with the remaining features( starting with the columns we cannot remove, adding features one at a time and only keeping them if they increase the CFCS score. This takes ~6 hours so I didn't include the forward selection process in this notebook, I have the list of features I got as a result of forward selection and I just drop the remaining columns. 
# 
# ---

# In[1]:


import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

pd.options.display.max_columns = 100
pd.options.display.max_rows = 100

print("✅ Libraries loaded")


# In[2]:


# Configuration
RISK_CATEGORIES = ['heat_stress', 'unseasonably_cold', 'excess_precip', 'drought']
SIGNIFICANCE_THRESHOLD = 0.5

# Data paths
DATA_PATH = '../../data/'
OUTPUT_PATH = '/kaggle/working/'

# Load data
df = pd.read_csv(f'{DATA_PATH}corn_climate_risk_futures_daily_master.csv')
df['date_on'] = pd.to_datetime(df['date_on'])
market_share_df = pd.read_csv(f'{DATA_PATH}corn_regional_market_share.csv')

print(f"📊 Dataset: {len(df):,} rows")
print(f"📅 Date range: {df['date_on'].min()} to {df['date_on'].max()}")
print(f"🌍 Countries: {df['country_name'].nunique()}")
print(f"📍 Regions: {df['region_name'].nunique()}")


# ---
# ## 📊 Helper Functions

# In[3]:


def compute_cfcs(df, verbose=True):
    """
    Compute CFCS score for a dataframe.
    CFCS = (0.5 × Avg_Sig_Corr) + (0.3 × Max_Corr) + (0.2 × Sig_Count%)
    """
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
    sig_corrs = abs_corrs[abs_corrs >= SIGNIFICANCE_THRESHOLD]

    avg_sig = sig_corrs.mean() if len(sig_corrs) > 0 else 0
    max_corr = abs_corrs.max() if len(abs_corrs) > 0 else 0
    sig_pct = len(sig_corrs) / len(correlations) * 100 if len(correlations) > 0 else 0

    avg_sig_score = min(100, avg_sig * 100)
    max_score = min(100, max_corr * 100)

    cfcs = (0.5 * avg_sig_score) + (0.3 * max_score) + (0.2 * sig_pct)

    result = {
        'cfcs': round(cfcs, 2),
        'avg_sig_corr': round(avg_sig, 4),
        'max_corr': round(max_corr, 4),
        'sig_count': len(sig_corrs),
        'total': len(correlations),
        'sig_pct': round(sig_pct, 4),
        'n_features': len(climate_cols)
    }

    if verbose:
        print(f"CFCS: {result['cfcs']} | Sig: {result['sig_count']}/{result['total']} ({result['sig_pct']:.2f}%) | Features: {result['n_features']}")

    return result


def analyze_feature_contributions(df, climate_cols, futures_cols):
    """
    Analyze contribution of each climate feature.
    Returns DataFrame with sig_count, max_corr, etc for each feature.
    """
    feature_stats = {col: {'sig_count': 0, 'total': 0, 'max_corr': 0, 'sig_corrs': []} 
                     for col in climate_cols}

    for country in df['country_name'].unique():
        df_country = df[df['country_name'] == country]

        for month in df_country['date_on_month'].unique():
            df_month = df_country[df_country['date_on_month'] == month]

            for clim in climate_cols:
                for fut in futures_cols:
                    if df_month[clim].std() > 0 and df_month[fut].std() > 0:
                        corr = df_month[[clim, fut]].corr().iloc[0, 1]

                        feature_stats[clim]['total'] += 1

                        if abs(corr) >= SIGNIFICANCE_THRESHOLD:
                            feature_stats[clim]['sig_count'] += 1
                            feature_stats[clim]['sig_corrs'].append(abs(corr))

                        if abs(corr) > feature_stats[clim]['max_corr']:
                            feature_stats[clim]['max_corr'] = abs(corr)

    results = []
    for col, stats in feature_stats.items():
        avg_sig = np.mean(stats['sig_corrs']) if stats['sig_corrs'] else 0
        results.append({
            'feature': col,
            'sig_count': stats['sig_count'],
            'total': stats['total'],
            'sig_pct': stats['sig_count'] / stats['total'] * 100 if stats['total'] > 0 else 0,
            'max_corr': round(stats['max_corr'], 4),
            'avg_sig_corr': round(avg_sig, 4)
        })

    return pd.DataFrame(results).sort_values('sig_count', ascending=False)

print("✅ Helper functions defined")


# ---
# ## 🔧 Phase 1: Base Feature Engineering

# In[4]:


# Create working copy
merged_df = df.copy()

# Add time features
merged_df['day_of_year'] = merged_df['date_on'].dt.dayofyear
merged_df['quarter'] = merged_df['date_on'].dt.quarter

# Merge market share
merged_df = merged_df.merge(
    market_share_df[['region_id', 'percent_country_production']], 
    on='region_id', how='left'
)
merged_df['percent_country_production'] = merged_df['percent_country_production'].fillna(1.0)

# Track all created features
ALL_NEW_FEATURES = []

print("✅ Base setup complete")


# In[5]:


# Base Risk Scores
for risk_type in RISK_CATEGORIES:
    low_col = f'climate_risk_cnt_locations_{risk_type}_risk_low'
    med_col = f'climate_risk_cnt_locations_{risk_type}_risk_medium' 
    high_col = f'climate_risk_cnt_locations_{risk_type}_risk_high'

    total = merged_df[low_col] + merged_df[med_col] + merged_df[high_col]
    risk_score = (merged_df[med_col] + 2 * merged_df[high_col]) / (total + 1e-6)
    weighted = risk_score * (merged_df['percent_country_production'] / 100)

    merged_df[f'climate_risk_{risk_type}_score'] = risk_score
    merged_df[f'climate_risk_{risk_type}_weighted'] = weighted
    ALL_NEW_FEATURES.extend([f'climate_risk_{risk_type}_score', f'climate_risk_{risk_type}_weighted'])

print(f"✅ Base risk scores: {len(ALL_NEW_FEATURES)} features")


# ---
# ## 🔧 Phase 2: Advanced Rolling Features

# In[6]:


# Sort for time series operations
merged_df = merged_df.sort_values(['region_id', 'date_on'])

# Rolling MA and Max (7, 14, 30, 60 days)
for window in [7, 14, 30, 60]:
    for risk_type in RISK_CATEGORIES:
        score_col = f'climate_risk_{risk_type}_score'

        # Moving Average
        ma_col = f'climate_risk_{risk_type}_ma_{window}d'
        merged_df[ma_col] = (
            merged_df.groupby('region_id')[score_col]
            .transform(lambda x: x.rolling(window, min_periods=1).mean())
        )
        ALL_NEW_FEATURES.append(ma_col)

        # Rolling Max
        max_col = f'climate_risk_{risk_type}_max_{window}d'
        merged_df[max_col] = (
            merged_df.groupby('region_id')[score_col]
            .transform(lambda x: x.rolling(window, min_periods=1).max())
        )
        ALL_NEW_FEATURES.append(max_col)

print(f"✅ Rolling features: {len(ALL_NEW_FEATURES)} total")


# ---
# ## 🔧 Phase 3: Lag Features (Weather Affects Prices with Delay)

# In[7]:


# Lag features - weather today affects prices in future
for lag in [7, 14, 30]:
    for risk_type in RISK_CATEGORIES:
        score_col = f'climate_risk_{risk_type}_score'

        lag_col = f'climate_risk_{risk_type}_lag_{lag}d'
        merged_df[lag_col] = merged_df.groupby('region_id')[score_col].shift(lag)
        ALL_NEW_FEATURES.append(lag_col)

print(f"✅ Lag features added: {len(ALL_NEW_FEATURES)} total")


# ---
# ## 🔧 Phase 4: EMA Features (More Weight to Recent Data)

# In[8]:


# Exponential Moving Averages
for span in [14, 30]:
    for risk_type in RISK_CATEGORIES:
        score_col = f'climate_risk_{risk_type}_score'

        ema_col = f'climate_risk_{risk_type}_ema_{span}d'
        merged_df[ema_col] = (
            merged_df.groupby('region_id')[score_col]
            .transform(lambda x: x.ewm(span=span, min_periods=1).mean())
        )
        ALL_NEW_FEATURES.append(ema_col)

print(f"✅ EMA features added: {len(ALL_NEW_FEATURES)} total")


# ---
# ## 🔧 Phase 5: Volatility Features (Risk Variability)

# In[9]:


# Rolling Standard Deviation (volatility)
for window in [14, 30]:
    for risk_type in RISK_CATEGORIES:
        score_col = f'climate_risk_{risk_type}_score'

        vol_col = f'climate_risk_{risk_type}_vol_{window}d'
        merged_df[vol_col] = (
            merged_df.groupby('region_id')[score_col]
            .transform(lambda x: x.rolling(window, min_periods=2).std())
        )
        ALL_NEW_FEATURES.append(vol_col)

print(f"✅ Volatility features added: {len(ALL_NEW_FEATURES)} total")


# ---
# ## 🔧 Phase 6: Cumulative Stress Features

# In[10]:


# Cumulative sum (total stress over period)
for window in [30, 60]:
    for risk_type in RISK_CATEGORIES:
        score_col = f'climate_risk_{risk_type}_score'

        cum_col = f'climate_risk_{risk_type}_cumsum_{window}d'
        merged_df[cum_col] = (
            merged_df.groupby('region_id')[score_col]
            .transform(lambda x: x.rolling(window, min_periods=1).sum())
        )
        ALL_NEW_FEATURES.append(cum_col)

print(f"✅ Cumulative features added: {len(ALL_NEW_FEATURES)} total")


# ---
# ## 🔧 Phase 7: Non-linear Features (Extreme Events)

# In[11]:


# Non-linear transformations
for risk_type in RISK_CATEGORIES:
    score_col = f'climate_risk_{risk_type}_score'

    # Squared - emphasizes extreme values
    sq_col = f'climate_risk_{risk_type}_squared'
    merged_df[sq_col] = merged_df[score_col] ** 2
    ALL_NEW_FEATURES.append(sq_col)

    # Log transform - compresses high values
    log_col = f'climate_risk_{risk_type}_log'
    merged_df[log_col] = np.log1p(merged_df[score_col])
    ALL_NEW_FEATURES.append(log_col)

print(f"✅ Non-linear features added: {len(ALL_NEW_FEATURES)} total")


# ---
# ## 🔧 Phase 8: Interaction Features (Combined Stress)

# In[12]:


# Composite indices
score_cols = [f'climate_risk_{r}_score' for r in RISK_CATEGORIES]

# Temperature stress (max of heat/cold)
merged_df['climate_risk_temperature_stress'] = merged_df[[
    'climate_risk_heat_stress_score', 'climate_risk_unseasonably_cold_score'
]].max(axis=1)
ALL_NEW_FEATURES.append('climate_risk_temperature_stress')

# Precipitation stress (max of wet/dry)
merged_df['climate_risk_precipitation_stress'] = merged_df[[
    'climate_risk_excess_precip_score', 'climate_risk_drought_score'
]].max(axis=1)
ALL_NEW_FEATURES.append('climate_risk_precipitation_stress')

# Overall stress (max of all)
merged_df['climate_risk_overall_stress'] = merged_df[score_cols].max(axis=1)
ALL_NEW_FEATURES.append('climate_risk_overall_stress')

# Combined stress (sum of all)
merged_df['climate_risk_combined_stress'] = merged_df[score_cols].sum(axis=1)
ALL_NEW_FEATURES.append('climate_risk_combined_stress')

# Difference features
merged_df['climate_risk_precip_drought_diff'] = (
    merged_df['climate_risk_excess_precip_score'] - merged_df['climate_risk_drought_score']
)
ALL_NEW_FEATURES.append('climate_risk_precip_drought_diff')

merged_df['climate_risk_temp_diff'] = (
    merged_df['climate_risk_heat_stress_score'] - merged_df['climate_risk_unseasonably_cold_score']
)
ALL_NEW_FEATURES.append('climate_risk_temp_diff')

# Ratio features
merged_df['climate_risk_precip_drought_ratio'] = (
    merged_df['climate_risk_excess_precip_score'] / 
    (merged_df['climate_risk_drought_score'] + 0.01)
)
ALL_NEW_FEATURES.append('climate_risk_precip_drought_ratio')

print(f"✅ Interaction features added: {len(ALL_NEW_FEATURES)} total")


# ---
# ## 🔧 Phase 9: Seasonal Features

# In[13]:


# Cyclical encoding of day of year
merged_df['climate_risk_season_sin'] = np.sin(2 * np.pi * merged_df['day_of_year'] / 365)
merged_df['climate_risk_season_cos'] = np.cos(2 * np.pi * merged_df['day_of_year'] / 365)
ALL_NEW_FEATURES.extend(['climate_risk_season_sin', 'climate_risk_season_cos'])

# Growing season weighted risk (Q2-Q3 higher weight)
growing_season_weight = merged_df['quarter'].map({1: 0.5, 2: 1.0, 3: 1.0, 4: 0.5})

for risk_type in ['drought', 'excess_precip']:  # Most relevant for growing season
    score_col = f'climate_risk_{risk_type}_score'
    seasonal_col = f'climate_risk_{risk_type}_seasonal'
    merged_df[seasonal_col] = merged_df[score_col] * growing_season_weight
    ALL_NEW_FEATURES.append(seasonal_col)

print(f"✅ Seasonal features added: {len(ALL_NEW_FEATURES)} total")


# ---
# ## 🔧 Phase 10: Momentum Features

# In[14]:


# Momentum/change features
for risk_type in RISK_CATEGORIES:
    score_col = f'climate_risk_{risk_type}_score'

    # Daily change
    c1 = f'climate_risk_{risk_type}_change_1d'
    merged_df[c1] = merged_df.groupby('region_id')[score_col].diff(1)
    ALL_NEW_FEATURES.append(c1)

    # Weekly change
    c7 = f'climate_risk_{risk_type}_change_7d'
    merged_df[c7] = merged_df.groupby('region_id')[score_col].diff(7)
    ALL_NEW_FEATURES.append(c7)

    # Acceleration
    acc = f'climate_risk_{risk_type}_acceleration'
    merged_df[acc] = merged_df.groupby('region_id')[c1].diff(1)
    ALL_NEW_FEATURES.append(acc)

print(f"✅ Momentum features added: {len(ALL_NEW_FEATURES)} total")


# ---
# ## 🔧 Phase 11: Country Aggregations

# In[15]:


# Country-level aggregations
for risk_type in RISK_CATEGORIES:
    score_col = f'climate_risk_{risk_type}_score'
    weighted_col = f'climate_risk_{risk_type}_weighted'

    country_agg = merged_df.groupby(['country_name', 'date_on']).agg({
        score_col: ['mean', 'max', 'std'],
        weighted_col: 'sum',
        'percent_country_production': 'sum'
    }).round(4)

    country_agg.columns = [f'country_{risk_type}_{"_".join(col).strip()}' for col in country_agg.columns]
    country_agg = country_agg.reset_index()

    new_cols = [c for c in country_agg.columns if c not in ['country_name', 'date_on']]
    ALL_NEW_FEATURES.extend(new_cols)

    merged_df = merged_df.merge(country_agg, on=['country_name', 'date_on'], how='left')

print(f"✅ Country aggregations added: {len(ALL_NEW_FEATURES)} total")


# In[16]:


# Since feature engineering creates some new NaN values due to lag etc. it might be tricky to
# match the IDs Kaggle expects.
# Although being far from optimal below approach guarantees exactly 219,161 rows while preserving all feature values.
#### STEPS FOLLOWED BELOW ####
# 1. Simulate what sample submission does to identify valid rows (by ID)
# 2. Fill all engineered features with 0 (edge-effect NaN)
# 3. Filter to only keep rows with valid IDs

REQUIRED_ROWS = 219161

print(f"\n📊 Before NaN handling: {len(merged_df):,} rows")

# Step 1: Identify valid IDs by simulating sample submission's approach
print("📊 Identifying valid IDs (simulating sample submission)...")

# Start fresh from original data
temp_df = pd.read_csv(f'{DATA_PATH}corn_climate_risk_futures_daily_master.csv')
temp_df['date_on'] = pd.to_datetime(temp_df['date_on'])

# Add basic features (same as sample submission)
temp_df['day_of_year'] = temp_df['date_on'].dt.dayofyear
temp_df['quarter'] = temp_df['date_on'].dt.quarter

# Merge market share
temp_df = temp_df.merge(
    market_share_df[['region_id', 'percent_country_production']], 
    on='region_id', how='left'
)
temp_df['percent_country_production'] = temp_df['percent_country_production'].fillna(1.0)

# Create base risk scores (same as sample submission)
for risk_type in RISK_CATEGORIES:
    low_col = f'climate_risk_cnt_locations_{risk_type}_risk_low'
    med_col = f'climate_risk_cnt_locations_{risk_type}_risk_medium' 
    high_col = f'climate_risk_cnt_locations_{risk_type}_risk_high'

    total = temp_df[low_col] + temp_df[med_col] + temp_df[high_col]
    risk_score = (temp_df[med_col] + 2 * temp_df[high_col]) / (total + 1e-6)
    weighted = risk_score * (temp_df['percent_country_production'] / 100)

    temp_df[f'climate_risk_{risk_type}_score'] = risk_score
    temp_df[f'climate_risk_{risk_type}_weighted'] = weighted

# Create composite indices
score_cols = [f'climate_risk_{r}_score' for r in RISK_CATEGORIES]
temp_df['climate_risk_temperature_stress'] = temp_df[['climate_risk_heat_stress_score', 'climate_risk_unseasonably_cold_score']].max(axis=1)
temp_df['climate_risk_precipitation_stress'] = temp_df[['climate_risk_excess_precip_score', 'climate_risk_drought_score']].max(axis=1)
temp_df['climate_risk_overall_stress'] = temp_df[score_cols].max(axis=1)
temp_df['climate_risk_combined_stress'] = temp_df[score_cols].mean(axis=1)

# Sort for rolling operations
temp_df = temp_df.sort_values(['region_id', 'date_on'])

# Create rolling features (7, 14, 30 days - same as sample submission)
for window in [7, 14, 30]:
    for risk_type in RISK_CATEGORIES:
        score_col = f'climate_risk_{risk_type}_score'
        temp_df[f'climate_risk_{risk_type}_ma_{window}d'] = (
            temp_df.groupby('region_id')[score_col]
            .transform(lambda x: x.rolling(window, min_periods=1).mean())
        )
        temp_df[f'climate_risk_{risk_type}_max_{window}d'] = (
            temp_df.groupby('region_id')[score_col]
            .transform(lambda x: x.rolling(window, min_periods=1).max())
        )

# Create momentum features (same as sample submission)
for risk_type in RISK_CATEGORIES:
    score_col = f'climate_risk_{risk_type}_score'
    temp_df[f'climate_risk_{risk_type}_change_1d'] = temp_df.groupby('region_id')[score_col].diff(1)
    temp_df[f'climate_risk_{risk_type}_change_7d'] = temp_df.groupby('region_id')[score_col].diff(7)
    temp_df[f'climate_risk_{risk_type}_acceleration'] = temp_df.groupby('region_id')[f'climate_risk_{risk_type}_change_1d'].diff(1)

# Create country aggregations (same as sample submission)
for risk_type in RISK_CATEGORIES:
    score_col = f'climate_risk_{risk_type}_score'
    weighted_col = f'climate_risk_{risk_type}_weighted'

    country_agg = temp_df.groupby(['country_name', 'date_on']).agg({
        score_col: ['mean', 'max', 'std'],
        weighted_col: 'sum',
        'percent_country_production': 'sum'
    }).round(4)

    country_agg.columns = [f'country_{risk_type}_{"_".join(col).strip()}' for col in country_agg.columns]
    country_agg = country_agg.reset_index()

    temp_df = temp_df.merge(country_agg, on=['country_name', 'date_on'], how='left')

# Now dropna to get valid IDs (this is what sample submission does)
valid_ids = temp_df.dropna()['ID'].tolist()
print(f"📊 Valid IDs from sample submission approach: {len(valid_ids):,}")

# Clean up
del temp_df

# Step 2: Fill all engineered features in merged_df with 0
print("📊 Filling engineered features with 0...")

for col in ALL_NEW_FEATURES:
    if col in merged_df.columns:
        merged_df[col] = merged_df[col].fillna(0)

# Also fill any remaining NaN in climate_risk columns
climate_cols = [c for c in merged_df.columns if c.startswith('climate_risk_')]
for col in climate_cols:
    if merged_df[col].isna().any():
        merged_df[col] = merged_df[col].fillna(0)

# Step 3: Filter to valid IDs
print("📊 Filtering to valid IDs...")

# First, drop rows with NaN in futures columns (non-trading days)
futures_cols = [c for c in merged_df.columns if c.startswith('futures_')]
baseline_df = merged_df.dropna(subset=futures_cols)

# Then filter to only valid IDs
baseline_df = baseline_df[baseline_df['ID'].isin(valid_ids)]

print(f"📊 After NaN handling: {len(baseline_df):,} rows")
print(f"📊 Expected rows: {REQUIRED_ROWS:,}")
print(f"📊 Match: {'✅' if len(baseline_df) == REQUIRED_ROWS else '❌'}")
print(f"📊 Total new features: {len(ALL_NEW_FEATURES)}")

# Final verification
if len(baseline_df) != REQUIRED_ROWS:
    diff = len(baseline_df) - REQUIRED_ROWS
    print(f"\n⚠️ Row count difference: {diff:+d}")


# ---
# ## 📊 Phase 12: Feature Analysis and Selection

# In[17]:


# Analyze feature contributions
print("📊 Analyzing feature contributions (this takes ~3 minutes)...")

climate_cols = [c for c in baseline_df.columns if c.startswith('climate_risk_')]
futures_cols = [c for c in baseline_df.columns if c.startswith('futures_')]

print(f"   Climate features: {len(climate_cols)}")
print(f"   Futures features: {len(futures_cols)}")

feature_analysis = analyze_feature_contributions(baseline_df, climate_cols, futures_cols)


# In[18]:


# Show top features
print("\n🔝 TOP 25 Features by Significant Correlation Count:")
print("="*80)
print(feature_analysis.head(25).to_string(index=False))


# In[19]:


# Show bottom features (candidates for removal)
print("\n❌ BOTTOM 25 Features (candidates for removal):")
print("="*80)
print(feature_analysis.tail(25).to_string(index=False))


# In[20]:


# Identify features to remove
zero_sig_features = feature_analysis[feature_analysis['sig_count'] == 0]['feature'].tolist()

# Keep original cnt_locations columns (required by competition)
original_cols = [c for c in zero_sig_features if 'cnt_locations' in c]
FEATURES_TO_REMOVE = [c for c in zero_sig_features if c not in original_cols]

print(f"\n📊 Feature Selection Summary:")
print(f"   Total climate features: {len(climate_cols)}")
print(f"   Features with 0 significant correlations: {len(zero_sig_features)}")
print(f"   Features to remove: {len(FEATURES_TO_REMOVE)}")
print(f"   Total significant correlations: {feature_analysis['sig_count'].sum()}")


# ---
# ## 📊 Phase 13: Create Optimized Dataset

# In[21]:


# Create optimized dataset by removing weak features
optimized_df = baseline_df.copy()

cols_before = len([c for c in optimized_df.columns if c.startswith('climate_risk_')])
optimized_df = optimized_df.drop(columns=FEATURES_TO_REMOVE, errors='ignore')
cols_after = len([c for c in optimized_df.columns if c.startswith('climate_risk_')])

print(f"📊 Climate features: {cols_before} → {cols_after} (removed {cols_before - cols_after})")


# ---
# ## 📊 Phase 14: Score Comparison

# ---
# ## 📊 Phase 15: Final Submission

# In[22]:


best_df = optimized_df
#best_score = optimized_score
best_name = 'optimized'


# In[23]:


print(list(best_df.columns))


# # Adding External Data

# In[24]:


df = best_df.copy()


# In[25]:


PROTECTED_COLS = {
    'ID','crop_name','country_name','country_code','region_name','region_id',
    'harvest_period','growing_season_year','date_on',
    'climate_risk_cnt_locations_heat_stress_risk_low',
    'climate_risk_cnt_locations_heat_stress_risk_medium',
    'climate_risk_cnt_locations_heat_stress_risk_high',
    'climate_risk_cnt_locations_unseasonably_cold_risk_low',
    'climate_risk_cnt_locations_unseasonably_cold_risk_medium',
    'climate_risk_cnt_locations_unseasonably_cold_risk_high',
    'climate_risk_cnt_locations_excess_precip_risk_low',
    'climate_risk_cnt_locations_excess_precip_risk_medium',
    'climate_risk_cnt_locations_excess_precip_risk_high',
    'climate_risk_cnt_locations_drought_risk_low',
    'climate_risk_cnt_locations_drought_risk_medium',
    'climate_risk_cnt_locations_drought_risk_high',
    'futures_close_ZC_1','futures_close_ZC_2','futures_close_ZW_1','futures_close_ZS_1',
    'futures_zc1_ret_pct','futures_zc1_ret_log',
    'futures_zc_term_spread','futures_zc_term_ratio',
    'futures_zc1_ma_20','futures_zc1_ma_60','futures_zc1_ma_120',
    'futures_zc1_vol_20','futures_zc1_vol_60',
    'futures_zw_zc_spread','futures_zc_zw_ratio',
    'futures_zs_zc_spread','futures_zc_zs_ratio',
    'date_on_year','date_on_month','date_on_year_month',
    'day_of_year','quarter','percent_country_production'
}


# In[26]:


# Skip external data if not found
try:
    extra_data = pd.read_csv('/kaggle/input/extra-climate-date-daily/extra_climate_data.csv')
    # ... (rest of external processing would go here)
    # Since we can't easily indent the rest of the script, we'll just raise if missing and handle in a modified flow
    # But simpler: let's just STOP here if missing and save optimized_df
except FileNotFoundError:
    print("⚠️ External data not found. Using optimized_df as final submission.")
    submission = optimized_df.copy()
    submission = submission.fillna(0) # Safety
    submission.to_csv('submission.csv', index=False)
    print(f"Saved submission.csv with {len(submission)} rows.")
    exit(0) # Stop execution

extra_data.head()


# In[27]:


df["date_on"] = pd.to_datetime(df["date_on"])
extra_data["date"] = pd.to_datetime(extra_data["date"])


# In[28]:


extra_data = extra_data.rename(
    columns=lambda c: c if c == "date" else f"climate_risk_{c}"
)


# In[29]:


import pandas as pd

da = extra_data.copy()
da["date"] = pd.to_datetime(da["date"])
da["ym"] = da["date"].dt.to_period("M")

DAILY_COLS = []
MONTHLY_COLS = []

for col in da.columns:
    if col in ["date", "ym"]:
        continue

    # number of unique NON-ZERO values per month
    uniq_per_month = (
        da.groupby("ym")[col]
          .apply(lambda s: s[s != 0].nunique())
    )

    if (uniq_per_month > 1).any():
        DAILY_COLS.append(col)
    else:
        MONTHLY_COLS.append(col)

print("DAILY COLUMNS:")
print(DAILY_COLS)

print("\nMONTHLY COLUMNS:")
print(MONTHLY_COLS)


# In[30]:


da = da.sort_values("date")

for col in DAILY_COLS:
    da[f"{col}_lag_7d"]  = da[col].shift(7)
    da[f"{col}_lag_14d"] = da[col].shift(14)
    da[f"{col}_lag_30d"] = da[col].shift(30)


# In[31]:


da = da.sort_values("date")

for col in DAILY_COLS:
    da[f"{col}_ema_7d"]  = da[col].ewm(span=7, adjust=False).mean()
    da[f"{col}_ema_14d"] = da[col].ewm(span=14, adjust=False).mean()
    da[f"{col}_ema_30d"] = da[col].ewm(span=30, adjust=False).mean()


# In[32]:


da = da.sort_values("date")

# Daily cumulative stress (recent memory)
for col in DAILY_COLS:
    da[f"{col}_cum_30d"] = da[col].rolling(30, min_periods=1).sum()

# Monthly cumulative stress (long memory)
for col in MONTHLY_COLS:
    da[f"{col}_cum"] = da[col].expanding().sum()


# In[33]:


da = da.sort_values("date")

for col in DAILY_COLS:
    da[f"{col}_vol_7d"]  = da[col].rolling(7,  min_periods=1).std()
    da[f"{col}_vol_14d"] = da[col].rolling(14, min_periods=1).std()
    da[f"{col}_vol_30d"] = da[col].rolling(30, min_periods=1).std()


# In[34]:


DERIV_DAYS = 30  

def rolling_slope(arr):
    x = np.arange(len(arr))
    return np.polyfit(x, arr, 1)[0]

for col in DAILY_COLS:
    da[f"{col}_deriv_{DERIV_DAYS}d"] = (
        da[col]
        .rolling(DERIV_DAYS, min_periods=DERIV_DAYS)
        .apply(rolling_slope, raw=True)
    )


# In[35]:


#da = da.sort_values("date").copy()
da["ym"] = da["date"].dt.to_period("M")

DERIV_MONTHS = 3  # <-- change to 3, 6, etc.

def rolling_slope(arr):
    x = np.arange(len(arr))
    return np.polyfit(x, arr, 1)[0]

for col in MONTHLY_COLS:
    # monthly series (one value per month)
    monthly = (
        da.groupby("ym")[col]
          .first()
          .sort_index()
    )

    # slope across months
    monthly_deriv = (
        monthly
        .rolling(DERIV_MONTHS, min_periods=DERIV_MONTHS)
        .apply(rolling_slope, raw=True)
    )

    # map back to daily rows
    da[f"{col}_deriv_{DERIV_MONTHS}m"] = da["ym"].map(monthly_deriv)

DERIV_MONTHS = 6

for col in MONTHLY_COLS:
    # monthly series (one value per month)
    monthly = (
        da.groupby("ym")[col]
          .first()
          .sort_index()
    )

    # slope across months
    monthly_deriv = (
        monthly
        .rolling(DERIV_MONTHS, min_periods=DERIV_MONTHS)
        .apply(rolling_slope, raw=True)
    )

    # map back to daily rows
    da[f"{col}_deriv_{DERIV_MONTHS}m"] = da["ym"].map(monthly_deriv)

da.drop(columns="ym", inplace=True)


# In[36]:


for col in DAILY_COLS + MONTHLY_COLS:
    da[f"{col}_sq"] = da[col] ** 2

for col in DAILY_COLS + MONTHLY_COLS:
    da[f"{col}_exp"] = np.exp(da[col]) - 1


# In[37]:


# extract year-month
da["ym"] = da["date"].dt.to_period("M")

for col in MONTHLY_COLS:
    # monthly series (one value per month)
    monthly = (
        da.groupby("ym")[col]
          .first()
          .sort_index()
    )

    # true month lags
    monthly_lag_3 = monthly.shift(3)
    monthly_lag_6 = monthly.shift(6)

    # map back to daily rows
    da[f"{col}_lag_3m"] = da["ym"].map(monthly_lag_3)
    da[f"{col}_lag_6m"] = da["ym"].map(monthly_lag_6)

for col in MONTHLY_COLS:
    # monthly series (1 value per month)
    monthly = (
        da.groupby("ym")[col]
          .first()
          .sort_index()
    )

    # true monthly EMAs
    monthly_ema_3 = monthly.ewm(span=3, adjust=False).mean()
    monthly_ema_6 = monthly.ewm(span=6, adjust=False).mean()

    # map back to daily rows
    da[f"{col}_ema_3m"] = da["ym"].map(monthly_ema_3)
    da[f"{col}_ema_6m"] = da["ym"].map(monthly_ema_6)


for col in MONTHLY_COLS:
    # one value per month
    monthly = (
        da.groupby("ym")[col]
          .first()
          .sort_index()
    )

    # true monthly rolling volatility
    monthly_vol_3 = monthly.rolling(3, min_periods=1).std()
    monthly_vol_6 = monthly.rolling(6, min_periods=1).std()

    # map back to daily rows
    da[f"{col}_vol_3m"] = da["ym"].map(monthly_vol_3)
    da[f"{col}_vol_6m"] = da["ym"].map(monthly_vol_6)

# optional cleanup
da.drop(columns="ym", inplace=True)


# In[38]:


for col in DAILY_COLS:
    hi = da[col].quantile(0.75)
    lo = da[col].quantile(0.25)

    da[f"{col}_strong"] = (da[col] >= hi).astype(int)
    da[f"{col}_weak"]   = (da[col] <= lo).astype(int)


# In[39]:


da["ym"] = da["date"].dt.to_period("M")

for col in MONTHLY_COLS:
    # one value per month
    monthly = (
        da.groupby("ym")[col]
          .first()
          .sort_index()
    )

    hi = monthly.quantile(0.75)
    lo = monthly.quantile(0.25)

    monthly_strong = (monthly >= hi).astype(int)
    monthly_weak   = (monthly <= lo).astype(int)

    # map back to daily rows
    da[f"{col}_strong"] = da["ym"].map(monthly_strong)
    da[f"{col}_weak"]   = da["ym"].map(monthly_weak)

da.drop(columns="ym", inplace=True)


# In[ ]:





# In[40]:


from itertools import combinations

cols = [
    "climate_risk_zwnd200_anomaly_ema_6m", "climate_risk_zwnd200_standardized_ema_6m", "climate_risk_repac_slpa_cum", "climate_risk_epac_850_ema_6m", "climate_risk_cpac_850_ema_6m",
    "climate_risk_tahiti_original_ema_6m", "climate_risk_tahiti_anomaly_ema_6m", "climate_risk_ONI_cum", "climate_risk_soi_standardized_ema_6m", "climate_risk_nino_anom_cum"
]

for c1, c2 in combinations(cols, 2):
    da[f"{c1}_x_{c2}"] = da[c1] * da[c2]


# In[41]:


'''da["climate_risk_zwnd200_x_repac"] = da["climate_risk_zwnd200_anomaly_ema_6m"] * da["climate_risk_repac_slpa_cum"]
da["climate_risk_epac_x_cpac"] = da["climate_risk_epac_850_ema_6m"] * da["climate_risk_cpac_850_ema_6m"]
da["climate_risk_ONI_x_tahiti"] = da["climate_risk_ONI_cum"] * da["climate_risk_tahiti_anomaly_ema_6m"]
da["climate_risk_soi_x_soi"] = da["climate_risk_soi_standardized_ema_6m"] * da["climate_risk_nino_anom_cum"]'''


# In[42]:


df = df.merge(
    da,
    how="left",
    left_on="date_on",
    right_on="date"
)


# In[43]:


df.head()


# # Column List
# as discussed earlier, I got this list of columns from dropping features with sig correlations <400 and then forward selection the rest. 

# In[44]:


cols = ['ID', 'crop_name', 'country_name', 'country_code', 'region_name', 'region_id', 'harvest_period', 'growing_season_year', 'date_on', 'climate_risk_cnt_locations_heat_stress_risk_low', 'climate_risk_cnt_locations_heat_stress_risk_medium', 'climate_risk_cnt_locations_heat_stress_risk_high', 'climate_risk_cnt_locations_unseasonably_cold_risk_low', 'climate_risk_cnt_locations_unseasonably_cold_risk_medium', 'climate_risk_cnt_locations_unseasonably_cold_risk_high', 'climate_risk_cnt_locations_excess_precip_risk_low', 'climate_risk_cnt_locations_excess_precip_risk_medium', 'climate_risk_cnt_locations_excess_precip_risk_high', 'climate_risk_cnt_locations_drought_risk_low', 'climate_risk_cnt_locations_drought_risk_medium', 'climate_risk_cnt_locations_drought_risk_high', 'futures_close_ZC_1', 'futures_close_ZC_2', 'futures_close_ZW_1', 'futures_close_ZS_1', 'futures_zc1_ret_pct', 'futures_zc1_ret_log', 'futures_zc_term_spread', 'futures_zc_term_ratio', 'futures_zc1_ma_20', 'futures_zc1_ma_60', 'futures_zc1_ma_120', 'futures_zc1_vol_20', 'futures_zc1_vol_60', 'futures_zw_zc_spread', 'futures_zc_zw_ratio', 'futures_zs_zc_spread', 'futures_zc_zs_ratio', 'date_on_year', 'date_on_month', 'date_on_year_month', 'day_of_year', 'quarter', 'percent_country_production', 'climate_risk_ONI', 'climate_risk_nino34_ssta', 'climate_risk_nino4_sst', 'climate_risk_nino4_ssta', 'climate_risk_nino_total', 'climate_risk_nino_anom', 'climate_risk_rnino_anom', 'climate_risk_tahiti_anomaly', 'climate_risk_tahiti_original', 'climate_risk_epac_850', 'climate_risk_cpac_850', 'climate_risk_zwnd200_anomaly', 'climate_risk_zwnd200_standardized', 'climate_risk_nino4_sst_lag_7d', 'climate_risk_nino4_sst_lag_14d', 'climate_risk_nino4_sst_lag_30d', 'climate_risk_nino4_ssta_lag_7d', 'climate_risk_nino4_ssta_lag_14d', 'climate_risk_nino4_ssta_lag_30d', 'climate_risk_nino4_sst_ema_7d', 'climate_risk_nino4_sst_ema_14d', 'climate_risk_nino4_sst_ema_30d', 'climate_risk_nino4_ssta_ema_7d', 'climate_risk_nino4_ssta_ema_14d', 'climate_risk_nino4_ssta_ema_30d', 'climate_risk_nino4_sst_cum_30d', 'climate_risk_nino4_ssta_cum_30d', 'climate_risk_cpolr_cum', 'climate_risk_ONI_cum', 'climate_risk_nino_total_cum', 'climate_risk_nino_clim_adjust_cum', 'climate_risk_nino_anom_cum', 'climate_risk_rnino_anom_cum', 'climate_risk_olr_anomaly_cum', 'climate_risk_olr_standardized_cum', 'climate_risk_soi_anomaly_cum', 'climate_risk_soi_standardized_cum', 'climate_risk_repac_slpa_cum', 'climate_risk_reqsoi_cum', 'climate_risk_natl_cum', 'climate_risk_satl_cum', 'climate_risk_trop_cum', 'climate_risk_tahiti_anomaly_cum', 'climate_risk_tahiti_original_cum', 'climate_risk_darwin_original_cum', 'climate_risk_epac_850_cum', 'climate_risk_cpac_850_cum', 'climate_risk_zwnd200_anomaly_cum', 'climate_risk_zwnd200_standardized_cum', 'climate_risk_soi_anomaly_lag_6m', 'climate_risk_soi_standardized_lag_6m', 'climate_risk_epac_850_lag_6m', 'climate_risk_cpac_850_lag_6m', 'climate_risk_zwnd200_anomaly_lag_6m', 'climate_risk_zwnd200_standardized_lag_6m', 'climate_risk_soi_anomaly_ema_3m', 'climate_risk_soi_anomaly_ema_6m', 'climate_risk_soi_standardized_ema_3m', 'climate_risk_soi_standardized_ema_6m', 'climate_risk_tahiti_anomaly_ema_3m', 'climate_risk_tahiti_anomaly_ema_6m', 'climate_risk_tahiti_original_ema_3m', 'climate_risk_tahiti_original_ema_6m', 'climate_risk_epac_850_ema_3m', 'climate_risk_epac_850_ema_6m', 'climate_risk_cpac_850_ema_3m', 'climate_risk_cpac_850_ema_6m', 'climate_risk_zwnd200_anomaly_ema_3m', 'climate_risk_zwnd200_anomaly_ema_6m', 'climate_risk_zwnd200_standardized_ema_3m', 'climate_risk_zwnd200_standardized_ema_6m', 'climate_risk_nino_clim_adjust_vol_3m', 'climate_risk_nino_clim_adjust_vol_6m', 'climate_risk_zwnd200_anomaly_ema_6m_x_climate_risk_zwnd200_standardized_ema_6m', 'climate_risk_zwnd200_anomaly_ema_6m_x_climate_risk_epac_850_ema_6m', 'climate_risk_zwnd200_anomaly_ema_6m_x_climate_risk_cpac_850_ema_6m', 'climate_risk_zwnd200_anomaly_ema_6m_x_climate_risk_tahiti_original_ema_6m', 'climate_risk_zwnd200_anomaly_ema_6m_x_climate_risk_tahiti_anomaly_ema_6m', 'climate_risk_zwnd200_anomaly_ema_6m_x_climate_risk_soi_standardized_ema_6m', 'climate_risk_zwnd200_standardized_ema_6m_x_climate_risk_epac_850_ema_6m', 'climate_risk_zwnd200_standardized_ema_6m_x_climate_risk_cpac_850_ema_6m', 'climate_risk_zwnd200_standardized_ema_6m_x_climate_risk_tahiti_original_ema_6m', 'climate_risk_zwnd200_standardized_ema_6m_x_climate_risk_tahiti_anomaly_ema_6m', 'climate_risk_zwnd200_standardized_ema_6m_x_climate_risk_soi_standardized_ema_6m', 'climate_risk_repac_slpa_cum_x_climate_risk_tahiti_original_ema_6m', 'climate_risk_repac_slpa_cum_x_climate_risk_tahiti_anomaly_ema_6m', 'climate_risk_repac_slpa_cum_x_climate_risk_ONI_cum', 'climate_risk_repac_slpa_cum_x_climate_risk_nino_anom_cum', 'climate_risk_epac_850_ema_6m_x_climate_risk_cpac_850_ema_6m', 'climate_risk_epac_850_ema_6m_x_climate_risk_tahiti_original_ema_6m', 'climate_risk_epac_850_ema_6m_x_climate_risk_tahiti_anomaly_ema_6m', 'climate_risk_epac_850_ema_6m_x_climate_risk_ONI_cum', 'climate_risk_epac_850_ema_6m_x_climate_risk_soi_standardized_ema_6m', 'climate_risk_cpac_850_ema_6m_x_climate_risk_tahiti_original_ema_6m', 'climate_risk_cpac_850_ema_6m_x_climate_risk_tahiti_anomaly_ema_6m', 'climate_risk_cpac_850_ema_6m_x_climate_risk_ONI_cum', 'climate_risk_cpac_850_ema_6m_x_climate_risk_soi_standardized_ema_6m', 'climate_risk_tahiti_original_ema_6m_x_climate_risk_tahiti_anomaly_ema_6m', 'climate_risk_tahiti_original_ema_6m_x_climate_risk_ONI_cum', 'climate_risk_tahiti_original_ema_6m_x_climate_risk_soi_standardized_ema_6m', 'climate_risk_tahiti_anomaly_ema_6m_x_climate_risk_soi_standardized_ema_6m', 'climate_risk_ONI_cum_x_climate_risk_nino_anom_cum']


# In[45]:


df = df[cols]


# In[46]:


df.shape


# In[47]:


df.head()


# In[48]:


print(list(df.columns))


# In[49]:


# Validation
REQUIRED_ROWS = 219161
submission = df.copy()

# Safety: fill any remaining nulls
if submission.isnull().sum().sum() > 0:
    print("⚠️ Filling remaining nulls with 0...")
    submission = submission.fillna(0)

print("\n" + "="*60)
print("✅ SUBMISSION VALIDATION")
print("="*60)

checks = [
    ('Row count', len(submission) == REQUIRED_ROWS, f"{len(submission):,}/{REQUIRED_ROWS:,}"),
    ('ID column', 'ID' in submission.columns, str('ID' in submission.columns)),
    ('No nulls', submission.isnull().sum().sum() == 0, f"{submission.isnull().sum().sum()} nulls"),
]

for name, passed, detail in checks:
    print(f"{'✅' if passed else '❌'} {name}: {detail}")

print("="*60)


# In[ ]:





# In[50]:


submission.head()


# In[51]:


# Save submission
output_file = f'{OUTPUT_PATH}submission.csv'
submission.to_csv(output_file, index=False)

climate_features = [c for c in submission.columns if c.startswith('climate_risk_')]

print(len(climate_features))


# In[ ]:




