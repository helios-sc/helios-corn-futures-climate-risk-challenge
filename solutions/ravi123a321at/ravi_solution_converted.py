#!/usr/bin/env python
# coding: utf-8

# # Helios Corn Futures Climate Challenge - Submission
# 
# **Features:** 34 (22 Baseline + 11 Phenology + 1 Soil Moisture)
# 
# **Local CFCS Score:** 50.96

# In[ ]:


import pandas as pd
import numpy as np
import warnings
import os

warnings.filterwarnings('ignore')


# ## 1. Feature Engineering Functions

# In[ ]:


def load_data():
    print("Loading data...")
    df = pd.read_csv('corn_climate_risk_futures_daily_master.csv')
    df['date_on'] = pd.to_datetime(df['date_on'])

    market_share = pd.read_csv('corn_regional_market_share.csv')
    df = df.merge(
        market_share[['region_id', 'percent_country_production']], 
        on='region_id', how='left'
    )
    df['percent_country_production'] = df['percent_country_production'].fillna(0)
    return df

def create_baseline_features(df):
    print("Creating baseline features...")
    risk_categories = ['heat_stress', 'unseasonably_cold', 'excess_precip', 'drought']

    for risk in risk_categories:
        low = df[f'climate_risk_cnt_locations_{risk}_risk_low']
        med = df[f'climate_risk_cnt_locations_{risk}_risk_medium']
        high = df[f'climate_risk_cnt_locations_{risk}_risk_high']

        total = low + med + high
        score = (med * 1.0 + high * 2.0) / (total.replace(0, 1))
        weighted_score = score * (df['percent_country_production'] / 100.0)

        df[f'climate_risk_{risk}_score'] = score
        df[f'climate_risk_{risk}_weighted'] = weighted_score

    return df

def aggregate_baseline_countries(df):
    print("Aggregating baseline to country level...")
    risk_categories = ['heat_stress', 'unseasonably_cold', 'excess_precip', 'drought']

    agg_funcs = {}
    for risk in risk_categories:
        agg_funcs[f'climate_risk_{risk}_score'] = ['mean', 'max', 'std']
        agg_funcs[f'climate_risk_{risk}_weighted'] = ['sum']

    country_groups = df.groupby(['country_name', 'date_on']).agg(agg_funcs)
    country_groups.columns = [f'country_{c[0].replace("climate_risk_", "").replace("_score", "").replace("_weighted", "")}_{c[1]}' for c in country_groups.columns]
    country_groups = country_groups.reset_index()
    df = df.merge(country_groups, on=['country_name', 'date_on'], how='left')

    # HHI Concentration
    print("Calculating Spatial Concentration (HHI)...")
    for risk in risk_categories:
        region_val = df[f'climate_risk_{risk}_weighted']
        country_sum = df[f'country_{risk}_sum']
        share = region_val / country_sum.replace(0, np.nan)
        share_sq = share ** 2

        hhi_col_name = f'temp_share_sq_{risk}'
        df[hhi_col_name] = share_sq
        hhi_agg = df.groupby(['country_name', 'date_on'])[hhi_col_name].sum().reset_index()
        hhi_agg.rename(columns={hhi_col_name: f'country_{risk}_concentration'}, inplace=True)
        df = df.merge(hhi_agg, on=['country_name', 'date_on'], how='left')
        df.drop(columns=[hhi_col_name], inplace=True)
        df[f'country_{risk}_concentration'] = df[f'country_{risk}_concentration'].fillna(0)

    return df


# In[ ]:


def create_phenology_features(df):
    print("Creating PHENOLOGY-WEIGHTED STRESS...")

    nh_countries = ['United States', 'Ukraine', 'Russia', 'India', 'China', 'Canada', 'European Union', 'Mexico']
    month_weights = {4: 0.2, 5: 0.5, 6: 1.0, 7: 1.5, 8: 1.0, 9: 0.4}

    df['month'] = df['date_on'].dt.month
    df['pheno_weight'] = df['month'].map(month_weights).fillna(0.0)
    base_signal = df['climate_risk_drought_weighted']

    all_countries = sorted(df['country_name'].unique())
    new_features = []

    for country in all_countries:
        feat_name = f'climate_risk_drought_exposure_pheno_{country}'

        if country in nh_countries:
            is_country = (df['country_name'] == country)
            df[feat_name] = 0.0
            df.loc[is_country, feat_name] = base_signal.loc[is_country] * df.loc[is_country, 'pheno_weight']
        else:
            df[feat_name] = 0.0

        new_features.append(feat_name)

    print(f"Created {len(new_features)} phenology features.")
    return df, new_features

def aggregate_phenology_daily_to_monthly(df, pheno_cols):
    print("Aggregating Phenology to Monthly Mean...")
    df['ym_str'] = df['date_on'].dt.strftime('%Y-%m')

    for col in pheno_cols:
        target_country = col.replace('climate_risk_drought_exposure_pheno_', '')
        mask = df['country_name'] == target_country
        if mask.any():
            monthly_means = df.loc[mask].groupby('ym_str')[col].transform('mean')
            df.loc[mask, col] = monthly_means

    return df


# In[ ]:


def create_soil_moisture_features(df):
    """Refined Soil Moisture Features"""
    print("Creating Refined Soil Moisture Features...")

    soil_file = 'external_soil_data.csv'
    if not os.path.exists(soil_file):
        print(f"  WARNING: {soil_file} not found. Skipping soil features.")
        return df, []

    soil_df = pd.read_csv(soil_file)
    soil_df['date_on'] = pd.to_datetime(soil_df['date'])

    # Weighted root-zone moisture (0.7 root + 0.3 surface)
    soil_df['soil_moisture_weighted'] = (
        0.7 * soil_df['soil_moisture_root'].fillna(0) + 
        0.3 * soil_df['soil_moisture_surf'].fillna(0)
    )

    # Z-score by country
    country_stats = soil_df.groupby('country_name')['soil_moisture_weighted'].agg(['mean', 'std']).reset_index()
    country_stats.columns = ['country_name', 'sm_mean', 'sm_std']
    soil_df = soil_df.merge(country_stats, on='country_name', how='left')

    soil_df['country_soil_moisture_zscore'] = (
        (soil_df['soil_moisture_weighted'] - soil_df['sm_mean']) / 
        soil_df['sm_std'].replace(0, 1)
    ).clip(-3, 3).fillna(0)

    # Aggregate to country-level
    soil_agg = soil_df.groupby(['date_on', 'country_name'])['country_soil_moisture_zscore'].mean().reset_index()
    df = df.merge(soil_agg, on=['date_on', 'country_name'], how='left')
    df['country_soil_moisture_zscore'] = df['country_soil_moisture_zscore'].fillna(0)

    print("  Created 1 soil moisture feature: country_soil_moisture_zscore")
    return df, ['country_soil_moisture_zscore']


# ## 2. Run Feature Engineering

# In[ ]:


# Load and process data
df = load_data()
df = create_baseline_features(df)
df = aggregate_baseline_countries(df)

baseline_cols = [c for c in df.columns if c.startswith('country_')]
print(f"Baseline features: {len(baseline_cols)}")

df, pheno_cols = create_phenology_features(df)
df = aggregate_phenology_daily_to_monthly(df, pheno_cols)

df, soil_cols = create_soil_moisture_features(df)

print(f"\nTotal Features: {len(baseline_cols) + len(pheno_cols) + len(soil_cols)}")


# In[ ]:


# Save engineered features
meta = ['date_on', 'country_name', 'region_id', 'crop_name']
final_cols = list(set(meta + baseline_cols + pheno_cols + soil_cols))
df[final_cols].to_csv('engineered_features_pheno.csv', index=False)
print("Features saved.")


# ## 3. Produce Submission

# In[ ]:


print("Generating Submission...")
# template_df = pd.read_csv("sample submission.csv")
print("Creating template from master data...")
template_df = pd.read_csv("corn_climate_risk_futures_daily_master.csv")[['country_name', 'region_id', 'date_on', 'crop_name']]
feat_df = pd.read_csv("engineered_features_pheno.csv")
feat_df = feat_df.loc[:, ~feat_df.columns.duplicated()]

keys = ['country_name', 'region_id', 'date_on', 'crop_name']
feature_cols = [c for c in feat_df.columns if (c.startswith('country_') or c.startswith('climate_risk_')) and c not in keys]

features_subset = feat_df[keys + feature_cols].copy()
template_df['date_on'] = pd.to_datetime(template_df['date_on'])
features_subset['date_on'] = pd.to_datetime(features_subset['date_on'])

submission_df = template_df.merge(features_subset, on=keys, how='left')
submission_df.to_csv("submission.csv", index=False)
print(f"Submission Saved: {len(submission_df)} rows.")

