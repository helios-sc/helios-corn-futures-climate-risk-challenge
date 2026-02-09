import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ------------------------------------------------------------------------------
# 1. SETUP & SORTING
# ------------------------------------------------------------------------------
print("Step 1: Loading & Sorting Data...")
df = pd.read_csv('../../data/corn_climate_risk_futures_daily_master.csv')
market_share_df = pd.read_csv('../../data/corn_regional_market_share.csv')

if 'id' in df.columns:
    df = df.rename(columns={'id': 'ID'})
df['date_on'] = pd.to_datetime(df['date_on'])

# CRITICAL: Sort immediately
df = df.sort_values(['region_id', 'date_on'])

# Merge Weights
merged_daily_df = df.merge(
    market_share_df[['region_id', 'percent_country_production', 'region_name']],
    on='region_id',
    how='left'
)
if 'region_name_x' in merged_daily_df.columns:
    merged_daily_df.rename(columns={'region_name_x': 'region_name'}, inplace=True)
    merged_daily_df.drop(columns=['region_name_y'], inplace=True, errors='ignore')

merged_daily_df['percent_country_production'] = merged_daily_df['percent_country_production'].fillna(1.0)

# ------------------------------------------------------------------------------
# 2. HEMISPHERIC GATING (The Winner Logic)
# ------------------------------------------------------------------------------
print("Step 2: Hemispheric Gating (Noise Filter)...")

merged_daily_df['__month'] = merged_daily_df['date_on'].dt.month

# Define Active Seasons (When Corn is actually vulnerable)
merged_daily_df['__is_US_active'] = merged_daily_df['__month'].isin([5, 6, 7, 8, 9, 10])
merged_daily_df['__is_BR_active'] = merged_daily_df['__month'].isin([10, 11, 12, 1, 2, 3, 4, 5])

# ------------------------------------------------------------------------------
# 3. SIGNAL SHARPENING (Power Law Risk Scoring) - NEW!
# ------------------------------------------------------------------------------
print("Step 3: Engineering Power-Law Risk Scores (Signal Sharpening)...")

risk_categories = ['heat_stress', 'unseasonably_cold', 'excess_precip', 'drought']
risk_weights = {'heat_stress': 10.0, 'drought': 6.0, 'unseasonably_cold': 2.0, 'excess_precip': 2.0}

merged_daily_df['harvest_period'] = merged_daily_df['harvest_period'].fillna('Unknown')

for risk_type in risk_categories:
    low_col = f'climate_risk_cnt_locations_{risk_type}_risk_low'
    med_col = f'climate_risk_cnt_locations_{risk_type}_risk_medium'
    high_col = f'climate_risk_cnt_locations_{risk_type}_risk_high'

    # 1. Biological Weighting
    high_weight = risk_weights[risk_type]
    total = merged_daily_df[low_col] + merged_daily_df[med_col] + merged_daily_df[high_col]
    
    # Base Score Calculation
    risk_score = (merged_daily_df[med_col] + high_weight * merged_daily_df[high_col]) / (total + 1e-6)
    
    # 2. **POWER LAW TRANSFORMATION** (The Upgrade)
    # Squaring the risk score suppresses low-level noise (0.1 -> 0.01)
    # and amplifies distinct signals (0.8 -> 0.64, 1.5 -> 2.25).
    # We apply this BEFORE the multiplier to keep the scale reasonable.
    risk_score_sharpened = np.power(risk_score, 2.0)
    
    # 3. Phenological Amplifiers
    multiplier = pd.Series(1.0, index=merged_daily_df.index)
    if risk_type == 'heat_stress':
        multiplier = np.where(merged_daily_df['harvest_period'].str.contains('Growing|Planting', case=False, regex=True), 1.5, multiplier)
        multiplier = np.where(merged_daily_df['harvest_period'] == 'Harvest', 0.5, multiplier)
    elif risk_type == 'drought':
        multiplier = np.where(merged_daily_df['harvest_period'].str.contains('Growing', case=False, regex=True), 1.5, multiplier)
    elif risk_type == 'excess_precip':
        multiplier = np.where(merged_daily_df['harvest_period'].isin(['Planting', 'Harvest']), 2.0, multiplier)
        multiplier = np.where(merged_daily_df['harvest_period'].str.contains('Growing', case=False, regex=True), 0.5, multiplier)
    elif risk_type == 'unseasonably_cold':
        multiplier = np.where(merged_daily_df['harvest_period'] == 'Planting', 2.0, multiplier)
    
    final_score = risk_score_sharpened * multiplier
    
    # 4. Hemispheric Gating (Apply zero-mask)
    is_us = merged_daily_df['country_name'] == 'United States'
    is_br = merged_daily_df['country_name'] == 'Brazil'
    
    active_factor = np.ones(len(merged_daily_df))
    active_factor = np.where(is_us & (~merged_daily_df['__is_US_active']), 0.0, active_factor)
    active_factor = np.where(is_br & (~merged_daily_df['__is_BR_active']), 0.0, active_factor)
    
    # Save Columns
    merged_daily_df[f'climate_risk_{risk_type}_score'] = final_score * active_factor
    merged_daily_df[f'climate_risk_{risk_type}_weighted'] = (final_score * active_factor) * (merged_daily_df['percent_country_production'] / 100)

# ------------------------------------------------------------------------------
# 4. MACRO-MARKET FACTORS
# ------------------------------------------------------------------------------
print("Step 4: Engineering Macro Factors...")

# Trend (Bull/Bear)
price = merged_daily_df['futures_close_ZC_1'].fillna(0)
ma60 = merged_daily_df['futures_zc1_ma_60'].fillna(price)
trend_mult = np.where(price > ma60, 1.2, 0.8)

# Scarcity (Backwardation)
ratio = merged_daily_df['futures_zc_term_ratio'].fillna(1.0)
scarcity_mult = np.where(ratio < 1.0, 1.5, 0.8)

# Wheat Substitution
zw_zc_spread = merged_daily_df['futures_zw_zc_spread'].fillna(0)
# Safe rolling calc
spread_mean = merged_daily_df.groupby('region_id')['futures_zw_zc_spread'].rolling(60, min_periods=1).mean().reset_index(0, drop=True)
spread_std = merged_daily_df.groupby('region_id')['futures_zw_zc_spread'].rolling(60, min_periods=1).std().reset_index(0, drop=True).fillna(1.0)
wheat_z = (zw_zc_spread - spread_mean) / (spread_std + 1e-6)
wheat_mult = (1.0 + (0.2 * wheat_z)).clip(0.5, 1.5).fillna(1.0)

merged_daily_df['__master_market_factor'] = trend_mult * scarcity_mult * wheat_mult

# Apply Master Factor
for risk in ['heat_stress', 'drought']:
    base_col = f'climate_risk_{risk}_weighted'
    merged_daily_df[f'climate_risk_{risk}_MACRO_ADJ'] = merged_daily_df[base_col] * merged_daily_df['__master_market_factor']

# ------------------------------------------------------------------------------
# 5. POWER BELT & ACREAGE
# ------------------------------------------------------------------------------
print("Step 5: Power Belt & Acreage Battle...")

us_belt = ['Iowa', 'Illinois', 'Nebraska', 'Minnesota', 'Indiana', 'Kansas']
br_belt = ['Mato Grosso', 'Paraná', 'Goiás', 'Mato Grosso do Sul']
merged_daily_df['__is_power_belt'] = merged_daily_df['region_name'].isin(us_belt + br_belt)

for risk in ['heat_stress', 'drought']:
    base_col = f'climate_risk_{risk}_weighted'
    merged_daily_df[f'climate_risk_{risk}_POWER_BELT'] = np.where(
        merged_daily_df['__is_power_belt'], 
        merged_daily_df[base_col] * merged_daily_df['__master_market_factor'], 
        0
    )

# Acreage Battle (Soy vs Corn)
zs_zc_spread = merged_daily_df['futures_zs_zc_spread'].fillna(0)
soy_mean = merged_daily_df.groupby('region_id')['futures_zs_zc_spread'].rolling(60, min_periods=1).mean().reset_index(0, drop=True)
soy_std = merged_daily_df.groupby('region_id')['futures_zs_zc_spread'].rolling(60, min_periods=1).std().reset_index(0, drop=True).fillna(1.0)
soy_z = (zs_zc_spread - soy_mean) / (soy_std + 1e-6)
soy_mult = (1.0 + (0.3 * soy_z)).clip(0.8, 1.6).fillna(1.0)

# Gated Acreage Risks
merged_daily_df['climate_risk_ACREAGE_BATTLE_WET'] = (
    merged_daily_df['climate_risk_excess_precip_weighted'] * 
    soy_mult * 
    np.where(merged_daily_df['harvest_period'] == 'Planting', 1.0, 0.0)
)
merged_daily_df['climate_risk_ACREAGE_BATTLE_COLD'] = (
    merged_daily_df['climate_risk_unseasonably_cold_weighted'] * 
    soy_mult * 
    np.where(merged_daily_df['harvest_period'] == 'Planting', 1.0, 0.0)
)

# ------------------------------------------------------------------------------
# 6. COMPOUND, PHENO & SEASONALITY
# ------------------------------------------------------------------------------
print("Step 6: Compound, Pheno & Seasonality...")

# Compound Volatility
# We use SQUARED Volatility here to emphasize "Regime Change" over noise
heat = merged_daily_df['climate_risk_heat_stress_weighted']
drought = merged_daily_df['climate_risk_drought_weighted']
vol = merged_daily_df['futures_zc1_vol_20'].fillna(0)
merged_daily_df['climate_risk_COMPOUND_VOL_ADJ'] = (heat * drought) * (1 + (np.power(vol, 2) * 20))

# Seasonality
day_of_year = merged_daily_df['date_on'].dt.dayofyear
merged_daily_df['climate_risk_SEASON_sin'] = np.sin(2 * np.pi * day_of_year / 365.0)
merged_daily_df['climate_risk_SEASON_cos'] = np.cos(2 * np.pi * day_of_year / 365.0)

# Pheno
unique_phases = [p for p in merged_daily_df['harvest_period'].unique() if p != 'Unknown']
for risk_type in risk_categories:
    base_col = f'climate_risk_{risk_type}_weighted'
    for phase in unique_phases:
        phase_clean = "".join(x for x in phase if x.isalnum()).upper()
        feature_name = f'__pheno_{risk_type}_{phase_clean}'
        merged_daily_df[feature_name] = np.where(merged_daily_df['harvest_period'] == phase, merged_daily_df[base_col], 0)

# ------------------------------------------------------------------------------
# 7. DEEP MEMORY & ANOMALIES
# ------------------------------------------------------------------------------
print("Step 7: Engineering Deep Memory...")
merged_daily_df = merged_daily_df.sort_values(['region_id', 'date_on'])

deep_memory_risks = [
    'heat_stress', 'drought', 'COMPOUND_VOL_ADJ', 
    'heat_stress_MACRO_ADJ', 'drought_MACRO_ADJ',
    'heat_stress_POWER_BELT', 'drought_POWER_BELT',
    'ACREAGE_BATTLE_WET', 'ACREAGE_BATTLE_COLD'
]

for risk_type in deep_memory_risks:
    if 'MACRO_ADJ' in risk_type or 'POWER_BELT' in risk_type or 'ACREAGE' in risk_type:
        col = f'climate_risk_{risk_type}'
    elif risk_type == 'COMPOUND_VOL_ADJ':
        col = 'climate_risk_COMPOUND_VOL_ADJ'
    else:
        col = f'climate_risk_{risk_type}_weighted'
    
    # Chronic (60-day)
    merged_daily_df[f'__deep_{risk_type}_cum60'] = merged_daily_df.groupby('region_id')[col].rolling(60, min_periods=1).sum().reset_index(0, drop=True)
    # Acute (15-day)
    merged_daily_df[f'__acute_{risk_type}_cum15'] = merged_daily_df.groupby('region_id')[col].rolling(15, min_periods=1).sum().reset_index(0, drop=True)

# ------------------------------------------------------------------------------
# 8. AGGREGATION
# ------------------------------------------------------------------------------
print("Step 8: Aggregating to Country Level...")

country_agg_cols = {'percent_country_production': 'sum'}
for risk_type in risk_categories:
    country_agg_cols[f'climate_risk_{risk_type}_score'] = ['mean', 'max', 'std']
    country_agg_cols[f'climate_risk_{risk_type}_weighted'] = 'sum'

sum_cols = [c for c in merged_daily_df.columns if 
            c.startswith('__pheno_') or 
            'MACRO_ADJ' in c or 
            'POWER_BELT' in c or 
            'ACREAGE_BATTLE' in c or
            'COMPOUND_VOL_ADJ' in c]

for c in sum_cols: country_agg_cols[c] = 'sum'

mean_cols = [c for c in merged_daily_df.columns if c.startswith('__deep_') or c.startswith('__acute_')]
for c in mean_cols: country_agg_cols[c] = 'mean'

country_agg_cols['climate_risk_SEASON_sin'] = 'mean'
country_agg_cols['climate_risk_SEASON_cos'] = 'mean'

country_agg = merged_daily_df.groupby(['country_name', 'date_on']).agg(country_agg_cols).round(4)

# Flatten
new_cols = []
for c in country_agg.columns:
    if isinstance(c, tuple):
        base, stat = c
        if '__pheno_' in base:
            clean = base.replace('__pheno_', '').upper()
            name = f'climate_risk_country_PHENO_{clean}_sum'
        elif '__deep_' in base:
            clean = base.replace('__deep_', '').upper()
            name = f'climate_risk_country_DEEP_{clean}_{stat}'
        elif '__acute_' in base:
            clean = base.replace('__acute_', '').upper()
            name = f'climate_risk_country_ACUTE_{clean}_{stat}'
        elif 'MACRO_ADJ' in base or 'POWER_BELT' in base or 'COMPOUND' in base or 'ACREAGE' in base:
            name = f'{base}_{stat}'
        elif 'SEASON' in base:
             name = f'{base}' 
        else:
            clean = base.replace('climate_risk_', '')
            name = f'climate_risk_country_{clean}_{stat}'
        new_cols.append(name)
    else:
        new_cols.append(c)

country_agg.columns = new_cols
country_agg = country_agg.reset_index()

merged_daily_df = merged_daily_df.merge(country_agg, on=['country_name', 'date_on'], how='left')

# ------------------------------------------------------------------------------
# 9. FINAL CLEANUP & ORGANIZER LOGIC
# ------------------------------------------------------------------------------
print("Step 9: Organizer Logic & Final Cleanup...")
# We run this LAST so it calculates diffs on the fully engineered features if needed,
# or simply to generate the row count mask.

windows = [7, 14, 30]
for risk_type in risk_categories:
    score_col = f'climate_risk_{risk_type}_score'
    for window in windows:
        grouped = merged_daily_df.groupby('region_id')[score_col]
        merged_daily_df[f'climate_risk_{risk_type}_ma_{window}d'] = grouped.rolling(window, min_periods=1).mean().reset_index(0, drop=True)
        merged_daily_df[f'climate_risk_{risk_type}_max_{window}d'] = grouped.rolling(window, min_periods=1).max().reset_index(0, drop=True)
    grouped_diff = merged_daily_df.groupby('region_id')[score_col]
    merged_daily_df[f'climate_risk_{risk_type}_change_1d'] = grouped_diff.diff(1)
    merged_daily_df[f'climate_risk_{risk_type}_change_7d'] = grouped_diff.diff(7)
    merged_daily_df[f'climate_risk_{risk_type}_acceleration'] = merged_daily_df.groupby('region_id')[f'climate_risk_{risk_type}_change_1d'].diff(1)

cols = ['ID', 'date_on', 'country_name', 'region_name'] + \
       [c for c in merged_daily_df.columns if c.startswith('futures_')] + \
       [c for c in merged_daily_df.columns if c.startswith('climate_risk_')]

df_clean = merged_daily_df[cols].copy()
df_clean = df_clean.loc[:, ~df_clean.columns.str.startswith('__')]

fill_zero_cols = [c for c in df_clean.columns if 
                  '_PHENO_' in c or '_DEEP_' in c or '_ACUTE_' in c or 
                  'COMPOUND' in c or 'MACRO_ADJ' in c or 'POWER_BELT' in c or
                  'ACREAGE' in c or 'SEASON' in c]
df_clean[fill_zero_cols] = df_clean[fill_zero_cols].fillna(0)

df_clean = df_clean.dropna()

print(f"Final Row Count: {len(df_clean)}")
if len(df_clean) == 219161:
    print("✅ MATCH: Row count is strictly 219,161.")
else:
    print(f"❌ MISMATCH: Got {len(df_clean)} (Expected 219,161)")

df_clean.to_csv('submission_signal_sharpening.csv', index=False)