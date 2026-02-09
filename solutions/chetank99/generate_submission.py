import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# Load data
df = pd.read_csv(DATA_DIR / "corn_climate_risk_futures_daily_master.csv")
df['date_on'] = pd.to_datetime(df['date_on'])
market_share_df = pd.read_csv(DATA_DIR / "corn_regional_market_share.csv")

df = df.merge(
    market_share_df[['region_id', 'percent_country_production']],
    on='region_id',
    how='left'
)
df['percent_country_production'] = df['percent_country_production'].fillna(1.0)
df['month'] = df['date_on'].dt.month

# Drought score from climate_risk_cnt_* columns
drought_low = 'climate_risk_cnt_locations_drought_risk_low'
drought_med = 'climate_risk_cnt_locations_drought_risk_medium'
drought_high = 'climate_risk_cnt_locations_drought_risk_high'
total_locations = df[drought_low] + df[drought_med] + df[drought_high]
df['drought_score'] = (df[drought_med] + 2 * df[drought_high]) / (total_locations + 1e-6)
df = df.sort_values(['region_id', 'date_on']).reset_index(drop=True)

# Country aggregates
arg_daily = df[df['country_name'] == 'Argentina'].groupby('date_on')['drought_score'].mean().reset_index()
arg_daily.columns = ['date_on', 'arg_mean']
brazil_daily = df[df['country_name'] == 'Brazil'].groupby('date_on')['drought_score'].mean().reset_index()
brazil_daily.columns = ['date_on', 'brazil_mean']
global_daily = df.groupby('date_on')['drought_score'].mean().reset_index()
global_daily.columns = ['date_on', 'global_mean']

# 120-day backward-looking rolling mean
arg_daily['arg_post'] = arg_daily['arg_mean'].rolling(window=120, min_periods=30).mean()
brazil_daily['brazil_post'] = brazil_daily['brazil_mean'].rolling(window=120, min_periods=30).mean()
global_daily['global_post'] = global_daily['global_mean'].rolling(window=120, min_periods=30).mean()

df = df.merge(arg_daily[['date_on', 'arg_post']], on='date_on', how='left')
df = df.merge(brazil_daily[['date_on', 'brazil_post']], on='date_on', how='left')
df = df.merge(global_daily[['date_on', 'global_post']], on='date_on', how='left')

for col in ['arg_post', 'brazil_post', 'global_post']:
    df[col] = df[col].fillna(0)

# Ratio features
df['ratio'] = df['arg_post'] / (df['global_post'] + 1e-8)
df['ratio_sq'] = df['ratio'] ** 2
df['ratio_cube'] = df['ratio'] ** 3
df['sync'] = df['arg_post'] * df['brazil_post']

# Month-specific features
selected_features = []
for month in [1, 2, 11, 12]:
    suffix = f'_M{month:02d}'
    df[f'climate_risk_ratio{suffix}'] = np.where(df['month'] == month, df['ratio'], 0)
    df[f'climate_risk_ratio_sq{suffix}'] = np.where(df['month'] == month, df['ratio_sq'], 0)
    selected_features.extend([f'climate_risk_ratio{suffix}', f'climate_risk_ratio_sq{suffix}'])

df['climate_risk_ratio_cube_M01'] = np.where(df['month'] == 1, df['ratio_cube'], 0)
selected_features.append('climate_risk_ratio_cube_M01')

for month in [1, 2]:
    df[f'climate_risk_sync_M{month:02d}'] = np.where(df['month'] == month, df['sync'], 0)
    selected_features.append(f'climate_risk_sync_M{month:02d}')

# Valid IDs (replicating sample notebook's dropna logic)
risk_categories = ['heat_stress', 'unseasonably_cold', 'excess_precip', 'drought']
temp_df = df.copy()

for risk_type in risk_categories:
    low_col = f'climate_risk_cnt_locations_{risk_type}_risk_low'
    med_col = f'climate_risk_cnt_locations_{risk_type}_risk_medium'
    high_col = f'climate_risk_cnt_locations_{risk_type}_risk_high'
    total = temp_df[low_col] + temp_df[med_col] + temp_df[high_col]
    temp_df[f'risk_{risk_type}_score'] = (temp_df[med_col] + 2 * temp_df[high_col]) / (total + 1e-6)

temp_df = temp_df.sort_values(['region_id', 'date_on'])
for window in [7, 14, 30]:
    for risk_type in risk_categories:
        score_col = f'risk_{risk_type}_score'
        temp_df[f'risk_{risk_type}_ma_{window}d'] = (
            temp_df.groupby('region_id')[score_col]
            .rolling(window=window, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
        temp_df[f'risk_{risk_type}_max_{window}d'] = (
            temp_df.groupby('region_id')[score_col]
            .rolling(window=window, min_periods=1)
            .max()
            .reset_index(level=0, drop=True)
        )

for risk_type in risk_categories:
    score_col = f'risk_{risk_type}_score'
    temp_df[f'risk_{risk_type}_change_1d'] = temp_df.groupby('region_id')[score_col].diff(1)
    temp_df[f'risk_{risk_type}_change_7d'] = temp_df.groupby('region_id')[score_col].diff(7)
    temp_df[f'risk_{risk_type}_accel'] = temp_df.groupby('region_id')[f'risk_{risk_type}_change_1d'].diff(1)

for risk_type in risk_categories:
    score_col = f'risk_{risk_type}_score'
    country_agg = temp_df.groupby(['country_name', 'date_on']).agg({
        score_col: ['mean', 'max', 'std']
    }).round(4)
    country_agg.columns = [f'country_{risk_type}_{c[1]}' for c in country_agg.columns]
    country_agg = country_agg.reset_index()
    temp_df = temp_df.merge(country_agg, on=['country_name', 'date_on'], how='left')

valid_ids = set(temp_df.dropna()['ID'].tolist())

# Generate submission
futures_cols = [c for c in df.columns if c.startswith('futures_')]
required_cols = ['ID', 'date_on', 'country_name', 'region_name'] + futures_cols + selected_features
submission_df = df[required_cols].copy()

for feat in selected_features:
    submission_df[feat] = submission_df[feat].fillna(0.0)

submission_df = submission_df[submission_df['ID'].isin(valid_ids)]

output_path = OUTPUT_DIR / "submission.csv"
submission_df.to_csv(output_path, index=False)

print(f"features: {len(selected_features)}")
print(f"rows: {len(submission_df):,}")
print(f"file saved: {output_path}")
