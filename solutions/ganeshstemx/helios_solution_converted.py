#!/usr/bin/env python
# coding: utf-8

# # Helios Corn Futures Climate Challenge

# In[2]:


import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

pd.options.display.max_columns = 100
pd.options.display.max_rows = 100

print("="*80)
print(" HELIOS CORN FUTURES CLIMATE CHALLENGE")
print("="*80)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Risk categories and levels
RISK_CATEGORIES = ['heat_stress', 'unseasonably_cold', 'excess_precip', 'drought']
RISK_LEVELS = ['low', 'medium', 'high']

# Feature analysis settings
SIGNIFICANCE_THRESHOLD = 0.6
TOP_N_FEATURES = 5  # Use top N features for scoring

# Feature selection strategy: 'sig_count', 'max_corr', 'avg_sig_corr', or 'weighted'
FEATURE_SELECTION_STRATEGY = 'sig_count'

# Correlation thresholds for feature removal
CORRELATION_THRESHOLD_GENERAL = 0.98  # Remove features with >= 98% correlation
CORRELATION_THRESHOLD_SPECIFIC = 0.70  # Remove features correlated with specific features in FEATURES_TO_REMOVE list

# Specific features to remove (and their >= 70% correlated counterparts)
FEATURES_TO_REMOVE = [
    "climate_risk_country_quartile_std_excess_precip_risk_medium",
    "climate_risk_quartile_agg_excess_precip_risk_medium_std",
    "climate_risk_quartile_agg_heat_stress_risk_medium_mean"
]


# In[3]:


# Data paths
DATA_PATH = './'
OUTPUT_PATH = './'


# In[4]:


# ============================================================================
# FEATURE ENGINEERING FUNCTIONS
# ============================================================================

def quarter_to_date(year, quarter):
    """Convert year-quarter to the first date of that quarter."""
    month = (quarter - 1) * 3 + 1  # Q1=1, Q2=4, Q3=7, Q4=10
    return pd.Timestamp(year=year, month=month, day=1)


def create_time_bins(df, date_col, quarters_per_bin, bin_name,
                     start_year, start_quarter, end_year, end_quarter, verbose=True):
    """
    Create time-based bins for temporal grouping.

    Parameters:
    -----------
    df : DataFrame - Input dataframe
    date_col : str - Name of the date column
    quarters_per_bin : int - Number of quarters per bin
    bin_name : str - Name for the bin column (e.g., 'decile', 'tredecile')
    start_year, start_quarter : int - Start of date range
    end_year, end_quarter : int - End of date range
    verbose : bool - Whether to print configuration details

    Returns:
    --------
    Series with bin assignments
    """
    total_quarters = (end_year - start_year) * 4 + (end_quarter - start_quarter) + 1
    num_bins = int(np.ceil(total_quarters / quarters_per_bin))

    if verbose:
        print(f"\n{bin_name.upper()} Configuration:")
        print(f"Quarters per bin: {quarters_per_bin}")
        print(f"Number of bins: {num_bins}")

    # Generate bin boundaries
    bin_boundaries = []
    current_year, current_quarter = start_year, start_quarter
    bin_boundaries.append(quarter_to_date(current_year, current_quarter))

    for _ in range(num_bins):
        new_quarter = current_quarter + quarters_per_bin
        new_year = current_year + (new_quarter - 1) // 4
        new_quarter = ((new_quarter - 1) % 4) + 1
        bin_boundaries.append(quarter_to_date(new_year, new_quarter))
        current_year, current_quarter = new_year, new_quarter

    # Extend last boundary to cover remaining dates
    bin_boundaries[-1] = pd.Timestamp(year=end_year + 1, month=1, day=1)

    if verbose:
        print(f"Bin boundaries:")
        for i in range(num_bins):
            print(f"Bin {i}: {bin_boundaries[i].date()} to {bin_boundaries[i+1].date()}")

    return pd.cut(df[date_col], bins=bin_boundaries, labels=False, include_lowest=True).fillna(0).astype(int)


def create_groupby_agg_features(df, source_cols, groupby_cols, agg_funcs,
                                feature_prefix, created_features_list):
    """
    Create groupby aggregation features for climate risk data.

    Parameters:
    -----------
    df : DataFrame - Input dataframe (modified in place)
    source_cols : list - Source columns to aggregate (climate_risk_cnt_locations_*)
    groupby_cols : list - Columns to group by
    agg_funcs : list - Aggregation functions ['max', 'mean', 'std', etc.]
    feature_prefix : str - Prefix for feature names (e.g., 'climate_risk_groupby_date_decile')
    created_features_list : list - List to append created feature names to

    Returns:
    --------
    int - Number of features created
    """
    feature_count = 0
    for source_col in source_cols:
        # Extract risk type name from source column
        risk_name = source_col.replace('climate_risk_cnt_locations_', '')

        for agg_func in agg_funcs:
            feat_name = f'{feature_prefix}_{risk_name}_{agg_func}'
            df[feat_name] = df.groupby(groupby_cols)[source_col].transform(agg_func)
            df[feat_name] = df[feat_name].replace([np.inf, -np.inf], 0).fillna(0)
            created_features_list.append(feat_name)
            feature_count += 1

    return feature_count


def create_spatial_std_features(df, source_cols, date_col, feature_prefix, created_features_list):
    """
    Create spatial standard deviation features (std across regions on same date).

    Parameters:
    -----------
    df : DataFrame - Input dataframe (modified in place)
    source_cols : list - Source columns to compute std for
    date_col : str - Date column name for grouping
    feature_prefix : str - Prefix for feature names (e.g., 'climate_risk_spatial_std_date_decile')
    created_features_list : list - List to append created feature names to

    Returns:
    --------
    int - Number of features created
    """
    feature_count = 0
    for source_col in source_cols:
        risk_name = source_col.replace('climate_risk_cnt_locations_', '')
        feat_name = f'{feature_prefix}_{risk_name}'
        df[feat_name] = df.groupby(date_col)[source_col].transform('std')
        df[feat_name] = df[feat_name].replace([np.inf, -np.inf], 0).fillna(0)
        created_features_list.append(feat_name)
        feature_count += 1

    return feature_count


def create_groupby_std_features(df, source_cols, groupby_cols, feature_prefix, created_features_list):
    """
    Create groupby standard deviation features (std within specified groups).

    Parameters:
    -----------
    df : DataFrame - Input dataframe (modified in place)
    source_cols : list - Source columns to compute std for
    groupby_cols : list - Columns to group by (e.g., ['country_name', 'time_bin'])
    feature_prefix : str - Prefix for feature names
    created_features_list : list - List to append created feature names to

    Returns:
    --------
    int - Number of features created
    """
    feature_count = 0
    for source_col in source_cols:
        risk_name = source_col.replace('climate_risk_cnt_locations_', '')
        feat_name = f'{feature_prefix}_{risk_name}'
        df[feat_name] = df.groupby(groupby_cols)[source_col].transform('std')
        df[feat_name] = df[feat_name].replace([np.inf, -np.inf], 0).fillna(0)
        created_features_list.append(feat_name)
        feature_count += 1

    return feature_count


def create_categorical_encoding(df, source_col, feature_name, created_features_list):
    """
    Create encoded feature from categorical column.

    Parameters:
    -----------
    df : DataFrame - Input dataframe (modified in place)
    source_col : str - Source categorical column
    feature_name : str - Name for the encoded feature
    created_features_list : list - List to append created feature names to

    Returns:
    --------
    int - Number of features created (always 1)
    """
    category_map = {val: idx for idx, val in enumerate(df[source_col].unique())}
    df[feature_name] = df[source_col].map(category_map)
    created_features_list.append(feature_name)
    return 1


def create_risk_score_features(df, risk_categories, feature_prefix, created_features_list,
                                weights=(1, 2, 3)):
    """
    Create weighted risk score features combining low/medium/high counts.

    Score = (low * w1 + medium * w2 + high * w3) / (low + medium + high + epsilon)

    Parameters:
    -----------
    df : DataFrame - Input dataframe (modified in place)
    risk_categories : list - Risk category names (e.g., ['heat_stress', 'drought'])
    feature_prefix : str - Prefix for feature names
    created_features_list : list - List to append created feature names to
    weights : tuple - Weights for (low, medium, high) risk levels

    Returns:
    --------
    int - Number of features created
    """
    feature_count = 0
    w_low, w_med, w_high = weights

    for risk_type in risk_categories:
        low_col = f'climate_risk_cnt_locations_{risk_type}_risk_low'
        med_col = f'climate_risk_cnt_locations_{risk_type}_risk_medium'
        high_col = f'climate_risk_cnt_locations_{risk_type}_risk_high'

        # Weighted risk score
        total = df[low_col] + df[med_col] + df[high_col]
        feat_name = f'{feature_prefix}_{risk_type}_score'
        df[feat_name] = (df[low_col] * w_low + df[med_col] * w_med + df[high_col] * w_high) / (total + 1e-6)
        df[feat_name] = df[feat_name].replace([np.inf, -np.inf], 0).fillna(0)
        created_features_list.append(feat_name)
        feature_count += 1

        # High risk ratio
        feat_name = f'{feature_prefix}_{risk_type}_high_ratio'
        df[feat_name] = df[high_col] / (total + 1e-6)
        df[feat_name] = df[feat_name].replace([np.inf, -np.inf], 0).fillna(0)
        created_features_list.append(feat_name)
        feature_count += 1

    return feature_count


def create_cross_risk_features(df, risk_pairs, feature_prefix, created_features_list):
    """
    Create cross-risk interaction features (compound stress indicators).

    Parameters:
    -----------
    df : DataFrame - Input dataframe (modified in place)
    risk_pairs : list of tuples - Pairs of risk types to combine, e.g., [('heat_stress', 'drought')]
    feature_prefix : str - Prefix for feature names
    created_features_list : list - List to append created feature names to

    Returns:
    --------
    int - Number of features created
    """
    feature_count = 0

    for risk1, risk2 in risk_pairs:
        for level in ['high', 'medium']:
            col1 = f'climate_risk_cnt_locations_{risk1}_risk_{level}'
            col2 = f'climate_risk_cnt_locations_{risk2}_risk_{level}'

            # Product interaction (compound stress)
            feat_name = f'{feature_prefix}_{risk1}_{risk2}_{level}_product'
            df[feat_name] = df[col1] * df[col2]
            df[feat_name] = df[feat_name].replace([np.inf, -np.inf], 0).fillna(0)
            created_features_list.append(feat_name)
            feature_count += 1

            # Sum interaction (combined stress)
            feat_name = f'{feature_prefix}_{risk1}_{risk2}_{level}_sum'
            df[feat_name] = df[col1] + df[col2]
            df[feat_name] = df[feat_name].replace([np.inf, -np.inf], 0).fillna(0)
            created_features_list.append(feat_name)
            feature_count += 1

            # Max interaction (dominant stress)
            feat_name = f'{feature_prefix}_{risk1}_{risk2}_{level}_max'
            df[feat_name] = df[[col1, col2]].max(axis=1)
            df[feat_name] = df[feat_name].replace([np.inf, -np.inf], 0).fillna(0)
            created_features_list.append(feat_name)
            feature_count += 1

    return feature_count


def create_country_agg_features(df, source_cols, date_col, country_col, agg_funcs,
                                 feature_prefix, created_features_list):
    """
    Create country-level aggregation features (aggregate regional risks to country level).

    Parameters:
    -----------
    df : DataFrame - Input dataframe (modified in place)
    source_cols : list - Source columns to aggregate
    date_col : str - Date column name
    country_col : str - Country column name
    agg_funcs : list - Aggregation functions ['mean', 'max', 'sum', 'std']
    feature_prefix : str - Prefix for feature names
    created_features_list : list - List to append created feature names to

    Returns:
    --------
    int - Number of features created
    """
    feature_count = 0

    for source_col in source_cols:
        risk_name = source_col.replace('climate_risk_cnt_locations_', '')

        for agg_func in agg_funcs:
            feat_name = f'{feature_prefix}_{risk_name}_{agg_func}'
            df[feat_name] = df.groupby([country_col, date_col])[source_col].transform(agg_func)
            df[feat_name] = df[feat_name].replace([np.inf, -np.inf], 0).fillna(0)
            created_features_list.append(feat_name)
            feature_count += 1

    return feature_count


# In[5]:


# Load data
print("\nLoading data...")
df = pd.read_csv(f'{DATA_PATH}corn_climate_risk_futures_daily_master.csv')
df['date_on'] = pd.to_datetime(df['date_on'])
market_share_df = pd.read_csv(f'{DATA_PATH}corn_regional_market_share.csv')

print(f"Dataset: {len(df):,} rows")
print(f"Date range: {df['date_on'].min()} to {df['date_on'].max()}")
print(f"Countries: {df['country_name'].nunique()}")
print(f"Regions: {df['region_name'].nunique()}")

# Create working copy
merged_df = df.copy()

# Add basic temporal features
merged_df['year'] = merged_df['date_on'].dt.year
merged_df['month'] = merged_df['date_on'].dt.month
merged_df['day_of_year'] = merged_df['date_on'].dt.dayofyear
merged_df['quarter'] = merged_df['date_on'].dt.quarter
merged_df['week_of_year'] = merged_df['date_on'].dt.isocalendar().week

# Merge market share
merged_df = merged_df.merge(
    market_share_df[['region_id', 'percent_country_production']],
    on='region_id', how='left'
)
merged_df['percent_country_production'] = merged_df['percent_country_production'].fillna(1.0)


# In[6]:


# Track all created features
ALL_NEW_FEATURES = []

print(f"\nBase setup complete: {len(merged_df):,} rows")

print("\n" + "="*80)
print(" CREATING CLIMATE RISK FEATURES")
print("="*80)

# Get all climate risk count columns
climate_cols = [c for c in merged_df.columns if c.startswith('climate_risk_cnt_locations_')]
print(f"\nBase climate risk columns: {len(climate_cols)}")

# ============================================================================
# TIME-BASED BINNING CONFIGURATION
# ============================================================================
# Derive date range dynamically from the dataset
_min_date = merged_df['date_on'].min()
_max_date = merged_df['date_on'].max()

_start_year = _min_date.year
_start_quarter = (_min_date.month - 1) // 3 + 1
_end_year = _max_date.year
_end_quarter = (_max_date.month - 1) // 3 + 1

total_quarters = (_end_year - _start_year) * 4 + (_end_quarter - _start_quarter) + 1
print(f"\nData date range: {_min_date.date()} to {_max_date.date()}")
print(f"Start: {_start_year} Q{_start_quarter}, End: {_end_year} Q{_end_quarter}")
print(f"Total quarters in data: {total_quarters}")

TIME_BIN_CONFIG = {
    'start_year': _start_year,
    'start_quarter': _start_quarter,
    'end_year': _end_year,
    'end_quarter': _end_quarter,
    # Use ceiling division to produce exactly N bins (last bin may be smaller)
    'quarters_per_tertile': max(1, -(-total_quarters // 3)),       # → 3 bins
    'quarters_per_quartile': max(1, -(-total_quarters // 4)),      # → 4 bins
    'quarters_per_quintile': max(1, -(-total_quarters // 5)),      # → 5 bins
    'quarters_per_sextile': max(1, -(-total_quarters // 6)),       # → 6 bins
    'quarters_per_octile': max(1, -(-total_quarters // 8)),        # → 8 bins
    'quarters_per_decile': max(1, -(-total_quarters // 10)),       # → 10 bins
    'quarters_per_tredecile': max(1, -(-total_quarters // 13)),    # → 13 bins
    'quarters_per_vigintile': max(1, -(-total_quarters // 20)),    # → 20 bins
}


# In[7]:


# ============================================================================
# CREATE ALL TIME BINS
# ============================================================================
print("\nCreating time bins for all -ile divisions.")

BIN_TYPES = [
    "tertile",
    "quartile",
    "quintile",
    "sextile",
    "octile",
    "decile",
    "tredecile",
    "vigintile",
]

COMMON_KWARGS = dict(
    start_year=TIME_BIN_CONFIG["start_year"],
    start_quarter=TIME_BIN_CONFIG["start_quarter"],
    end_year=TIME_BIN_CONFIG["end_year"],
    end_quarter=TIME_BIN_CONFIG["end_quarter"],
)

for bin_name in BIN_TYPES:
    merged_df[f"climate_risk_time_bin_{bin_name}"] = create_time_bins(
        merged_df,
        "date_on",
        quarters_per_bin=TIME_BIN_CONFIG[f"quarters_per_{bin_name}"],
        bin_name=bin_name,
        **COMMON_KWARGS,
    )


# In[8]:


# DATE DECILE GROUPBY FEATURES
# ============================================================================
print("\nDate Decile Groupby Features")
feature_count = 0

# Define high-impact risk features for decile aggregation
HIGH_IMPACT_RISK_COLS = [
    'climate_risk_cnt_locations_heat_stress_risk_high',
    'climate_risk_cnt_locations_drought_risk_high',
    'climate_risk_cnt_locations_drought_risk_medium',
    'climate_risk_cnt_locations_excess_precip_risk_medium',
    'climate_risk_cnt_locations_heat_stress_risk_medium'
]

# Create groupby aggregation features using generic function
feature_count += create_groupby_agg_features(
    df=merged_df,
    source_cols=HIGH_IMPACT_RISK_COLS,
    groupby_cols=['climate_risk_time_bin_decile', 'country_name'],
    agg_funcs=['max', 'mean'],
    feature_prefix='climate_risk_decile_agg',
    created_features_list=ALL_NEW_FEATURES
)


# Create spatial std features (variation across regions on same date)
print("Creating spatial std features...")
feature_count += create_spatial_std_features(
    df=merged_df,
    source_cols=HIGH_IMPACT_RISK_COLS,
    date_col='date_on',
    feature_prefix='climate_risk_spatial_std',
    created_features_list=ALL_NEW_FEATURES
)

# Create country-decile std features (variation within country-period groups)
print("Creating country-decile std features...")
feature_count += create_groupby_std_features(
    df=merged_df,
    source_cols=HIGH_IMPACT_RISK_COLS,
    groupby_cols=['country_name', 'climate_risk_time_bin_decile'],
    feature_prefix='climate_risk_country_decile_std',
    created_features_list=ALL_NEW_FEATURES
)

print(f"Created {feature_count} decile-based features")


# In[9]:


# LOW-RISK LEVEL FEATURES (Cold Stress and Low-Severity Risks)
# ============================================================================
print("\nLow-Risk Level Features")
feature_count = 0

# Define low-severity and cold stress risk columns
LOW_SEVERITY_RISK_COLS = [
    'climate_risk_cnt_locations_unseasonably_cold_risk_high',
    'climate_risk_cnt_locations_unseasonably_cold_risk_medium',
    'climate_risk_cnt_locations_unseasonably_cold_risk_low',
    'climate_risk_cnt_locations_heat_stress_risk_low',
    'climate_risk_cnt_locations_drought_risk_low',
    'climate_risk_cnt_locations_excess_precip_risk_low',
]

# Decile-based aggregations (region-level)
print("Creating decile aggregation features...")
feature_count += create_groupby_agg_features(
    df=merged_df,
    source_cols=LOW_SEVERITY_RISK_COLS,
    groupby_cols=['country_name', 'region_name', 'climate_risk_time_bin_decile'],
    agg_funcs=['max', 'mean'],
    feature_prefix='climate_risk_decile_region_agg',
    created_features_list=ALL_NEW_FEATURES
)

# Spatial std features for low-severity risks
print("Creating spatial std features...")
feature_count += create_spatial_std_features(
    df=merged_df,
    source_cols=LOW_SEVERITY_RISK_COLS,
    date_col='date_on',
    feature_prefix='climate_risk_low_severity_spatial_std',
    created_features_list=ALL_NEW_FEATURES
)

# Country-decile std features for low-severity risks
print("Creating country-decile std features...")
feature_count += create_groupby_std_features(
    df=merged_df,
    source_cols=LOW_SEVERITY_RISK_COLS,
    groupby_cols=['country_name', 'climate_risk_time_bin_decile'],
    feature_prefix='climate_risk_low_severity_country_decile_std',
    created_features_list=ALL_NEW_FEATURES
)

print(f"Created {feature_count} low-severity risk features")


# In[10]:


# HIGH-IMPACT TREDECILE FEATURES
# ============================================================================
print("\nHigh-Impact Tredecile Features")
feature_count = 0

# Define medium and high severity risk columns for tredecile analysis
MEDIUM_HIGH_SEVERITY_RISK_COLS = [
    # Heat Stress Risk
    'climate_risk_cnt_locations_heat_stress_risk_medium',
    'climate_risk_cnt_locations_heat_stress_risk_high',
    # Excess Precipitation Risk
    'climate_risk_cnt_locations_excess_precip_risk_medium',
    'climate_risk_cnt_locations_excess_precip_risk_high',
    # Drought Risk
    'climate_risk_cnt_locations_drought_risk_medium',
    'climate_risk_cnt_locations_drought_risk_high'
]

# Tredecile-based country-weekly aggregations
print("Creating tredecile country-weekly aggregation features...")
feature_count += create_groupby_agg_features(
    df=merged_df,
    source_cols=MEDIUM_HIGH_SEVERITY_RISK_COLS,
    groupby_cols=['climate_risk_time_bin_tredecile', 'country_name'],
    agg_funcs=['max', 'min', 'std', 'var', 'mean'],
    feature_prefix='climate_risk_tredecile_country_weekly_agg',
    created_features_list=ALL_NEW_FEATURES
)

print(f"Created {feature_count} high-impact tredecile features")


# In[11]:


# CATEGORICAL ENCODED FEATURES
# ============================================================================
print("\nCategorical Encoded Features")
feature_count = 0

# Encode harvest period (categorical → numerical)
print("Creating harvest period encoding...")
feature_count += create_categorical_encoding(
    df=merged_df,
    source_col='harvest_period',
    feature_name='climate_risk_harvest_period_encoded',
    created_features_list=ALL_NEW_FEATURES
)

print(f"Created {feature_count} categorical encoded features")


# In[12]:


# RISK SCORE FEATURES
# ============================================================================
print("\nRisk Score Features")
feature_count = 0

# Create weighted risk scores for all risk categories
print("Creating risk score features...")
feature_count += create_risk_score_features(
    df=merged_df,
    risk_categories=RISK_CATEGORIES,
    feature_prefix='climate_risk',
    created_features_list=ALL_NEW_FEATURES,
    weights=(1, 2, 3)  # low=1, medium=2, high=3
)

print(f"Created {feature_count} risk score features")


# In[13]:


# CROSS-RISK INTERACTION FEATURES
# ============================================================================
print("\nCross-Risk Interaction Features")
feature_count = 0

# Define meaningful risk pairs (compound stress scenarios)
RISK_PAIRS = [
    ('heat_stress', 'drought'),           # Hot + dry = severe crop stress
    ('heat_stress', 'excess_precip'),     # Hot + wet = disease risk
    ('unseasonably_cold', 'excess_precip'), # Cold + wet = frost/flooding
    ('drought', 'heat_stress'),           # Already covered above, skip
]
# Remove duplicate pair
RISK_PAIRS = [
    ('heat_stress', 'drought'),
    ('heat_stress', 'excess_precip'),
    ('unseasonably_cold', 'excess_precip'),
]

print("Creating cross-risk interaction features...")
feature_count += create_cross_risk_features(
    df=merged_df,
    risk_pairs=RISK_PAIRS,
    feature_prefix='climate_risk_interaction',
    created_features_list=ALL_NEW_FEATURES
)

print(f"Created {feature_count} cross-risk interaction features")


# In[14]:


# COUNTRY-LEVEL AGGREGATION FEATURES
# ============================================================================
print("\nCountry-Level Aggregation Features")
feature_count = 0

# Select key risk columns for country-level aggregation
COUNTRY_AGG_RISK_COLS = [
    'climate_risk_cnt_locations_heat_stress_risk_high',
    'climate_risk_cnt_locations_drought_risk_high',
    'climate_risk_cnt_locations_excess_precip_risk_high',
    'climate_risk_cnt_locations_unseasonably_cold_risk_high',
]

print("Creating country-level aggregation features...")
feature_count += create_country_agg_features(
    df=merged_df,
    source_cols=COUNTRY_AGG_RISK_COLS,
    date_col='date_on',
    country_col='country_name',
    agg_funcs=['mean', 'max', 'sum'],
    feature_prefix='climate_risk_country_daily',
    created_features_list=ALL_NEW_FEATURES
)

print(f"Created {feature_count} country-level aggregation features")


# In[15]:


# QUANTILE-BASED FEATURES
# ============================================================================
quantile_config = {
    "tertile": "climate_risk_time_bin_tertile",
    "quartile": "climate_risk_time_bin_quartile",
    "quintile": "climate_risk_time_bin_quintile",
    "sextile": "climate_risk_time_bin_sextile",
    "octile": "climate_risk_time_bin_octile",
    "decile": "climate_risk_time_bin_decile",
    "tredecile": "climate_risk_time_bin_tredecile",
    "vigintile": "climate_risk_time_bin_vigintile",
}


print("\nQuantile-Based Features")
feature_count = 0

for quantile_name, quantile_col in quantile_config.items():

    print(f"\n Creating {quantile_name}-based features...")

    # Aggregation features
    feature_count += create_groupby_agg_features(
        df=merged_df,
        source_cols=HIGH_IMPACT_RISK_COLS,
        groupby_cols=[quantile_col, 'country_name'],
        agg_funcs=['max', 'mean', 'std'],
        feature_prefix=f'climate_risk_{quantile_name}_agg',
        created_features_list=ALL_NEW_FEATURES
    )

    # Country–quantile standard deviation features
    feature_count += create_groupby_std_features(
        df=merged_df,
        source_cols=HIGH_IMPACT_RISK_COLS,
        groupby_cols=['country_name', quantile_col],
        feature_prefix=f'climate_risk_country_{quantile_name}_std',
        created_features_list=ALL_NEW_FEATURES
    )

    print(f" Completed {quantile_name} features")

print(f"Created {feature_count} quartile-based features")


# In[16]:


# COUNTRY-RELATIVE BINNING FEATURES
# ============================================================================
print("\n Country-Relative Binning Features")
feature_count = 0

def create_country_relative_bins(df, source_cols, country_col, date_col, feature_prefix,
                                  created_features_list):
    """
    Create bins relative to country-level distribution on same date.
    """
    feat_count = 0

    for source_col in source_cols:
        risk_name = source_col.replace('climate_risk_cnt_locations_', '')
        feat_name = f'{feature_prefix}_country_rel_bin_{risk_name}'

        # Calculate country-date mean and std
        country_mean = df.groupby([country_col, date_col])[source_col].transform('mean')
        country_std = df.groupby([country_col, date_col])[source_col].transform('std')

        # Relative deviation from country mean
        relative_val = (df[source_col] - country_mean) / (country_std + 1e-6)

        # Bin the relative values
        bins = [-np.inf, -2, -1, 1, 2, np.inf]
        df[feat_name] = pd.cut(relative_val, bins=bins, labels=[0, 1, 2, 3, 4])
        df[feat_name] = df[feat_name].fillna(2).astype(int)
        created_features_list.append(feat_name)
        feat_count += 1

    return feat_count

# Country-relative bins for high-impact risks
print("Creating country-relative bin features...")
feature_count += create_country_relative_bins(
    df=merged_df,
    source_cols=HIGH_IMPACT_RISK_COLS,
    country_col='country_name',
    date_col='date_on',
    feature_prefix='climate_risk',
    created_features_list=ALL_NEW_FEATURES
)

print(f"Created {feature_count} country-relative bin features")


# In[17]:


#  COUNTRY TIME-BIN STD FEATURES
# ============================================================================

print("\n Enhanced Country Time-Bin Std Features")
feature_count = 0

# Focus on drought and heat stress
DROUGHT_HEAT_COLS = [
    'climate_risk_cnt_locations_drought_risk_high',
    'climate_risk_cnt_locations_drought_risk_medium',
    'climate_risk_cnt_locations_heat_stress_risk_high',
    'climate_risk_cnt_locations_heat_stress_risk_medium',
]

print("\n Enhanced Country Time-Bin Std Features")
feature_count = 0

for quantile_name, quantile_col in quantile_config.items():

    print(f"Creating country-{quantile_name} std for drought/heat...")

    feature_count += create_groupby_std_features(
        df=merged_df,
        source_cols=DROUGHT_HEAT_COLS,
        groupby_cols=['country_name', quantile_col],
        feature_prefix=f'climate_risk_country_{quantile_name}_std',
        created_features_list=ALL_NEW_FEATURES
    )

print(f"Created {feature_count} enhanced country time-bin std features")


# In[18]:


#  REGION-WEIGHTED RISK FEATURES
# ============================================================================
# Weight climate risk by regional production importance
print("\n Region-Weighted Risk Features")
feature_count = 0

def create_production_weighted_features(df, source_cols, weight_col, feature_prefix,
                                        created_features_list):
    """
    Create features weighted by regional production importance.

    Parameters:
    -----------
    df : DataFrame - Input dataframe (modified in place)
    source_cols : list - Source columns to weight
    weight_col : str - Column containing weights (e.g., percent_country_production)
    feature_prefix : str - Prefix for feature names
    created_features_list : list - List to append created feature names to

    Returns:
    --------
    int - Number of features created
    """
    feat_count = 0

    for source_col in source_cols:
        risk_name = source_col.replace('climate_risk_cnt_locations_', '')

        # Production-weighted risk
        feat_name = f'{feature_prefix}_weighted_{risk_name}'
        df[feat_name] = df[source_col] * (df[weight_col] / 100)
        df[feat_name] = df[feat_name].replace([np.inf, -np.inf], 0).fillna(0)
        created_features_list.append(feat_name)
        feat_count += 1

    return feat_count

# Create production-weighted features for high-impact risks
print("Creating production-weighted risk features...")
feature_count += create_production_weighted_features(
    df=merged_df,
    source_cols=HIGH_IMPACT_RISK_COLS,
    weight_col='percent_country_production',
    feature_prefix='climate_risk',
    created_features_list=ALL_NEW_FEATURES
)

# Aggregate weighted features by country-time bin
print("Creating weighted aggregations by country-quintile...")
weighted_cols = [f'climate_risk_weighted_{c.replace("climate_risk_cnt_locations_", "")}'
                 for c in HIGH_IMPACT_RISK_COLS]

print("\nQuantile-Based Weighted Risk Aggregations")
feature_count = 0

valid_weighted_cols = [c for c in weighted_cols if c in merged_df.columns]

for quantile_name, quantile_col in quantile_config.items():

    print(f"Creating {quantile_name} weighted risk aggregation features...")

    feature_count += create_groupby_agg_features(
        df=merged_df,
        source_cols=valid_weighted_cols,
        groupby_cols=['country_name', quantile_col],
        agg_funcs=['mean', 'max', 'min', 'std', 'var', 'sum'],
        feature_prefix=f'climate_risk_{quantile_name}_weighted_agg',
        created_features_list=ALL_NEW_FEATURES
    )

print(f"Created {feature_count} quantile-based weighted risk features")


# In[19]:


#  SEASONAL AGGREGATION FEATURES
# ============================================================================
# Capture seasonal patterns in climate risk (no lagging, just groupby month/quarter)
print("\n Seasonal Aggregation Features")
feature_count = 0

# Month-based aggregations for drought/heat (seasonal patterns)
print("Creating month-country aggregations...")
feature_count += create_groupby_agg_features(
    df=merged_df,
    source_cols=DROUGHT_HEAT_COLS,
    groupby_cols=['country_name', 'month'],
    agg_funcs=['mean', 'max', 'min', 'std', 'var', 'sum'],
    feature_prefix='climate_risk_monthly_agg',
    created_features_list=ALL_NEW_FEATURES
)

# Quarter-based aggregations
print("Creating quarter-country aggregations...")
feature_count += create_groupby_agg_features(
    df=merged_df,
    source_cols=DROUGHT_HEAT_COLS,
    groupby_cols=['country_name', 'quarter'],
    agg_funcs=['mean', 'max', 'min', 'std', 'var', 'sum'],
    feature_prefix='climate_risk_quarterly_agg',
    created_features_list=ALL_NEW_FEATURES
)


print(f"Created {feature_count} seasonal aggregation features")


# In[20]:


# RISK RATIO AND PROPORTION FEATURES
# ============================================================================
# Capture relative intensity of different risk levels
print("\n Risk Ratio and Proportion Features")
feature_count = 0

def create_risk_ratio_features(df, risk_categories, feature_prefix, created_features_list):
    """
    Create ratio features between different risk levels.

    Parameters:
    -----------
    df : DataFrame - Input dataframe (modified in place)
    risk_categories : list - Risk category names
    feature_prefix : str - Prefix for feature names
    created_features_list : list - List to append created feature names to

    Returns:
    --------
    int - Number of features created
    """
    feat_count = 0

    for risk_type in risk_categories:
        low_col = f'climate_risk_cnt_locations_{risk_type}_risk_low'
        med_col = f'climate_risk_cnt_locations_{risk_type}_risk_medium'
        high_col = f'climate_risk_cnt_locations_{risk_type}_risk_high'

        total = df[low_col] + df[med_col] + df[high_col] + 1e-6

        # High to medium ratio
        feat_name = f'{feature_prefix}_{risk_type}_high_med_ratio'
        df[feat_name] = df[high_col] / (df[med_col] + 1e-6)
        df[feat_name] = df[feat_name].replace([np.inf, -np.inf], 0).fillna(0).clip(0, 100)
        created_features_list.append(feat_name)
        feat_count += 1

        # High proportion of total
        feat_name = f'{feature_prefix}_{risk_type}_high_proportion'
        df[feat_name] = df[high_col] / total
        df[feat_name] = df[feat_name].replace([np.inf, -np.inf], 0).fillna(0)
        created_features_list.append(feat_name)
        feat_count += 1

        # Medium + High proportion (elevated risk)
        feat_name = f'{feature_prefix}_{risk_type}_elevated_proportion'
        df[feat_name] = (df[med_col] + df[high_col]) / total
        df[feat_name] = df[feat_name].replace([np.inf, -np.inf], 0).fillna(0)
        created_features_list.append(feat_name)
        feat_count += 1

        # High dominance (high vs low+med)
        feat_name = f'{feature_prefix}_{risk_type}_high_dominance'
        df[feat_name] = df[high_col] / (df[low_col] + df[med_col] + 1e-6)
        df[feat_name] = df[feat_name].replace([np.inf, -np.inf], 0).fillna(0).clip(0, 100)
        created_features_list.append(feat_name)
        feat_count += 1

    return feat_count

# Create risk ratio features for all categories
print("Creating risk ratio features...")
feature_count += create_risk_ratio_features(
    df=merged_df,
    risk_categories=RISK_CATEGORIES,
    feature_prefix='climate_risk_ratio',
    created_features_list=ALL_NEW_FEATURES
)

# Aggregate ratio features by time bins
RATIO_COLS = [c for c in merged_df.columns if 'ratio' in c and c.startswith('climate_risk_ratio')]
# Focus on top drought / heat stress ratios
RATIO_COLS_FOCUSED = RATIO_COLS[:8]

print("\nQuantile-Based Risk Ratio Aggregations")
feature_count = 0

for quantile_name, quantile_col in quantile_config.items():

    print(f"Aggregating {len(RATIO_COLS_FOCUSED)} ratio features by country-{quantile_name}...")

    feature_count += create_groupby_agg_features(
        df=merged_df,
        source_cols=RATIO_COLS_FOCUSED,
        groupby_cols=['country_name', quantile_col],
        agg_funcs=['mean', 'max', 'min', 'std', 'var', 'sum'],
        feature_prefix=f'climate_risk_{quantile_name}_ratio_agg',
        created_features_list=ALL_NEW_FEATURES
    )

print(f"Created {feature_count} quantile-based risk ratio features")


# In[21]:


#  DROUGHT-HEAT COMPOUND STRESS FEATURES
# ============================================================================
# Specific focus on drought + heat interactions
print("\n Drought-Heat Compound Stress Features")
feature_count = 0

def create_compound_drought_heat_features(df, feature_prefix, created_features_list):
    """
    Create specialized compound features for drought + heat stress.
    """
    feat_count = 0

    drought_high = df['climate_risk_cnt_locations_drought_risk_high']
    drought_med = df['climate_risk_cnt_locations_drought_risk_medium']
    heat_high = df['climate_risk_cnt_locations_heat_stress_risk_high']
    heat_med = df['climate_risk_cnt_locations_heat_stress_risk_medium']

    # Combined high stress (drought_high * heat_high)
    feat_name = f'{feature_prefix}_drought_heat_high_product'
    df[feat_name] = drought_high * heat_high
    df[feat_name] = df[feat_name].replace([np.inf, -np.inf], 0).fillna(0)
    created_features_list.append(feat_name)
    feat_count += 1

    # Combined elevated stress (med+high for both)
    feat_name = f'{feature_prefix}_drought_heat_elevated_sum'
    df[feat_name] = (drought_high + drought_med) + (heat_high + heat_med)
    df[feat_name] = df[feat_name].replace([np.inf, -np.inf], 0).fillna(0)
    created_features_list.append(feat_name)
    feat_count += 1

    # Max of drought/heat high
    feat_name = f'{feature_prefix}_drought_heat_high_max'
    df[feat_name] = np.maximum(drought_high, heat_high)
    created_features_list.append(feat_name)
    feat_count += 1

    # Min of drought/heat high (both present = severe)
    feat_name = f'{feature_prefix}_drought_heat_high_min'
    df[feat_name] = np.minimum(drought_high, heat_high)
    created_features_list.append(feat_name)
    feat_count += 1

    # Geometric mean of high risks
    feat_name = f'{feature_prefix}_drought_heat_high_geomean'
    df[feat_name] = np.sqrt(drought_high * heat_high + 1e-6)
    df[feat_name] = df[feat_name].replace([np.inf, -np.inf], 0).fillna(0)
    created_features_list.append(feat_name)
    feat_count += 1

    # Drought dominance ratio
    feat_name = f'{feature_prefix}_drought_vs_heat_ratio'
    df[feat_name] = drought_high / (heat_high + 1e-6)
    df[feat_name] = df[feat_name].replace([np.inf, -np.inf], 0).fillna(0).clip(0, 100)
    created_features_list.append(feat_name)
    feat_count += 1

    return feat_count

print("Creating drought-heat compound features...")
feature_count += create_compound_drought_heat_features(
    df=merged_df,
    feature_prefix='climate_risk_compound',
    created_features_list=ALL_NEW_FEATURES
)

# Aggregate compound features by time bins
COMPOUND_COLS = [c for c in merged_df.columns if c.startswith('climate_risk_compound_')]

SELECTED_QUANTILES = ["quartile", "quintile", "sextile"]

for q in SELECTED_QUANTILES:
    print(f"Aggregating compound features by {q}...")

    feature_count += create_groupby_agg_features(
        df=merged_df,
        source_cols=COMPOUND_COLS,
        groupby_cols=['country_name', quantile_config[q]],
        agg_funcs=['mean', 'max', 'std'],   # drop min/var/sum for stability
        feature_prefix=f'climate_risk_compound_{q}_agg',
        created_features_list=ALL_NEW_FEATURES
    )

print(f"Created {feature_count} drought-heat compound stress features")


# In[22]:


#  WEIGHTED FEATURES WITH QUARTILE-COUNTRY-MONTH AGG
# ============================================================================

print("\n Weighted + Quartile-Country-Month Aggregation")
feature_count = 0

# First ensure weighted features exist for key risk types
WEIGHTED_MEDIUM_COLS = [
    'climate_risk_weighted_drought_risk_medium',
    'climate_risk_weighted_excess_precip_risk_medium',
    'climate_risk_weighted_heat_stress_risk_medium',
    'climate_risk_weighted_drought_risk_high',
    'climate_risk_weighted_heat_stress_risk_high',
]

# Check which exist, create if needed
for col in WEIGHTED_MEDIUM_COLS:
    base_col = col.replace('climate_risk_weighted_', 'climate_risk_cnt_locations_')
    if col not in merged_df.columns and base_col in merged_df.columns:
        merged_df[col] = merged_df[base_col] * (merged_df['percent_country_production'] / 100)
        merged_df[col] = merged_df[col].replace([np.inf, -np.inf], 0).fillna(0)
        ALL_NEW_FEATURES.append(col)
        feature_count += 1

EXISTING_WEIGHTED_COLS = [c for c in WEIGHTED_MEDIUM_COLS if c in merged_df.columns]

# Aggregate weighted features by quartile-country-month
print("Creating weighted quartile-country-month aggregations...")
feature_count += create_groupby_agg_features(
    df=merged_df,
    source_cols=EXISTING_WEIGHTED_COLS,
    groupby_cols=['climate_risk_time_bin_quartile', 'country_name', 'month'],
    agg_funcs=['mean', 'max', 'min', 'std', 'var', 'sum'],
    feature_prefix='climate_risk_weighted_quartile_country_month_agg',
    created_features_list=ALL_NEW_FEATURES
)

# Also try sextile-country-month for weighted
print("Creating weighted sextile-country-month aggregations...")
feature_count += create_groupby_agg_features(
    df=merged_df,
    source_cols=EXISTING_WEIGHTED_COLS[:3],  # Focus on medium risks
    groupby_cols=['climate_risk_time_bin_sextile', 'country_name', 'month'],
    agg_funcs=['mean', 'max', 'min', 'std', 'var', 'sum'],
    feature_prefix='climate_risk_weighted_sextile_country_month_agg',
    created_features_list=ALL_NEW_FEATURES
)

print(f"Created {feature_count} weighted quartile-country-month features")


# In[23]:


#  MEDIUM RISK COMPOUND FEATURES
# ============================================================================
# Focus on medium risk level compounds
print("\n Medium Risk Compound Features")
feature_count = 0

def create_medium_risk_compounds(df, feature_prefix, created_features_list):
    """
    Create compound features for medium risk levels.
    """
    feat_count = 0

    drought_med = df['climate_risk_cnt_locations_drought_risk_medium']
    heat_med = df['climate_risk_cnt_locations_heat_stress_risk_medium']
    excess_med = df['climate_risk_cnt_locations_excess_precip_risk_medium']
    cold_med = df['climate_risk_cnt_locations_unseasonably_cold_risk_medium']

    # Drought × Heat medium
    feat_name = f'{feature_prefix}_drought_heat_med_product'
    df[feat_name] = drought_med * heat_med
    df[feat_name] = df[feat_name].replace([np.inf, -np.inf], 0).fillna(0)
    created_features_list.append(feat_name)
    feat_count += 1

    feat_name = f'{feature_prefix}_drought_heat_med_sum'
    df[feat_name] = drought_med + heat_med
    created_features_list.append(feat_name)
    feat_count += 1

    feat_name = f'{feature_prefix}_drought_heat_med_max'
    df[feat_name] = np.maximum(drought_med, heat_med)
    created_features_list.append(feat_name)
    feat_count += 1

    feat_name = f'{feature_prefix}_drought_heat_med_min'
    df[feat_name] = np.minimum(drought_med, heat_med)
    created_features_list.append(feat_name)
    feat_count += 1

    feat_name = f'{feature_prefix}_drought_heat_med_geomean'
    df[feat_name] = np.sqrt(drought_med * heat_med + 1e-6)
    df[feat_name] = df[feat_name].replace([np.inf, -np.inf], 0).fillna(0)
    created_features_list.append(feat_name)
    feat_count += 1

    # Drought × Excess Precip medium
    feat_name = f'{feature_prefix}_drought_excess_med_product'
    df[feat_name] = drought_med * excess_med
    df[feat_name] = df[feat_name].replace([np.inf, -np.inf], 0).fillna(0)
    created_features_list.append(feat_name)
    feat_count += 1

    feat_name = f'{feature_prefix}_drought_excess_med_sum'
    df[feat_name] = drought_med + excess_med
    created_features_list.append(feat_name)
    feat_count += 1

    feat_name = f'{feature_prefix}_drought_excess_med_geomean'
    df[feat_name] = np.sqrt(drought_med * excess_med + 1e-6)
    df[feat_name] = df[feat_name].replace([np.inf, -np.inf], 0).fillna(0)
    created_features_list.append(feat_name)
    feat_count += 1

    # Heat × Excess Precip medium
    feat_name = f'{feature_prefix}_heat_excess_med_product'
    df[feat_name] = heat_med * excess_med
    df[feat_name] = df[feat_name].replace([np.inf, -np.inf], 0).fillna(0)
    created_features_list.append(feat_name)
    feat_count += 1

    feat_name = f'{feature_prefix}_heat_excess_med_sum'
    df[feat_name] = heat_med + excess_med
    created_features_list.append(feat_name)
    feat_count += 1

    # All medium combined
    feat_name = f'{feature_prefix}_all_med_sum'
    df[feat_name] = drought_med + heat_med + excess_med + cold_med
    created_features_list.append(feat_name)
    feat_count += 1

    feat_name = f'{feature_prefix}_all_med_max'
    df[feat_name] = df[['climate_risk_cnt_locations_drought_risk_medium',
                        'climate_risk_cnt_locations_heat_stress_risk_medium',
                        'climate_risk_cnt_locations_excess_precip_risk_medium',
                        'climate_risk_cnt_locations_unseasonably_cold_risk_medium']].max(axis=1)
    created_features_list.append(feat_name)
    feat_count += 1

    return feat_count

print("Creating medium risk compound features...")
feature_count += create_medium_risk_compounds(
    df=merged_df,
    feature_prefix='climate_risk_compound_med',
    created_features_list=ALL_NEW_FEATURES
)

# Aggregate medium compounds by quintile
COMPOUND_MED_COLS = [c for c in merged_df.columns if c.startswith('climate_risk_compound_med_')]

print("Aggregating medium compounds by quintile...")
feature_count += create_groupby_agg_features(
    df=merged_df,
    source_cols=COMPOUND_MED_COLS,
    groupby_cols=['country_name', 'climate_risk_time_bin_quintile'],
    agg_funcs=['mean', 'max', 'sum'],
    feature_prefix='climate_risk_compound_med_quintile_agg',
    created_features_list=ALL_NEW_FEATURES
)

print("Aggregating medium compounds by sextile...")
feature_count += create_groupby_agg_features(
    df=merged_df,
    source_cols=COMPOUND_MED_COLS,
    groupby_cols=['country_name', 'climate_risk_time_bin_sextile'],
    agg_funcs=['mean', 'max', 'min', 'std', 'var', 'sum'],
    feature_prefix='climate_risk_compound_med_sextile_agg',
    created_features_list=ALL_NEW_FEATURES
)

print(f"Created {feature_count} medium risk compound features")


# In[24]:


#  RATIO FEATURES WITH QUARTILE-COUNTRY-MONTH AGG
# ============================================================================
print("\n Ratio + Quartile-Country-Month Aggregation")
feature_count = 0

# Get existing ratio columns
RATIO_COLS = [c for c in merged_df.columns if c.startswith('climate_risk_ratio_')]

quantile_config_2 = {
    "quartile": "climate_risk_time_bin_quartile",
    "sextile": "climate_risk_time_bin_sextile",
}

if RATIO_COLS:
    print(f"Found {len(RATIO_COLS)} ratio columns")

    for quantile_name, quantile_col in quantile_config_2.items():

        print(f"Creating ratio {quantile_name}-country-month aggregations...")

        feature_count += create_groupby_agg_features(
            df=merged_df,
            source_cols=RATIO_COLS[:8],  # focus on top ratios
            groupby_cols=[quantile_col, 'country_name', 'month'],
            agg_funcs=['mean', 'max', 'std'],  # ❗ safer for ratios
            feature_prefix=f'climate_risk_ratio_{quantile_name}_country_month_agg',
            created_features_list=ALL_NEW_FEATURES
        )

print(f"Created {feature_count} ratio quartile-country-month features")


# In[25]:


#  CROSS-LEVEL COMPOUND FEATURES
# ============================================================================
# Combine high and medium risk levels (high × medium interactions)
print("\n Cross-Level Compound Features")
feature_count = 0

def create_cross_level_compounds(df, feature_prefix, created_features_list):
    """
    Create compound features combining high and medium risk levels.
    """
    feat_count = 0

    drought_high = df['climate_risk_cnt_locations_drought_risk_high']
    drought_med = df['climate_risk_cnt_locations_drought_risk_medium']
    heat_high = df['climate_risk_cnt_locations_heat_stress_risk_high']
    heat_med = df['climate_risk_cnt_locations_heat_stress_risk_medium']
    excess_high = df['climate_risk_cnt_locations_excess_precip_risk_high']
    excess_med = df['climate_risk_cnt_locations_excess_precip_risk_medium']

    # Drought high × Heat medium
    feat_name = f'{feature_prefix}_drought_high_heat_med_product'
    df[feat_name] = drought_high * heat_med
    df[feat_name] = df[feat_name].replace([np.inf, -np.inf], 0).fillna(0)
    created_features_list.append(feat_name)
    feat_count += 1

    # Heat high × Drought medium
    feat_name = f'{feature_prefix}_heat_high_drought_med_product'
    df[feat_name] = heat_high * drought_med
    df[feat_name] = df[feat_name].replace([np.inf, -np.inf], 0).fillna(0)
    created_features_list.append(feat_name)
    feat_count += 1

    # Drought high × Excess medium
    feat_name = f'{feature_prefix}_drought_high_excess_med_product'
    df[feat_name] = drought_high * excess_med
    df[feat_name] = df[feat_name].replace([np.inf, -np.inf], 0).fillna(0)
    created_features_list.append(feat_name)
    feat_count += 1

    # Excess high × Drought medium
    feat_name = f'{feature_prefix}_excess_high_drought_med_product'
    df[feat_name] = excess_high * drought_med
    df[feat_name] = df[feat_name].replace([np.inf, -np.inf], 0).fillna(0)
    created_features_list.append(feat_name)
    feat_count += 1

    # Heat high × Excess medium
    feat_name = f'{feature_prefix}_heat_high_excess_med_product'
    df[feat_name] = heat_high * excess_med
    df[feat_name] = df[feat_name].replace([np.inf, -np.inf], 0).fillna(0)
    created_features_list.append(feat_name)
    feat_count += 1

    # Combined elevated risk (high + medium for each type)
    feat_name = f'{feature_prefix}_drought_elevated'
    df[feat_name] = drought_high + drought_med
    created_features_list.append(feat_name)
    feat_count += 1

    feat_name = f'{feature_prefix}_heat_elevated'
    df[feat_name] = heat_high + heat_med
    created_features_list.append(feat_name)
    feat_count += 1

    feat_name = f'{feature_prefix}_excess_elevated'
    df[feat_name] = excess_high + excess_med
    created_features_list.append(feat_name)
    feat_count += 1

    # Elevated × Elevated
    feat_name = f'{feature_prefix}_drought_heat_elevated_product'
    df[feat_name] = (drought_high + drought_med) * (heat_high + heat_med)
    df[feat_name] = df[feat_name].replace([np.inf, -np.inf], 0).fillna(0)
    created_features_list.append(feat_name)
    feat_count += 1

    return feat_count

print("Creating cross-level compound features...")
feature_count += create_cross_level_compounds(
    df=merged_df,
    feature_prefix='climate_risk_crosslevel',
    created_features_list=ALL_NEW_FEATURES
)

# Aggregate cross-level compounds
CROSSLEVEL_COLS = [c for c in merged_df.columns if c.startswith('climate_risk_crosslevel_')]

print("Aggregating cross-level compounds by quintile...")
feature_count += create_groupby_agg_features(
    df=merged_df,
    source_cols=CROSSLEVEL_COLS,
    groupby_cols=['country_name', 'climate_risk_time_bin_quintile'],
    agg_funcs=['mean', 'max', 'sum'],
    feature_prefix='climate_risk_crosslevel_quintile_agg',
    created_features_list=ALL_NEW_FEATURES
)

print("Aggregating cross-level compounds by quartile-country-month...")
feature_count += create_groupby_agg_features(
    df=merged_df,
    source_cols=CROSSLEVEL_COLS[:5],
    groupby_cols=['climate_risk_time_bin_quartile', 'country_name', 'month'],
    agg_funcs=['mean', 'max', 'min', 'std', 'var', 'sum'],
    feature_prefix='climate_risk_crosslevel_quartile_month_agg',
    created_features_list=ALL_NEW_FEATURES
)

print(f"Created {feature_count} cross-level compound features")


# In[26]:


#  WEIGHTED COMPOUND FEATURES
# ============================================================================
# Apply production weighting to compound features
print("\n Weighted Compound Features")
feature_count = 0

def create_weighted_compounds(df, feature_prefix, created_features_list):
    """
    Create compound features using production-weighted risk values.
    """
    feat_count = 0

    weight = df['percent_country_production'] / 100

    drought_high_w = df['climate_risk_cnt_locations_drought_risk_high'] * weight
    drought_med_w = df['climate_risk_cnt_locations_drought_risk_medium'] * weight
    heat_high_w = df['climate_risk_cnt_locations_heat_stress_risk_high'] * weight
    heat_med_w = df['climate_risk_cnt_locations_heat_stress_risk_medium'] * weight
    excess_med_w = df['climate_risk_cnt_locations_excess_precip_risk_medium'] * weight

    # Weighted drought × heat high
    feat_name = f'{feature_prefix}_w_drought_heat_high_product'
    df[feat_name] = drought_high_w * heat_high_w
    df[feat_name] = df[feat_name].replace([np.inf, -np.inf], 0).fillna(0)
    created_features_list.append(feat_name)
    feat_count += 1

    feat_name = f'{feature_prefix}_w_drought_heat_high_geomean'
    df[feat_name] = np.sqrt(drought_high_w * heat_high_w + 1e-6)
    df[feat_name] = df[feat_name].replace([np.inf, -np.inf], 0).fillna(0)
    created_features_list.append(feat_name)
    feat_count += 1

    feat_name = f'{feature_prefix}_w_drought_heat_high_min'
    df[feat_name] = np.minimum(drought_high_w, heat_high_w)
    created_features_list.append(feat_name)
    feat_count += 1

    feat_name = f'{feature_prefix}_w_drought_heat_high_max'
    df[feat_name] = np.maximum(drought_high_w, heat_high_w)
    created_features_list.append(feat_name)
    feat_count += 1

    # Weighted drought × heat medium
    feat_name = f'{feature_prefix}_w_drought_heat_med_product'
    df[feat_name] = drought_med_w * heat_med_w
    df[feat_name] = df[feat_name].replace([np.inf, -np.inf], 0).fillna(0)
    created_features_list.append(feat_name)
    feat_count += 1

    feat_name = f'{feature_prefix}_w_drought_heat_med_geomean'
    df[feat_name] = np.sqrt(drought_med_w * heat_med_w + 1e-6)
    df[feat_name] = df[feat_name].replace([np.inf, -np.inf], 0).fillna(0)
    created_features_list.append(feat_name)
    feat_count += 1

    # Weighted drought × excess medium
    feat_name = f'{feature_prefix}_w_drought_excess_med_product'
    df[feat_name] = drought_med_w * excess_med_w
    df[feat_name] = df[feat_name].replace([np.inf, -np.inf], 0).fillna(0)
    created_features_list.append(feat_name)
    feat_count += 1

    feat_name = f'{feature_prefix}_w_drought_excess_med_geomean'
    df[feat_name] = np.sqrt(drought_med_w * excess_med_w + 1e-6)
    df[feat_name] = df[feat_name].replace([np.inf, -np.inf], 0).fillna(0)
    created_features_list.append(feat_name)
    feat_count += 1

    return feat_count

print("Creating weighted compound features...")
feature_count += create_weighted_compounds(
    df=merged_df,
    feature_prefix='climate_risk_wcompound',
    created_features_list=ALL_NEW_FEATURES
)

# Aggregate weighted compounds
WCOMPOUND_COLS = [c for c in merged_df.columns if c.startswith('climate_risk_wcompound_')]

print("Aggregating weighted compounds by quintile...")
feature_count += create_groupby_agg_features(
    df=merged_df,
    source_cols=WCOMPOUND_COLS,
    groupby_cols=['country_name', 'climate_risk_time_bin_quintile'],
    agg_funcs=['mean', 'max', 'sum'],
    feature_prefix='climate_risk_wcompound_quintile_agg',
    created_features_list=ALL_NEW_FEATURES
)

MONTH_QUANTILES = {
    "quartile": "climate_risk_time_bin_quartile",
}

for q_name, q_col in MONTH_QUANTILES.items():
    print(f"Aggregating weighted compounds by {q_name}-country-month...")

    feature_count += create_groupby_agg_features(
        df=merged_df,
        source_cols=WCOMPOUND_COLS,
        groupby_cols=[q_col, 'country_name', 'month'],
        agg_funcs=['mean', 'max', 'std'],  # ❗ drop min/var/sum
        feature_prefix=f'climate_risk_wcompound_{q_name}_country_month_agg',
        created_features_list=ALL_NEW_FEATURES
    )

print(f"Created {feature_count} weighted compound features")


# In[27]:


#  CROSSBIN FOR WEIGHTED FEATURES
# ============================================================================
# Apply crossbin pattern to weighted features
print("\n Crossbin for Weighted Features")
feature_count = 0

def create_weighted_crossbin_features(df, source_cols, timebin_pairs, feature_prefix, created_features_list):
    """
    Create crossbin deviation features for weighted columns.
    """
    feat_count = 0

    for coarse_bin, fine_bin in timebin_pairs:
        coarse_name = coarse_bin.replace('climate_risk_time_bin_', '')
        fine_name = fine_bin.replace('climate_risk_time_bin_', '')

        for source_col in source_cols:
            if source_col not in df.columns:
                continue
            risk_name = source_col.replace('climate_risk_weighted_', '')

            # Coarse bin mean
            coarse_mean = df.groupby(coarse_bin)[source_col].transform('mean')

            # Deviation from coarse bin mean
            feat_name = f'{feature_prefix}_{coarse_name}_{fine_name}_dev_{risk_name}'
            df[feat_name] = df[source_col] - coarse_mean
            df[feat_name] = df[feat_name].replace([np.inf, -np.inf], 0).fillna(0)
            created_features_list.append(feat_name)
            feat_count += 1

    return feat_count

WEIGHTED_CROSSBIN_COLS = [
    'climate_risk_weighted_drought_risk_medium',
    'climate_risk_weighted_drought_risk_high',
    'climate_risk_weighted_heat_stress_risk_high',
    'climate_risk_weighted_excess_precip_risk_medium',
]

CROSSBIN_PAIRS = [
    ('climate_risk_time_bin_sextile', 'climate_risk_time_bin_tredecile'),
    ('climate_risk_time_bin_quartile', 'climate_risk_time_bin_decile'),
]

print("Creating weighted crossbin features...")
feature_count += create_weighted_crossbin_features(
    df=merged_df,
    source_cols=WEIGHTED_CROSSBIN_COLS,
    timebin_pairs=CROSSBIN_PAIRS,
    feature_prefix='climate_risk_wcrossbin',
    created_features_list=ALL_NEW_FEATURES
)

print(f"Created {feature_count} weighted crossbin features")


# In[28]:


#  COMPOUND FEATURES WITH QUARTILE-COUNTRY-MONTH
# ============================================================================

print("\n Compound + Quartile-Country-Month")
feature_count = 0

# Get all compound columns
ALL_COMPOUND_COLS = [c for c in merged_df.columns if 'compound' in c and not '_agg_' in c]
ALL_COMPOUND_COLS = ALL_COMPOUND_COLS[:10]

if ALL_COMPOUND_COLS:
    print(f"Found {len(ALL_COMPOUND_COLS)} compound base columns")

    print("Creating compound quartile-country-month aggregations...")
    feature_count += create_groupby_agg_features(
        df=merged_df,
        source_cols=ALL_COMPOUND_COLS,
        groupby_cols=['climate_risk_time_bin_quartile', 'country_name', 'month'],
        agg_funcs=['mean', 'max', 'min', 'std', 'var', 'sum'],
        feature_prefix='climate_risk_compound_quartile_month_agg',
        created_features_list=ALL_NEW_FEATURES
    )

print(f"Created {feature_count} compound quartile-country-month features")


# In[29]:


#  DROUGHT-EXCESS COMPOUND FEATURES - TWO
# ============================================================================

print("\n Expanded Drought-Excess Compound Features")
feature_count = 0

def create_drought_excess_compounds(df, feature_prefix, created_features_list):
    """
    Create expanded drought × excess precip compound features.
    """
    feat_count = 0

    drought_high = df['climate_risk_cnt_locations_drought_risk_high']
    drought_med = df['climate_risk_cnt_locations_drought_risk_medium']
    drought_low = df['climate_risk_cnt_locations_drought_risk_low']
    excess_high = df['climate_risk_cnt_locations_excess_precip_risk_high']
    excess_med = df['climate_risk_cnt_locations_excess_precip_risk_medium']
    excess_low = df['climate_risk_cnt_locations_excess_precip_risk_low']
    weight = df['percent_country_production'] / 100

    # High × High
    feat_name = f'{feature_prefix}_drought_excess_high_product'
    df[feat_name] = drought_high * excess_high
    df[feat_name] = df[feat_name].replace([np.inf, -np.inf], 0).fillna(0)
    created_features_list.append(feat_name)
    feat_count += 1

    feat_name = f'{feature_prefix}_drought_excess_high_geomean'
    df[feat_name] = np.sqrt(drought_high * excess_high + 1e-6)
    df[feat_name] = df[feat_name].replace([np.inf, -np.inf], 0).fillna(0)
    created_features_list.append(feat_name)
    feat_count += 1

    feat_name = f'{feature_prefix}_drought_excess_high_min'
    df[feat_name] = np.minimum(drought_high, excess_high)
    created_features_list.append(feat_name)
    feat_count += 1

    feat_name = f'{feature_prefix}_drought_excess_high_max'
    df[feat_name] = np.maximum(drought_high, excess_high)
    created_features_list.append(feat_name)
    feat_count += 1

    # Medium × Medium
    feat_name = f'{feature_prefix}_drought_excess_med_min'
    df[feat_name] = np.minimum(drought_med, excess_med)
    created_features_list.append(feat_name)
    feat_count += 1

    feat_name = f'{feature_prefix}_drought_excess_med_max'
    df[feat_name] = np.maximum(drought_med, excess_med)
    created_features_list.append(feat_name)
    feat_count += 1

    # Cross level: High × Medium
    feat_name = f'{feature_prefix}_drought_high_excess_med_geomean'
    df[feat_name] = np.sqrt(drought_high * excess_med + 1e-6)
    df[feat_name] = df[feat_name].replace([np.inf, -np.inf], 0).fillna(0)
    created_features_list.append(feat_name)
    feat_count += 1

    feat_name = f'{feature_prefix}_drought_med_excess_high_geomean'
    df[feat_name] = np.sqrt(drought_med * excess_high + 1e-6)
    df[feat_name] = df[feat_name].replace([np.inf, -np.inf], 0).fillna(0)
    created_features_list.append(feat_name)
    feat_count += 1

    # Combined elevated (high + med)
    drought_elevated = drought_high + drought_med
    excess_elevated = excess_high + excess_med

    feat_name = f'{feature_prefix}_drought_excess_elevated_product'
    df[feat_name] = drought_elevated * excess_elevated
    df[feat_name] = df[feat_name].replace([np.inf, -np.inf], 0).fillna(0)
    created_features_list.append(feat_name)
    feat_count += 1

    feat_name = f'{feature_prefix}_drought_excess_elevated_geomean'
    df[feat_name] = np.sqrt(drought_elevated * excess_elevated + 1e-6)
    df[feat_name] = df[feat_name].replace([np.inf, -np.inf], 0).fillna(0)
    created_features_list.append(feat_name)
    feat_count += 1

    # Weighted versions
    feat_name = f'{feature_prefix}_w_drought_excess_med_min'
    df[feat_name] = np.minimum(drought_med * weight, excess_med * weight)
    created_features_list.append(feat_name)
    feat_count += 1

    feat_name = f'{feature_prefix}_w_drought_excess_med_max'
    df[feat_name] = np.maximum(drought_med * weight, excess_med * weight)
    created_features_list.append(feat_name)
    feat_count += 1

    feat_name = f'{feature_prefix}_w_drought_excess_elevated_geomean'
    df[feat_name] = np.sqrt((drought_elevated * weight) * (excess_elevated * weight) + 1e-6)
    df[feat_name] = df[feat_name].replace([np.inf, -np.inf], 0).fillna(0)
    created_features_list.append(feat_name)
    feat_count += 1

    return feat_count

print("Creating expanded drought-excess compound features...")
feature_count += create_drought_excess_compounds(
    df=merged_df,
    feature_prefix='climate_risk_de_compound',
    created_features_list=ALL_NEW_FEATURES
)

# Aggregate drought-excess compounds
DE_COMPOUND_COLS = [c for c in merged_df.columns if c.startswith('climate_risk_de_compound_')]

print("Aggregating drought-excess compounds by quintile...")
feature_count += create_groupby_agg_features(
    df=merged_df,
    source_cols=DE_COMPOUND_COLS,
    groupby_cols=['country_name', 'climate_risk_time_bin_quintile'],
    agg_funcs=['mean', 'max', 'sum'],
    feature_prefix='climate_risk_de_compound_quintile_agg',
    created_features_list=ALL_NEW_FEATURES
)

print("Aggregating drought-excess compounds by sextile...")
feature_count += create_groupby_agg_features(
    df=merged_df,
    source_cols=DE_COMPOUND_COLS,
    groupby_cols=['country_name', 'climate_risk_time_bin_sextile'],
    agg_funcs=['mean', 'max', 'sum'],
    feature_prefix='climate_risk_de_compound_sextile_agg',
    created_features_list=ALL_NEW_FEATURES
)

print("Aggregating drought-excess compounds by quartile-country-month...")
feature_count += create_groupby_agg_features(
    df=merged_df,
    source_cols=DE_COMPOUND_COLS[:8],
    groupby_cols=['climate_risk_time_bin_quartile', 'country_name', 'month'],
    agg_funcs=['mean', 'max', 'min', 'std', 'var', 'sum'],
    feature_prefix='climate_risk_de_compound_quartile_month_agg',
    created_features_list=ALL_NEW_FEATURES
)

print(f"Created {feature_count} expanded drought-excess compound features")


# In[30]:


#  EXPLORE ALL-RISK MAX FEATURES
# ============================================================================

print("\n Expanded All-Risk Max Features")
feature_count = 0

def create_all_risk_max_features(df, feature_prefix, created_features_list):
    """
    Create all-risk max features at different levels.
    """
    feat_count = 0

    # High risk columns
    high_cols = [
        'climate_risk_cnt_locations_drought_risk_high',
        'climate_risk_cnt_locations_heat_stress_risk_high',
        'climate_risk_cnt_locations_excess_precip_risk_high',
        'climate_risk_cnt_locations_unseasonably_cold_risk_high',
    ]

    # Medium risk columns
    med_cols = [
        'climate_risk_cnt_locations_drought_risk_medium',
        'climate_risk_cnt_locations_heat_stress_risk_medium',
        'climate_risk_cnt_locations_excess_precip_risk_medium',
        'climate_risk_cnt_locations_unseasonably_cold_risk_medium',
    ]

    weight = df['percent_country_production'] / 100

    # All high max
    feat_name = f'{feature_prefix}_all_high_max'
    df[feat_name] = df[high_cols].max(axis=1)
    created_features_list.append(feat_name)
    feat_count += 1

    feat_name = f'{feature_prefix}_all_high_sum'
    df[feat_name] = df[high_cols].sum(axis=1)
    created_features_list.append(feat_name)
    feat_count += 1

    feat_name = f'{feature_prefix}_all_high_mean'
    df[feat_name] = df[high_cols].mean(axis=1)
    created_features_list.append(feat_name)
    feat_count += 1

    # All medium min (complement to max)
    feat_name = f'{feature_prefix}_all_med_min'
    df[feat_name] = df[med_cols].min(axis=1)
    created_features_list.append(feat_name)
    feat_count += 1

    feat_name = f'{feature_prefix}_all_med_mean'
    df[feat_name] = df[med_cols].mean(axis=1)
    created_features_list.append(feat_name)
    feat_count += 1

    # Weighted versions
    feat_name = f'{feature_prefix}_w_all_med_max'
    df[feat_name] = (df[med_cols].values * weight.values.reshape(-1, 1)).max(axis=1)
    created_features_list.append(feat_name)
    feat_count += 1

    feat_name = f'{feature_prefix}_w_all_med_sum'
    df[feat_name] = (df[med_cols].values * weight.values.reshape(-1, 1)).sum(axis=1)
    created_features_list.append(feat_name)
    feat_count += 1

    feat_name = f'{feature_prefix}_w_all_high_max'
    df[feat_name] = (df[high_cols].values * weight.values.reshape(-1, 1)).max(axis=1)
    created_features_list.append(feat_name)
    feat_count += 1

    # Combined elevated (high + med) for each risk, then max
    drought_elev = df['climate_risk_cnt_locations_drought_risk_high'] + df['climate_risk_cnt_locations_drought_risk_medium']
    heat_elev = df['climate_risk_cnt_locations_heat_stress_risk_high'] + df['climate_risk_cnt_locations_heat_stress_risk_medium']
    excess_elev = df['climate_risk_cnt_locations_excess_precip_risk_high'] + df['climate_risk_cnt_locations_excess_precip_risk_medium']
    cold_elev = df['climate_risk_cnt_locations_unseasonably_cold_risk_high'] + df['climate_risk_cnt_locations_unseasonably_cold_risk_medium']

    feat_name = f'{feature_prefix}_all_elevated_max'
    df[feat_name] = pd.concat([drought_elev, heat_elev, excess_elev, cold_elev], axis=1).max(axis=1)
    created_features_list.append(feat_name)
    feat_count += 1

    feat_name = f'{feature_prefix}_all_elevated_sum'
    df[feat_name] = drought_elev + heat_elev + excess_elev + cold_elev
    created_features_list.append(feat_name)
    feat_count += 1

    # Geomean of all mediums
    feat_name = f'{feature_prefix}_all_med_geomean'
    df[feat_name] = (df[med_cols].prod(axis=1) + 1e-6) ** 0.25  # 4th root for 4 columns
    df[feat_name] = df[feat_name].replace([np.inf, -np.inf], 0).fillna(0)
    created_features_list.append(feat_name)
    feat_count += 1

    return feat_count

print("Creating all-risk max features...")
feature_count += create_all_risk_max_features(
    df=merged_df,
    feature_prefix='climate_risk_allrisk',
    created_features_list=ALL_NEW_FEATURES
)

# Aggregate all-risk features
ALLRISK_COLS = [c for c in merged_df.columns if c.startswith('climate_risk_allrisk_')]

print("Aggregating all-risk features by quintile...")
feature_count += create_groupby_agg_features(
    df=merged_df,
    source_cols=ALLRISK_COLS,
    groupby_cols=['country_name', 'climate_risk_time_bin_quintile'],
    agg_funcs=['mean', 'max', 'sum'],
    feature_prefix='climate_risk_allrisk_quintile_agg',
    created_features_list=ALL_NEW_FEATURES
)

print("Aggregating all-risk features by sextile...")
feature_count += create_groupby_agg_features(
    df=merged_df,
    source_cols=ALLRISK_COLS,
    groupby_cols=['country_name', 'climate_risk_time_bin_sextile'],
    agg_funcs=['mean', 'max', 'sum'],
    feature_prefix='climate_risk_allrisk_sextile_agg',
    created_features_list=ALL_NEW_FEATURES
)

print(f"Created {feature_count} expanded all-risk max features")


# In[31]:


#  GEOMEAN FOCUS FEATURES
# ============================================================================
# Geomean is very effective - create more geomean-based features
print("\n Geomean Focus Features")
feature_count = 0

def create_geomean_features(df, feature_prefix, created_features_list):
    """
    Create geomean-based compound features for various risk combinations.
    """
    feat_count = 0

    drought_high = df['climate_risk_cnt_locations_drought_risk_high']
    drought_med = df['climate_risk_cnt_locations_drought_risk_medium']
    heat_high = df['climate_risk_cnt_locations_heat_stress_risk_high']
    heat_med = df['climate_risk_cnt_locations_heat_stress_risk_medium']
    excess_high = df['climate_risk_cnt_locations_excess_precip_risk_high']
    excess_med = df['climate_risk_cnt_locations_excess_precip_risk_medium']
    cold_high = df['climate_risk_cnt_locations_unseasonably_cold_risk_high']
    cold_med = df['climate_risk_cnt_locations_unseasonably_cold_risk_medium']
    weight = df['percent_country_production'] / 100

    # Heat × Excess geomean (not yet tried much)
    feat_name = f'{feature_prefix}_heat_excess_med_geomean'
    df[feat_name] = np.sqrt(heat_med * excess_med + 1e-6)
    df[feat_name] = df[feat_name].replace([np.inf, -np.inf], 0).fillna(0)
    created_features_list.append(feat_name)
    feat_count += 1

    feat_name = f'{feature_prefix}_heat_excess_high_geomean'
    df[feat_name] = np.sqrt(heat_high * excess_high + 1e-6)
    df[feat_name] = df[feat_name].replace([np.inf, -np.inf], 0).fillna(0)
    created_features_list.append(feat_name)
    feat_count += 1

    # Cold × Drought geomean
    feat_name = f'{feature_prefix}_cold_drought_med_geomean'
    df[feat_name] = np.sqrt(cold_med * drought_med + 1e-6)
    df[feat_name] = df[feat_name].replace([np.inf, -np.inf], 0).fillna(0)
    created_features_list.append(feat_name)
    feat_count += 1

    # Triple geomean: drought × heat × excess
    feat_name = f'{feature_prefix}_drought_heat_excess_med_geomean'
    df[feat_name] = (drought_med * heat_med * excess_med + 1e-6) ** (1/3)
    df[feat_name] = df[feat_name].replace([np.inf, -np.inf], 0).fillna(0)
    created_features_list.append(feat_name)
    feat_count += 1

    feat_name = f'{feature_prefix}_drought_heat_excess_high_geomean'
    df[feat_name] = (drought_high * heat_high * excess_high + 1e-6) ** (1/3)
    df[feat_name] = df[feat_name].replace([np.inf, -np.inf], 0).fillna(0)
    created_features_list.append(feat_name)
    feat_count += 1

    # Weighted geomeans
    feat_name = f'{feature_prefix}_w_heat_excess_med_geomean'
    df[feat_name] = np.sqrt((heat_med * weight) * (excess_med * weight) + 1e-6)
    df[feat_name] = df[feat_name].replace([np.inf, -np.inf], 0).fillna(0)
    created_features_list.append(feat_name)
    feat_count += 1

    feat_name = f'{feature_prefix}_w_drought_heat_excess_med_geomean'
    df[feat_name] = ((drought_med * weight) * (heat_med * weight) * (excess_med * weight) + 1e-6) ** (1/3)
    df[feat_name] = df[feat_name].replace([np.inf, -np.inf], 0).fillna(0)
    created_features_list.append(feat_name)
    feat_count += 1

    # Elevated geomeans
    drought_elev = drought_high + drought_med
    heat_elev = heat_high + heat_med
    excess_elev = excess_high + excess_med

    feat_name = f'{feature_prefix}_drought_heat_elevated_geomean'
    df[feat_name] = np.sqrt(drought_elev * heat_elev + 1e-6)
    df[feat_name] = df[feat_name].replace([np.inf, -np.inf], 0).fillna(0)
    created_features_list.append(feat_name)
    feat_count += 1

    feat_name = f'{feature_prefix}_drought_excess_elevated_geomean'
    df[feat_name] = np.sqrt(drought_elev * excess_elev + 1e-6)
    df[feat_name] = df[feat_name].replace([np.inf, -np.inf], 0).fillna(0)
    created_features_list.append(feat_name)
    feat_count += 1

    feat_name = f'{feature_prefix}_heat_excess_elevated_geomean'
    df[feat_name] = np.sqrt(heat_elev * excess_elev + 1e-6)
    df[feat_name] = df[feat_name].replace([np.inf, -np.inf], 0).fillna(0)
    created_features_list.append(feat_name)
    feat_count += 1

    return feat_count

print("Creating geomean focus features...")
feature_count += create_geomean_features(
    df=merged_df,
    feature_prefix='climate_risk_geomean',
    created_features_list=ALL_NEW_FEATURES
)

# Aggregate geomean features
GEOMEAN_COLS = [c for c in merged_df.columns if c.startswith('climate_risk_geomean_')]

print("Aggregating geomean features by quintile...")
feature_count += create_groupby_agg_features(
    df=merged_df,
    source_cols=GEOMEAN_COLS,
    groupby_cols=['country_name', 'climate_risk_time_bin_quintile'],
    agg_funcs=['mean', 'max', 'sum'],
    feature_prefix='climate_risk_geomean_quintile_agg',
    created_features_list=ALL_NEW_FEATURES
)

print("Aggregating geomean features by sextile...")
feature_count += create_groupby_agg_features(
    df=merged_df,
    source_cols=GEOMEAN_COLS,
    groupby_cols=['country_name', 'climate_risk_time_bin_sextile'],
    agg_funcs=['mean', 'max', 'min', 'std', 'var', 'sum'],
    feature_prefix='climate_risk_geomean_sextile_agg',
    created_features_list=ALL_NEW_FEATURES
)

print("Aggregating geomean features by quartile-country-month...")
feature_count += create_groupby_agg_features(
    df=merged_df,
    source_cols=GEOMEAN_COLS,
    groupby_cols=['climate_risk_time_bin_quartile', 'country_name', 'month'],
    agg_funcs=['mean', 'max', 'min', 'std', 'var', 'sum'],
    feature_prefix='climate_risk_geomean_quartile_month_agg',
    created_features_list=ALL_NEW_FEATURES
)

print(f"Created {feature_count} geomean focus features")


# In[32]:


#  MEDIUM COMPOUND EXPANDED AGGREGATIONS
# ============================================================================

print("\n Medium Compound Expanded Aggregations")
feature_count = 0

# Get medium compound columns
MED_COMPOUND_BASE_COLS = [c for c in merged_df.columns if c.startswith('climate_risk_compound_med_') and '_agg_' not in c]

if MED_COMPOUND_BASE_COLS:
    print(f"Found {len(MED_COMPOUND_BASE_COLS)} medium compound base columns")

    # Aggregate by quartile (different from quintile/sextile already done)
    print("Aggregating medium compounds by quartile...")
    feature_count += create_groupby_agg_features(
        df=merged_df,
        source_cols=MED_COMPOUND_BASE_COLS,
        groupby_cols=['country_name', 'climate_risk_time_bin_quartile'],
        agg_funcs=['mean', 'max', 'sum'],
        feature_prefix='climate_risk_compound_med_quartile_agg',
        created_features_list=ALL_NEW_FEATURES
    )

    # Aggregate by quartile-country-month
    print("Aggregating medium compounds by quartile-country-month...")
    feature_count += create_groupby_agg_features(
        df=merged_df,
        source_cols=MED_COMPOUND_BASE_COLS[:8],
        groupby_cols=['climate_risk_time_bin_quartile', 'country_name', 'month'],
        agg_funcs=['mean', 'max', 'min', 'std', 'var', 'sum'],
        feature_prefix='climate_risk_compound_med_quartile_month_agg',
        created_features_list=ALL_NEW_FEATURES
    )

print(f"Created {feature_count} medium compound expanded aggregation features")


# In[33]:


#  TERTILE AGGREGATIONS
# ============================================================================
# Tertile aggregation for compound_med and allrisk (expanding time bins)
print("\n Tertile Aggregations")
feature_count = 0

# Get compound med columns (base, not aggregated)
COMPOUND_MED_BASE = [c for c in merged_df.columns if c.startswith('climate_risk_compound_med_') and '_agg_' not in c]
ALLRISK_BASE = [c for c in merged_df.columns if c.startswith('climate_risk_allrisk_') and '_agg_' not in c]
WCOMPOUND_BASE = [c for c in merged_df.columns if c.startswith('climate_risk_wcompound_') and '_agg_' not in c]

if COMPOUND_MED_BASE:
    print(f"Found {len(COMPOUND_MED_BASE)} compound_med base columns")
    feature_count += create_groupby_agg_features(
        df=merged_df,
        source_cols=COMPOUND_MED_BASE,
        groupby_cols=['country_name', 'climate_risk_time_bin_tertile'],
        agg_funcs=['mean', 'max', 'min', 'std', 'var', 'sum'],
        feature_prefix='climate_risk_compound_med_tertile_agg',
        created_features_list=ALL_NEW_FEATURES
    )

if ALLRISK_BASE:
    print(f"Found {len(ALLRISK_BASE)} allrisk base columns")
    feature_count += create_groupby_agg_features(
        df=merged_df,
        source_cols=ALLRISK_BASE,
        groupby_cols=['country_name', 'climate_risk_time_bin_tertile'],
        agg_funcs=['mean', 'max', 'min', 'std', 'var', 'sum'],
        feature_prefix='climate_risk_allrisk_tertile_agg',
        created_features_list=ALL_NEW_FEATURES
    )

if WCOMPOUND_BASE:
    print(f"Found {len(WCOMPOUND_BASE)} wcompound base columns")
    feature_count += create_groupby_agg_features(
        df=merged_df,
        source_cols=WCOMPOUND_BASE,
        groupby_cols=['country_name', 'climate_risk_time_bin_tertile'],
        agg_funcs=['mean', 'max', 'min', 'std', 'var', 'sum'],
        feature_prefix='climate_risk_wcompound_tertile_agg',
        created_features_list=ALL_NEW_FEATURES
    )

print(f"Created {feature_count} tertile aggregation features")


# In[34]:


#  OCTILE AGGREGATIONS
# ============================================================================
print("\n Octile Aggregations")
feature_count = 0

if COMPOUND_MED_BASE:
    feature_count += create_groupby_agg_features(
        df=merged_df,
        source_cols=COMPOUND_MED_BASE[:12],
        groupby_cols=['country_name', 'climate_risk_time_bin_octile'],
        agg_funcs=['mean', 'max', 'min', 'std', 'var', 'sum'],
        feature_prefix='climate_risk_compound_med_octile_agg',
        created_features_list=ALL_NEW_FEATURES
    )

if ALLRISK_BASE:
    feature_count += create_groupby_agg_features(
        df=merged_df,
        source_cols=ALLRISK_BASE[:8],
        groupby_cols=['country_name', 'climate_risk_time_bin_octile'],
        agg_funcs=['mean', 'max', 'min', 'std', 'var', 'sum'],
        feature_prefix='climate_risk_allrisk_octile_agg',
        created_features_list=ALL_NEW_FEATURES
    )

print(f"Created {feature_count} octile aggregation features")


# In[35]:


#  PRODUCT-BASED COMPOUND FEATURES
# ============================================================================

print("\n Product-Based Compound Features")
feature_count = 0

def create_product_compounds(df, feature_prefix, created_features_list):
    """
    Create product-based compound features (multiplication of risk levels).
    """
    feat_count = 0

    drought_med = df['climate_risk_cnt_locations_drought_risk_medium']
    heat_med = df['climate_risk_cnt_locations_heat_stress_risk_medium']
    excess_med = df['climate_risk_cnt_locations_excess_precip_risk_medium']
    cold_med = df['climate_risk_cnt_locations_unseasonably_cold_risk_medium']
    drought_high = df['climate_risk_cnt_locations_drought_risk_high']
    heat_high = df['climate_risk_cnt_locations_heat_stress_risk_high']
    excess_high = df['climate_risk_cnt_locations_excess_precip_risk_high']
    cold_high = df['climate_risk_cnt_locations_unseasonably_cold_risk_high']
    weight = df['percent_country_production'] / 100

    # Product of medium risks (scaled)
    feat_name = f'{feature_prefix}_drought_heat_med_product'
    df[feat_name] = drought_med * heat_med / 100
    df[feat_name] = df[feat_name].replace([np.inf, -np.inf], 0).fillna(0)
    created_features_list.append(feat_name)
    feat_count += 1

    feat_name = f'{feature_prefix}_drought_excess_med_product'
    df[feat_name] = drought_med * excess_med / 100
    df[feat_name] = df[feat_name].replace([np.inf, -np.inf], 0).fillna(0)
    created_features_list.append(feat_name)
    feat_count += 1

    feat_name = f'{feature_prefix}_heat_excess_med_product'
    df[feat_name] = heat_med * excess_med / 100
    df[feat_name] = df[feat_name].replace([np.inf, -np.inf], 0).fillna(0)
    created_features_list.append(feat_name)
    feat_count += 1

    feat_name = f'{feature_prefix}_drought_cold_med_product'
    df[feat_name] = drought_med * cold_med / 100
    df[feat_name] = df[feat_name].replace([np.inf, -np.inf], 0).fillna(0)
    created_features_list.append(feat_name)
    feat_count += 1

    # Triple products (scaled more)
    feat_name = f'{feature_prefix}_drought_heat_excess_med_product'
    df[feat_name] = drought_med * heat_med * excess_med / 10000
    df[feat_name] = df[feat_name].replace([np.inf, -np.inf], 0).fillna(0)
    created_features_list.append(feat_name)
    feat_count += 1

    # Weighted products
    feat_name = f'{feature_prefix}_w_drought_heat_med_product'
    df[feat_name] = (drought_med * weight) * (heat_med * weight)
    df[feat_name] = df[feat_name].replace([np.inf, -np.inf], 0).fillna(0)
    created_features_list.append(feat_name)
    feat_count += 1

    feat_name = f'{feature_prefix}_w_drought_excess_med_product'
    df[feat_name] = (drought_med * weight) * (excess_med * weight)
    df[feat_name] = df[feat_name].replace([np.inf, -np.inf], 0).fillna(0)
    created_features_list.append(feat_name)
    feat_count += 1

    feat_name = f'{feature_prefix}_w_heat_excess_med_product'
    df[feat_name] = (heat_med * weight) * (excess_med * weight)
    df[feat_name] = df[feat_name].replace([np.inf, -np.inf], 0).fillna(0)
    created_features_list.append(feat_name)
    feat_count += 1

    # High risk products (scaled)
    feat_name = f'{feature_prefix}_drought_heat_high_product'
    df[feat_name] = drought_high * heat_high / 100
    df[feat_name] = df[feat_name].replace([np.inf, -np.inf], 0).fillna(0)
    created_features_list.append(feat_name)
    feat_count += 1

    feat_name = f'{feature_prefix}_drought_excess_high_product'
    df[feat_name] = drought_high * excess_high / 100
    df[feat_name] = df[feat_name].replace([np.inf, -np.inf], 0).fillna(0)
    created_features_list.append(feat_name)
    feat_count += 1

    # Mixed high-med products
    feat_name = f'{feature_prefix}_drought_high_heat_med_product'
    df[feat_name] = drought_high * heat_med / 100
    df[feat_name] = df[feat_name].replace([np.inf, -np.inf], 0).fillna(0)
    created_features_list.append(feat_name)
    feat_count += 1

    feat_name = f'{feature_prefix}_drought_med_excess_high_product'
    df[feat_name] = drought_med * excess_high / 100
    df[feat_name] = df[feat_name].replace([np.inf, -np.inf], 0).fillna(0)
    created_features_list.append(feat_name)
    feat_count += 1

    return feat_count

print("Creating product compounds...")
feature_count += create_product_compounds(
    df=merged_df,
    feature_prefix='climate_risk_product',
    created_features_list=ALL_NEW_FEATURES
)

# Aggregate products by quartile 
PRODUCT_COLS = [c for c in merged_df.columns if c.startswith('climate_risk_product_')]
if PRODUCT_COLS:
    print(f"Aggregating {len(PRODUCT_COLS)} product features by quartile...")
    feature_count += create_groupby_agg_features(
        df=merged_df,
        source_cols=PRODUCT_COLS,
        groupby_cols=['country_name', 'climate_risk_time_bin_quartile'],
        agg_funcs=['mean', 'max', 'min', 'std', 'var', 'sum'],
        feature_prefix='climate_risk_product_quartile_agg',
        created_features_list=ALL_NEW_FEATURES
    )

    print(f"Aggregating product features by quintile...")
    feature_count += create_groupby_agg_features(
        df=merged_df,
        source_cols=PRODUCT_COLS,
        groupby_cols=['country_name', 'climate_risk_time_bin_quintile'],
        agg_funcs=['mean', 'max', 'min', 'std', 'var', 'sum'],
        feature_prefix='climate_risk_product_quintile_agg',
        created_features_list=ALL_NEW_FEATURES
    )

print(f"Created {feature_count} product-based features")


# In[36]:


#  DECILE AGGREGATIONS
# ============================================================================
print("\n Decile Aggregations")
feature_count = 0

if COMPOUND_MED_BASE:
    feature_count += create_groupby_agg_features(
        df=merged_df,
        source_cols=COMPOUND_MED_BASE[:10],
        groupby_cols=['country_name', 'climate_risk_time_bin_decile'],
        agg_funcs=['mean', 'max', 'min', 'std', 'var', 'sum'],
        feature_prefix='climate_risk_compound_med_decile_agg',
        created_features_list=ALL_NEW_FEATURES
    )

if WCOMPOUND_BASE:
    feature_count += create_groupby_agg_features(
        df=merged_df,
        source_cols=WCOMPOUND_BASE[:8],
        groupby_cols=['country_name', 'climate_risk_time_bin_decile'],
        agg_funcs=['mean', 'max', 'min', 'std', 'var', 'sum'],
        feature_prefix='climate_risk_wcompound_decile_agg',
        created_features_list=ALL_NEW_FEATURES
    )

print(f"Created {feature_count} decile aggregation features")


# In[37]:


#  QUARTILE-COUNTRY AGGREGATIONS
# ============================================================================
# Week-level aggregation for compound features
print("\n Quartile-Country-Week Aggregations")
feature_count = 0

if COMPOUND_MED_BASE:
    feature_count += create_groupby_agg_features(
        df=merged_df,
        source_cols=COMPOUND_MED_BASE[:10],
        groupby_cols=['climate_risk_time_bin_quartile', 'country_name'],
        agg_funcs=['mean', 'max', 'min', 'std', 'var', 'sum'],
        feature_prefix='climate_risk_compound_med_quartile_country_names_agg',
        created_features_list=ALL_NEW_FEATURES
    )

if ALLRISK_BASE:
    feature_count += create_groupby_agg_features(
        df=merged_df,
        source_cols=ALLRISK_BASE[:6],
        groupby_cols=['climate_risk_time_bin_quartile', 'country_name'],
        agg_funcs=['mean', 'max', 'min', 'std', 'var', 'sum'],
        feature_prefix='climate_risk_allrisk_quartile_country_names_agg',
        created_features_list=ALL_NEW_FEATURES
    )

print(f"Created {feature_count} quartile-country-week features")


# In[38]:


#  SEXTILE-COUNTRY AGGREGATIONS WITH SUM
# ============================================================================
# Sextile with sum is performing well
print("\n Sextile-Country Aggregations with Sum")
feature_count = 0

if COMPOUND_MED_BASE:
    feature_count += create_groupby_agg_features(
        df=merged_df,
        source_cols=COMPOUND_MED_BASE,
        groupby_cols=['country_name', 'climate_risk_time_bin_sextile'],
        agg_funcs=['mean', 'max', 'min', 'std', 'var', 'sum'],
        feature_prefix='climate_risk_compound_med_sextile_sum_agg',
        created_features_list=ALL_NEW_FEATURES
    )

if WCOMPOUND_BASE:
    feature_count += create_groupby_agg_features(
        df=merged_df,
        source_cols=WCOMPOUND_BASE,
        groupby_cols=['country_name', 'climate_risk_time_bin_sextile'],
        agg_funcs=['mean', 'max', 'min', 'std', 'var', 'sum'],
        feature_prefix='climate_risk_wcompound_sextile_sum_agg',
        created_features_list=ALL_NEW_FEATURES
    )

print(f"Created {feature_count} sextile-country sum features")


# In[39]:


#  WEIGHTED SUM COMPOUND FEATURES
# ============================================================================
# Create weighted sum features (weight * sum of risks)
print("\n Weighted Sum Compound Features")
feature_count = 0

def create_weighted_sum_features(df, feature_prefix, created_features_list):
    """
    Create features that are weighted sums of multiple risk types.
    """
    feat_count = 0

    drought_med = df['climate_risk_cnt_locations_drought_risk_medium']
    heat_med = df['climate_risk_cnt_locations_heat_stress_risk_medium']
    excess_med = df['climate_risk_cnt_locations_excess_precip_risk_medium']
    cold_med = df['climate_risk_cnt_locations_unseasonably_cold_risk_medium']
    drought_high = df['climate_risk_cnt_locations_drought_risk_high']
    heat_high = df['climate_risk_cnt_locations_heat_stress_risk_high']
    excess_high = df['climate_risk_cnt_locations_excess_precip_risk_high']
    cold_high = df['climate_risk_cnt_locations_unseasonably_cold_risk_high']
    weight = df['percent_country_production'] / 100

    # Weighted sum of all medium risks
    feat_name = f'{feature_prefix}_w_all_med_sum'
    df[feat_name] = (drought_med + heat_med + excess_med + cold_med) * weight
    df[feat_name] = df[feat_name].replace([np.inf, -np.inf], 0).fillna(0)
    created_features_list.append(feat_name)
    feat_count += 1

    # Weighted sum of all high risks
    feat_name = f'{feature_prefix}_w_all_high_sum'
    df[feat_name] = (drought_high + heat_high + excess_high + cold_high) * weight
    df[feat_name] = df[feat_name].replace([np.inf, -np.inf], 0).fillna(0)
    created_features_list.append(feat_name)
    feat_count += 1

    # Weighted drought+heat (common pairing)
    feat_name = f'{feature_prefix}_w_drought_heat_med_sum'
    df[feat_name] = (drought_med + heat_med) * weight
    df[feat_name] = df[feat_name].replace([np.inf, -np.inf], 0).fillna(0)
    created_features_list.append(feat_name)
    feat_count += 1

    # Weighted drought+excess
    feat_name = f'{feature_prefix}_w_drought_excess_med_sum'
    df[feat_name] = (drought_med + excess_med) * weight
    df[feat_name] = df[feat_name].replace([np.inf, -np.inf], 0).fillna(0)
    created_features_list.append(feat_name)
    feat_count += 1

    # Weighted heat+excess+cold (non-drought)
    feat_name = f'{feature_prefix}_w_non_drought_med_sum'
    df[feat_name] = (heat_med + excess_med + cold_med) * weight
    df[feat_name] = df[feat_name].replace([np.inf, -np.inf], 0).fillna(0)
    created_features_list.append(feat_name)
    feat_count += 1

    # Double weighted (weight squared)
    feat_name = f'{feature_prefix}_w2_all_med_sum'
    df[feat_name] = (drought_med + heat_med + excess_med + cold_med) * (weight ** 2)
    df[feat_name] = df[feat_name].replace([np.inf, -np.inf], 0).fillna(0)
    created_features_list.append(feat_name)
    feat_count += 1

    # Combined high+med weighted
    feat_name = f'{feature_prefix}_w_all_combined_sum'
    df[feat_name] = (drought_med + drought_high + heat_med + heat_high + excess_med + excess_high) * weight
    df[feat_name] = df[feat_name].replace([np.inf, -np.inf], 0).fillna(0)
    created_features_list.append(feat_name)
    feat_count += 1

    # Weighted max of medium risks
    feat_name = f'{feature_prefix}_w_max_med'
    df[feat_name] = df[['climate_risk_cnt_locations_drought_risk_medium',
                        'climate_risk_cnt_locations_heat_stress_risk_medium',
                        'climate_risk_cnt_locations_excess_precip_risk_medium',
                        'climate_risk_cnt_locations_unseasonably_cold_risk_medium']].max(axis=1) * weight
    df[feat_name] = df[feat_name].replace([np.inf, -np.inf], 0).fillna(0)
    created_features_list.append(feat_name)
    feat_count += 1

    return feat_count

print("Creating weighted sum features...")
feature_count += create_weighted_sum_features(
    df=merged_df,
    feature_prefix='climate_risk_wsum',
    created_features_list=ALL_NEW_FEATURES
)

# Aggregate weighted sum features
WSUM_COLS = [c for c in merged_df.columns if c.startswith('climate_risk_wsum_')]
if WSUM_COLS:
    print(f"Aggregating {len(WSUM_COLS)} weighted sum features by quartile...")
    feature_count += create_groupby_agg_features(
        df=merged_df,
        source_cols=WSUM_COLS,
        groupby_cols=['country_name', 'climate_risk_time_bin_quartile'],
        agg_funcs=['mean', 'max', 'min', 'std', 'var', 'sum'],
        feature_prefix='climate_risk_wsum_quartile_agg',
        created_features_list=ALL_NEW_FEATURES
    )

    print(f"Aggregating weighted sum features by quintile...")
    feature_count += create_groupby_agg_features(
        df=merged_df,
        source_cols=WSUM_COLS,
        groupby_cols=['country_name', 'climate_risk_time_bin_quintile'],
        agg_funcs=['mean', 'max', 'min', 'std', 'var', 'sum'],
        feature_prefix='climate_risk_wsum_quintile_agg',
        created_features_list=ALL_NEW_FEATURES
    )

print(f"Created {feature_count} weighted sum features")


# In[40]:


#  MISSING TIME-BIN AGGREGATIONS FOR TOP PERFORMERS (FEATURES)
# ============================================================================

print("\n Missing Time-Bin Aggregations")
feature_count = 0

# --- allrisk: missing quartile, sextile, tertile ---
ALLRISK_BASE = [c for c in merged_df.columns if c.startswith('climate_risk_allrisk_') and '_agg_' not in c]
if ALLRISK_BASE:
    print("Adding allrisk + quartile aggregations...")
    feature_count += create_groupby_agg_features(
        df=merged_df,
        source_cols=ALLRISK_BASE,
        groupby_cols=['country_name', 'climate_risk_time_bin_quartile'],
        agg_funcs=['sum', 'mean', 'max', 'std', 'var'],
        feature_prefix='climate_risk_allrisk_quartile_agg',
        created_features_list=ALL_NEW_FEATURES
    )

    print("Adding allrisk + sextile aggregations...")
    feature_count += create_groupby_agg_features(
        df=merged_df,
        source_cols=ALLRISK_BASE,
        groupby_cols=['country_name', 'climate_risk_time_bin_sextile'],
        agg_funcs=['sum', 'mean', 'max', 'std', 'var'],
        feature_prefix='climate_risk_allrisk_sextile_agg',
        created_features_list=ALL_NEW_FEATURES
    )

    print("Adding allrisk + tertile aggregations...")
    feature_count += create_groupby_agg_features(
        df=merged_df,
        source_cols=ALLRISK_BASE,
        groupby_cols=['country_name', 'climate_risk_time_bin_tertile'],
        agg_funcs=['sum', 'mean', 'max', 'std', 'var'],
        feature_prefix='climate_risk_allrisk_tertile_agg',
        created_features_list=ALL_NEW_FEATURES
    )

# --- compound_med: missing quartile ---
COMPOUND_MED_BASE = [c for c in merged_df.columns if c.startswith('climate_risk_compound_med_') and '_agg_' not in c]
if COMPOUND_MED_BASE:
    print("Adding compound_med + quartile aggregations...")
    feature_count += create_groupby_agg_features(
        df=merged_df,
        source_cols=COMPOUND_MED_BASE,
        groupby_cols=['country_name', 'climate_risk_time_bin_quartile'],
        agg_funcs=['sum', 'mean', 'max', 'std', 'var'],
        feature_prefix='climate_risk_compound_med_quartile_agg',
        created_features_list=ALL_NEW_FEATURES
    )

# --- wsum: missing sextile, tertile ---
WSUM_BASE = [c for c in merged_df.columns if c.startswith('climate_risk_wsum_') and '_agg_' not in c]
if WSUM_BASE:
    print("Adding wsum + sextile aggregations...")
    feature_count += create_groupby_agg_features(
        df=merged_df,
        source_cols=WSUM_BASE,
        groupby_cols=['country_name', 'climate_risk_time_bin_sextile'],
        agg_funcs=['sum', 'mean', 'max', 'std', 'var'],
        feature_prefix='climate_risk_wsum_sextile_agg',
        created_features_list=ALL_NEW_FEATURES
    )

    print("Adding wsum + tertile aggregations...")
    feature_count += create_groupby_agg_features(
        df=merged_df,
        source_cols=WSUM_BASE,
        groupby_cols=['country_name', 'climate_risk_time_bin_tertile'],
        agg_funcs=['sum', 'mean', 'max', 'std', 'var'],
        feature_prefix='climate_risk_wsum_tertile_agg',
        created_features_list=ALL_NEW_FEATURES
    )

# --- de_compound: missing quartile, quintile ---
DE_COMPOUND_BASE = [c for c in merged_df.columns if c.startswith('climate_risk_de_compound_') and '_agg_' not in c]
if DE_COMPOUND_BASE:
    print("Adding de_compound + quartile aggregations...")
    feature_count += create_groupby_agg_features(
        df=merged_df,
        source_cols=DE_COMPOUND_BASE,
        groupby_cols=['country_name', 'climate_risk_time_bin_quartile'],
        agg_funcs=['sum', 'mean', 'max', 'min', 'std', 'var'],
        feature_prefix='climate_risk_de_compound_quartile_agg',
        created_features_list=ALL_NEW_FEATURES
    )

    print("Adding de_compound + quintile aggregations...")
    feature_count += create_groupby_agg_features(
        df=merged_df,
        source_cols=DE_COMPOUND_BASE,
        groupby_cols=['country_name', 'climate_risk_time_bin_quintile'],
        agg_funcs=['sum', 'mean', 'max', 'min', 'std', 'var'],
        feature_prefix='climate_risk_de_compound_quintile_agg',
        created_features_list=ALL_NEW_FEATURES
    )

# --- wcompound: missing quartile, quintile ---
WCOMPOUND_BASE = [c for c in merged_df.columns if c.startswith('climate_risk_wcompound_') and '_agg_' not in c]
if WCOMPOUND_BASE:
    print("Adding wcompound + quartile aggregations...")
    feature_count += create_groupby_agg_features(
        df=merged_df,
        source_cols=WCOMPOUND_BASE,
        groupby_cols=['country_name', 'climate_risk_time_bin_quartile'],
        agg_funcs=['sum', 'mean', 'max', 'std', 'var'],
        feature_prefix='climate_risk_wcompound_quartile_agg',
        created_features_list=ALL_NEW_FEATURES
    )

    print("Adding wcompound + quintile aggregations...")
    feature_count += create_groupby_agg_features(
        df=merged_df,
        source_cols=WCOMPOUND_BASE,
        groupby_cols=['country_name', 'climate_risk_time_bin_quintile'],
        agg_funcs=['sum', 'mean', 'max', 'std', 'var'],
        feature_prefix='climate_risk_wcompound_quintile_agg',
        created_features_list=ALL_NEW_FEATURES
    )

# --- product: missing quintile, sextile ---
PRODUCT_BASE = [c for c in merged_df.columns if c.startswith('climate_risk_product_') and '_agg_' not in c]
if PRODUCT_BASE:
    print("Adding product + quintile aggregations...")
    feature_count += create_groupby_agg_features(
        df=merged_df,
        source_cols=PRODUCT_BASE,
        groupby_cols=['country_name', 'climate_risk_time_bin_quintile'],
        agg_funcs=['sum', 'mean', 'max', 'std', 'var'],
        feature_prefix='climate_risk_product_quintile_agg',
        created_features_list=ALL_NEW_FEATURES
    )

    print("Adding product + sextile aggregations...")
    feature_count += create_groupby_agg_features(
        df=merged_df,
        source_cols=PRODUCT_BASE,
        groupby_cols=['country_name', 'climate_risk_time_bin_sextile'],
        agg_funcs=['sum', 'mean', 'max', 'std', 'var'],
        feature_prefix='climate_risk_product_sextile_agg',
        created_features_list=ALL_NEW_FEATURES
    )

print(f"Created {feature_count} missing time-bin aggregation features")


# In[41]:


# SUMMARY OF FEATURE ENGINEERING
# ============================================================================
print("\n" + "="*80)
print("FEATURE ENGINEERING SUMMARY")
print("="*80)
print(f"Total new features created: {len(ALL_NEW_FEATURES)}")
print(f"Original climate risk columns: {len(climate_cols)}")
print(f"Grand total climate features: {len(climate_cols) + len(ALL_NEW_FEATURES)}")


# In[42]:


# REMOVE HIGHLY CORRELATED FEATURES (>= 99% correlation)
# ============================================================================
print("\n" + "="*80)
print("REMOVING HIGHLY CORRELATED FEATURES")
print("="*80)

def remove_highly_correlated_features(df, feature_cols, threshold=0.99):
    """
    Remove features that have >= threshold correlation with other features.
    Keeps the first feature encountered, removes subsequent correlated features.
    """
    n_features = len(feature_cols)
    print(f"Calculating correlation matrix for {n_features} features")

    # Sample rows if too many (correlation is stable with ~50k samples)
    if len(df) > 50000:
        sample_df = df[feature_cols].sample(n=100000, random_state=42)
    else:
        sample_df = df[feature_cols]

    # Convert to numpy float32 (faster and less memory)
    data = sample_df.values.astype(np.float32)

    # Handle NaN values
    data = np.nan_to_num(data, nan=0.0)

    # Standardize columns (required for correlation via dot product)
    means = np.mean(data, axis=0)
    stds = np.std(data, axis=0)
    stds[stds == 0] = 1  # Avoid division by zero

    data_norm = (data - means) / stds

    # Compute correlation matrix using matrix multiplication (much faster)
    print(f"Computing correlations...")
    corr_matrix = (data_norm.T @ data_norm) / len(data_norm)

    # Take absolute values
    corr_matrix = np.abs(corr_matrix)

    # Find features to drop using vectorized operations
    features_to_drop = set()
    correlation_pairs = []

    # Process upper triangle only
    for i in range(n_features):
        if i in features_to_drop:
            continue
        for j in range(i + 1, n_features):
            if j in features_to_drop:
                continue
            if corr_matrix[i, j] >= threshold:
                features_to_drop.add(j)
                correlation_pairs.append((feature_cols[i], feature_cols[j], corr_matrix[i, j]))

    features_to_drop_names = [feature_cols[i] for i in features_to_drop]
    features_to_keep = [f for f in feature_cols if f not in features_to_drop_names]

    return features_to_keep, features_to_drop_names, correlation_pairs

# Get all climate risk features (original + engineered)
all_climate_features = [c for c in merged_df.columns if c.startswith('climate_risk_')]
print(f"Total climate features before removal: {len(all_climate_features)}")

# ============================================================================
# Remove FEATURES_TO_REMOVE features and their highly correlated counterparts
# ============================================================================
# Find features with >= CORRELATION_THRESHOLD_SPECIFIC correlation with FEATURES_TO_REMOVE features
print(f"\n  Removing FEATURES_TO_REMOVE features and their correlated counterparts (>= {CORRELATION_THRESHOLD_SPECIFIC*100:.0f}%)...")
print(f"FEATURES_TO_REMOVE features: {len(FEATURES_TO_REMOVE)}")

# Check which FEATURES_TO_REMOVE features exist in the dataframe
existing_features_to_remove = [f for f in FEATURES_TO_REMOVE if f in merged_df.columns]
print(f"Found in dataframe: {len(existing_features_to_remove)}")

features_to_remove_with_correlated = set(existing_features_to_remove)

if existing_features_to_remove:
    # Only compute correlations for FEATURES_TO_REMOVE vs all features
    print("Computing correlations for FEATURES_TO_REMOVE features.")

    # Sample rows if too many
    if len(merged_df) > 50000:
        sample_df = merged_df.sample(n=100000, random_state=42)
    else:
        sample_df = merged_df

    # Get data for all features and FEATURES_TO_REMOVE features
    # all_data = sample_df[all_climate_features].values.astype(np.float32)
    all_data = sample_df[all_climate_features].copy()
    all_data = np.nan_to_num(all_data, nan=0.0)

    # Standardize
    means = np.mean(all_data, axis=0)
    stds = np.std(all_data, axis=0)
    stds[stds == 0] = 1
    all_data_norm = (all_data - means) / stds

    # For each FEATURES_TO_REMOVE feature, find correlated features
    for target_feat in existing_features_to_remove:
        if target_feat in all_climate_features:
            target_idx = all_climate_features.index(target_feat)
            target_col = all_data_norm[:, target_idx]

            # Compute correlation with all features
            corrs = (all_data_norm.T @ target_col) / len(target_col)
            corrs = np.abs(corrs)

            # Find features with >= threshold correlation
            for i, corr_val in enumerate(corrs):
                if corr_val >= CORRELATION_THRESHOLD_SPECIFIC and all_climate_features[i] != target_feat:
                    features_to_remove_with_correlated.add(all_climate_features[i])

print(f"Total features to remove (including {CORRELATION_THRESHOLD_SPECIFIC*100:.0f}% correlated): {len(features_to_remove_with_correlated)}")

# Show what's being removed
if len(features_to_remove_with_correlated) > len(existing_features_to_remove):
    print(f"\n  Additional correlated features being removed:")
    additional = features_to_remove_with_correlated - set(existing_features_to_remove)
    for feat in list(additional)[:20]:  # Show first 20
        print(f"- {feat}")
    if len(additional) > 20:
        print(f"... and {len(additional) - 20} more")

# Remove from dataframe
features_to_remove_list = [f for f in features_to_remove_with_correlated if f in merged_df.columns]
if features_to_remove_list:
    merged_df = merged_df.drop(columns=features_to_remove_list)
    ALL_NEW_FEATURES = [f for f in ALL_NEW_FEATURES if f not in features_to_remove_with_correlated]
    print(f"\n  Dropped {len(features_to_remove_list)} features from dataframe")

# Update all_climate_features list
all_climate_features = [c for c in merged_df.columns if c.startswith('climate_risk_')]
print(f"Remaining climate features: {len(all_climate_features)}")

# ============================================================================
# Remove remaining highly correlated features
# ============================================================================
print(f"\n  Now removing remaining features with >= {CORRELATION_THRESHOLD_GENERAL*100:.0f}% correlation...")

# Remove highly correlated features
features_to_keep, features_to_drop, correlation_pairs = remove_highly_correlated_features(
    merged_df,
    all_climate_features,
    threshold=CORRELATION_THRESHOLD_GENERAL
)

print(f"Features to drop (>= {CORRELATION_THRESHOLD_GENERAL*100:.0f}% correlation): {len(features_to_drop)}")
print(f"Features to keep: {len(features_to_keep)}")

# Show some examples of dropped correlations
if len(correlation_pairs) > 0:
    print(f"\n  Sample highly correlated pairs (showing first 10):")
    for feat1, feat2, corr_val in correlation_pairs[:10]:
        print(f"{feat1[:50]:50s} <-> {feat2[:50]:50s} : {corr_val:.4f}")

# Drop the highly correlated features from the dataframe
if features_to_drop:
    merged_df = merged_df.drop(columns=features_to_drop)
    # Also remove from ALL_NEW_FEATURES list
    ALL_NEW_FEATURES = [f for f in ALL_NEW_FEATURES if f not in features_to_drop]
    print(f"\n  Dropped {len(features_to_drop)} highly correlated features from dataframe")

# Update climate_cols to reflect removed features
climate_cols = [c for c in merged_df.columns if c.startswith('climate_risk_cnt_locations_')]
print(f"Remaining climate features: {len([c for c in merged_df.columns if c.startswith('climate_risk_')])}")


# In[43]:


# FEATURE ANALYSIS
# ============================================================================
print("\n" + "="*80)
print(" FEATURE ANALYSIS")
print("="*80)

def analyze_feature_contributions(df, climate_cols, futures_cols):
    """Analyze which climate features have significant correlations with futures."""

    # Initialize stats for all features
    feature_stats = {col: {'sig_count': 0, 'total_count': 0, 'max_corr': 0, 'sig_corrs': []}
                     for col in climate_cols}

    # Pre-group data once (much faster than filtering repeatedly)
    print("Grouping data by country and month...")
    grouped = df.groupby(['country_name', 'date_on_month'])

    total_groups = len(grouped)
    print(f"Processing {total_groups} groups...")

    # Process each group once
    for group_idx, ((country, month), group_df) in enumerate(grouped):
        if group_idx % 100 == 0:
            print(f"Progress: {group_idx}/{total_groups} groups processed", end='\r')

        if len(group_df) < 2:
            continue

        # Convert to numpy arrays for faster computation
        climate_data = group_df[climate_cols].values
        futures_data = group_df[futures_cols].values

        # Pre-compute standard deviations for all columns
        climate_std = np.std(climate_data, axis=0)
        futures_std = np.std(futures_data, axis=0)

        # Find which columns have variance
        valid_climate = climate_std > 0
        valid_futures = futures_std > 0

        # For each valid climate feature
        for i, clim_col in enumerate(climate_cols):
            if not valid_climate[i]:
                continue

            # For each valid futures column
            for j, fut_col in enumerate(futures_cols):
                if not valid_futures[j]:
                    continue

                # Compute correlation using numpy (faster than pandas)
                corr = np.corrcoef(climate_data[:, i], futures_data[:, j])[0, 1]

                if not np.isnan(corr):
                    abs_corr = abs(corr)
                    feature_stats[clim_col]['total_count'] += 1
                    feature_stats[clim_col]['max_corr'] = max(feature_stats[clim_col]['max_corr'], abs_corr)

                    if abs_corr >= SIGNIFICANCE_THRESHOLD:
                        feature_stats[clim_col]['sig_count'] += 1
                        feature_stats[clim_col]['sig_corrs'].append(abs_corr)

    print(f"Progress: {total_groups}/{total_groups} groups processed")

    # Convert to DataFrame
    results = []
    for clim_col in climate_cols:
        stats = feature_stats[clim_col]
        avg_sig_corr = np.mean(stats['sig_corrs']) if stats['sig_corrs'] else 0
        results.append({
            'feature': clim_col,
            'sig_count': stats['sig_count'],
            'total_count': stats['total_count'],
            'max_corr': round(stats['max_corr'], 4),
            'avg_sig_corr': round(avg_sig_corr, 4)
        })

    return pd.DataFrame(results).sort_values('sig_count', ascending=False)

# Prepare baseline dataset
futures_cols = [c for c in merged_df.columns if c.startswith('futures_')]
baseline_df = merged_df.dropna(subset=futures_cols)

# Filter to valid rows (Recreate ALL features in temp_df)
print("\nIdentifying valid IDs (matching sample submission approach)...")
temp_df = pd.read_csv(f'{DATA_PATH}corn_climate_risk_futures_daily_master.csv')
temp_df['date_on'] = pd.to_datetime(temp_df['date_on'])

# Add basic features
temp_df['day_of_year'] = temp_df['date_on'].dt.dayofyear
temp_df['quarter'] = temp_df['date_on'].dt.quarter

# Merge market share
temp_df = temp_df.merge(
    market_share_df[['region_id', 'percent_country_production']],
    on='region_id', how='left'
)
temp_df['percent_country_production'] = temp_df['percent_country_production'].fillna(1.0)

# Create base risk scores
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

# Create rolling features (7, 14, 30 days)
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

# Create momentum features
for risk_type in RISK_CATEGORIES:
    score_col = f'climate_risk_{risk_type}_score'
    temp_df[f'climate_risk_{risk_type}_change_1d'] = temp_df.groupby('region_id')[score_col].diff(1)
    temp_df[f'climate_risk_{risk_type}_change_7d'] = temp_df.groupby('region_id')[score_col].diff(7)
    temp_df[f'climate_risk_{risk_type}_acceleration'] = temp_df.groupby('region_id')[f'climate_risk_{risk_type}_change_1d'].diff(1)

# Create country aggregations
for risk_type in RISK_CATEGORIES:
    score_col = f'climate_risk_{risk_type}_score'
    weighted_col = f'climate_risk_{risk_type}_weighted'

    country_agg = temp_df.groupby(['country_name', 'date_on']).agg({
        score_col: ['mean', 'max', 'min', 'std', 'var', 'sum'],
        weighted_col: 'sum',
        'percent_country_production': 'sum'
    }).round(4)

    country_agg.columns = [f'country_{risk_type}_{"_".join(col).strip()}' for col in country_agg.columns]
    country_agg = country_agg.reset_index()

    temp_df = temp_df.merge(country_agg, on=['country_name', 'date_on'], how='left')

# Get valid IDs
valid_ids = temp_df.dropna()['ID'].tolist()
print(f"Valid IDs from sample submission approach: {len(valid_ids):,}")

# Clean up
del temp_df

# Filter baseline_df to valid IDs
baseline_df = baseline_df[baseline_df['ID'].isin(valid_ids)]

print(f"\nBaseline dataset: {len(baseline_df):,} rows")

# Analyze features
print("\nAnalyzing feature contributions...")
climate_features = [c for c in baseline_df.columns if c.startswith('climate_risk_')]
print(f"Climate features to analyze (before removal): {len(climate_features)}")

# Time bin columns are auxiliary (used for grouping, not as direct features)
AUXILIARY_FEATURES_TO_REMOVE = [
    "climate_risk_time_bin_tertile",
    "climate_risk_time_bin_quartile",
    "climate_risk_time_bin_quintile",
    "climate_risk_time_bin_sextile",
    "climate_risk_time_bin_octile",
    "climate_risk_time_bin_decile",
    "climate_risk_time_bin_tredecile",
    "climate_risk_time_bin_vigintile",

    "climate_risk_quartile_agg_heat_stress_risk_high_max",
    "climate_risk_tertile_agg_heat_stress_risk_high_max",
    "climate_risk_country_sextile_std_drought_risk_high",
    "climate_risk_quartile_agg_drought_risk_high_mean"
]

print(f"\nRemoving {len(AUXILIARY_FEATURES_TO_REMOVE)} auxiliary features...")
features_to_drop = [f for f in AUXILIARY_FEATURES_TO_REMOVE if f in baseline_df.columns]
print(f"Found {len(features_to_drop)} features to drop from dataframe")
baseline_df = baseline_df.drop(columns=features_to_drop)

# Update climate_features list
climate_features = [c for c in baseline_df.columns if c.startswith('climate_risk_')]
print(f"Climate features to analyze (after removal): {len(climate_features)}")

feature_analysis = analyze_feature_contributions(baseline_df, climate_features, futures_cols)

# Show top features
print("\nTOP 50 Features by Significant Correlation Count:")
print("="*80)
print(feature_analysis.head(50).to_string(index=False))


# In[44]:


# Show top features
print("\nBottom 300 Features by Significant Correlation Count:")
print("="*80)
print(feature_analysis.tail(300).to_string(index=False))


# In[45]:


# SCORING WITH TOP 10 FEATURES
# ============================================================================
print("\n" + "="*80)
print(f" SCORING WITH TOP {TOP_N_FEATURES} FEATURES")
print("="*80)

print(f"\nFeature selection strategy: {FEATURE_SELECTION_STRATEGY}")

if FEATURE_SELECTION_STRATEGY == 'sig_count':
    top_features = feature_analysis.nlargest(TOP_N_FEATURES, 'sig_count')['feature'].tolist()
elif FEATURE_SELECTION_STRATEGY == 'max_corr':
    top_features = feature_analysis.nlargest(TOP_N_FEATURES, 'max_corr')['feature'].tolist()
elif FEATURE_SELECTION_STRATEGY == 'avg_sig_corr':
    top_features = feature_analysis.nlargest(TOP_N_FEATURES, 'avg_sig_corr')['feature'].tolist()
elif FEATURE_SELECTION_STRATEGY == 'weighted':
    # Weighted combination: normalize each metric and combine
    fa = feature_analysis.copy()
    # Normalize each metric to [0, 1]
    fa['sig_count_norm'] = fa['sig_count'] / (fa['sig_count'].max() + 1e-6)
    fa['max_corr_norm'] = fa['max_corr'] / (fa['max_corr'].max() + 1e-6)
    fa['avg_sig_corr_norm'] = fa['avg_sig_corr'] / (fa['avg_sig_corr'].max() + 1e-6)
    # Weighted score (adjust weights as needed)
    fa['combined_score'] = (0.4 * fa['sig_count_norm'] +
                            0.3 * fa['max_corr_norm'] +
                            0.3 * fa['avg_sig_corr_norm'])
    top_features = fa.nlargest(TOP_N_FEATURES, 'combined_score')['feature'].tolist()
    print(f"\nWeighted scoring (sig_count: 0.4, max_corr: 0.3, avg_sig_corr: 0.3)")
else:
    # Default to sig_count
    top_features = feature_analysis.nlargest(TOP_N_FEATURES, 'sig_count')['feature'].tolist()

print(f"\nSelected top {TOP_N_FEATURES} features:")
for i, feat in enumerate(top_features, 1):
    row = feature_analysis[feature_analysis['feature'] == feat].iloc[0]
    print(f" {i}. {feat}")
    print(f"  sig_count: {row['sig_count']}, max_corr: {row['max_corr']:.4f}, avg_sig_corr: {row['avg_sig_corr']:.4f}")

# Compute CFCS score
def compute_cfcs(df, climate_cols=None):
    """Compute CFCS score."""
    if climate_cols is None:
        climate_cols = [c for c in df.columns if c.startswith('climate_risk_')]

    futures_cols = [c for c in df.columns if c.startswith('futures_')]

    feature_stats = {col: {'sig_count': 0, 'total': 0, 'max_corr': 0, 'sig_corrs': []}
                     for col in climate_cols}

    # Pre-group data once
    grouped = df.groupby(['country_name', 'date_on_month'])

    # Process each group once
    for (country, month), group_df in grouped:
        if len(group_df) < 2:
            continue

        # Convert to numpy arrays
        climate_data = group_df[climate_cols].values
        futures_data = group_df[futures_cols].values

        # Pre-compute standard deviations
        climate_std = np.std(climate_data, axis=0)
        futures_std = np.std(futures_data, axis=0)

        valid_climate = climate_std > 0
        valid_futures = futures_std > 0

        # For each valid climate feature
        for i, clim_col in enumerate(climate_cols):
            if not valid_climate[i]:
                continue

            for j in range(len(futures_cols)):
                if not valid_futures[j]:
                    continue

                # Compute correlation using numpy
                corr = np.corrcoef(climate_data[:, i], futures_data[:, j])[0, 1]

                if not np.isnan(corr):
                    abs_corr = abs(corr)
                    feature_stats[clim_col]['total'] += 1
                    feature_stats[clim_col]['max_corr'] = max(feature_stats[clim_col]['max_corr'], abs_corr)

                    if abs_corr >= SIGNIFICANCE_THRESHOLD:
                        feature_stats[clim_col]['sig_count'] += 1
                        feature_stats[clim_col]['sig_corrs'].append(abs_corr)

    avg_sig_corr = np.mean([np.mean(stats['sig_corrs']) if stats['sig_corrs'] else 0
                             for stats in feature_stats.values()])

    total_sig_count = sum(stats['sig_count'] for stats in feature_stats.values())
    total_correlations = sum(stats['total'] for stats in feature_stats.values())
    sig_pct = (total_sig_count / total_correlations * 100) if total_correlations > 0 else 0

    max_corr = max(stats['max_corr'] for stats in feature_stats.values()) if feature_stats else 0

    cfcs = (0.5 * avg_sig_corr) + (0.3 * max_corr) + (0.2 * sig_pct / 100)

    return {
        'cfcs': round(cfcs, 6),
        'avg_sig_corr': round(avg_sig_corr, 6),
        'max_corr': round(max_corr, 6),
        'sig_count': total_sig_count,
        'total': total_correlations,
        'sig_pct': round(sig_pct, 4),
        'n_features': len(climate_cols)
    }

# Compute score with top features only
top_features_score = compute_cfcs(baseline_df, top_features)

print(f"\nCFCS Score (Top {TOP_N_FEATURES} features):")
print(f" CFCS: {top_features_score['cfcs']}")
print(f" Avg Sig Corr: {top_features_score['avg_sig_corr']}")
print(f" Max Corr: {top_features_score['max_corr']}")
print(f" Sig Count: {top_features_score['sig_count']}/{top_features_score['total']} ({top_features_score['sig_pct']:.2f}%)")
print(f" Features: {top_features_score['n_features']}")


# In[46]:


# CREATE SUBMISSION
# ============================================================================
print("\n" + "="*80)
print(" CREATING SUBMISSION")
print("="*80)

# Get futures columns
futures_cols = [c for c in baseline_df.columns if c.startswith('futures_')]

# Filter to only include top N features (not all climate_risk features)
required_cols = ['ID', 'date_on', 'country_name', 'region_name'] + futures_cols + top_features
submission = baseline_df[required_cols].copy()

# Fill any remaining nulls with 0
if submission.isnull().sum().sum() > 0:
    submission = submission.fillna(0)

# Save submission
import os
os.makedirs(OUTPUT_PATH, exist_ok=True)
output_file = f'{OUTPUT_PATH}submission.csv'
submission.to_csv(output_file, index=False)

climate_cols = [c for c in submission.columns if c.startswith('climate_risk_')]
futures_cols = [c for c in submission.columns if c.startswith('futures_')]

print(f"\nSubmission saved: {output_file}")
print(f" Rows: {len(submission):,}")
print(f" Columns: {len(submission.columns)}")
print(f" Climate features: {len(climate_cols)}")
print(f" Futures columns: {len(futures_cols)}")

print("\n" + "="*80)
print(" COMPLETE!")
print("="*80)
print(f"\nSummary:")
print(f" Total features created: {len(ALL_NEW_FEATURES)}")
print(f" Top {TOP_N_FEATURES} features selected for submission")
print(f" CFCS Score: {top_features_score['cfcs']}")
print("="*80)

