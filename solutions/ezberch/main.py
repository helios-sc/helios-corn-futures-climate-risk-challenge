import pandas as pd
import numpy as np
import warnings
import os
import gc
from concurrent.futures import ProcessPoolExecutor, as_completed

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs): return x

# --- CONFIGURATION ---
LB_START_DATE = '2023-12-20' 
# Windows: Added 7 for ultra-fast reactions, 14 for fast, kept long windows for trends
WINDOWS = [7, 14, 30, 60, 120, 240, 365, 450, 500]
# Powers: Added 3.0 for extreme tail events
POWERS = [0.25, 0.5, 1.0, 2.0, 3.0]
REDUNDANCY_CAP = 0.90 
TARGET_PHALANX = 80 
# Added 0.20 floor to capture weaker but unique signals
FLOORS_TO_TEST = [0.20, 0.25, 0.30, 0.35, 0.40]
# Increased K to find more candidates for optimizer
PRE_SELECT_K = 1500 
MAX_WORKERS = 4 

warnings.filterwarnings('ignore')

def reduce_mem_usage(df, verbose=False):
    """
    Downcasts Integers to save memory. 
    Floats are kept as float64 to maintain exact scoring precision.
    """
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object and col_type != '<M8[ns]':
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            # SKIPPING float downcasting to preserve precision
    if verbose:
        end_mem = df.memory_usage().sum() / 1024**2
        print(f'Memory usage reduced to {end_mem:.2f} MB ({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)')
    return df

def load_data():
    potential_dirs = ['../../data', '../../../data', 'data', '/kaggle/input/forecasting-the-future-the-helios-corn-climate-challenge']
    for d in potential_dirs:
        if os.path.exists(d) and os.path.exists(os.path.join(d, 'corn_climate_risk_futures_daily_master.csv')):
            df = pd.read_csv(os.path.join(d, 'corn_climate_risk_futures_daily_master.csv'))
            df['date_on'] = pd.to_datetime(df['date_on'])
            return df, pd.read_csv(os.path.join(d, 'corn_regional_market_share.csv'))
    raise FileNotFoundError("Data not found.")

def generate_b66_skeleton(df, market_share_df):
    orig_climate_cols = [c for c in df.columns if c.startswith('climate_risk_')]
    b_df = df.copy()
    b_df = b_df.merge(market_share_df[['region_id', 'percent_country_production']], on='region_id', how='left')
    b_df['percent_country_production'] = b_df['percent_country_production'].fillna(1.0)
    
    risk_cats = ['heat_stress', 'unseasonably_cold', 'excess_precip', 'drought']
    for r in risk_cats:
        l, m, h = [f'climate_risk_cnt_locations_{r}_risk_{x}' for x in ['low', 'medium', 'high']]
        if m not in b_df.columns: continue
        s_col = f'climate_risk_{r}_score'
        b_df[s_col] = (b_df[m] + 2 * b_df[h]) / (b_df[l] + b_df[m] + b_df[h] + 1e-6)
        w_col = f'climate_risk_{r}_weighted'
        b_df[w_col] = b_df[s_col] * (b_df['percent_country_production'] / 100)

    b_df = b_df.sort_values(['region_id', 'date_on'])
    
    for w in [7, 14, 30]:
        for r in risk_cats:
            s_col = f'climate_risk_{r}_score'
            for stat in ['ma', 'max']:
                col_name = f'climate_risk_{r}_{stat}_{w}d'
                if stat == 'ma':
                    b_df[col_name] = b_df.groupby('region_id')[s_col].rolling(window=w, min_periods=1).mean().reset_index(level=0, drop=True)
                else:
                    b_df[col_name] = b_df.groupby('region_id')[s_col].rolling(window=w, min_periods=1).max().reset_index(level=0, drop=True)

    for r in risk_cats:
        s_col = f'climate_risk_{r}_score'
        for suff in ['change_1d', 'change_7d', 'acceleration']:
            col_name = f'climate_risk_{r}_{suff}'
            if suff == 'change_1d':
                b_df[col_name] = b_df.groupby('region_id')[s_col].diff(1)
            elif suff == 'change_7d':
                b_df[col_name] = b_df.groupby('region_id')[s_col].diff(7)
            else:
                b_df[col_name] = b_df.groupby('region_id')[f'climate_risk_{r}_change_1d'].diff(1)

    for r in risk_cats:
        s_col = f'climate_risk_{r}_score'
        w_col = f'climate_risk_{r}_weighted'
        country_agg = b_df.groupby(['country_name', 'date_on']).agg({
            s_col: ['mean', 'max', 'std'],
            w_col: 'sum',
            'percent_country_production': 'sum'
        })
        country_agg.columns = [f'country_{r}_{"_".join(col).strip()}' for col in country_agg.columns]
        b_df = b_df.merge(country_agg.reset_index(), on=['country_name', 'date_on'], how='left')

    filtered_df = b_df.dropna()
    cr_baseline = [c for c in filtered_df.columns if c.startswith('climate_risk_') and c not in orig_climate_cols]
    return reduce_mem_usage(filtered_df), cr_baseline

def generate_base_daily_features(df, share, use_external=False):
    """Generates the base daily dataframe (aggregated regions, interactions) WITHOUT expanding variants yet."""
    df_calc = df.copy()
    df_calc = df_calc.merge(share[['region_id', 'percent_country_production']], on='region_id', how='left')
    df_calc['month'] = df_calc['date_on'].dt.month
    
    # REVERTED TO STANDARD PHENOLOGY (Non-Circular) - Proven superior
    PHENO_MAP = {'US': (7.2, 1.4, 'United States'), 'BR_Sum': (2.5, 3.0, 'Brazil'), 
                 'BR_Saf': (5.5, 2.5, 'Brazil'), 'AR': (1.5, 2.0, 'Argentina'), 'UA': (7.0, 1.5, 'Ukraine')}
    
    risk_types = ['heat_stress', 'drought', 'excess_precip', 'unseasonably_cold']
    daily = pd.DataFrame({'date_on': df_calc['date_on'].unique()})
    
    # Critical Sort
    daily = daily.sort_values('date_on').reset_index(drop=True)
    
    # 1. Standard Phenology Aggregation
    for pid, (pk, wd, country) in PHENO_MAP.items():
        c_df = df_calc[df_calc['country_name'] == country].copy()
        # Standard Gaussian (No circular min())
        c_df['m_weight'] = np.exp(-((c_df['month'] - pk)**2) / (2 * wd**2))
        
        for r in risk_types:
            cl, cm, ch = [f'climate_risk_cnt_locations_{r}_risk_{x}' for x in ['low', 'medium', 'high']]
            if cm not in c_df.columns: continue
            score = (c_df[cm] + 2 * c_df[ch]) / (c_df[cl] + c_df[cm] + c_df[ch] + 1e-6)
            c_df['v'] = score * (c_df['percent_country_production'] / 100) * c_df['m_weight']
            agg = c_df.groupby('date_on')['v'].sum().reset_index().rename(columns={'v': f'{pid}_{r}'})
            daily = daily.merge(agg, on='date_on', how='left')
            
    # 2. US Special Windows (Planting & Harvest)
    us_df = df_calc[df_calc['country_name'] == 'United States'].copy()
    
    # Planting: April/May (Standard Weighting)
    us_df['plant_weight'] = np.exp(-((us_df['month'] - 4.5)**2) / (2 * 1.0**2))
    # Harvest: October (Standard Weighting)
    us_df['harv_weight'] = np.exp(-((us_df['month'] - 10.0)**2) / (2 * 1.0**2))
    
    for r in ['excess_precip', 'unseasonably_cold']:
        if f'climate_risk_cnt_locations_{r}_risk_medium' not in us_df.columns: continue
        cl, cm, ch = [f'climate_risk_cnt_locations_{r}_risk_{x}' for x in ['low', 'medium', 'high']]
        score = (us_df[cm] + 2 * us_df[ch]) / (us_df[cl] + us_df[cm] + us_df[ch] + 1e-6)
        
        # Planting Features
        us_df['v_p'] = score * (us_df['percent_country_production'] / 100) * us_df['plant_weight']
        agg_p = us_df.groupby('date_on')['v_p'].sum().reset_index().rename(columns={'v_p': f'US_Planting_{r}'})
        daily = daily.merge(agg_p, on='date_on', how='left')
        
        # Harvest Features
        us_df['v_h'] = score * (us_df['percent_country_production'] / 100) * us_df['harv_weight']
        agg_h = us_df.groupby('date_on')['v_h'].sum().reset_index().rename(columns={'v_h': f'US_Harvest_{r}'})
        daily = daily.merge(agg_h, on='date_on', how='left')

    daily = daily.fillna(0)
    
    # 3. Compound Logic
    for r in risk_types:
        cols = [c for c in daily.columns if c.endswith(f'_{r}') and 'Inter' not in c and 'Global' not in c and 'Planting' not in c and 'Harvest' not in c]
        if cols: daily[f'Global_{r}'] = daily[cols].sum(axis=1)
        
        north_cols = [c for c in cols if 'US' in c or 'UA' in c]
        south_cols = [c for c in cols if 'BR' in c or 'AR' in c]
        if north_cols: daily[f'North_{r}'] = daily[north_cols].sum(axis=1)
        if south_cols: daily[f'South_{r}'] = daily[south_cols].sum(axis=1)
    
    # Interactions
    daily['Global_Heat_Drought'] = daily.get('Global_heat_stress',0) * daily.get('Global_drought',0)
    daily['US_Heat_Drought'] = daily.get('US_heat_stress',0) * daily.get('US_drought',0)
    daily['AR_Heat_Drought'] = daily.get('AR_heat_stress', 0) * daily.get('AR_drought', 0)
    
    daily['BR_Heat_Drought'] = (daily.get('BR_Sum_heat_stress', 0) + daily.get('BR_Saf_heat_stress', 0)) * \
                               (daily.get('BR_Sum_drought', 0) + daily.get('BR_Saf_drought', 0))
    
    daily['North_x_South_Failure'] = (daily.get('North_heat_stress',0) + daily.get('North_drought',0)) * \
                                     (daily.get('South_heat_stress',0) + daily.get('South_drought',0))
    
    # 4. External Data
    if use_external:
        # Load ONI (Existing)
        if os.path.exists('external_oni.csv'):
            oni = pd.read_csv('external_oni.csv')
            oni['date_on'] = pd.to_datetime(oni['date_on'])
            oni = oni.sort_values('date_on')
            daily = daily.merge(oni, on='date_on', how='left')
        
        # Load MJO/PDO (New)
        if os.path.exists('external_indices.csv'):
            indices = pd.read_csv('external_indices.csv')
            indices['date_on'] = pd.to_datetime(indices['date_on'])
            daily = daily.merge(indices, on='date_on', how='left')
            
        daily = daily.ffill().bfill() # Fill any gaps after merging all external sources
        
        # --- A. ONI (Nino/Nina) ---
        if 'climate_risk_ONI_index' in daily.columns:
            # La Niña Multiplier (cold phase)
            daily['La_Nina_Multiplier'] = daily['climate_risk_ONI_index'].apply(lambda x: abs(x) if x < -0.5 else 0)
            # El Niño Multiplier (warm phase)
            daily['El_Nino_Multiplier'] = daily['climate_risk_ONI_index'].apply(lambda x: x if x > 0.5 else 0)
            
            # La Niña Interactions
            daily['climate_risk_Global_Stress_x_Nina'] = daily['Global_Heat_Drought'] * daily['La_Nina_Multiplier']
            daily['climate_risk_US_Stress_x_Nina'] = daily['US_Heat_Drought'] * daily['La_Nina_Multiplier']
            daily['climate_risk_South_Stress_x_Nina'] = (daily.get('South_drought', 0) + daily.get('South_heat_stress', 0)) * daily['La_Nina_Multiplier']
            
            # El Niño Interactions
            daily['climate_risk_Global_Stress_x_Nino'] = daily['Global_Heat_Drought'] * daily['El_Nino_Multiplier']
            daily['climate_risk_US_Stress_x_Nino'] = daily['US_Heat_Drought'] * daily['El_Nino_Multiplier']
            daily['climate_risk_South_Stress_x_Nino'] = (daily.get('South_drought', 0) + daily.get('South_heat_stress', 0)) * daily['El_Nino_Multiplier']
            
            # Priming Features
            daily['climate_risk_US_Stress_x_Nina_Primed'] = daily['US_Heat_Drought'] * daily['La_Nina_Multiplier'].shift(90).fillna(0)
            daily['climate_risk_US_Stress_x_Nina_Primed_120'] = daily['US_Heat_Drought'] * daily['La_Nina_Multiplier'].shift(120).fillna(0)
            daily['climate_risk_Global_Stress_x_Nina_Primed'] = daily['Global_Heat_Drought'] * daily['La_Nina_Multiplier'].shift(90).fillna(0)
            daily['climate_risk_US_Stress_x_Nino_Primed'] = daily['US_Heat_Drought'] * daily['El_Nino_Multiplier'].shift(90).fillna(0)
            
            if 'US_Planting_excess_precip' in daily.columns:
                daily['climate_risk_US_Wet_Planting_x_Nina'] = daily['US_Planting_excess_precip'] * daily['La_Nina_Multiplier']
                daily['climate_risk_US_Wet_Planting_x_Nino'] = daily['US_Planting_excess_precip'] * daily['El_Nino_Multiplier']
                
            # Momentum/Accel
            daily['climate_risk_Nina_Momentum_Gate'] = daily.get('Global_heat_stress',0) * daily['climate_risk_ONI_momentum'].apply(lambda x: abs(x) if x < 0 else 0)
            daily['climate_risk_Nino_Momentum_Gate'] = daily.get('Global_heat_stress',0) * daily['climate_risk_ONI_momentum'].apply(lambda x: x if x > 0 else 0)
            daily['climate_risk_ONI_acceleration'] = daily['climate_risk_ONI_momentum'].diff(7).fillna(0)
            daily['climate_risk_Stress_x_ONI_Accel'] = daily['Global_Heat_Drought'] * daily['climate_risk_ONI_acceleration'].abs()

        # --- B. MJO (Madden-Julian Oscillation) ---
        # Phase: 1-8. Amplitude: Strength.
        if 'phase' in daily.columns and 'amplitude' in daily.columns:
            # Amplitude Modulation
            daily['MJO_Amp'] = daily['amplitude'].fillna(0)
            daily['climate_risk_Global_Stress_x_MJO_Amp'] = daily['Global_Heat_Drought'] * daily['MJO_Amp']
            
            # Phase Interactions (One-hot encoding implicit via conditions)
            # Phases 8, 1, 2: Often wet for Western US / dangerous for some regions
            daily['MJO_Phase_812'] = daily['phase'].isin([8, 1, 2]).astype(int)
            # Phases 4, 5, 6: Often dry
            daily['MJO_Phase_456'] = daily['phase'].isin([4, 5, 6]).astype(int)
            
            # Interaction with US Precip/Drought
            if 'US_excess_precip' in daily.columns:
                 daily['climate_risk_US_Wet_x_MJO_812'] = daily['US_excess_precip'] * daily['MJO_Phase_812'] * daily['MJO_Amp']
            if 'US_drought' in daily.columns:
                 daily['climate_risk_US_Drought_x_MJO_456'] = daily['US_drought'] * daily['MJO_Phase_456'] * daily['MJO_Amp']

        # --- C. PDO (Pacific Decadal Oscillation) ---
        if 'pdo_index' in daily.columns:
            daily['PDO_Index'] = daily['pdo_index'].fillna(0)
            # Positive PDO: Warm Pacific Coast
            daily['PDO_Positive'] = daily['PDO_Index'].apply(lambda x: x if x > 0 else 0)
            daily['PDO_Negative'] = daily['PDO_Index'].apply(lambda x: abs(x) if x < 0 else 0)
            
            # Constructive Interference (PDO x ONI)
            # When PDO and ONI are same sign, effects can be amplified
            if 'climate_risk_ONI_index' in daily.columns:
                daily['climate_risk_PDO_x_ONI_Resonance'] = daily['PDO_Index'] * daily['climate_risk_ONI_index']
                # Amplified Stress when both are 'hot' (El Nino + +PDO)
                daily['climate_risk_Global_Stress_x_Nino_PDO'] = daily['Global_Heat_Drought'] * daily['El_Nino_Multiplier'] * daily['PDO_Positive']

    # 5. Time Shifts - expanded lag windows
    if 'US_heat_stress' in daily.columns:
        daily['climate_risk_US_PreStressed_Heat'] = daily['US_heat_stress'] * daily.get('US_drought', pd.Series(0, index=daily.index)).shift(30).fillna(0)
        daily['climate_risk_US_PreStressed_Heat_7d'] = daily['US_heat_stress'] * daily.get('US_drought', pd.Series(0, index=daily.index)).shift(7).fillna(0)
    if 'Global_heat_stress' in daily.columns:
        daily['climate_risk_Global_PreStressed_Heat'] = daily['Global_heat_stress'] * daily.get('Global_drought', pd.Series(0, index=daily.index)).shift(30).fillna(0)
        daily['climate_risk_Global_PreStressed_Heat_120d'] = daily['Global_heat_stress'] * daily.get('Global_drought', pd.Series(0, index=daily.index)).shift(120).fillna(0)
    if 'South_heat_stress' in daily.columns:
        daily['climate_risk_South_PreStressed_Heat'] = daily['South_heat_stress'] * daily.get('South_drought', pd.Series(0, index=daily.index)).shift(30).fillna(0)
    
    # 6. Cross-hemispheric divergence - NEW
    if 'North_heat_stress' in daily.columns and 'South_heat_stress' in daily.columns:
        daily['climate_risk_Hemisphere_Divergence'] = abs(daily['North_heat_stress'] - daily['South_heat_stress'])
        daily['climate_risk_Cold_Wet_Compound'] = (daily.get('North_unseasonably_cold', 0) + daily.get('US_Planting_excess_precip', 0)) * \
                                                   (daily.get('South_excess_precip', 0) + daily.get('South_unseasonably_cold', 0))
        
    return reduce_mem_usage(daily)

def expand_single_feature(base_series, col_name, date_series):
    """Generates variants for a SINGLE column. Returns a DataFrame."""
    expanded = {}
    
    # Expanded lag windows: added 7, 14, 120 for better temporal coverage
    for lag in [0, 7, 14, 30, 60, 90, 120]:
        s_base = base_series.shift(lag).fillna(0)
        suffix = f"_L{lag}" if lag > 0 else ""
        for pwr in POWERS:
            sig = s_base ** pwr
            mom = sig.diff(7).fillna(0)
            for w in WINDOWS:
                expanded[f"climate_risk_EMA_{col_name}{suffix}_P{pwr}_W{w}"] = sig.ewm(span=w).mean()
                if w >= 60:
                    expanded[f"climate_risk_STD_{col_name}{suffix}_P{pwr}_W{w}"] = sig.rolling(w).std().fillna(0)
                    expanded[f"climate_risk_TURB_{col_name}{suffix}_P{pwr}_W{w}"] = mom.rolling(w).std().fillna(0)
    
    return pd.DataFrame(expanded)

def get_corrs_batch(df_features, df_targets, df_metadata):
    """Calculates correlations for a batch of features against targets."""
    local_df = pd.concat([df_features, df_targets, df_metadata], axis=1)
    f_cols = df_targets.columns.tolist()
    c_cols = df_features.columns.tolist()
    
    results = {c: [] for c in c_cols}
    groups = local_df.groupby(['crop_name', 'country_name', 'date_on_month'])
    
    for _, group in groups:
        if len(group) < 5: continue
        
        cols_to_use = c_cols + f_cols
        num = group[cols_to_use].select_dtypes(include=np.number).astype(np.float64)
        
        v_std = num.std()
        valid_cols = v_std[v_std > 1e-5].index
        
        valid_targets = [f for f in f_cols if f in valid_cols]
        valid_cands = [c for c in c_cols if c in valid_cols]
        
        if not valid_targets or not valid_cands:
            continue
            
        corr_mat = num[valid_cands + valid_targets].corr()
        
        for c in valid_cands:
            res = corr_mat.loc[c, valid_targets].values
            results[c].append(res)
            
    final_corrs = {}
    for c in c_cols:
        if results[c]:
            final_corrs[c] = np.concatenate(results[c])
        else:
            final_corrs[c] = np.array([], dtype=np.float64)
            
    return final_corrs

def process_batch_task(base_col, base_data, train_dates, train_targets, train_meta):
    """Worker function for parallel processing."""
    # 1. Expand
    expanded_df = expand_single_feature(base_data[base_col], base_col, base_data['date_on'])
    
    # 2. Align with Train
    batch_with_date = pd.concat([base_data[['date_on']], expanded_df], axis=1)
    aligned_batch = train_dates.merge(batch_with_date, on='date_on', how='left')
    
    feat_batch = aligned_batch.drop(columns=['date_on']).fillna(0)
    
    # 3. Calculate Correlations
    corrs = get_corrs_batch(feat_batch, train_targets, train_meta)
    return corrs

def calculate_cfcs(all_corrs_flat):
    if len(all_corrs_flat) == 0: return 0, 0, 0, 0
    flat = np.floor(np.array(all_corrs_flat, dtype=np.float64) * 100000) / 100000
    abs_flat = np.abs(flat)
    sig = abs_flat[abs_flat >= 0.5]
    avg_s = np.mean(sig) if len(sig) > 0 else 0
    max_c = np.max(abs_flat)
    pct = len(sig) / len(flat)
    score = (50 * min(1.0, avg_s)) + (30 * min(1.0, max_c)) + (20 * pct)
    return score, avg_s, max_c, pct

def run_optimizer_round(density_floor, candidates, densities, cache, raw_cols, name_to_idx, red_matrix, king):
    selected = [king]
    sniper_pool = [c for c in candidates if c != king and densities.get(c, 0) >= density_floor]
    
    current_flat = np.concatenate([cache[f] for f in raw_cols + selected if len(cache[f]) > 0])
    current_score = calculate_cfcs(current_flat)[0]
    
    # Forward Selection
    for i in range(TARGET_PHALANX - 1):
        best_cand, best_new_score = None, -1
        selected_indices = [name_to_idx[s] for s in selected]
        
        for c in sniper_pool:
            c_idx = name_to_idx[c]
            is_redundant = False
            for s_idx in selected_indices:
                if abs(red_matrix[c_idx, s_idx]) > REDUNDANCY_CAP:
                    is_redundant = True; break
            if is_redundant: continue
            
            trial_flat = np.concatenate([current_flat, cache[c]])
            score = calculate_cfcs(trial_flat)[0]
            if score > best_new_score: best_new_score, best_cand = score, c
            
        if best_cand:
            selected.append(best_cand)
            sniper_pool.remove(best_cand)
            current_flat = np.concatenate([current_flat, cache[best_cand]])
            current_score = best_new_score
        else:
            break
    
    # Backward Elimination
    final_list = raw_cols + selected
    improved = True
    while improved:
        improved = False
        if len(final_list) < 20: break 
        for feat in [f for f in final_list if f not in raw_cols and f != king]:
            temp_list = [f for f in final_list if f != feat]
            temp_flat = np.concatenate([cache[f] for f in temp_list if len(cache[f]) > 0])
            score = calculate_cfcs(temp_flat)[0]
            if score > current_score + 1e-6:
                current_score = score
                final_list = temp_list
                current_flat = temp_flat
                improved = True
                break
    return current_score, [f for f in final_list if f not in raw_cols]

def run_pipeline(use_external=False, label="Experiment"):
    print(f"\n{'='*30}\nRUNNING: {label} (Parallel + Precise + Harvest + Lagged ONI)\n{'='*30}")
    
    df_raw, share = load_data()
    skeleton, baseline_cr_cols = generate_b66_skeleton(df_raw, share)
    skeleton = reduce_mem_usage(skeleton)
    
    base_daily = generate_base_daily_features(df_raw, share, use_external=use_external)
    
    f_cols = [c for c in skeleton.columns if c.startswith('futures_')]
    target_df = skeleton[['date_on'] + f_cols].copy()
    meta_df = skeleton[['date_on', 'crop_name', 'country_name', 'date_on_month']].copy()
    
    raw_cols = [c for c in df_raw.columns if c.startswith('climate_risk_cnt_locations_')]
    base_candidates = [c for c in base_daily.columns if c != 'date_on']
    
    print("Step 1: Correlation Mining (Parallel Batches)...")
    cache = {} 
    
    priv_mask = skeleton['date_on'] < LB_START_DATE
    train_skel = skeleton.loc[priv_mask].reset_index(drop=True)
    train_targets = target_df.loc[priv_mask, f_cols].reset_index(drop=True)
    train_meta = meta_df.loc[priv_mask, ['crop_name', 'country_name', 'date_on_month']].reset_index(drop=True)
    train_dates = skeleton.loc[priv_mask, ['date_on']].reset_index(drop=True)

    # A. Raw Cols
    print(f"Processing {len(raw_cols)} Raw Features...")
    raw_corrs = get_corrs_batch(train_skel[raw_cols], train_targets, train_meta)
    cache.update(raw_corrs)
    
    # B. Expanded Features (Parallel)
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {}
        for base_col in base_candidates:
            f = executor.submit(process_batch_task, base_col, base_daily, train_dates, train_targets, train_meta)
            futures[f] = base_col
            
        for f in tqdm(as_completed(futures), total=len(base_candidates), desc="Batches"):
            res = f.result()
            cache.update(res)

    print(f"Step 1.5: Filtering for top {PRE_SELECT_K} candidates...")
    candidates = list(cache.keys())
    scores_dict = {c: np.max(np.abs(cache[c])) if len(cache[c]) > 0 else 0 for c in candidates}
    sorted_cands = sorted([c for c in candidates if c not in raw_cols], key=lambda x: scores_dict[x], reverse=True)
    active_candidates = sorted_cands[:PRE_SELECT_K]
    
    best_max, king = -1, None
    densities = {}
    for c in active_candidates:
        arr = cache[c]
        densities[c] = len(arr[np.abs(arr) >= 0.5]) / len(arr)
        m = np.max(np.abs(arr))
        if m > best_max: best_max, king = m, c

    # --- STEP 2: RECONSTRUCT POOL ---
    print("Step 2: Reconstructing Reduced Pool & Redundancy Matrix...")
    reconstructed_features = []
    
    for base_col in tqdm(base_candidates, desc="Rebuilding Pool"):
        relevant_cols = [ac for ac in active_candidates if base_col in ac]
        if not relevant_cols: continue
        
        batch_expanded = expand_single_feature(base_daily[base_col], base_col, base_daily['date_on'])
        cols_to_keep = [c for c in batch_expanded.columns if c in active_candidates]
        
        if cols_to_keep:
            batch_with_date = pd.concat([base_daily[['date_on']], batch_expanded[cols_to_keep]], axis=1)
            reconstructed_features.append(batch_with_date)

    if reconstructed_features:
        pool_df = pd.DataFrame({'date_on': base_daily['date_on']})
        for pdf in reconstructed_features:
            pool_df = pool_df.merge(pdf, on='date_on', how='left')
    else:
        pool_df = pd.DataFrame({'date_on': base_daily['date_on']})

    pool_subset_cols = [c for c in pool_df.columns if c in active_candidates]
    if not pool_subset_cols:
        redundancy_matrix = np.eye(len(active_candidates))
    else:
        red_aligned = skeleton[['date_on']].merge(pool_df, on='date_on', how='left')
        redundancy_matrix = np.corrcoef(red_aligned[active_candidates].fillna(0).values.T)
        del red_aligned
        gc.collect()
        
    name_to_idx = {name: i for i, name in enumerate(active_candidates)}
    
    best_overall_score, best_features, best_floor = -1, [], -1
    for floor in FLOORS_TO_TEST:
        score, feats = run_optimizer_round(floor, active_candidates, densities, cache, raw_cols, name_to_idx, redundancy_matrix, king)
        if score > best_overall_score: best_overall_score, best_features, best_floor = score, feats, floor

    print(f"🏆 Best Floor: {best_floor} -> Score {best_overall_score:.4f}")
    
    skeleton_clean = skeleton.drop(columns=[c for c in baseline_cr_cols if c in skeleton.columns], errors='ignore')
    final_sub = skeleton_clean.merge(pool_df[['date_on'] + best_features], on='date_on', how='left')
    
    final_cr_cols = [c for c in final_sub.columns if c.startswith('climate_risk_')]
    final_sub[final_cr_cols] = final_sub[final_cr_cols].fillna(0)
    
    def score_ds(df_sub, feats):
        if len(df_sub) == 0: return (0,0,0,0)
        t_sub = df_sub[f_cols]
        m_sub = df_sub[['crop_name', 'country_name', 'date_on_month']]
        f_sub = df_sub[feats]
        res_dict = get_corrs_batch(f_sub, t_sub, m_sub)
        flat = np.concatenate(list(res_dict.values())) if res_dict else []
        return calculate_cfcs(flat)
    
    res_p = score_ds(final_sub[final_sub['date_on'] < LB_START_DATE], final_cr_cols)
    res_b = score_ds(final_sub[final_sub['date_on'] >= LB_START_DATE], final_cr_cols)
    res_f = score_ds(final_sub, final_cr_cols)
    
    print("\n" + "="*110 + f"\n{'Metric':<18} | {'Private (Train)':<18} | {'Public (Test)':<18} | {'Full (All)':<18}\n" + "-" * 110)
    print(f"{'CFCS Score':<18} | {res_p[0]:<18.4f} | {res_b[0]:<18.4f} | {res_f[0]:<18.4f}")
    print(f"{'Avg Sig Score':<18} | {res_p[1]*100:<18.2f} | {res_b[1]*100:<18.2f} | {res_f[1]*100:<18.2f}")
    print(f"{'Max Corr Score':<18} | {res_p[2]*100:<18.2f} | {res_b[2]*100:<18.2f} | {res_f[2]*100:<18.2f}")
    print(f"{'Sig Count %':<18} | {res_p[3]*100:<18.2f} | {res_b[3]*100:<18.2f} | {res_f[3]*100:<18.2f}\n" + "="*110)
    
    filename = f"submission_{label.lower().replace(' ', '_')}.csv"
    final_sub.to_csv(filename, index=False)
    print(f"Saved: {filename}")

if __name__ == "__main__":
    # RUN EXTERNAL FIRST as requested for faster feedback
    run_pipeline(use_external=True, label="With External")
    run_pipeline(use_external=False, label="No External")