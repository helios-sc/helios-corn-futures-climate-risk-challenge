"""
Helios Corn Futures Climate Challenge - LightGBM Parameter Tuning with L9 Orthogonal Array
===========================================================================================

L9直交表（3水準4因子）を使用して大域的なパラメータ探索を行う。
その後、Optunaで局所最適化を行う。
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
import itertools
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# =============================================================================
# 1. データ読み込みと特徴量エンジニアリング（前回と同じ）
# =============================================================================
print("=" * 60)
print("1. Loading Data and Feature Engineering")
print("=" * 60)

DATA_PATH = './'

df = pd.read_csv(DATA_PATH + 'corn_climate_risk_futures_daily_master.csv')
df['date_on'] = pd.to_datetime(df['date_on'])
market_share_df = pd.read_csv(DATA_PATH + 'corn_regional_market_share.csv')

print(f"Dataset loaded: {len(df):,} rows")

# 作業用コピー
merged_daily_df = df.copy()
merged_daily_df['day_of_year'] = merged_daily_df['date_on'].dt.dayofyear
merged_daily_df['quarter'] = merged_daily_df['date_on'].dt.quarter
merged_daily_df['week_of_year'] = merged_daily_df['date_on'].dt.isocalendar().week.astype(int)
merged_daily_df['month'] = merged_daily_df['date_on'].dt.month

# マーケットシェアのマージ
market_share_unique = market_share_df.drop_duplicates(subset='region_id', keep='first')
merged_daily_df = merged_daily_df.merge(
    market_share_unique[['region_id', 'percent_country_production']], 
    on='region_id', 
    how='left'
)
merged_daily_df['percent_country_production'] = merged_daily_df['percent_country_production'].fillna(1.0)

# リスクカテゴリ
risk_categories = ['heat_stress', 'unseasonably_cold', 'excess_precip', 'drought']

# 基本リスクスコア
for risk_type in risk_categories:
    low_col = f'climate_risk_cnt_locations_{risk_type}_risk_low'
    med_col = f'climate_risk_cnt_locations_{risk_type}_risk_medium' 
    high_col = f'climate_risk_cnt_locations_{risk_type}_risk_high'
    
    total_locations = merged_daily_df[low_col] + merged_daily_df[med_col] + merged_daily_df[high_col]
    risk_score = (merged_daily_df[med_col] + 2 * merged_daily_df[high_col]) / (total_locations + 1e-6)
    weighted_risk = risk_score * (merged_daily_df['percent_country_production'] / 100)
    
    merged_daily_df[f'climate_risk_{risk_type}_score'] = risk_score
    merged_daily_df[f'climate_risk_{risk_type}_weighted'] = weighted_risk
    merged_daily_df[f'climate_risk_{risk_type}_high_ratio'] = merged_daily_df[high_col] / (total_locations + 1e-6)
    merged_daily_df[f'climate_risk_{risk_type}_elevated_ratio'] = (merged_daily_df[med_col] + merged_daily_df[high_col]) / (total_locations + 1e-6)

# 複合リスク指標
temp_scores = [f'climate_risk_{risk}_score' for risk in ['heat_stress', 'unseasonably_cold']]
precip_scores = [f'climate_risk_{risk}_score' for risk in ['excess_precip', 'drought']]
all_risk_scores = [f'climate_risk_{risk}_score' for risk in risk_categories]

merged_daily_df['climate_risk_temperature_stress'] = merged_daily_df[temp_scores].max(axis=1)
merged_daily_df['climate_risk_precipitation_stress'] = merged_daily_df[precip_scores].max(axis=1)
merged_daily_df['climate_risk_overall_stress'] = merged_daily_df[all_risk_scores].max(axis=1)
merged_daily_df['climate_risk_combined_stress'] = merged_daily_df[all_risk_scores].mean(axis=1)
merged_daily_df['climate_risk_total_stress'] = merged_daily_df[all_risk_scores].sum(axis=1)

weighted_scores = [f'climate_risk_{risk}_weighted' for risk in risk_categories]
merged_daily_df['climate_risk_total_weighted_stress'] = merged_daily_df[weighted_scores].sum(axis=1)

# 季節性特徴量
harvest_dummies = pd.get_dummies(merged_daily_df['harvest_period'], prefix='climate_risk_harvest')
merged_daily_df = pd.concat([merged_daily_df, harvest_dummies], axis=1)

harvest_periods = merged_daily_df['harvest_period'].unique()
for risk_type in risk_categories:
    score_col = f'climate_risk_{risk_type}_score'
    for period in harvest_periods:
        if pd.notna(period):
            period_mask = (merged_daily_df['harvest_period'] == period).astype(float)
            col_name = f'climate_risk_{risk_type}_{period.lower().replace(" ", "_").replace("-", "_")}'
            merged_daily_df[col_name] = merged_daily_df[score_col] * period_mask

growing_mask = merged_daily_df['harvest_period'].isin(['Planting', 'Growing', 'Vegetative', 'Reproductive']).astype(float)
for risk_type in risk_categories:
    score_col = f'climate_risk_{risk_type}_score'
    merged_daily_df[f'climate_risk_{risk_type}_growing_season'] = merged_daily_df[score_col] * growing_mask

for risk_type in risk_categories:
    score_col = f'climate_risk_{risk_type}_score'
    summer_mask = merged_daily_df['month'].isin([6, 7, 8]).astype(float)
    merged_daily_df[f'climate_risk_{risk_type}_summer'] = merged_daily_df[score_col] * summer_mask
    winter_mask = merged_daily_df['month'].isin([12, 1, 2]).astype(float)
    merged_daily_df[f'climate_risk_{risk_type}_winter'] = merged_daily_df[score_col] * winter_mask

# ソート
merged_daily_df = merged_daily_df.sort_values(['region_id', 'date_on']).reset_index(drop=True)

# ローリング特徴量
windows = [7, 14, 30]
for window in windows:
    for risk_type in risk_categories:
        score_col = f'climate_risk_{risk_type}_score'
        merged_daily_df[f'climate_risk_{risk_type}_ma_{window}d'] = (
            merged_daily_df.groupby('region_id')[score_col]
            .rolling(window=window, min_periods=1).mean()
            .reset_index(level=0, drop=True)
        )
        merged_daily_df[f'climate_risk_{risk_type}_max_{window}d'] = (
            merged_daily_df.groupby('region_id')[score_col]
            .rolling(window=window, min_periods=1).max()
            .reset_index(level=0, drop=True)
        )
        merged_daily_df[f'climate_risk_{risk_type}_std_{window}d'] = (
            merged_daily_df.groupby('region_id')[score_col]
            .rolling(window=window, min_periods=2).std()
            .reset_index(level=0, drop=True)
        )
        merged_daily_df[f'climate_risk_{risk_type}_min_{window}d'] = (
            merged_daily_df.groupby('region_id')[score_col]
            .rolling(window=window, min_periods=1).min()
            .reset_index(level=0, drop=True)
        )

# ラグ特徴量
lag_days = [1, 3, 7, 14, 21]
for lag in lag_days:
    for risk_type in risk_categories:
        score_col = f'climate_risk_{risk_type}_score'
        merged_daily_df[f'climate_risk_{risk_type}_lag_{lag}d'] = (
            merged_daily_df.groupby('region_id')[score_col].shift(lag)
        )

# モメンタム特徴量
for risk_type in risk_categories:
    score_col = f'climate_risk_{risk_type}_score'
    merged_daily_df[f'climate_risk_{risk_type}_change_1d'] = merged_daily_df.groupby('region_id')[score_col].diff(1)
    merged_daily_df[f'climate_risk_{risk_type}_change_7d'] = merged_daily_df.groupby('region_id')[score_col].diff(7)
    merged_daily_df[f'climate_risk_{risk_type}_change_14d'] = merged_daily_df.groupby('region_id')[score_col].diff(14)
    merged_daily_df[f'climate_risk_{risk_type}_acceleration'] = (
        merged_daily_df.groupby('region_id')[f'climate_risk_{risk_type}_change_1d'].diff(1)
    )

# 国レベル集約
original_len = len(merged_daily_df)
for risk_type in risk_categories:
    score_col = f'climate_risk_{risk_type}_score'
    weighted_col = f'climate_risk_{risk_type}_weighted'
    
    country_agg = merged_daily_df.groupby(['country_name', 'date_on']).agg({
        score_col: ['mean', 'max', 'std', 'min'],
        weighted_col: 'sum'
    }).round(4)
    
    country_agg.columns = [
        f'climate_risk_country_{risk_type}_mean',
        f'climate_risk_country_{risk_type}_max',
        f'climate_risk_country_{risk_type}_std',
        f'climate_risk_country_{risk_type}_min',
        f'climate_risk_country_{risk_type}_weighted_sum'
    ]
    country_agg = country_agg.reset_index()
    merged_daily_df = merged_daily_df.merge(country_agg, on=['country_name', 'date_on'], how='left')

# グローバル集約
major_producers = ['United States', 'Brazil', 'Argentina', 'China']
for risk_type in risk_categories:
    weighted_col = f'climate_risk_{risk_type}_weighted'
    score_col = f'climate_risk_{risk_type}_score'
    major_df = merged_daily_df[merged_daily_df['country_name'].isin(major_producers)]
    
    global_agg = major_df.groupby('date_on').agg({
        weighted_col: 'sum',
        score_col: ['mean', 'max']
    }).round(4)
    global_agg.columns = [
        f'climate_risk_global_{risk_type}_weighted',
        f'climate_risk_global_{risk_type}_mean',
        f'climate_risk_global_{risk_type}_max'
    ]
    global_agg = global_agg.reset_index()
    merged_daily_df = merged_daily_df.merge(global_agg, on='date_on', how='left')

# 非線形変換
for risk_type in risk_categories:
    score_col = f'climate_risk_{risk_type}_score'
    merged_daily_df[f'climate_risk_{risk_type}_squared'] = merged_daily_df[score_col] ** 2
    merged_daily_df[f'climate_risk_{risk_type}_cubed'] = merged_daily_df[score_col] ** 3
    merged_daily_df[f'climate_risk_{risk_type}_high_flag'] = (merged_daily_df[score_col] > 1.0).astype(int)
    merged_daily_df[f'climate_risk_{risk_type}_extreme_flag'] = (merged_daily_df[score_col] > 1.5).astype(int)
    merged_daily_df[f'climate_risk_{risk_type}_log'] = np.log1p(merged_daily_df[score_col])
    merged_daily_df[f'climate_risk_{risk_type}_sqrt'] = np.sqrt(merged_daily_df[score_col])

# 相互作用
merged_daily_df['climate_risk_heat_drought_interaction'] = (
    merged_daily_df['climate_risk_heat_stress_score'] * merged_daily_df['climate_risk_drought_score']
)
merged_daily_df['climate_risk_cold_wet_interaction'] = (
    merged_daily_df['climate_risk_unseasonably_cold_score'] * merged_daily_df['climate_risk_excess_precip_score']
)
merged_daily_df['climate_risk_temp_precip_interaction'] = (
    merged_daily_df['climate_risk_temperature_stress'] * merged_daily_df['climate_risk_precipitation_stress']
)
merged_daily_df['climate_risk_all_interaction'] = (
    merged_daily_df['climate_risk_heat_stress_score'] * 
    merged_daily_df['climate_risk_unseasonably_cold_score'] *
    merged_daily_df['climate_risk_excess_precip_score'] *
    merged_daily_df['climate_risk_drought_score']
)
merged_daily_df['climate_risk_temp_to_precip_ratio'] = (
    merged_daily_df['climate_risk_temperature_stress'] / 
    (merged_daily_df['climate_risk_precipitation_stress'] + 1e-6)
)

print(f"Feature engineering complete. Shape: {merged_daily_df.shape}")

# =============================================================================
# 2. L9直交表によるパラメータチューニング
# =============================================================================
print("\n" + "=" * 60)
print("2. L9 Orthogonal Array Parameter Tuning")
print("=" * 60)

# 特徴量カラムの準備
climate_feature_cols = [c for c in merged_daily_df.columns 
                        if c.startswith('climate_risk_') 
                        and merged_daily_df[c].dtype in ['float64', 'int64', 'int32', 'uint8']]
merged_daily_df[climate_feature_cols] = merged_daily_df[climate_feature_cols].fillna(0)

print(f"Climate features: {len(climate_feature_cols)}")

# ターゲット変数（代表的なものを1つ選択してチューニング）
TARGET_VAR = 'futures_zc_term_spread'

# 学習データの準備
train_mask = merged_daily_df[TARGET_VAR].notna()
X_train = merged_daily_df.loc[train_mask, climate_feature_cols].values
y_train = merged_daily_df.loc[train_mask, TARGET_VAR].values

print(f"Training samples: {len(X_train)}")

# L9直交表（3水準4因子）
# 因子: num_leaves, learning_rate, feature_fraction, min_child_samples
L9_ARRAY = [
    [0, 0, 0, 0],
    [0, 1, 1, 1],
    [0, 2, 2, 2],
    [1, 0, 1, 2],
    [1, 1, 2, 0],
    [1, 2, 0, 1],
    [2, 0, 2, 1],
    [2, 1, 0, 2],
    [2, 2, 1, 0],
]

# 各因子の水準
FACTOR_LEVELS = {
    'num_leaves': [31, 63, 127],
    'learning_rate': [0.01, 0.03, 0.05],
    'feature_fraction': [0.6, 0.7, 0.8],
    'min_child_samples': [10, 20, 50],
}

FACTOR_NAMES = list(FACTOR_LEVELS.keys())

def evaluate_params(params, X, y, n_splits=5):
    """パラメータを評価してCV相関を返す"""
    lgb_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': params['num_leaves'],
        'learning_rate': params['learning_rate'],
        'feature_fraction': params['feature_fraction'],
        'min_child_samples': params['min_child_samples'],
        'bagging_fraction': 0.7,
        'bagging_freq': 5,
        'verbose': -1,
        'n_jobs': -1,
        'seed': 42
    }
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    correlations = []
    
    for train_idx, val_idx in tscv.split(X):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        
        train_data = lgb.Dataset(X_tr, label=y_tr)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        model = lgb.train(
            lgb_params,
            train_data,
            num_boost_round=500,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )
        
        val_pred = model.predict(X_val)
        if np.std(val_pred) > 0 and np.std(y_val) > 0:
            corr = np.corrcoef(val_pred, y_val)[0, 1]
            correlations.append(corr)
    
    return np.mean(correlations) if correlations else 0

# L9実験の実行
print("\nRunning L9 experiments...")
print("-" * 60)

l9_results = []

for exp_idx, levels in enumerate(L9_ARRAY):
    params = {
        FACTOR_NAMES[i]: FACTOR_LEVELS[FACTOR_NAMES[i]][levels[i]]
        for i in range(len(FACTOR_NAMES))
    }
    
    print(f"Exp {exp_idx + 1}/9: {params}")
    
    cv_corr = evaluate_params(params, X_train, y_train)
    
    l9_results.append({
        'exp': exp_idx + 1,
        **params,
        'cv_corr': cv_corr
    })
    
    print(f"  → CV Correlation: {cv_corr:.4f}")

# 結果をDataFrameに
l9_df = pd.DataFrame(l9_results)
print("\n" + "=" * 60)
print("L9 Results Summary")
print("=" * 60)
print(l9_df.to_string(index=False))

# 最良のパラメータ
best_idx = l9_df['cv_corr'].idxmax()
best_params_l9 = {
    'num_leaves': l9_df.loc[best_idx, 'num_leaves'],
    'learning_rate': l9_df.loc[best_idx, 'learning_rate'],
    'feature_fraction': l9_df.loc[best_idx, 'feature_fraction'],
    'min_child_samples': l9_df.loc[best_idx, 'min_child_samples'],
}

print(f"\nBest L9 params: {best_params_l9}")
print(f"Best CV Correlation: {l9_df.loc[best_idx, 'cv_corr']:.4f}")

# 因子効果の分析
print("\n" + "=" * 60)
print("Factor Effects Analysis")
print("=" * 60)

for factor in FACTOR_NAMES:
    print(f"\n{factor}:")
    for level_idx, level_val in enumerate(FACTOR_LEVELS[factor]):
        # この水準を使った実験の平均
        mask = l9_df[factor] == level_val
        avg_corr = l9_df.loc[mask, 'cv_corr'].mean()
        print(f"  Level {level_val}: avg CV corr = {avg_corr:.4f}")

# =============================================================================
# 3. Optunaによる局所最適化（オプション）
# =============================================================================
print("\n" + "=" * 60)
print("3. Optuna Fine-tuning (Optional)")
print("=" * 60)

USE_OPTUNA = True  # Optunaを使用するかどうか

if USE_OPTUNA:
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        def objective(trial):
            params = {
                'num_leaves': trial.suggest_int('num_leaves', 
                    max(15, best_params_l9['num_leaves'] - 32),
                    min(255, best_params_l9['num_leaves'] + 32)),
                'learning_rate': trial.suggest_float('learning_rate',
                    max(0.005, best_params_l9['learning_rate'] - 0.02),
                    min(0.1, best_params_l9['learning_rate'] + 0.02)),
                'feature_fraction': trial.suggest_float('feature_fraction',
                    max(0.4, best_params_l9['feature_fraction'] - 0.15),
                    min(0.95, best_params_l9['feature_fraction'] + 0.15)),
                'min_child_samples': trial.suggest_int('min_child_samples',
                    max(5, best_params_l9['min_child_samples'] - 15),
                    min(100, best_params_l9['min_child_samples'] + 30)),
            }
            
            return evaluate_params(params, X_train, y_train, n_splits=5)
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=30, show_progress_bar=True)
        
        best_params_optuna = study.best_params
        print(f"\nOptuna best params: {best_params_optuna}")
        print(f"Optuna best CV Correlation: {study.best_value:.4f}")
        
        # Optunaの結果がL9より良ければ採用
        if study.best_value > l9_df.loc[best_idx, 'cv_corr']:
            FINAL_PARAMS = best_params_optuna
            print("Using Optuna params (better than L9)")
        else:
            FINAL_PARAMS = best_params_l9
            print("Using L9 params (better than Optuna)")
            
    except ImportError:
        print("Optuna not available. Using L9 params.")
        FINAL_PARAMS = best_params_l9
else:
    FINAL_PARAMS = best_params_l9
    print("Skipping Optuna. Using L9 params.")

print(f"\nFinal params: {FINAL_PARAMS}")

# =============================================================================
# 4. 最終モデルによる予測特徴量生成
# =============================================================================
print("\n" + "=" * 60)
print("4. Generate Prediction Features with Optimized Params")
print("=" * 60)

# 全ターゲット変数
target_vars = [
    'futures_close_ZC_1',
    'futures_close_ZC_2',
    'futures_close_ZW_1',
    'futures_close_ZS_1',
    'futures_zc1_ret_pct',
    'futures_zc1_ret_log',
    'futures_zc_term_spread',
    'futures_zc_term_ratio',
    'futures_zc1_ma_20',
    'futures_zc1_ma_60',
    'futures_zc1_ma_120',
    'futures_zc1_vol_20',
    'futures_zc1_vol_60',
    'futures_zw_zc_spread',
    'futures_zc_zw_ratio',
    'futures_zs_zc_spread',
    'futures_zc_zs_ratio'
]

# 最終LightGBMパラメータ
final_lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': int(FINAL_PARAMS['num_leaves']),
    'learning_rate': FINAL_PARAMS['learning_rate'],
    'feature_fraction': FINAL_PARAMS['feature_fraction'],
    'min_child_samples': int(FINAL_PARAMS['min_child_samples']),
    'bagging_fraction': 0.7,
    'bagging_freq': 5,
    'verbose': -1,
    'n_jobs': -1,
    'seed': 42
}

print(f"Final LightGBM params: {final_lgb_params}")

N_SPLITS = 10
lgb_pred_features = []

for target in target_vars:
    print(f"\nTraining for: {target}")
    
    train_mask = merged_daily_df[target].notna()
    if train_mask.sum() == 0:
        print(f"  Skipping - no valid data")
        continue
    
    X_all = merged_daily_df[climate_feature_cols].values
    y_all = merged_daily_df[target].values
    X_train = X_all[train_mask]
    y_train = y_all[train_mask]
    
    ensemble_predictions = np.zeros(len(merged_daily_df))
    ensemble_count = np.zeros(len(merged_daily_df))
    
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    fold_correlations = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        
        train_data = lgb.Dataset(X_tr, label=y_tr)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        model = lgb.train(
            final_lgb_params,
            train_data,
            num_boost_round=500,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )
        
        val_pred = model.predict(X_val)
        if np.std(val_pred) > 0 and np.std(y_val) > 0:
            fold_corr = np.corrcoef(val_pred, y_val)[0, 1]
            fold_correlations.append(fold_corr)
        
        all_pred = model.predict(X_all)
        ensemble_predictions += all_pred
        ensemble_count += 1
    
    ensemble_predictions = ensemble_predictions / ensemble_count
    
    feature_name = f'climate_risk_lgb_pred_{target.replace("futures_", "")}'
    merged_daily_df[feature_name] = ensemble_predictions
    lgb_pred_features.append(feature_name)
    
    avg_corr = np.mean(fold_correlations) if fold_correlations else 0
    print(f"  Average CV Correlation: {avg_corr:.4f}")

print(f"\nCreated {len(lgb_pred_features)} prediction features")

# ラグ付き予測特徴量
print("\nCreating lagged prediction features...")
merged_daily_df = merged_daily_df.sort_values(['region_id', 'date_on']).reset_index(drop=True)

pred_lag_days = [1, 3, 7, 14]
for pred_feature in lgb_pred_features:
    for lag in pred_lag_days:
        lag_feature_name = f'{pred_feature}_lag_{lag}d'
        merged_daily_df[lag_feature_name] = (
            merged_daily_df.groupby('region_id')[pred_feature].shift(lag)
        )

for pred_feature in lgb_pred_features:
    merged_daily_df[f'{pred_feature}_change_1d'] = merged_daily_df.groupby('region_id')[pred_feature].diff(1)
    merged_daily_df[f'{pred_feature}_change_7d'] = merged_daily_df.groupby('region_id')[pred_feature].diff(7)

for pred_feature in lgb_pred_features[:5]:
    merged_daily_df[f'{pred_feature}_ma_7d'] = (
        merged_daily_df.groupby('region_id')[pred_feature]
        .rolling(window=7, min_periods=1).mean()
        .reset_index(level=0, drop=True)
    )

print("Lagged prediction features created!")

# =============================================================================
# 5. 評価
# =============================================================================
print("\n" + "=" * 60)
print("5. Evaluation")
print("=" * 60)

def compute_monthly_climate_futures_correlations(df):
    climate_cols = [c for c in df.columns if c.startswith("climate_risk_")]
    futures_cols = [c for c in df.columns if c.startswith("futures_")]
    results = []

    for comm in df["crop_name"].unique():
        df_comm = df[df["crop_name"] == comm]
        for country in sorted(df_comm["country_name"].unique()):
            df_country = df_comm[df_comm["country_name"] == country]
            for month in sorted(df_country["date_on_month"].unique()):
                df_month = df_country[df_country["date_on_month"] == month]
                for clim in climate_cols:
                    for fut in futures_cols:
                        if df_month[clim].std() > 0 and df_month[fut].std() > 0:
                            corr = df_month[[clim, fut]].corr().iloc[0, 1]
                        else:
                            corr = None
                        results.append({
                            "crop_name": comm,
                            "country_name": country,
                            "month": month,
                            "climate_variable": clim,
                            "futures_variable": fut,
                            "correlation": corr
                        })

    results_df = pd.DataFrame(results)
    results_df['correlation'] = results_df['correlation'].round(5)
    return results_df

def calculate_cfcs_score(correlations_df):
    valid_corrs = correlations_df["correlation"].dropna()
    if len(valid_corrs) == 0:
        return {'cfcs_score': 0.0}
    
    abs_corrs = valid_corrs.abs()
    max_abs_corr = abs_corrs.max()
    significant_mask = abs_corrs >= 0.5
    significant_corrs = abs_corrs[significant_mask]
    significant_count = len(significant_corrs)
    total_count = len(valid_corrs)
    
    if significant_count > 0:
        avg_sig_corr = significant_corrs.mean()
        avg_sig_score = min(100, avg_sig_corr * 100)
    else:
        avg_sig_corr = 0.0
        avg_sig_score = 0.0
    
    max_corr_score = min(100, max_abs_corr * 100)
    sig_count_score = (significant_count / total_count) * 100
    cfcs = (0.5 * avg_sig_score) + (0.3 * max_corr_score) + (0.2 * sig_count_score)
    
    return {
        'cfcs_score': round(cfcs, 2),
        'avg_significant_correlation': round(avg_sig_corr, 4),
        'max_abs_correlation': round(max_abs_corr, 4),
        'significant_correlations_pct': round(sig_count_score, 2),
        'significant_correlations': significant_count,
        'total_correlations': total_count
    }

print("Computing correlations...")
monthly_corr_df = compute_monthly_climate_futures_correlations(merged_daily_df)
score_results = calculate_cfcs_score(monthly_corr_df)

print("\n" + "=" * 60)
print("CLIMATE-FUTURES CORRELATION SCORE (CFCS)")
print("=" * 60)
print(f"\nFinal CFCS Score: {score_results['cfcs_score']}")
print(f"\nComponent Breakdown:")
print(f"  Avg Significant |Corr|: {score_results['avg_significant_correlation']:.4f}")
print(f"  Max |Corr|: {score_results['max_abs_correlation']:.4f}")
print(f"  Significant %: {score_results['significant_correlations_pct']:.2f}%")

# =============================================================================
# 6. 提出ファイル作成
# =============================================================================
print("\n" + "=" * 60)
print("6. Create Submission")
print("=" * 60)

# Valid IDsの取得（前回と同じ処理）
print("Getting valid IDs...")
temp_df = pd.read_csv(DATA_PATH + 'corn_climate_risk_futures_daily_master.csv')
temp_df['date_on'] = pd.to_datetime(temp_df['date_on'])
temp_df = temp_df.merge(market_share_df[['region_id', 'percent_country_production']], on='region_id', how='left')
temp_df['percent_country_production'] = temp_df['percent_country_production'].fillna(1.0)

for risk_type in risk_categories:
    low_col = f'climate_risk_cnt_locations_{risk_type}_risk_low'
    med_col = f'climate_risk_cnt_locations_{risk_type}_risk_medium'
    high_col = f'climate_risk_cnt_locations_{risk_type}_risk_high'
    total = temp_df[low_col] + temp_df[med_col] + temp_df[high_col]
    temp_df[f'climate_risk_{risk_type}_score'] = (temp_df[med_col] + 2 * temp_df[high_col]) / (total + 1e-6)
    temp_df[f'climate_risk_{risk_type}_weighted'] = temp_df[f'climate_risk_{risk_type}_score'] * (temp_df['percent_country_production'] / 100)

score_cols_temp = [f'climate_risk_{r}_score' for r in risk_categories]
temp_df['climate_risk_temperature_stress'] = temp_df[['climate_risk_heat_stress_score', 'climate_risk_unseasonably_cold_score']].max(axis=1)
temp_df['climate_risk_precipitation_stress'] = temp_df[['climate_risk_excess_precip_score', 'climate_risk_drought_score']].max(axis=1)
temp_df['climate_risk_overall_stress'] = temp_df[score_cols_temp].max(axis=1)
temp_df['climate_risk_combined_stress'] = temp_df[score_cols_temp].mean(axis=1)

temp_df = temp_df.sort_values(['region_id', 'date_on'])
for window in [7, 14, 30]:
    for risk_type in risk_categories:
        score_col = f'climate_risk_{risk_type}_score'
        temp_df[f'climate_risk_{risk_type}_ma_{window}d'] = temp_df.groupby('region_id')[score_col].transform(lambda x: x.rolling(window, min_periods=1).mean())
        temp_df[f'climate_risk_{risk_type}_max_{window}d'] = temp_df.groupby('region_id')[score_col].transform(lambda x: x.rolling(window, min_periods=1).max())

for risk_type in risk_categories:
    score_col = f'climate_risk_{risk_type}_score'
    temp_df[f'climate_risk_{risk_type}_change_1d'] = temp_df.groupby('region_id')[score_col].diff(1)
    temp_df[f'climate_risk_{risk_type}_change_7d'] = temp_df.groupby('region_id')[score_col].diff(7)
    temp_df[f'climate_risk_{risk_type}_acceleration'] = temp_df.groupby('region_id')[f'climate_risk_{risk_type}_change_1d'].diff(1)

for risk_type in risk_categories:
    score_col = f'climate_risk_{risk_type}_score'
    weighted_col = f'climate_risk_{risk_type}_weighted'
    country_agg = temp_df.groupby(['country_name', 'date_on']).agg({
        score_col: ['mean', 'max', 'std'], weighted_col: 'sum', 'percent_country_production': 'sum'
    }).round(4)
    country_agg.columns = [f'country_{risk_type}_{"_".join(col).strip()}' for col in country_agg.columns]
    country_agg = country_agg.reset_index()
    temp_df = temp_df.merge(country_agg, on=['country_name', 'date_on'], how='left')

valid_ids = temp_df.dropna()['ID'].tolist()
del temp_df
print(f"Valid IDs: {len(valid_ids):,}")

# 提出データの作成
submission_df = merged_daily_df[merged_daily_df['ID'].isin(valid_ids)].copy()
climate_cols_to_fill = [c for c in submission_df.columns if c.startswith('climate_risk_')]
submission_df[climate_cols_to_fill] = submission_df[climate_cols_to_fill].fillna(0)

print(f"Submission shape: {submission_df.shape}")

# 日付フォーマット
submission_df["date_on"] = pd.to_datetime(submission_df["date_on"]).dt.strftime("%Y-%m-%d")

# 保存
submission_df.to_csv('submission.csv', index=False)
print("\nSubmission saved!")

# 最終サマリー
print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
print(f"CFCS Score: {score_results['cfcs_score']}")
print(f"Optimized Params: {FINAL_PARAMS}")
print(f"Rows: {len(submission_df):,}")
print(f"Climate Features: {len([c for c in submission_df.columns if c.startswith('climate_risk_')])}")
print("=" * 60)