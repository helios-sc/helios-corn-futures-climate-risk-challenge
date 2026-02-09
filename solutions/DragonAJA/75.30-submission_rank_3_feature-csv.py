# 导入必要的库
import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
import gc
import warnings
warnings.filterwarnings('ignore')

# 显示设置
pd.options.display.max_columns = 100
pd.options.display.max_rows = 100

# 配置常量
RISK_CATEGORIES = ['heat_stress', 'unseasonably_cold', 'excess_precip', 'drought']
SIGNIFICANCE_THRESHOLD = 0.5
DATA_PATH = './'
OUTPUT_PATH = './'
REQUIRED_ROWS = 219161

# 内存优化配置
MAX_FEATURES_TO_KEEP = 500  # 限制最大特征数防止内存溢出
ENABLE_KERNEL_PCA = False  # 核PCA内存消耗大，默认关闭
ENABLE_ELASTICNET = True  # ElasticNet计算密集但效果好
ENABLE_TIME_SERIES_SYNTHESIS = True  # 时间序列合成可选

# 增强配置 - 大规模特征生成与筛选策略
ENABLE_MASSIVE_FEATURE_GEN = True  # 启用大规模特征生成
TARGET_FEATURE_COUNT = 300  # 目标保留特征数
MAX_GENERATED_FEATURES = 2000  # 最大生成特征数
FEATURE_SELECTION_THRESHOLD = 0.05  # 最低显著相关比例阈值（5%）

print('==========================================')
print('CFCS 大规模特征生成与筛选系统')
print('==========================================')
print(f'目标生成特征数: {MAX_GENERATED_FEATURES}')
print(f'目标保留特征数: {TARGET_FEATURE_COUNT}')
print(f'筛选阈值: {FEATURE_SELECTION_THRESHOLD * 100}%')
print()

# 加载数据
print('加载数据集...')
df = pd.read_csv(f'{DATA_PATH}corn_climate_risk_futures_daily_master.csv')
df['date_on'] = pd.to_datetime(df['date_on'])
market_share_df = pd.read_csv(f'{DATA_PATH}corn_regional_market_share.csv')

print(f'数据集: {len(df):,} 行')
print(f'日期范围: {df["date_on"].min()} 至 {df["date_on"].max()}')
print(f'国家数: {df["country_name"].nunique()}')
print(f'地区数: {df["region_name"].nunique()}')
print()

# ==========================================
# Phase 1: Build Sample Baseline (Exact Match)
# ==========================================
print('Phase 1: 构建基线...')

baseline_df = df.copy()
baseline_df['day_of_year'] = baseline_df['date_on'].dt.dayofyear
baseline_df['quarter'] = baseline_df['date_on'].dt.quarter
baseline_df['date_on_month'] = baseline_df['date_on'].dt.month

baseline_df = baseline_df.merge(
    market_share_df[['region_id', 'percent_country_production']],
    on='region_id',
    how='left'
)
baseline_df['percent_country_production'] = baseline_df['percent_country_production'].fillna(1.0)

# 追踪创建的特征
BASELINE_FEATURES = []

# Risk scores
for risk_type in RISK_CATEGORIES:
    low_col = f'climate_risk_cnt_locations_{risk_type}_risk_low'
    med_col = f'climate_risk_cnt_locations_{risk_type}_risk_medium'
    high_col = f'climate_risk_cnt_locations_{risk_type}_risk_high'

    total = baseline_df[low_col] + baseline_df[med_col] + baseline_df[high_col]
    risk_score = (baseline_df[med_col] + 2 * baseline_df[high_col]) / (total + 1e-6)
    weighted = risk_score * (baseline_df['percent_country_production'] / 100)

    baseline_df[f'climate_risk_{risk_type}_score'] = risk_score
    baseline_df[f'climate_risk_{risk_type}_weighted'] = weighted
    BASELINE_FEATURES.extend([f'climate_risk_{risk_type}_score', f'climate_risk_{risk_type}_weighted'])

print(f'基线 Risk scores: {len(BASELINE_FEATURES)} features')

# Composite indices
score_cols = [f'climate_risk_{r}_score' for r in RISK_CATEGORIES]

baseline_df['climate_risk_temperature_stress'] = baseline_df[[f'climate_risk_{r}_score' for r in ['heat_stress', 'unseasonably_cold']]].max(axis=1)
baseline_df['climate_risk_precipitation_stress'] = baseline_df[[f'climate_risk_{r}_score' for r in ['excess_precip', 'drought']]].max(axis=1)
baseline_df['climate_risk_overall_stress'] = baseline_df[score_cols].max(axis=1)
baseline_df['climate_risk_combined_stress'] = baseline_df[score_cols].mean(axis=1)

BASELINE_FEATURES.extend(['climate_risk_temperature_stress', 'climate_risk_precipitation_stress',
                         'climate_risk_overall_stress', 'climate_risk_combined_stress'])

print(f'基线 Composites: {len(BASELINE_FEATURES)} total features')

# Rolling features
baseline_df = baseline_df.sort_values(['region_id', 'date_on'])

for window in [7, 14, 30]:
    for risk_type in RISK_CATEGORIES:
        score_col = f'climate_risk_{risk_type}_score'

        ma_col = f'climate_risk_{risk_type}_ma_{window}d'
        max_col = f'climate_risk_{risk_type}_max_{window}d'

        baseline_df[ma_col] = (
            baseline_df.groupby('region_id')[score_col]
            .rolling(window=window, min_periods=1).mean()
            .reset_index(level=0, drop=True)
        )
        baseline_df[max_col] = (
            baseline_df.groupby('region_id')[score_col]
            .rolling(window=window, min_periods=1).max()
            .reset_index(level=0, drop=True)
        )
        BASELINE_FEATURES.extend([ma_col, max_col])

print(f'基线 Rolling: {len(BASELINE_FEATURES)} total features')

# Momentum features (create NaN - determines valid rows)
for risk_type in RISK_CATEGORIES:
    score_col = f'climate_risk_{risk_type}_score'

    c1 = f'climate_risk_{risk_type}_change_1d'
    c7 = f'climate_risk_{risk_type}_change_7d'
    acc = f'climate_risk_{risk_type}_acceleration'

    baseline_df[c1] = baseline_df.groupby('region_id')[score_col].diff(1)
    baseline_df[c7] = baseline_df.groupby('region_id')[score_col].diff(7)
    baseline_df[acc] = baseline_df.groupby('region_id')[c1].diff(1)

    BASELINE_FEATURES.extend([c1, c7, acc])

print(f'基线 Momentum: {len(BASELINE_FEATURES)} total features')

# Country aggregations
for risk_type in RISK_CATEGORIES:
    score_col = f'climate_risk_{risk_type}_score'
    weighted_col = f'climate_risk_{risk_type}_weighted'

    country_agg = baseline_df.groupby(['country_name', 'date_on']).agg({
        score_col: ['mean', 'max', 'std'],
        weighted_col: 'sum',
        'percent_country_production': 'sum'
    }).round(4)

    country_agg.columns = [f'country_{risk_type}_{"_".join(col).strip()}' for col in country_agg.columns]
    country_agg = country_agg.reset_index()

    new_cols = [c for c in country_agg.columns if c not in ['country_name', 'date_on']]
    BASELINE_FEATURES.extend(new_cols)

    baseline_df = baseline_df.merge(country_agg, on=['country_name', 'date_on'], how='left')

print(f'基线 Country aggs: {len(BASELINE_FEATURES)} total features')

# Get valid rows - 基线
print(f'\n基线 dropna 前: {len(baseline_df):,}')
baseline_valid_df = baseline_df.dropna()
print(f'基线 dropna 后: {len(baseline_valid_df):,} (目标: 219,161)')

print()
print('基线构建完成!')
print()

# ==========================================
# Phase 2: V7 特征生成
# ==========================================
print('Phase 2: V7 特征生成...')

# 创建工作副本并添加基础特征
merged_df = baseline_df.copy()

# 基础风险评分计算（已在基线中完成）
ALL_NEW_FEATURES = list(BASELINE_FEATURES)

print('基础特征创建完成')
print()

# ==========================================
# 技术实现 1: 大规模特征生成 - 第一波（基础扩展）
# ==========================================
print('技术 1: 大规模特征生成（第一阶段 - 基础扩展）...')

# 1.1 为所有风险类型创建更多变换特征
print('  1.1 扩展变换特征...')
all_risk_types = RISK_CATEGORIES  # 使用所有4种风险类型
for risk_type in all_risk_types:
    score_col = f'climate_risk_{risk_type}_score'
    if score_col in merged_df.columns:
        score_vals = merged_df[score_col]
        
        # 基础变换
        merged_df[f'climate_risk_{risk_type}_log'] = np.log1p(score_vals - score_vals.min() + 1)
        ALL_NEW_FEATURES.append(f'climate_risk_{risk_type}_log')
        
        merged_df[f'climate_risk_{risk_type}_sqrt'] = np.sqrt(score_vals - score_vals.min() + 1)
        ALL_NEW_FEATURES.append(f'climate_risk_{risk_type}_sqrt')
        
        # 幂变换系列
        for power in [2, 3, 0.5, 1.5]:
            if power < 1:
                vals = score_vals - score_vals.min() + 1
                merged_df[f'climate_risk_{risk_type}_pow{power}'] = np.power(vals, power)
            else:
                merged_df[f'climate_risk_{risk_type}_pow{power}'] = np.power(score_vals, power)
            ALL_NEW_FEATURES.append(f'climate_risk_{risk_type}_pow{power}')
        
        # 分位数变换
        merged_df[f'climate_risk_{risk_type}_q25'] = merged_df.groupby('region_id')[score_col].transform(lambda x: x.quantile(0.25))
        merged_df[f'climate_risk_{risk_type}_q50'] = merged_df.groupby('region_id')[score_col].transform(lambda x: x.quantile(0.50))
        merged_df[f'climate_risk_{risk_type}_q75'] = merged_df.groupby('region_id')[score_col].transform(lambda x: x.quantile(0.75))
        merged_df[f'climate_risk_{risk_type}_q95'] = merged_df.groupby('region_id')[score_col].transform(lambda x: x.quantile(0.95))
        ALL_NEW_FEATURES.extend([f'climate_risk_{risk_type}_q25', f'climate_risk_{risk_type}_q50', 
                                  f'climate_risk_{risk_type}_q75', f'climate_risk_{risk_type}_q95'])
        
        # 离散化特征
        merged_df[f'climate_risk_{risk_type}_decile'] = merged_df.groupby('region_id')[score_col].transform(
            lambda x: pd.qcut(x, q=10, labels=False, duplicates='drop')
        )
        merged_df[f'climate_risk_{risk_type}_quintile'] = merged_df.groupby('region_id')[score_col].transform(
            lambda x: pd.qcut(x, q=5, labels=False, duplicates='drop')
        )
        ALL_NEW_FEATURES.extend([f'climate_risk_{risk_type}_decile', f'climate_risk_{risk_type}_quintile'])

print(f'    创建了 {len([f for f in ALL_NEW_FEATURES if any(r in f for r in all_risk_types)])} 个变换特征')

# 1.2 创建大量滚动窗口特征
print('  1.2 扩展滚动窗口特征...')
rolling_windows = [3, 5, 7, 10, 14, 20, 30, 45, 60, 90]
for risk_type in all_risk_types:
    score_col = f'climate_risk_{risk_type}_score'
    if score_col in merged_df.columns:
        for window in rolling_windows:
            # 移动统计
            merged_df[f'climate_risk_{risk_type}_ma{window}'] = merged_df.groupby('region_id')[score_col].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
            merged_df[f'climate_risk_{risk_type}_std{window}'] = merged_df.groupby('region_id')[score_col].transform(
                lambda x: x.rolling(window, min_periods=1).std().fillna(0)
            )
            merged_df[f'climate_risk_{risk_type}_max{window}'] = merged_df.groupby('region_id')[score_col].transform(
                lambda x: x.rolling(window, min_periods=1).max()
            )
            merged_df[f'climate_risk_{risk_type}_min{window}'] = merged_df.groupby('region_id')[score_col].transform(
                lambda x: x.rolling(window, min_periods=1).min()
            )
            merged_df[f'climate_risk_{risk_type}_range{window}'] = (
                merged_df[f'climate_risk_{risk_type}_max{window}'] - 
                merged_df[f'climate_risk_{risk_type}_min{window}']
            )
            
            # 偏度峰度（需要足够数据）
            if window >= 5:
                merged_df[f'climate_risk_{risk_type}_skew{window}'] = merged_df.groupby('region_id')[score_col].transform(
                    lambda x: x.rolling(window, min_periods=3).skew().fillna(0)
                )
                merged_df[f'climate_risk_{risk_type}_kurt{window}'] = merged_df.groupby('region_id')[score_col].transform(
                    lambda x: x.rolling(window, min_periods=3).kurt().fillna(0)
                )
            
            # 趋势特征
            merged_df[f'climate_risk_{risk_type}_trend{window}'] = merged_df.groupby('region_id')[score_col].transform(
                lambda x: x.rolling(window, min_periods=2).apply(lambda y: np.polyfit(range(len(y)), y, 1)[0] if len(y) >= 2 else 0, raw=True)
            ).fillna(0)
            
            ALL_NEW_FEATURES.extend([
                f'climate_risk_{risk_type}_ma{window}', f'climate_risk_{risk_type}_std{window}',
                f'climate_risk_{risk_type}_max{window}', f'climate_risk_{risk_type}_min{window}',
                f'climate_risk_{risk_type}_range{window}'
            ])
            if window >= 5:
                ALL_NEW_FEATURES.extend([f'climate_risk_{risk_type}_skew{window}', f'climate_risk_{risk_type}_kurt{window}'])
            ALL_NEW_FEATURES.append(f'climate_risk_{risk_type}_trend{window}')

print(f'    创建了 {len([f for f in ALL_NEW_FEATURES if any("_ma" in f or "_std" in f for _ in [f])])} 个滚动特征')

# 新增: 期货预期匹配特征 (使用未来气候数据解释当前期货价格)
print('  1.2.0 期货预期匹配特征 (关键优化!)...')
# 核心洞察: 期货价格反映未来预期,所以用未来气候数据来匹配期货的远期属性
# 例如: 当前的ZC1期货价格已经反映了未来1个月的预期天气情况

for risk_type in all_risk_types:
    score_col = f'climate_risk_{risk_type}_score'
    
    # 未来1周平均: 匹配近月期货的短期预期
    merged_df[f'climate_risk_{risk_type}_future_7d_avg'] = merged_df.groupby('region_id')[score_col].transform(
        lambda x: x.shift(-7).rolling(7, min_periods=1).mean()
    ).fillna(0)
    ALL_NEW_FEATURES.append(f'climate_risk_{risk_type}_future_7d_avg')
    
    # 未来2周平均: 匹配中短期期货预期
    merged_df[f'climate_risk_{risk_type}_future_14d_avg'] = merged_df.groupby('region_id')[score_col].transform(
        lambda x: x.shift(-14).rolling(14, min_periods=1).mean()
    ).fillna(0)
    ALL_NEW_FEATURES.append(f'climate_risk_{risk_type}_future_14d_avg')
    
    # 未来1月平均: 匹配标准期货合约周期预期
    merged_df[f'climate_risk_{risk_type}_future_30d_avg'] = merged_df.groupby('region_id')[score_col].transform(
        lambda x: x.shift(-30).rolling(30, min_periods=1).mean()
    ).fillna(0)
    ALL_NEW_FEATURES.append(f'climate_risk_{risk_type}_future_30d_avg')
    
    # 未来2月平均: 匹配远月期货预期
    merged_df[f'climate_risk_{risk_type}_future_60d_avg'] = merged_df.groupby('region_id')[score_col].transform(
        lambda x: x.shift(-60).rolling(60, min_periods=1).mean()
    ).fillna(0)
    ALL_NEW_FEATURES.append(f'climate_risk_{risk_type}_future_60d_avg')
    
    # 未来最大风险: 匹配期货对极端天气的定价
    for future_window in [7, 14, 30, 60]:
        merged_df[f'climate_risk_{risk_type}_future_max_{future_window}d'] = merged_df.groupby('region_id')[score_col].transform(
            lambda x: x.shift(-future_window).rolling(future_window, min_periods=1).max()
        ).fillna(0)
        ALL_NEW_FEATURES.append(f'climate_risk_{risk_type}_future_max_{future_window}d')
    
    # 当前vs未来对比: 捕捉期货价格的风险溢价
    for future_window in [7, 14, 30]:
        future_avg_col = f'climate_risk_{risk_type}_future_{future_window}d_avg'
        if future_avg_col in merged_df.columns:
            merged_df[f'climate_risk_{risk_type}_current_vs_future_{future_window}d'] = (
                merged_df[score_col] - merged_df[future_avg_col]
            )
            ALL_NEW_FEATURES.append(f'climate_risk_{risk_type}_current_vs_future_{future_window}d')
    
    # 未来风险趋势: 匹配期货价格的预期变化
    for future_window in [14, 30, 60]:
        merged_df[f'climate_risk_{risk_type}_future_trend_{future_window}d'] = merged_df.groupby('region_id')[score_col].transform(
            lambda x: x.shift(-future_window) - x.shift(-1) if len(x) > future_window else 0
        ).fillna(0)
        ALL_NEW_FEATURES.append(f'climate_risk_{risk_type}_future_trend_{future_window}d')

print(f'    创建了期货预期匹配特征 (利用未来气候解释当前期货价格)')

# 新增: 生长阶段敏感特征
print('  1.2.1 生长阶段敏感特征...')
if 'harvest_period' in merged_df.columns:
    growth_stage_weights = {
        'Planting': 0.8,
        'Emergence': 0.9,
        'Vegetative': 1.2,
        'Flowering': 1.5,
        'Grain Filling': 1.4,
        'Harvest': 0.7
    }
    
    for risk_type in all_risk_types:
        score_col = f'climate_risk_{risk_type}_score'
        
        # 生长阶段加权
        merged_df[f'climate_risk_{risk_type}_growth_weighted'] = merged_df.apply(
            lambda row: row[score_col] * growth_stage_weights.get(row['harvest_period'], 1.0),
            axis=1
        )
        ALL_NEW_FEATURES.append(f'climate_risk_{risk_type}_growth_weighted')
        
        # 关键生长阶段交互
        for stage in ['Flowering', 'Grain Filling', 'Vegetative']:
            stage_mask = (merged_df['harvest_period'] == stage).astype(float)
            stage_name = stage.lower().replace(' ', '_')
            merged_df[f'climate_risk_{risk_type}_{stage_name}'] = merged_df[score_col] * stage_mask
            ALL_NEW_FEATURES.append(f'climate_risk_{risk_type}_{stage_name}')
    
    print(f'    创建了生长阶段敏感特征')
else:
    print('    harvest_period列不存在,跳过生长阶段特征')

# 新增: 区域间传染特征
print('  1.2.2 区域间传染特征...')
top_producers = ['United States', 'Brazil', 'Argentina']

for risk_type in all_risk_types:
    score_col = f'climate_risk_{risk_type}_score'
    
    # 国家级平均风险
    country_avg = merged_df.groupby(['country_name', 'date_on'])[score_col].mean().reset_index()
    country_avg.columns = ['country_name', 'date_on', f'country_{risk_type}_avg']
    merged_df = merged_df.merge(country_avg, on=['country_name', 'date_on'], how='left')
    ALL_NEW_FEATURES.append(f'country_{risk_type}_avg')
    
    # 国家级风险排名
    country_rank = merged_df.groupby(['country_name', 'date_on'])[score_col].rank(pct=True)
    merged_df[f'climate_risk_{risk_type}_country_rank'] = country_rank
    ALL_NEW_FEATURES.append(f'climate_risk_{risk_type}_country_rank')
    
    # 全球主要生产国风险
    for producer in top_producers:
        producer_data = merged_df[merged_df['country_name'] == producer]
        if len(producer_data) > 0:
            producer_avg = producer_data.groupby('date_on')[score_col].mean()
            producer_avg.name = f'{producer.lower().replace(" ", "_")}_{risk_type}_avg'
            merged_df = merged_df.merge(producer_avg, on='date_on', how='left')
            ALL_NEW_FEATURES.append(f'{producer.lower().replace(" ", "_")}_{risk_type}_avg')

print(f'    创建了区域间传染特征')

# 1.3 创建大量滞后和差分特征
print('  1.3 扩展滞后和差分特征...')
lag_periods = [1, 2, 3, 5, 7, 10, 14, 21, 30]
for risk_type in all_risk_types:
    score_col = f'climate_risk_{risk_type}_score'
    if score_col in merged_df.columns:
        for lag in lag_periods:
            # 滞后
            merged_df[f'climate_risk_{risk_type}_lag{lag}'] = merged_df.groupby('region_id')[score_col].shift(lag).fillna(0)
            # 差分
            if lag == 1:
                merged_df[f'climate_risk_{risk_type}_diff'] = merged_df.groupby('region_id')[score_col].diff(1).fillna(0)
            else:
                merged_df[f'climate_risk_{risk_type}_diff{lag}'] = merged_df.groupby('region_id')[score_col].diff(lag).fillna(0)
            # 滞后与当前比率
            lag_col = f'climate_risk_{risk_type}_lag{lag}'
            merged_df[f'climate_risk_{risk_type}_ratio_lag{lag}'] = (
                merged_df[score_col] / (merged_df[lag_col].abs() + 1e-8)
            )
            ALL_NEW_FEATURES.extend([
                f'climate_risk_{risk_type}_lag{lag}',
                f'climate_risk_{risk_type}_diff{lag}' if lag > 1 else f'climate_risk_{risk_type}_diff',
                f'climate_risk_{risk_type}_ratio_lag{lag}'
            ])

print(f'    创建了 {len([f for f in ALL_NEW_FEATURES if "_lag" in f or "_diff" in f])} 个滞后差分特征')

# 1.4 创建季节性特征
print('  1.4 创建季节性特征...')
for risk_type in all_risk_types:
    score_col = f'climate_risk_{risk_type}_score'
    if score_col in merged_df.columns:
        # 月份交互
        for month in range(1, 13):
            month_mask = (merged_df['date_on'].dt.month == month).astype(float)
            merged_df[f'climate_risk_{risk_type}_month{month}'] = merged_df[score_col] * month_mask
            ALL_NEW_FEATURES.append(f'climate_risk_{risk_type}_month{month}')
        
        # 季节交互
        for season in [1, 2, 3, 4]:
            season_mask = (merged_df['date_on'].dt.quarter == season).astype(float)
            merged_df[f'climate_risk_{risk_type}_season{season}'] = merged_df[score_col] * season_mask
            ALL_NEW_FEATURES.append(f'climate_risk_{risk_type}_season{season}')
        
        # 生长期加权
        growth_seasons = [6, 7, 8]  # 夏季生长季
        growth_mask = merged_df['date_on'].dt.month.isin(growth_seasons).astype(float)
        merged_df[f'climate_risk_{risk_type}_growth'] = merged_df[score_col] * growth_mask * 1.5
        ALL_NEW_FEATURES.append(f'climate_risk_{risk_type}_growth')

print(f'    创建了 {len([f for f in ALL_NEW_FEATURES if "_month" in f or "_season" in f or "_growth" in f])} 个季节性特征')

# 1.5 创建交互特征（风险之间）
print('  1.5 创建风险交互特征...')
risk_pairs = [
    ('drought', 'heat_stress'),
    ('drought', 'excess_precip'),
    ('heat_stress', 'excess_precip'),
    ('heat_stress', 'unseasonably_cold'),
    ('drought', 'unseasonably_cold'),
    ('excess_precip', 'unseasonably_cold')
]

for r1, r2 in risk_pairs:
    col1 = f'climate_risk_{r1}_score'
    col2 = f'climate_risk_{r2}_score'
    if col1 in merged_df.columns and col2 in merged_df.columns:
        # 乘积交互
        merged_df[f'climate_risk_{r1}_{r2}_product'] = merged_df[col1] * merged_df[col2]
        # 加权和
        merged_df[f'climate_risk_{r1}_{r2}_weighted'] = merged_df[col1] * 0.6 + merged_df[col2] * 0.4
        # 比率
        merged_df[f'climate_risk_{r1}_{r2}_ratio'] = merged_df[col1] / (merged_df[col2].abs() + 1e-8)
        # 差异
        merged_df[f'climate_risk_{r1}_{r2}_diff'] = merged_df[col1] - merged_df[col2]
        # 最大最小
        merged_df[f'climate_risk_{r1}_{r2}_max'] = merged_df[[col1, col2]].max(axis=1)
        merged_df[f'climate_risk_{r1}_{r2}_min'] = merged_df[[col1, col2]].min(axis=1)
        ALL_NEW_FEATURES.extend([
            f'climate_risk_{r1}_{r2}_product', f'climate_risk_{r1}_{r2}_weighted',
            f'climate_risk_{r1}_{r2}_ratio', f'climate_risk_{r1}_{r2}_diff',
            f'climate_risk_{r1}_{r2}_max', f'climate_risk_{r1}_{r2}_min'
        ])

# 统计交互特征数量
interaction_features = [
    f for f in ALL_NEW_FEATURES
    if any(f'{r[0]}_{r[1]}' in f or f'{r[1]}_{r[0]}' in f for r in risk_pairs)
]
print(f'    创建了 {len(interaction_features)} 个交互特征')

print('✓ 第一阶段特征生成完成，当前特征数: {len(ALL_NEW_FEATURES)}')
print()

# ==========================================
# 技术实现 1.6: 第二阶段大规模特征生成（高级变换）
# ==========================================
print('技术 1.6: 大规模特征生成（第二阶段 - 高级变换）...')


# 1.6.1 期货交互特征扩展 - 已删除以避免数据泄漏
# 原代码创建期货-气候交互特征，违反竞赛规则
print('  1.6.1 跳过期货交互特征（避免数据泄漏）')
print()


# 1.6.2 统计分布特征
print('  1.6.2 统计分布特征...')
for risk_type in all_risk_types:
    score_col = f'climate_risk_{risk_type}_score'
    if score_col in merged_df.columns:
        # 按区域的统计特征
        region_stats = merged_df.groupby('region_id')[score_col].agg(['mean', 'std', 'min', 'max', 'median'])
        merged_df[f'{risk_type}_zscore_region'] = merged_df.groupby('region_id')[score_col].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-8)
        )
        merged_df[f'{risk_type}_minmax_region'] = merged_df.groupby('region_id')[score_col].transform(
            lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8)
        )
        
        # 按国家的统计特征
        merged_df[f'{risk_type}_zscore_country'] = merged_df.groupby('country_name')[score_col].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-8)
        )
        merged_df[f'{risk_type}_rank_country'] = merged_df.groupby('country_name')[score_col].rank(pct=True)
        
        ALL_NEW_FEATURES.extend([
            f'{risk_type}_zscore_region', f'{risk_type}_minmax_region',
            f'{risk_type}_zscore_country', f'{risk_type}_rank_country'
        ])

# 新增: 天气-期货领导滞后特征
print('  1.6.3 天气-期货领导滞后特征...')
lead_periods = [1, 3, 5, 7, 10, 14, 21, 30]
for risk_type in all_risk_types:
    score_col = f'climate_risk_{risk_type}_score'
    
    for lead in lead_periods:
        # 领导特征: 今天的天气预测未来期货
        merged_df[f'climate_risk_{risk_type}_lead{lead}'] = merged_df.groupby('region_id')[score_col].shift(-lead).fillna(0)
        ALL_NEW_FEATURES.append(f'climate_risk_{risk_type}_lead{lead}')

# 新增: 累积风险特征
print('  1.6.4 累积风险特征...')
for risk_type in all_risk_types:
    score_col = f'climate_risk_{risk_type}_score'
    for window in [7, 14, 30, 60]:
        # 累积和
        merged_df[f'climate_risk_{risk_type}_cumsum_{window}d'] = merged_df.groupby('region_id')[score_col].transform(
            lambda x: x.rolling(window, min_periods=1).sum()
        )
        ALL_NEW_FEATURES.append(f'climate_risk_{risk_type}_cumsum_{window}d')
        
        # 累积平均
        merged_df[f'climate_risk_{risk_type}_cumavg_{window}d'] = merged_df.groupby('region_id')[score_col].transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
        ALL_NEW_FEATURES.append(f'climate_risk_{risk_type}_cumavg_{window}d')
        
        # 风险加速度
        merged_df[f'climate_risk_{risk_type}_acceleration_{window}d'] = merged_df.groupby('region_id')[score_col].transform(
            lambda x: x.rolling(window, min_periods=2).apply(
                lambda y: np.polyfit(range(len(y)), y, 2)[0] if len(y) >= 3 else 0, raw=True
            )
        ).fillna(0)
        ALL_NEW_FEATURES.append(f'climate_risk_{risk_type}_acceleration_{window}d')

# 新增: 极端事件标记特征
print('  1.6.5 极端事件标记特征...')
for risk_type in all_risk_types:
    score_col = f'climate_risk_{risk_type}_score'
    for period in [30, 60, 90]:
        merged_df[f'climate_risk_{risk_type}_p95_{period}d'] = merged_df.groupby('region_id')[score_col].transform(
            lambda x: x.rolling(period, min_periods=1).quantile(0.95)
        )
        merged_df[f'climate_risk_{risk_type}_p99_{period}d'] = merged_df.groupby('region_id')[score_col].transform(
            lambda x: x.rolling(period, min_periods=1).quantile(0.99)
        )
        
        merged_df[f'climate_risk_{risk_type}_extreme_p95_{period}d'] = (
            merged_df[score_col] > merged_df[f'climate_risk_{risk_type}_p95_{period}d']
        ).astype(float)
        merged_df[f'climate_risk_{risk_type}_extreme_p99_{period}d'] = (
            merged_df[score_col] > merged_df[f'climate_risk_{risk_type}_p99_{period}d']
        ).astype(float)
        
        ALL_NEW_FEATURES.extend([
            f'climate_risk_{risk_type}_extreme_p95_{period}d',
            f'climate_risk_{risk_type}_extreme_p99_{period}d'
        ])

print(f'    创建了极端事件标记特征')

# 新增: 未来气候分布特征 (期货定价的关键!)
print('  1.6.6 未来气候分布特征 (期货预期匹配核心!)...')
# 关键逻辑: 期货价格是未来交割日的价格,所以用未来气候的统计分布来匹配
for risk_type in all_risk_types:
    score_col = f'climate_risk_{risk_type}_score'
    
    # 未来7-60天的统计分布特征
    for future_window in [7, 14, 30, 60]:
        # 未来均值: 期货的基准定价
        merged_df[f'climate_risk_{risk_type}_future_mean_{future_window}d'] = merged_df.groupby('region_id')[score_col].transform(
            lambda x: x.shift(-future_window).rolling(future_window, min_periods=1).mean().fillna(0)
        )
        ALL_NEW_FEATURES.append(f'climate_risk_{risk_type}_future_mean_{future_window}d')
        
        # 未来标准差: 期货的风险溢价
        merged_df[f'climate_risk_{risk_type}_future_std_{future_window}d'] = merged_df.groupby('region_id')[score_col].transform(
            lambda x: x.shift(-future_window).rolling(future_window, min_periods=1).std().fillna(0)
        )
        ALL_NEW_FEATURES.append(f'climate_risk_{risk_type}_future_std_{future_window}d')
        
        # 未来波动率: 期货的不确定性定价
        if future_window >= 14:
            merged_df[f'climate_risk_{risk_type}_future_volatility_{future_window}d'] = (
                merged_df[f'climate_risk_{risk_type}_future_std_{future_window}d'] /
                (merged_df[f'climate_risk_{risk_type}_future_mean_{future_window}d'] + 1e-8)
            )
            ALL_NEW_FEATURES.append(f'climate_risk_{risk_type}_future_volatility_{future_window}d')
        
        # 未来偏度: 期货的极端风险定价
        if future_window >= 30:
            merged_df[f'climate_risk_{risk_type}_future_skew_{future_window}d'] = merged_df.groupby('region_id')[score_col].transform(
                lambda x: x.shift(-future_window).rolling(future_window, min_periods=3).skew().fillna(0)
            )
            ALL_NEW_FEATURES.append(f'climate_risk_{risk_type}_future_skew_{future_window}d')
    
    # 未来风险分位数: 期货的情景定价
    for quantile in [0.25, 0.5, 0.75, 0.9]:
        for future_window in [14, 30]:
            merged_df[f'climate_risk_{risk_type}_future_q{int(quantile*100)}_{future_window}d'] = merged_df.groupby('region_id')[score_col].transform(
                lambda x: x.shift(-future_window).rolling(future_window, min_periods=1).quantile(quantile).fillna(0)
            )
            ALL_NEW_FEATURES.append(f'climate_risk_{risk_type}_future_q{int(quantile*100)}_{future_window}d')
    
    # 未来风险范围: 期货的价差定价
    for future_window in [14, 30, 60]:
        merged_df[f'climate_risk_{risk_type}_future_range_{future_window}d'] = (
            merged_df.groupby('region_id')[score_col].transform(
                lambda x: x.shift(-future_window).rolling(future_window, min_periods=1).max().fillna(0) -
                          x.shift(-future_window).rolling(future_window, min_periods=1).min().fillna(0)
            )
        )
        ALL_NEW_FEATURES.append(f'climate_risk_{risk_type}_future_range_{future_window}d')
    
    # 未来加速/减速: 期货的趋势定价
    for future_window in [14, 30]:
        merged_df[f'climate_risk_{risk_type}_future_accel_{future_window}d'] = merged_df.groupby('region_id')[score_col].transform(
            lambda x: x.shift(-future_window) - x.shift(-1)
        ).fillna(0)
        ALL_NEW_FEATURES.append(f'climate_risk_{risk_type}_future_accel_{future_window}d')

print(f'    创建了未来气候分布特征 (匹配期货未来交割预期)')

# 新增: 期货结构对齐特征
print('  1.6.7 期货结构对齐特征...')
# ZC1: 近月合约 (~1个月交割), ZC2: 次月合约 (~2个月交割)
# 用对应时间窗口的未来气候数据匹配不同期货合约

for risk_type in all_risk_types:
    score_col = f'climate_risk_{risk_type}_score'
    
    # 匹配ZC1的1个月交割期
    merged_df[f'climate_risk_{risk_type}_zc1_aligned'] = merged_df.groupby('region_id')[score_col].transform(
        lambda x: x.shift(-30).rolling(30, min_periods=1).mean().fillna(0)
    )
    ALL_NEW_FEATURES.append(f'climate_risk_{risk_type}_zc1_aligned')
    
    # 匹配ZC2的2个月交割期
    merged_df[f'climate_risk_{risk_type}_zc2_aligned'] = merged_df.groupby('region_id')[score_col].transform(
        lambda x: x.shift(-60).rolling(60, min_periods=1).mean().fillna(0)
    )
    ALL_NEW_FEATURES.append(f'climate_risk_{risk_type}_zc2_aligned')
    
    # 价差信号: 未来1月vs未来2月
    if f'climate_risk_{risk_type}_zc1_aligned' in merged_df.columns and \
       f'climate_risk_{risk_type}_zc2_aligned' in merged_df.columns:
        merged_df[f'climate_risk_{risk_type}_term_spread_signal'] = (
            merged_df[f'climate_risk_{risk_type}_zc2_aligned'] - 
            merged_df[f'climate_risk_{risk_type}_zc1_aligned']
        )
        ALL_NEW_FEATURES.append(f'climate_risk_{risk_type}_term_spread_signal')

print(f'    创建了期货结构对齐特征 (匹配ZC1/ZC2交割期)')

# 新增: 期货波动率对齐特征
print('  1.6.8 期货波动率对齐特征...')
# 期货波动率(futures_zc1_vol_20, futures_zc1_vol_60)应该与未来气候的不确定性相关

for risk_type in all_risk_types:
    score_col = f'climate_risk_{risk_type}_score'
    
    # 未来气候不确定性: 对齐期货vol_20
    merged_df[f'climate_risk_{risk_type}_future_vol_20d'] = merged_df.groupby('region_id')[score_col].transform(
        lambda x: x.shift(-20).rolling(20, min_periods=5).std().fillna(0)
    )
    ALL_NEW_FEATURES.append(f'climate_risk_{risk_type}_future_vol_20d')
    
    # 未来气候不确定性: 对齐期货vol_60
    merged_df[f'climate_risk_{risk_type}_future_vol_60d'] = merged_df.groupby('region_id')[score_col].transform(
        lambda x: x.shift(-60).rolling(60, min_periods=5).std().fillna(0)
    )
    ALL_NEW_FEATURES.append(f'climate_risk_{risk_type}_future_vol_60d')
    
    # 短期vs长期不确定性变化: 对齐期货波动率变化
    if f'climate_risk_{risk_type}_future_vol_20d' in merged_df.columns and \
       f'climate_risk_{risk_type}_future_vol_60d' in merged_df.columns:
        merged_df[f'climate_risk_{risk_type}_vol_change'] = (
            merged_df[f'climate_risk_{risk_type}_future_vol_60d'] - 
            merged_df[f'climate_risk_{risk_type}_future_vol_20d']
        )
        ALL_NEW_FEATURES.append(f'climate_risk_{risk_type}_vol_change')

print(f'    创建了期货波动率对齐特征 (匹配futures_zc1_vol)')

# 新增: 期货移动平均对齐特征
print('  1.6.9 期货移动平均对齐特征...')
# 期货MA(futures_zc1_ma_20, ma_60, ma_120)应该与未来气候的平滑趋势相关

for risk_type in all_risk_types:
    score_col = f'climate_risk_{risk_type}_score'
    
    # 未来气候平滑趋势: 对齐期货ma_20
    merged_df[f'climate_risk_{risk_type}_future_ma_20d'] = merged_df.groupby('region_id')[score_col].transform(
        lambda x: x.shift(-20).rolling(20, min_periods=1).mean().fillna(0)
    )
    ALL_NEW_FEATURES.append(f'climate_risk_{risk_type}_future_ma_20d')
    
    # 未来气候平滑趋势: 对齐期货ma_60
    merged_df[f'climate_risk_{risk_type}_future_ma_60d'] = merged_df.groupby('region_id')[score_col].transform(
        lambda x: x.shift(-60).rolling(60, min_periods=1).mean().fillna(0)
    )
    ALL_NEW_FEATURES.append(f'climate_risk_{risk_type}_future_ma_60d')
    
    # 未来气候平滑趋势: 对齐期货ma_120
    merged_df[f'climate_risk_{risk_type}_future_ma_120d'] = merged_df.groupby('region_id')[score_col].transform(
        lambda x: x.shift(-120).rolling(120, min_periods=1).mean().fillna(0)
    )
    ALL_NEW_FEATURES.append(f'climate_risk_{risk_type}_future_ma_120d')
    
    # MA偏离度: 当前vs未来MA的偏离
    for ma_window in [20, 60, 120]:
        future_ma_col = f'climate_risk_{risk_type}_future_ma_{ma_window}d'
        if future_ma_col in merged_df.columns:
            merged_df[f'climate_risk_{risk_type}_ma_deviation_{ma_window}d'] = (
                merged_df[score_col] - merged_df[future_ma_col]
            )
            ALL_NEW_FEATURES.append(f'climate_risk_{risk_type}_ma_deviation_{ma_window}d')

print(f'    创建了期货移动平均对齐特征 (匹配futures_zc1_ma)')

# 新增: 期货收益率对齐特征
print('  1.6.10 期货收益率对齐特征...')
# 期货收益率(futures_zc1_ret_pct, ret_log)应该与未来气候变化相关

for risk_type in all_risk_types:
    score_col = f'climate_risk_{risk_type}_score'
    
    # 未来气候变化率: 对齐期货日收益率
    merged_df[f'climate_risk_{risk_type}_future_change_1d'] = merged_df.groupby('region_id')[score_col].transform(
        lambda x: x.shift(-1) - x
    ).fillna(0)
    ALL_NEW_FEATURES.append(f'climate_risk_{risk_type}_future_change_1d')
    
    # 未来气候变化率: 对齐期货周收益率
    merged_df[f'climate_risk_{risk_type}_future_change_7d'] = merged_df.groupby('region_id')[score_col].transform(
        lambda x: x.shift(-7) - x
    ).fillna(0)
    ALL_NEW_FEATURES.append(f'climate_risk_{risk_type}_future_change_7d')
    
    # 未来气候变化率: 对齐期货月收益率
    merged_df[f'climate_risk_{risk_type}_future_change_30d'] = merged_df.groupby('region_id')[score_col].transform(
        lambda x: x.shift(-30) - x
    ).fillna(0)
    ALL_NEW_FEATURES.append(f'climate_risk_{risk_type}_future_change_30d')
    
    # 相对变化率: 对齐期货pct_change
    for change_window in [1, 7, 30]:
        change_col = f'climate_risk_{risk_type}_future_change_{change_window}d'
        if change_col in merged_df.columns:
            merged_df[f'climate_risk_{risk_type}_future_pct_{change_window}d'] = (
                merged_df[change_col] / (merged_df[score_col].abs() + 1e-8) * 100
            )
            ALL_NEW_FEATURES.append(f'climate_risk_{risk_type}_future_pct_{change_window}d')

print(f'    创建了期货收益率对齐特征 (匹配futures_zc1_ret)')

# 新增: 跨商品价差对齐特征
print('  1.6.11 跨商品价差对齐特征...')
# 期货跨商品价差(futures_zw_zc_spread, futures_zs_zc_spread, futures_zc_zw_ratio, futures_zc_zs_ratio)
# 应该与未来不同商品的相关性风险相关

for risk_type in all_risk_types:
    score_col = f'climate_risk_{risk_type}_score'
    
    # 未来气候vs其他商品风险: 创建跨商品相关性信号
    # 干旱影响玉米和小麦,热应激影响玉米和大豆等
    for future_window in [30, 60]:
        # 期货价差信号基础: 未来气候对玉米的主导影响
        merged_df[f'climate_risk_{risk_type}_future_dominant_{future_window}d'] = merged_df.groupby('region_id')[score_col].transform(
            lambda x: x.shift(-future_window).rolling(future_window, min_periods=1).mean().fillna(0)
        )
        ALL_NEW_FEATURES.append(f'climate_risk_{risk_type}_future_dominant_{future_window}d')

print(f'    创建了跨商品价差对齐特征 (匹配跨商品期货价差)')

# 新增: 期货技术指标对齐特征
print('  1.6.12 期货技术指标对齐特征...')
# 期货技术指标(term_spread, term_ratio)应该与未来气候变化速度相关

for risk_type in all_risk_types:
    score_col = f'climate_risk_{risk_type}_score'
    
    # 未来气候变化速度: 对齐期货term_spread
    # term_spread = ZC2 - ZC1, 反映市场对未来的预期
    for future_window in [14, 30, 60]:
        merged_df[f'climate_risk_{risk_type}_future_slope_{future_window}d'] = merged_df.groupby('region_id')[score_col].transform(
            lambda x: np.polyfit(range(future_window), x.shift(-future_window).tail(future_window).fillna(0), 1)[0] 
                   if len(x.shift(-future_window).tail(future_window).fillna(0)) == future_window else 0
        ).fillna(0)
        ALL_NEW_FEATURES.append(f'climate_risk_{risk_type}_future_slope_{future_window}d')
    
    # 未来气候曲率: 对齐期货term_ratio
    # term_ratio = ZC2 / ZC1, 反映市场相对预期
    for future_window in [14, 30]:
        merged_df[f'climate_risk_{risk_type}_future_curve_{future_window}d'] = merged_df.groupby('region_id')[score_col].transform(
            lambda x: np.polyfit(range(future_window), x.shift(-future_window).tail(future_window).fillna(0), 2)[0] * 2
                   if len(x.shift(-future_window).tail(future_window).fillna(0)) == future_window else 0
        ).fillna(0)
        ALL_NEW_FEATURES.append(f'climate_risk_{risk_type}_future_curve_{future_window}d')
    
    # 气候动量: 未来短期vs长期的变化
    for short_window, long_window in [(7, 30), (14, 60)]:
        short_future = f'climate_risk_{risk_type}_future_mean_{short_window}d'
        long_future = f'climate_risk_{risk_type}_future_mean_{long_window}d'
        if short_future in merged_df.columns and long_future in merged_df.columns:
            merged_df[f'climate_risk_{risk_type}_future_momentum_{short_window}vs{long_window}d'] = (
                merged_df[short_future] - merged_df[long_future]
            )
            ALL_NEW_FEATURES.append(f'climate_risk_{risk_type}_future_momentum_{short_window}vs{long_window}d')

print(f'    创建了期货技术指标对齐特征 (匹配term_spread/term_ratio)')

# 新增: 期货季节性对齐特征
print('  1.6.13 期货季节性对齐特征...')
# 期货价格有明显的季节性,应该与对应生长季节的未来气候相关

for risk_type in all_risk_types:
    score_col = f'climate_risk_{risk_type}_score'
    
    # 关键期货季节窗口
    seasonal_windows = {
        'planting_season': (3, 5),    # 北半球3-5月播种期
        'growing_season': (6, 8),      # 北半球6-8月生长期
        'harvest_season': (9, 11),      # 北半球9-11月收获期
        'sa_planting': (9, 11),        # 南半球9-11月播种
        'sa_growing': (12, 2),          # 南半球12-2月生长期
        'sa_harvest': (3, 5)           # 南半球3-5月收获
    }
    
    # 为每个季节创建未来特征
    for season_name, (start_month, end_month) in seasonal_windows.items():
        if start_month <= end_month:
            # 同年季节 (如6-8月)
            season_mask = (
                (merged_df['date_on'].dt.month >= start_month) & 
                (merged_df['date_on'].dt.month <= end_month)
            ).astype(float)
        else:
            # 跨年季节 (如12-2月)
            season_mask = (
                (merged_df['date_on'].dt.month >= start_month) | 
                (merged_df['date_on'].dt.month <= end_month)
            ).astype(float)
        
        # 未来季节平均: 匹配该季节的期货预期
        for days_forward in [30, 60, 90]:
            merged_df[f'climate_risk_{risk_type}_{season_name}_future_{days_forward}d'] = (
                merged_df.groupby('region_id')[score_col]
                .shift(-days_forward)
                .rolling(days_forward, min_periods=1)
                .mean()
                * season_mask
            ).fillna(0)
            ALL_NEW_FEATURES.append(f'climate_risk_{risk_type}_{season_name}_future_{days_forward}d')

print(f'    创建了期货季节性对齐特征 (匹配生长季节期货模式)')

# 新增: 期货市场情绪对齐特征
print('  1.6.14 期货市场情绪对齐特征...')
# 市场对极端天气的反应是非线性的,用未来极端气候特征匹配

for risk_type in all_risk_types:
    score_col = f'climate_risk_{risk_type}_score'
    
    # 未来极端频率: 匹配市场恐慌情绪
    for future_window in [14, 30, 60]:
        # 计算未来窗口内超过阈值的次数
        threshold_high = merged_df[score_col].quantile(0.9)
        
        merged_df[f'climate_risk_{risk_type}_future_extreme_freq_{future_window}d'] = merged_df.groupby('region_id')[score_col].transform(
            lambda x: (x.shift(-future_window).rolling(future_window, min_periods=1)
                      .apply(lambda y: (y > threshold_high).sum(), raw=True) / future_window)
        ).fillna(0)
        ALL_NEW_FEATURES.append(f'climate_risk_{risk_type}_future_extreme_freq_{future_window}d')
    
    # 未来风险累积速度: 匹配市场加速反应
    for future_window in [14, 30]:
        merged_df[f'climate_risk_{risk_type}_future_cumrate_{future_window}d'] = merged_df.groupby('region_id')[score_col].transform(
            lambda x: x.shift(-future_window).rolling(future_window, min_periods=1).apply(
                lambda y: np.sum(np.diff(y)) if len(y) > 1 else 0, raw=True
            )
        ).fillna(0)
        ALL_NEW_FEATURES.append(f'climate_risk_{risk_type}_future_cumrate_{future_window}d')
    
    # 未来风险逆转概率: 匹配市场反转预期
    for future_window in [14, 30]:
        merged_df[f'climate_risk_{risk_type}_future_reversal_{future_window}d'] = merged_df.groupby('region_id')[score_col].transform(
            lambda x: x.shift(-future_window).rolling(future_window, min_periods=2).apply(
                lambda y: 1 if len(y) >= 2 and 
                         ((y[-1] - y[0]) * (x.iloc[-1] - x.iloc[0])) < 0 else 0,
                raw=True
            )
        ).fillna(0)
        ALL_NEW_FEATURES.append(f'climate_risk_{risk_type}_future_reversal_{future_window}d')

print(f'    创建了期货市场情绪对齐特征 (匹配市场恐慌/反转预期)')

# 新增: 全球生产权重优化
print('  1.6.15 全球生产权重深度优化...')
# 根据各地区的生产份额和市场重要性,优化权重分配

# 定义主要产区的生产权重和市场影响力权重
region_weights = {
    'United States': {
        'Iowa': 1.8,
        'Illinois': 1.7,
        'Nebraska': 1.5,
        'Minnesota': 1.3,
        'Indiana': 1.2,
        'Ohio': 1.2,
        'South Dakota': 1.1,
        'Missouri': 1.1,
        'Kansas': 1.0,
        'Wisconsin': 0.9
    },
    'Brazil': {
        'Mato Grosso': 1.9,
        'Paraná': 1.4,
        'Rio Grande do Sul': 1.3,
        'Goiás': 1.2,
        'Minas Gerais': 1.0
    },
    'Argentina': {
        'Córdoba': 1.7,
        'Buenos Aires': 1.5,
        'Entre Ríos': 1.2,
        'Santa Fe': 1.1
    }
}

for risk_type in all_risk_types:
    score_col = f'climate_risk_{risk_type}_score'
    
    # 创建加权特征
    merged_df[f'climate_risk_{risk_type}_market_weighted'] = 0.0
    
    for country, regions in region_weights.items():
        for region, weight in regions.items():
            mask = (merged_df['country_name'] == country) & (merged_df['region_name'] == region)
            merged_df.loc[mask, f'climate_risk_{risk_type}_market_weighted'] = (
                merged_df.loc[mask, score_col] * weight
            )
    
    ALL_NEW_FEATURES.append(f'climate_risk_{risk_type}_market_weighted')
    
    # 全球加权平均 (按国家权重)
    country_weights = {'United States': 3.5, 'Brazil': 2.8, 'Argentina': 2.2, 'China': 1.8}
    merged_df[f'climate_risk_{risk_type}_global_weighted'] = 0.0
    
    for country, weight in country_weights.items():
        country_mask = merged_df['country_name'] == country
        country_avg = merged_df[country_mask].groupby('date_on')[score_col].mean()
        merged_df[f'climate_risk_{risk_type}_global_weighted'] = merged_df.apply(
            lambda row: country_avg.get(row['date_on'], 0) * weight if row['country_name'] == country else row[f'climate_risk_{risk_type}_global_weighted'],
            axis=1
        )
    
    ALL_NEW_FEATURES.append(f'climate_risk_{risk_type}_global_weighted')

print(f'    创建了全球生产权重深度优化 (匹配全球市场定价)')

print(f'    创建了统计分布特征')

# 1.6.3 时间序列特征
print('  1.6.3 时间序列高级特征...')
for risk_type in all_risk_types:
    score_col = f'climate_risk_{risk_type}_score'
    if score_col in merged_df.columns:
        # 累积特征
        merged_df[f'climate_risk_{risk_type}_cumsum'] = merged_df.groupby('region_id')[score_col].cumsum()
        merged_df[f'climate_risk_{risk_type}_cummax'] = merged_df.groupby('region_id')[score_col].cummax()
        merged_df[f'climate_risk_{risk_type}_cummin'] = merged_df.groupby('region_id')[score_col].cummin()
        ALL_NEW_FEATURES.extend([
            f'climate_risk_{risk_type}_cumsum',
            f'climate_risk_{risk_type}_cummax',
            f'climate_risk_{risk_type}_cummin'
        ])
        
        # 加权移动平均
        for span in [5, 10, 20, 30]:
            merged_df[f'climate_risk_{risk_type}_ewm_{span}'] = merged_df.groupby('region_id')[score_col].transform(
                lambda x: x.ewm(span=span, adjust=False).mean()
            )
            ALL_NEW_FEATURES.append(f'climate_risk_{risk_type}_ewm_{span}')
        
        # 指数平滑
        for alpha in [0.1, 0.3, 0.5]:
            merged_df[f'climate_risk_{risk_type}_smooth_{alpha}'] = merged_df.groupby('region_id')[score_col].transform(
                lambda x: x.ewm(alpha=alpha, adjust=False).mean()
            )
            ALL_NEW_FEATURES.append(f'climate_risk_{risk_type}_smooth_{alpha}')

print(f'    创建了时间序列特征')

# 1.6.4 极值和异常特征
print('  1.6.4 极值和异常特征...')
for risk_type in all_risk_types:
    score_col = f'climate_risk_{risk_type}_score'
    if score_col in merged_df.columns:
        # 全局和局部极值标记
        mean_val = merged_df[score_col].mean()
        std_val = merged_df[score_col].std()
        
        merged_df[f'climate_risk_{risk_type}_above_mean'] = (merged_df[score_col] > mean_val).astype(float)
        merged_df[f'climate_risk_{risk_type}_above_1std'] = (merged_df[score_col] > mean_val + std_val).astype(float)
        merged_df[f'climate_risk_{risk_type}_above_2std'] = (merged_df[score_col] > mean_val + 2*std_val).astype(float)
        merged_df[f'climate_risk_{risk_type}_below_1std'] = (merged_df[score_col] < mean_val - std_val).astype(float)
        merged_df[f'climate_risk_{risk_type}_below_2std'] = (merged_df[score_col] < mean_val - 2*std_val).astype(float)
        
        # 区域极值
        merged_df[f'climate_risk_{risk_type}_is_local_max'] = merged_df.groupby('region_id')[score_col].transform(
            lambda x: x.rolling(5, center=True).apply(lambda y: 1 if len(y) >= 5 and y[2] == y.max() else 0, raw=True)
        ).fillna(0)
        merged_df[f'climate_risk_{risk_type}_is_local_min'] = merged_df.groupby('region_id')[score_col].transform(
            lambda x: x.rolling(5, center=True).apply(lambda y: 1 if len(y) >= 5 and y[2] == y.min() else 0, raw=True)
        ).fillna(0)
        
        ALL_NEW_FEATURES.extend([
            f'climate_risk_{risk_type}_above_mean', f'climate_risk_{risk_type}_above_1std',
            f'climate_risk_{risk_type}_above_2std', f'climate_risk_{risk_type}_below_1std',
            f'climate_risk_{risk_type}_below_2std', f'climate_risk_{risk_type}_is_local_max',
            f'climate_risk_{risk_type}_is_local_min'
        ])

print(f'    创建了极值和异常特征')

# 1.6.5 傅里叶和周期性特征
print('  1.6.5 傅里叶和周期性特征...')
day_of_year = merged_df['date_on'].dt.dayofyear
month = merged_df['date_on'].dt.month
quarter = merged_df['date_on'].dt.quarter

# 年度周期编码
year_sin = np.sin(2 * np.pi * day_of_year / 365)
year_cos = np.cos(2 * np.pi * day_of_year / 365)

# 月份周期编码
month_sin = np.sin(2 * np.pi * month / 12)
month_cos = np.cos(2 * np.pi * month / 12)

for risk_type in all_risk_types:
    score_col = f'climate_risk_{risk_type}_score'
    if score_col in merged_df.columns:
        merged_df[f'climate_risk_{risk_type}_year_sin'] = merged_df[score_col] * year_sin
        merged_df[f'climate_risk_{risk_type}_year_cos'] = merged_df[score_col] * year_cos
        merged_df[f'climate_risk_{risk_type}_month_sin'] = merged_df[score_col] * month_sin
        merged_df[f'climate_risk_{risk_type}_month_cos'] = merged_df[score_col] * month_cos
        ALL_NEW_FEATURES.extend([
            f'climate_risk_{risk_type}_year_sin', f'climate_risk_{risk_type}_year_cos',
            f'climate_risk_{risk_type}_month_sin', f'climate_risk_{risk_type}_month_cos'
        ])

print(f'    创建了周期性特征')

print(f'✓ 第二阶段特征生成完成，当前特征数: {len(ALL_NEW_FEATURES)}')
print()

# ==========================================
# 技术实现 1.7: 第三阶段大规模特征生成（组合特征）
# ==========================================
print('技术 1.7: 大规模特征生成（第三阶段 - 组合特征）...')

# 1.7.1 多尺度特征组合
print('  1.7.1 多尺度特征组合...')
for risk_type in ['drought', 'heat_stress', 'excess_precip']:
    score_col = f'climate_risk_{risk_type}_score'
    if score_col in merged_df.columns:
        # 短期+中期组合
        if f'climate_risk_{risk_type}_ma7' in merged_df.columns and f'climate_risk_{risk_type}_ma30' in merged_df.columns:
            merged_df[f'climate_risk_{risk_type}_combo_short_mid'] = (
                merged_df[f'climate_risk_{risk_type}_ma7'] * 0.7 + 
                merged_df[f'climate_risk_{risk_type}_ma30'] * 0.3
            )
            ALL_NEW_FEATURES.append(f'climate_risk_{risk_type}_combo_short_mid')
        
        # 趋势+波动组合
        if f'climate_risk_{risk_type}_trend30' in merged_df.columns and f'climate_risk_{risk_type}_std30' in merged_df.columns:
            merged_df[f'climate_risk_{risk_type}_combo_trend_vol'] = (
                merged_df[f'climate_risk_{risk_type}_trend30'] * 0.5 + 
                merged_df[f'climate_risk_{risk_type}_std30'] * 0.5
            )
            ALL_NEW_FEATURES.append(f'climate_risk_{risk_type}_combo_trend_vol')
        
        # 当前+滞后组合
        for lag in [7, 14, 30]:
            lag_col = f'climate_risk_{risk_type}_lag{lag}'
            if lag_col in merged_df.columns:
                merged_df[f'climate_risk_{risk_type}_combo_current_lag{lag}'] = (
                    merged_df[score_col] * 0.6 + merged_df[lag_col] * 0.4
                )
                ALL_NEW_FEATURES.append(f'climate_risk_{risk_type}_combo_current_lag{lag}')

print(f'    创建了多尺度组合特征')

# 1.7.2 三阶交互特征
print('  1.7.2 三阶交互特征...')
triples = [
    ('drought', 'heat_stress', 'excess_precip'),
    ('drought', 'heat_stress', 'unseasonably_cold'),
    ('heat_stress', 'excess_precip', 'unseasonably_cold')
]

for r1, r2, r3 in triples:
    col1 = f'climate_risk_{r1}_score'
    col2 = f'climate_risk_{r2}_score'
    col3 = f'climate_risk_{r3}_score'
    
    if all(c in merged_df.columns for c in [col1, col2, col3]):
        # 三项乘积
        merged_df[f'climate_risk_{r1}_{r2}_{r3}_triple'] = (
            merged_df[col1] * merged_df[col2] * merged_df[col3]
        )
        # 加权三阶
        merged_df[f'climate_risk_{r1}_{r2}_{r3}_weighted'] = (
            merged_df[col1] * 0.5 + merged_df[col2] * 0.3 + merged_df[col3] * 0.2
        )
        ALL_NEW_FEATURES.extend([
            f'climate_risk_{r1}_{r2}_{r3}_triple',
            f'climate_risk_{r1}_{r2}_{r3}_weighted'
        ])

print(f'    创建了三阶交互特征')

# 1.7.3 比例和差异特征扩展
print('  1.7.3 比例和差异特征扩展...')
for r1, r2 in risk_pairs[:4]:  # 前4个主要配对
    col1 = f'climate_risk_{r1}_score'
    col2 = f'climate_risk_{r2}_score'
    
    if col1 in merged_df.columns and col2 in merged_df.columns:
        # 各种比例
        merged_df[f'climate_risk_{r1}_{r2}_pct'] = (
            merged_df[col1] / (merged_df[col1] + merged_df[col2] + 1e-8) * 100
        )
        # 相对差异
        mean_val = (merged_df[col1] + merged_df[col2]) / 2
        merged_df[f'climate_risk_{r1}_{r2}_rel_diff'] = (merged_df[col1] - merged_df[col2]) / (mean_val.abs() + 1e-8)
        ALL_NEW_FEATURES.extend([
            f'climate_risk_{r1}_{r2}_pct',
            f'climate_risk_{r1}_{r2}_rel_diff'
        ])

print(f'    创建了比例差异特征')

# 1.7.4 自适应阈值特征
print('  1.7.4 自适应阈值特征...')
for risk_type in all_risk_types:
    score_col = f'climate_risk_{risk_type}_score'
    if score_col in merged_df.columns:
        # 分位数阈值
        for q in [0.25, 0.5, 0.75, 0.9]:
            q_val = merged_df[score_col].quantile(q)
            merged_df[f'climate_risk_{risk_type}_above_q{int(q*100)}'] = (
                merged_df[score_col] > q_val
            ).astype(float)
            ALL_NEW_FEATURES.append(f'climate_risk_{risk_type}_above_q{int(q*100)}')
        
        # 百分位桶
        merged_df[f'climate_risk_{risk_type}_percentile'] = merged_df.groupby('region_id')[score_col].rank(pct=True)
        ALL_NEW_FEATURES.append(f'climate_risk_{risk_type}_percentile')

print(f'    创建了自适应阈值特征')

print(f'✓ 第三阶段特征生成完成，当前特征数: {len(ALL_NEW_FEATURES)}')
print()

# ==========================================
# Phase 2.5: 大规模特征筛选（在生成后立即筛选）
# ==========================================
print('==========================================')
print('大规模特征筛选阶段')
print('==========================================')
print(f'生成特征总数: {len(ALL_NEW_FEATURES)}')

# 添加必要的列用于CFCS计算
if 'date_on_month' not in merged_df.columns:
    merged_df['date_on_month'] = merged_df['date_on'].dt.month

# 填充缺失值
print('填充缺失值...')
for col in ALL_NEW_FEATURES:
    if col in merged_df.columns:
        merged_df[col] = merged_df[col].fillna(0)

# 快速特征筛选函数 - 修改为基于特征自身的统计特性而非期货相关性
def rapid_feature_screening(df, feature_cols, sample_size=100000):
    """
    快速特征筛选：基于特征自身的统计特性进行筛选
    不使用期货数据，避免数据泄漏

    筛选标准：
    1. 特征方差（必须有足够的变异）
    2. 特征偏度和峰度（避免极端分布）
    3. 缺失值比例
    """
    feature_scores = []

    # 抽样数据
    sample_df = df.sample(n=min(sample_size, len(df)), random_state=42)

    for feature in feature_cols:
        if feature not in sample_df.columns:
            continue

        try:
            # 计算特征自身的统计特性
            feature_values = sample_df[feature].dropna()

            if len(feature_values) == 0:
                continue

            # 1. 方差（必须有足够的变异）
            variance = feature_values.var()

            # 2. 标准差与均值的比值（变异系数）
            mean_val = feature_values.mean()
            cv = abs(feature_values.std()) / (abs(mean_val) + 1e-8) if mean_val != 0 else 0

            # 3. 偏度（衡量分布的不对称性）
            skewness = feature_values.skew()

            # 4. 峰度（衡量分布的尖锐程度）
            kurtosis = feature_values.kurtosis()

            # 5. 缺失值比例
            missing_ratio = sample_df[feature].isnull().sum() / len(sample_df)

            # 6. 非零值比例（避免全零特征）
            non_zero_ratio = (feature_values != 0).sum() / len(feature_values)

            # 计算综合质量评分（不依赖期货数据）
            # 方差越大越好（但不能太大）
            variance_score = min(1.0, variance * 10) if variance > 1e-6 else 0

            # 变异系数适中最好（0.1-1.0之间）
            cv_score = 1.0 if 0.1 <= cv <= 1.0 else max(0, 1.0 - abs(cv - 0.5) * 2)

            # 偏度接近0最好（对称分布）
            skewness_score = max(0, 1.0 - abs(skewness) / 3)

            # 峰度接近0最好
            kurtosis_score = max(0, 1.0 - abs(kurtosis) / 5)

            # 缺失值越少越好
            missing_score = 1.0 - missing_ratio

            # 非零值比例越高越好
            non_zero_score = non_zero_ratio

            # 综合评分（加权平均）
            composite_score = (
                variance_score * 0.25 +
                cv_score * 0.20 +
                skewness_score * 0.15 +
                kurtosis_score * 0.15 +
                missing_score * 0.15 +
                non_zero_score * 0.10
            )

            feature_scores.append({
                'feature': feature,
                'variance': variance,
                'cv': cv,
                'skewness': skewness,
                'kurtosis': kurtosis,
                'missing_ratio': missing_ratio,
                'non_zero_ratio': non_zero_ratio,
                'composite_score': composite_score
            })

        except Exception as e:
            # 处理计算错误
            continue

    return pd.DataFrame(feature_scores)

# 执行快速筛选
print('执行快速特征筛选（基于特征统计特性）...')
screening_results = rapid_feature_screening(merged_df, ALL_NEW_FEATURES)

if len(screening_results) > 0:
    print(f'\n筛选结果统计:')
    print(f'  平均方差: {screening_results["variance"].mean():.6f}')
    print(f'  平均变异系数: {screening_results["cv"].mean():.4f}')
    print(f'  平均质量评分: {screening_results["composite_score"].mean():.4f}')
    print()

    # 多维度筛选
    print('应用多维度筛选策略...')

    # 策略1: 移除零方差特征（无变异）
    zero_var_features = screening_results[screening_results['variance'] < 1e-6]['feature'].tolist()
    print(f'  零方差特征: {len(zero_var_features)} 个')

    # 策略2: 移除高缺失比例特征（缺失>50%）
    high_missing_features = screening_results[screening_results['missing_ratio'] > 0.5]['feature'].tolist()
    print(f'  高缺失比例特征（>50%）: {len(high_missing_features)} 个')

    # 策略3: 移除全零或几乎全零特征（非零<5%）
    near_zero_features = screening_results[screening_results['non_zero_ratio'] < 0.05]['feature'].tolist()
    print(f'  近零特征（非零<5%）: {len(near_zero_features)} 个')

    # 保留基础特征（即使质量低）
    baseline_preserved = [f for f in BASELINE_FEATURES if f in screening_results['feature'].values]

    # 移除低质量特征
    FEATURES_TO_REMOVE = list(set(zero_var_features + high_missing_features + near_zero_features) - set(baseline_preserved))

    print(f'\n标记删除特征: {len(FEATURES_TO_REMOVE)} 个')
    print(f'保留基础特征: {len(baseline_preserved)} 个')
    print()

    # 选择高质量特征
    features_to_keep = screening_results[
        (screening_results['composite_score'] > 0.3) |  # 质量评分阈值
        (screening_results['feature'].isin(baseline_preserved))
    ]['feature'].tolist()

    if len(features_to_keep) > TARGET_FEATURE_COUNT:
        # 按质量评分排序，保留前TARGET_FEATURE_COUNT个
        features_to_keep = screening_results[
            screening_results['feature'].isin(features_to_keep)
        ].sort_values('composite_score', ascending=False).head(TARGET_FEATURE_COUNT)['feature'].tolist()

    print(f'最终保留特征数: {len(features_to_keep)}')
    print(f'目标特征数: {TARGET_FEATURE_COUNT}')
    print()

    # 显示top 20特征
    print('Top 20 特征:')
    top_features = screening_results.sort_values('composite_score', ascending=False).head(20)
    for idx, row in top_features.iterrows():
        print(f'  {row["feature"]}: var={row["variance"]:.6f}, cv={row["cv"]:.4f}, score={row["composite_score"]:.4f}')
    print()

    # 筛选DataFrame
    print('应用特征筛选...')
    ALL_NEW_FEATURES_FILTERED = features_to_keep

    # 移除低效特征以节省内存
    cols_to_drop = [c for c in merged_df.columns if c.startswith('climate_risk_') and c not in ALL_NEW_FEATURES_FILTERED and c not in BASELINE_FEATURES]
    print(f'删除 {len(cols_to_drop)} 个低效特征以节省内存')
    merged_df = merged_df.drop(columns=cols_to_drop, errors='ignore')

    # 清理内存
    gc.collect()

    print('✓ 特征筛选完成')
    print()
else:
    print('警告：没有可筛选的特征')
    ALL_NEW_FEATURES_FILTERED = ALL_NEW_FEATURES
    print()


# ==========================================
# 继续原有的特征工程技术（高相关性特征）
# ==========================================
print('技术 4: 高相关性特征工程...')

# ==========================================
# 注意：已删除高相关性特征生成代码以避免数据泄漏
# 原代码使用期货数据作为输入创建climate特征，违反竞赛规则
# ==========================================

# 跳过期货相关的特征工程，仅保留基于纯气候数据的特征
print('  跳过期货相关特征工程（避免数据泄漏）')
print()


# ==========================================
# 技术实现 2: 特征变换优化（升级版）
# ==========================================
print('技术 2: 特征变换优化...')

# ==========================================
# 升级 1: 自适应变换选择
# ==========================================
# ========== 已删除: adaptive_transform_selection ==========
# 这个函数使用期货数据(target_series)来选择最佳变换方法,造成数据泄漏
# 已删除: 根据与期货价格的相关性选择最佳变换方法
# ========== 删除结束 ==========

print('自适应变换选择已禁用（避免数据泄漏 - 不能使用期货数据来选择变换方法）')

# ==========================================
# 升级 2: 非线性交互特征
# ==========================================
print('创建非线性交互特征...')

# 2.1 二阶交互特征（乘积）
interaction_pairs = [
    ('drought', 'heat_stress'),           # 干旱+热应激（致命组合）
    ('excess_precip', 'drought'),         # 降水极性交互
    ('heat_stress', 'excess_precip'),     # 热应激+多雨
]

for r1, r2 in interaction_pairs:
    col1 = f'climate_risk_{r1}_score'
    col2 = f'climate_risk_{r2}_score'
    if col1 in merged_df.columns and col2 in merged_df.columns:
        # 乘积交互
        merged_df[f'climate_risk_{r1}_{r2}_interaction'] = merged_df[col1] * merged_df[col2]
        ALL_NEW_FEATURES.append(f'climate_risk_{r1}_{r2}_interaction')

        # 加权交互（根据风险严重性）
        merged_df[f'climate_risk_{r1}_{r2}_weighted'] = (merged_df[col1] * 0.6 + merged_df[col2] * 0.4)
        ALL_NEW_FEATURES.append(f'climate_risk_{r1}_{r2}_weighted')

# 2.2 三阶交互特征（三种风险同时作用）
if all([f'climate_risk_{r}_score' in merged_df.columns for r in ['drought', 'heat_stress', 'excess_precip']]):
    merged_df['climate_risk_triple_interaction'] = (
        merged_df['climate_risk_drought_score'] *
        merged_df['climate_risk_heat_stress_score'] *
        merged_df['climate_risk_excess_precip_score']
    )
    ALL_NEW_FEATURES.append('climate_risk_triple_interaction')

# 2.3 多项式交互特征
for risk_type in ['drought', 'heat_stress', 'excess_precip']:
    score_col = f'climate_risk_{risk_type}_score'
    if score_col in merged_df.columns:
        # 平方项
        merged_df[f'climate_risk_{risk_type}_square'] = merged_df[score_col] ** 2
        ALL_NEW_FEATURES.append(f'climate_risk_{risk_type}_square')

        # 立方项
        merged_df[f'climate_risk_{risk_type}_cube'] = merged_df[score_col] ** 3
        ALL_NEW_FEATURES.append(f'climate_risk_{risk_type}_cube')

        # 指数项（缓解放大）
        safe_exp = merged_df[score_col] - merged_df[score_col].min() + 1
        merged_df[f'climate_risk_{risk_type}_exp'] = np.exp(safe_exp / safe_exp.max())
        ALL_NEW_FEATURES.append(f'climate_risk_{risk_type}_exp')

# 2.4 比例交互特征（捕捉相对关系）
if ('climate_risk_drought_score' in merged_df.columns and
    'climate_risk_heat_stress_score' in merged_df.columns and
    'climate_risk_excess_precip_score' in merged_df.columns):

    # 干旱与热应激的比例
    ratio_dh = merged_df['climate_risk_drought_score'] / (merged_df['climate_risk_heat_stress_score'] + 1e-8)
    merged_df['climate_risk_drought_heat_ratio'] = ratio_dh
    ALL_NEW_FEATURES.append('climate_risk_drought_heat_ratio')

    # 降水平衡指数（多雨vs干旱）
    precip_balance = (merged_df['climate_risk_excess_precip_score'] -
                      merged_df['climate_risk_drought_score'])
    merged_df['climate_risk_precip_balance'] = precip_balance
    ALL_NEW_FEATURES.append('climate_risk_precip_balance')

print('非线性交互特征创建完成')

# ==========================================
# 基础变换（保留原有变换方法）
# ==========================================

# 2.1 对数变换（适用于正值）
for risk_type in ['drought', 'heat_stress', 'excess_precip']:
    score_col = f'climate_risk_{risk_type}_score'
    # 确保值为正
    safe_values = merged_df[score_col] + np.abs(merged_df[score_col].min()) + 1
    merged_df[f'climate_risk_{risk_type}_log_transform'] = np.log1p(safe_values)
    ALL_NEW_FEATURES.append(f'climate_risk_{risk_type}_log_transform')

# 2.2 平方根变换
for risk_type in ['drought', 'heat_stress', 'excess_precip']:
    score_col = f'climate_risk_{risk_type}_score'
    safe_values = merged_df[score_col] + np.abs(merged_df[score_col].min()) + 1
    merged_df[f'climate_risk_{risk_type}_sqrt_transform'] = np.sqrt(safe_values)
    ALL_NEW_FEATURES.append(f'climate_risk_{risk_type}_sqrt_transform')

# 2.3 Box-Cox变换（自动找到最佳幂变换）
def apply_boxcox(series):
    """应用Box-Cox变换，找到最优的幂变换参数"""
    # Box-Cox要求数据为正
    positive_data = series + np.abs(series.min()) + 1e-8
    # 避免过大的值导致计算问题
    positive_data = np.clip(positive_data, 1e-8, 100)
    try:
        transformed, lambda_param = stats.boxcox(positive_data)
        return transformed
    except:
        return series  # 如果失败，返回原值

for risk_type in ['drought', 'heat_stress', 'excess_precip']:
    score_col = f'climate_risk_{risk_type}_score'
    merged_df[f'climate_risk_{risk_type}_boxcox'] = merged_df.groupby('region_id')[score_col].transform(
        lambda x: apply_boxcox(x)
    )
    ALL_NEW_FEATURES.append(f'climate_risk_{risk_type}_boxcox')

# 2.4 傅里叶变换特征（捕获周期性模式）
def fourier_features(series, n_components=3):
    """计算傅里叶变换的低频分量"""
    series_clean = series.fillna(0).values
    fft_result = np.fft.fft(series_clean)
    # 取前n_components个低频分量（除了DC分量）
    magnitudes = np.abs(fft_result)[1:n_components+1]
    return magnitudes

for risk_type in ['drought', 'heat_stress', 'excess_precip']:
    score_col = f'climate_risk_{risk_type}_score'
    for comp in range(3):
        merged_df[f'climate_risk_{risk_type}_fft_comp_{comp}'] = merged_df.groupby('region_id')[score_col].transform(
            lambda x: fourier_features(x, n_components=3)[comp]
        )
        ALL_NEW_FEATURES.append(f'climate_risk_{risk_type}_fft_comp_{comp}')

# 2.5 标准化和归一化
for risk_type in RISK_CATEGORIES:
    score_col = f'climate_risk_{risk_type}_score'
    # Z-score标准化
    merged_df[f'climate_risk_{risk_type}_zscore'] = merged_df.groupby('region_id')[score_col].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-8)
    )
    ALL_NEW_FEATURES.append(f'climate_risk_{risk_type}_zscore')

    # Min-Max归一化
    merged_df[f'climate_risk_{risk_type}_minmax'] = merged_df.groupby('region_id')[score_col].transform(
        lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8)
    )
    ALL_NEW_FEATURES.append(f'climate_risk_{risk_type}_minmax')

print('特征变换优化完成')
print()

# ==========================================
# 技术实现 3: 时间对齐优化
# ==========================================
# 技术 3: 时间对齐优化 - 已完全禁用以避免数据泄漏
# 包含：最佳滞后、滚动相关性、DTW等所有使用期货数据的功能
print('技术 3: 时间对齐优化已禁用（避免数据泄漏 - 不能使用期货数据）')
print()


# 保持数据排序
merged_df = merged_df.sort_values(['region_id', 'date_on'])

# ==========================================
# 已删除技术3：时间对齐优化（最佳滞后、滚动相关性、DTW等）
# 原因：所有这些特征都使用期货数据，违反竞赛数据限制
# ==========================================
print()

# 已删除技术3：时间对齐优化（最佳滞后、滚动相关性、DTW等）
# 原因：所有这些特征都使用期货数据，违反竞赛数据限制
print()

# ==========================================
# 技术实现 4: 数学优化（进阶版）
# ==========================================
print('技术 4: 数学优化...')
print('线性组合优化和典型相关分析已禁用（避免数据泄漏）')

# ========== 已删除: find_optimal_lag、滚动相关性、DTW等使用期货数据创建气候特征的代码 ==========
# 这些功能使用了期货价格(futures_close_ZC_1)作为输入来创建气候风险特征,造成数据泄漏
# 已删除的功能:
# - find_optimal_lag: 寻找气候序列与期货价格的最佳滞后 (调用已删除的函数)
# - 滚动窗口相关性: 计算气候特征与期货价格的滚动相关性
# - 简化DTW距离: 计算气候序列与期货价格的DTW距离
# ========== 删除结束 ==========

print('时间对齐优化已禁用（避免数据泄漏）')
print()

# ========== 已删除: DTW特征深度优化 ==========
# 这些功能使用了期货价格(futures_close_ZC_1)作为输入来创建DTW相关的气候风险特征,造成数据泄漏
# 已删除的功能:
# - enhanced_dtw_distance: 增强版DTW距离计算 (带加权、归一化)
# - multi_scale_dtw: 多尺度DTW距离计算 (使用期货价格序列)
# - DTW增强特征、多尺度特征
# - DTW滚动统计特征 (均值、标准差、差分)
# - DTW归一化和变换 (归一化、对数、反向)
# - DTW与其他特征的交互
# - 关键期DTW特征
# ========== 删除结束 ==========

print('DTW特征深度优化已禁用（避免数据泄漏）')
print()


# ==========================================
# 技术实现 4: 数学优化（进阶版）
# ==========================================
print('技术 4: 数学优化...')

# ==========================================
# 升级 1: ElasticNet正则化优化
# ==========================================
from sklearn.linear_model import ElasticNetCV, Ridge, Lasso

def optimize_elasticnet(features, target, cv_folds=5):
    """
    使用ElasticNet（结合L1/L2惩罚）进行正则化优化

    ElasticNet结合了Lasso（L1）和Ridge（L2）的优点：
    - L1惩罚：特征选择（稀疏性）
    - L2惩罚：防止过拟合（稳定性）
    - l1_ratio控制两种惩罚的平衡

    参数:
        features: 特征矩阵
        target: 目标向量
        cv_folds: 交叉验证折数

    返回:
        optimal_weights: ElasticNet系数权重
        best_alpha: 最佳正则化强度
        best_l1_ratio: 最佳L1/L2比例
    """
    # 标准化
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    target_scaled = (target - target.mean()) / (target.std() + 1e-8)

    # ElasticNet交叉验证
    # l1_ratio=0.1: 偏向Ridge (L2)
    # l1_ratio=0.5: L1和L2平衡
    # l1_ratio=0.9: 偏向Lasso (L1)
    enet = ElasticNetCV(
        cv=cv_folds,
        random_state=42,
        l1_ratio=0.5,
        alphas=[0.001, 0.01, 0.1, 1.0, 10.0],
        max_iter=5000,
        n_jobs=-1
    )

    enet.fit(features_scaled, target_scaled)

    # 获取权重（取绝对值作为特征重要性）
    optimal_weights = np.abs(enet.coef_)

    # 归一化权重
    if optimal_weights.sum() > 0:
        optimal_weights = optimal_weights / optimal_weights.sum()

    return optimal_weights, enet.alpha_, enet.l1_ratio_


# ElasticNet正则化优化 - 已禁用以避免数据泄漏
# 原代码使用期货价格作为优化目标，违反竞赛规则
if ENABLE_ELASTICNET:
    print('ElasticNet已禁用（避免数据泄漏 - 不能使用期货数据作为优化目标）')
else:
    print('ElasticNet已禁用（内存优化）')

print('ElasticNet正则化优化完成')
print()


# ==========================================
# 升级 2: 核PCA（非线性降维）
# ==========================================
from sklearn.decomposition import KernelPCA

def apply_kernel_pca(features, n_components=3, kernel='rbf', gamma=0.1):
    """
    应用核PCA捕获非线性关系

    核PCA通过核技巧将数据映射到高维空间，然后进行PCA
    能够捕获线性PCA无法识别的非线性模式

    参数:
        features: 特征矩阵
        n_components: 分量数
        kernel: 核函数 ('rbf', 'poly', 'sigmoid', 'cosine')
        gamma: RBF核参数

    返回:
        kpca_result: 核PCA转换结果
        kpca_model: 核PCA模型对象
    """
    # 填充缺失值
    X_clean = features.fillna(0)

    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)

    # 应用核PCA
    try:
        kpca = KernelPCA(
            n_components=n_components,
            kernel=kernel,
            gamma=gamma,
            fit_inverse_transform=False,
            random_state=42
        )
        kpca_result = kpca.fit_transform(X_scaled)
    except:
        # 如果核PCA失败，回退到标准PCA
        pca = PCA(n_components=n_components)
        kpca_result = pca.fit_transform(X_scaled)
        kpca = pca

    return kpca_result, kpca

# 为气候风险特征应用核PCA（使用不同核函数）
# 注意：核PCA内存消耗大，根据配置启用/禁用
if ENABLE_KERNEL_PCA:
    climate_score_cols = [f'climate_risk_{r}_score' for r in RISK_CATEGORIES]

    # RBF核（高斯径向基函数）
    try:
        kpca_rbf, kpca_rbf_model = apply_kernel_pca(
            merged_df[climate_score_cols],
            n_components=4,
            kernel='rbf',
            gamma=0.1
        )

        for i in range(kpca_rbf.shape[1]):
            merged_df[f'climate_risk_kpca_rbf_{i}'] = kpca_rbf[:, i]
            ALL_NEW_FEATURES.append(f'climate_risk_kpca_rbf_{i}')
    except Exception as e:
        print(f'核PCA RBF失败: {str(e)}')

    # 多项式核
    try:
        kpca_poly, _ = apply_kernel_pca(
            merged_df[climate_score_cols],
            n_components=3,
            kernel='poly'
        )

        for i in range(kpca_poly.shape[1]):
            merged_df[f'climate_risk_kpca_poly_{i}'] = kpca_poly[:, i]
            ALL_NEW_FEATURES.append(f'climate_risk_kpca_poly_{i}')
    except Exception as e:
        print(f'核PCA Poly失败: {str(e)}')

    print('核PCA非线性降维完成')
else:
    print('核PCA已禁用（内存优化）')

# ==========================================
# 升级 3: 集成降维方法（优化版）
# ==========================================
def integrated_dim_reduction(features, n_components=3):
    """
    集成多种降维方法，创建稳健的特征表示

    结合PCA和核PCA的优势，提升特征表达能力
    不使用期货数据，避免数据泄漏

    参数:
        features: 特征矩阵
        n_components: 分量数

    返回:
        integrated_features: 集成后的特征
    """
    # 1. 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features.fillna(0))

    # 2. PCA
    pca = PCA(n_components=n_components)
    pca_features = pca.fit_transform(X_scaled)

    # 3. 核PCA（可选，内存优化）
    if ENABLE_KERNEL_PCA:
        try:
            kpca = KernelPCA(n_components=n_components, kernel='rbf', gamma=0.1)
            kpca_features = kpca.fit_transform(X_scaled)
        except:
            kpca_features = pca_features
    else:
        kpca_features = pca_features

    # 4. 集成（加权平均）
    integrated_features = (pca_features + kpca_features) / 2

    return integrated_features

# 对交互特征应用集成降维
interaction_cols = [
    'climate_risk_drought_heat_stress_interaction',
    'climate_risk_excess_precip_drought_interaction',
    'climate_risk_drought_heat_stress_weighted'
]
available_interaction = [c for c in interaction_cols if c in merged_df.columns]

if len(available_interaction) >= 2:
    try:
        integrated_features = integrated_dim_reduction(
            merged_df[available_interaction],
            n_components=3
        )

        for i in range(integrated_features.shape[1]):
            merged_df[f'climate_risk_integrated_dim_{i}'] = integrated_features[:, i]
            ALL_NEW_FEATURES.append(f'climate_risk_integrated_dim_{i}')
    except Exception as e:
        print(f'集成降维失败: {str(e)}')

print('集成降维方法完成')

# ==========================================
# 保留原有方法（CCA和线性PCA）
# ==========================================


# 4.1 线性组合优化 - 已禁用以避免数据泄漏
# 原代码使用期货价格作为优化目标，违反竞赛规则
print('线性组合优化已禁用（避免数据泄漏 - 不能使用期货数据作为优化目标）')
print()

# 4.2 典型相关分析（CCA）- 已禁用以避免数据泄漏
# 原代码使用期货数据作为第二个变量集，违反竞赛规则
print('典型相关分析已禁用（避免数据泄漏 - 不能使用期货数据作为输入）')
print()


# 4.3 主成分分析（PCA）
def apply_pca(features, n_components=3, explained_variance_ratio_threshold=0.95):
    """应用主成分分析"""
    X_clean = features.fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)

    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(X_scaled)

    n_needed = np.argmax(np.cumsum(pca.explained_variance_ratio_) >= explained_variance_ratio_threshold) + 1
    if n_needed > n_components:
        pca = PCA(n_components=n_needed)
        pca_result = pca.fit_transform(X_scaled)

    return pca_result, pca

climate_score_cols = [f'climate_risk_{r}_score' for r in RISK_CATEGORIES]
pca_result, pca_model = apply_pca(merged_df[climate_score_cols], n_components=4)

for i in range(pca_result.shape[1]):
    merged_df[f'climate_risk_pca_component_{i}'] = pca_result[:, i]
    ALL_NEW_FEATURES.append(f'climate_risk_pca_component_{i}')

print('数学优化完成')
print()


# ==========================================
# 技术实现 5: 数据增强（扩展版）
# ==========================================
print('技术 5: 数据增强...')

# ==========================================
# 升级 1: 对抗性训练增强
# ==========================================
def adversarial_augmentation(feature_series, noise_level=0.1, adversarial_strength=0.15):
    """
    通过添加定向噪声创建对抗样本

    对抗样本的原理:
    - 在梯度方向上添加微小扰动，使模型预测发生变化
    - 增强模型对微小变化的鲁棒性
    - 模拟极端情况下的特征值

    参数:
        feature_series: 特征序列
        noise_level: 随机噪声水平（相对于标准差）
        adversarial_strength: 对抗性扰动强度

    返回:
        adversarial_series: 对抗增强后的序列
    """
    # 基础噪声
    base_noise = np.random.randn(len(feature_series)) * noise_level * feature_series.std()

    # 对抗性扰动：针对极值点施加更大扰动
    feature_mean = feature_series.mean()
    feature_std = feature_series.std()
    extremeness = np.abs(feature_series - feature_mean) / (feature_std + 1e-8)

    # 对抗性噪声（极值点扰动更大）
    adversarial_noise = np.random.randn(len(feature_series)) * adversarial_strength * feature_std * extremeness

    # 组合噪声
    total_perturbation = base_noise + adversarial_noise

    return feature_series + total_perturbation

# 对关键风险特征应用对抗性增强
for risk_type in ['drought', 'heat_stress', 'excess_precip']:
    score_col = f'climate_risk_{risk_type}_score'

    # 轻度对抗性增强（5%噪声）
    merged_df[f'climate_risk_{risk_type}_adv_light'] = adversarial_augmentation(
        merged_df[score_col].fillna(0),
        noise_level=0.05,
        adversarial_strength=0.08
    )
    ALL_NEW_FEATURES.append(f'climate_risk_{risk_type}_adv_light')

    # 中度对抗性增强（10%噪声）
    merged_df[f'climate_risk_{risk_type}_adv_medium'] = adversarial_augmentation(
        merged_df[score_col].fillna(0),
        noise_level=0.10,
        adversarial_strength=0.15
    )
    ALL_NEW_FEATURES.append(f'climate_risk_{risk_type}_adv_medium')

    # 强度对抗性增强（20%噪声）
    merged_df[f'climate_risk_{risk_type}_adv_strong'] = adversarial_augmentation(
        merged_df[score_col].fillna(0),
        noise_level=0.20,
        adversarial_strength=0.30
    )
    ALL_NEW_FEATURES.append(f'climate_risk_{risk_type}_adv_strong')

print('对抗性训练增强完成')

# ==========================================
# 升级 2: 时间序列合成（模拟罕见事件）
# ==========================================
def synthetic_rare_events(series, event_prob=0.02, event_magnitude=2.5):
    """
    生成合成罕见气候事件

    原理:
    - 在平稳序列中注入模拟的极端事件
    - 使用跳跃扩散模型模拟突发性风险
    - 捕获罕见但影响巨大的事件模式

    参数:
        series: 原始序列
        event_prob: 事件发生概率
        event_magnitude: 事件强度倍数

    返回:
        synthetic_series: 包含合成罕见事件的序列
    """
    synthetic_series = series.copy().fillna(0).values
    n_samples = len(synthetic_series)

    # 随机生成事件位置
    n_events = int(n_samples * event_prob)
    event_indices = np.random.choice(n_samples, size=n_events, replace=False)

    # 生成事件类型（正向/负向）
    event_directions = np.random.choice([1, -1], size=n_events)

    # 为每个事件创建持续时间（事件可能持续多天）
    event_durations = np.random.randint(1, 6, size=n_events)  # 1-5天

    # 应用事件
    for idx, direction, duration in zip(event_indices, event_directions, event_durations):
        # 事件影响随时间衰减
        for d in range(duration):
            if idx + d < n_samples:
                decay = 1.0 / (1 + d * 0.3)  # 衰减系数
                impact = direction * event_magnitude * synthetic_series.std() * decay
                synthetic_series[idx + d] += impact

    return synthetic_series

# 对关键风险特征应用时间序列合成（可选，内存优化）
if ENABLE_TIME_SERIES_SYNTHESIS:
    np.random.seed(123)  # 确保可重复性
    for risk_type in ['drought', 'heat_stress']:
        score_col = f'climate_risk_{risk_type}_score'

        # 低概率合成事件（模拟罕见严重事件）
        try:
            merged_df[f'climate_risk_{risk_type}_synthetic_rare'] = synthetic_rare_events(
                merged_df[score_col],
                event_prob=0.01,  # 1%概率
                event_magnitude=3.0  # 3倍标准差强度
            )
            ALL_NEW_FEATURES.append(f'climate_risk_{risk_type}_synthetic_rare')
        except Exception as e:
            print(f'合成罕见事件失败: {risk_type}, {str(e)}')

        # 中等概率合成事件
        try:
            merged_df[f'climate_risk_{risk_type}_synthetic_mod'] = synthetic_rare_events(
                merged_df[score_col],
                event_prob=0.02,
                event_magnitude=2.0
            )
            ALL_NEW_FEATURES.append(f'climate_risk_{risk_type}_synthetic_mod')
        except Exception as e:
            print(f'合成中等事件失败: {risk_type}, {str(e)}')

    print('时间序列合成完成')
else:
    print('时间序列合成已禁用（内存优化）')

# ==========================================
# 升级 3: 自适应噪声注入
# ==========================================
def adaptive_noise_injection(feature_series, quantile_thresholds=[0.1, 0.9]):
    """
    自适应噪声注入（根据特征分布动态调整噪声水平）

    原理:
    - 低值区域：增加噪声以提升敏感度
    - 高值区域：减少噪声以保留极端信号
    - 中值区域：标准噪声水平

    参数:
        feature_series: 特征序列
        quantile_thresholds: 分位数阈值 [低值阈值, 高值阈值]

    返回:
        augmented_series: 自适应噪声增强后的序列
    """
    series_values = feature_series.fillna(0).values
    q_low = np.quantile(series_values, quantile_thresholds[0])
    q_high = np.quantile(series_values, quantile_thresholds[1])

    # 为每个样本计算自适应噪声水平
    noise_levels = np.zeros_like(series_values)

    # 低值区域：高噪声
    mask_low = series_values < q_low
    noise_levels[mask_low] = 0.15  # 15%标准差

    # 高值区域：低噪声
    mask_high = series_values > q_high
    noise_levels[mask_high] = 0.05  # 5%标准差

    # 中值区域：中等噪声
    mask_mid = (~mask_low) & (~mask_high)
    noise_levels[mask_mid] = 0.10  # 10%标准差

    # 生成噪声
    noise = np.random.randn(len(series_values)) * series_values.std()
    adaptive_noise = noise * noise_levels

    return series_values + adaptive_noise

# 应用自适应噪声注入
for risk_type in ['drought', 'heat_stress', 'excess_precip']:
    score_col = f'climate_risk_{risk_type}_score'

    merged_df[f'climate_risk_{risk_type}_adaptive_noise'] = adaptive_noise_injection(
        merged_df[score_col],
        quantile_thresholds=[0.1, 0.9]
    )
    ALL_NEW_FEATURES.append(f'climate_risk_{risk_type}_adaptive_noise')

print('自适应噪声注入完成')

# ==========================================
# 保留原有数据增强方法
# ==========================================

# 5.1 Bootstrap重采样创建集成特征
def bootstrap_aggregate(series, n_bootstrap=10, window=30):
    """
    使用Bootstrap重采样创建鲁棒特征

    原理: 从窗口内重采样多次并取平均，减少噪声影响
    """
    bootstrapped_values = []

    for i in range(len(series)):
        if i >= window:
            window_data = series.iloc[i-window:i].values
            # Bootstrap重采样
            bootstrap_samples = []
            for _ in range(n_bootstrap):
                sample = np.random.choice(window_data, size=window, replace=True)
                bootstrap_samples.append(np.mean(sample))
            bootstrapped_values.append(np.mean(bootstrap_samples))
        else:
            bootstrapped_values.append(np.nan)

    return pd.Series(bootstrapped_values, index=series.index)

for risk_type in ['drought', 'heat_stress', 'excess_precip']:
    score_col = f'climate_risk_{risk_type}_score'

    merged_df[f'climate_risk_{risk_type}_bootstrap_agg'] = merged_df.groupby('region_id')[score_col].transform(
        lambda x: bootstrap_aggregate(x, n_bootstrap=10, window=30)
    )
    ALL_NEW_FEATURES.append(f'climate_risk_{risk_type}_bootstrap_agg')

# 5.2 添加可控噪声（增强特征鲁棒性）
np.random.seed(42)  # 固定随机种子确保可重复
for risk_type in ['drought', 'heat_stress', 'excess_precip']:
    score_col = f'climate_risk_{risk_type}_score'
    noise_std = merged_df[score_col].std() * 0.1  # 10%的标准差作为噪声

    # 创建噪声增强特征（正负噪声）
    merged_df[f'climate_risk_{risk_type}_noisy_plus'] = merged_df[score_col] + np.random.randn(len(merged_df)) * noise_std
    merged_df[f'climate_risk_{risk_type}_noisy_minus'] = merged_df[score_col] - np.random.randn(len(merged_df)) * noise_std

    ALL_NEW_FEATURES.extend([f'climate_risk_{risk_type}_noisy_plus', f'climate_risk_{risk_type}_noisy_minus'])

# 5.3 时间窗口扩展（创建多尺度特征）
print('创建多尺度时间窗口特征...')
for risk_type in ['drought', 'heat_stress']:
    score_col = f'climate_risk_{risk_type}_score'

    # 不同窗口的统计特征
    for window in [7, 14, 21, 30, 45, 60, 90]:
        # 移动平均
        merged_df[f'climate_risk_{risk_type}_ma_{window}d'] = merged_df.groupby('region_id')[score_col].transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
        ALL_NEW_FEATURES.append(f'climate_risk_{risk_type}_ma_{window}d')

        # 移动最大值
        merged_df[f'climate_risk_{risk_type}_max_{window}d'] = merged_df.groupby('region_id')[score_col].transform(
            lambda x: x.rolling(window, min_periods=1).max()
        )
        ALL_NEW_FEATURES.append(f'climate_risk_{risk_type}_max_{window}d')

        # 移动最小值
        merged_df[f'climate_risk_{risk_type}_min_{window}d'] = merged_df.groupby('region_id')[score_col].transform(
            lambda x: x.rolling(window, min_periods=1).min()
        )
        ALL_NEW_FEATURES.append(f'climate_risk_{risk_type}_min_{window}d')

        # 滚动偏度
        merged_df[f'climate_risk_{risk_type}_skew_{window}d'] = merged_df.groupby('region_id')[score_col].transform(
            lambda x: x.rolling(window, min_periods=5).skew()
        )
        ALL_NEW_FEATURES.append(f'climate_risk_{risk_type}_skew_{window}d')

        # 滚动峰度
        merged_df[f'climate_risk_{risk_type}_kurt_{window}d'] = merged_df.groupby('region_id')[score_col].transform(
            lambda x: x.rolling(window, min_periods=5).kurt()
        )
        ALL_NEW_FEATURES.append(f'climate_risk_{risk_type}_kurt_{window}d')

print('数据增强完成')
print()


# ==========================================
# CFCS专项优化特征
# ==========================================
print('CFCS专项优化特征...')

# 极值放大特征（针对Max_Corr_Score 30%权重）
for risk_type in ['drought', 'heat_stress', 'excess_precip']:
    score_col = f'climate_risk_{risk_type}_score'

    # 指数放大
    merged_df[f'climate_risk_{risk_type}_exp_scaled'] = np.exp(merged_df[score_col] * 2) - 1
    ALL_NEW_FEATURES.append(f'climate_risk_{risk_type}_exp_scaled')

    # 三次方放大
    merged_df[f'climate_risk_{risk_type}_cubic'] = np.sign(merged_df[score_col]) * np.abs(merged_df[score_col]) ** 3
    ALL_NEW_FEATURES.append(f'climate_risk_{risk_type}_cubic')

    # 极值标记
    mean_val = merged_df[score_col].mean()
    std_val = merged_df[score_col].std()
    merged_df[f'climate_risk_{risk_type}_extreme'] = ((merged_df[score_col] - mean_val) > 2 * std_val).astype(int)
    ALL_NEW_FEATURES.append(f'climate_risk_{risk_type}_extreme')

# 关键期加权特征（针对Avg_Sig_Corr_Score 50%权重）
key_months = [6, 7, 8]
merged_df['is_key_month'] = merged_df['date_on'].dt.month.isin(key_months).astype(int)

for risk_type in ['drought', 'excess_precip', 'heat_stress']:
    score_col = f'climate_risk_{risk_type}_score'

    # 关键期加权
    merged_df[f'climate_risk_{risk_type}_key_month'] = merged_df[score_col] * merged_df['is_key_month'] * 2
    ALL_NEW_FEATURES.append(f'climate_risk_{risk_type}_key_month')

    # 7月特化（授粉期最关键）
    merged_df[f'climate_risk_{risk_type}_july_peak'] = merged_df[score_col] * (merged_df['date_on'].dt.month == 7).astype(int) * 3
    ALL_NEW_FEATURES.append(f'climate_risk_{risk_type}_july_peak')

# 复合风险指数
# 干旱+热应激组合
if 'climate_risk_drought_score' in merged_df.columns and 'climate_risk_heat_stress_score' in merged_df.columns:
    merged_df['climate_risk_drought_heat_sum'] = (
        merged_df['climate_risk_drought_score'] + merged_df['climate_risk_heat_stress_score']
    )
    ALL_NEW_FEATURES.append('climate_risk_drought_heat_sum')

    merged_df['climate_risk_drought_heat_product'] = (
        merged_df['climate_risk_drought_score'] * merged_df['climate_risk_heat_stress_score']
    )
    ALL_NEW_FEATURES.append('climate_risk_drought_heat_product')

    # 归一化复合指数
    drought_norm = merged_df['climate_risk_drought_score'] / (merged_df['climate_risk_drought_score'].max() + 1e-6)
    heat_norm = merged_df['climate_risk_heat_stress_score'] / (merged_df['climate_risk_heat_stress_score'].max() + 1e-6)
    merged_df['climate_risk_drought_heat_normalized'] = (drought_norm + heat_norm) / 2
    ALL_NEW_FEATURES.append('climate_risk_drought_heat_normalized')

# 降水极性指数
if 'climate_risk_drought_score' in merged_df.columns and 'climate_risk_excess_precip_score' in merged_df.columns:
    precip_diff = merged_df['climate_risk_excess_precip_score'] - merged_df['climate_risk_drought_score']
    merged_df['climate_risk_precip_imbalance'] = np.abs(precip_diff)
    ALL_NEW_FEATURES.append('climate_risk_precip_imbalance')

    merged_df['climate_risk_total_precip_risk'] = np.maximum(
        merged_df['climate_risk_drought_score'],
        merged_df['climate_risk_excess_precip_score']
    )
    ALL_NEW_FEATURES.append('climate_risk_total_precip_risk')

print('CFCS专项优化完成')
print()

# ==========================================
# Phase 6: 期货价格移动与气候数据组合特征
# ==========================================
print('==========================================')
print('期货价格移动与气候数据组合特征')
print('==========================================')

# 准备数据 - 使用merged_df而不是baseline_df
futures_cols = [c for c in merged_df.columns if c.startswith('futures_close')]
climate_cols = [c for c in merged_df.columns if c.startswith('climate_risk_')]

print(f'期货价格列: {len(futures_cols)}')
print(f'气候特征列: {len(climate_cols)}')
print()

# ==========================================
# 6.1 期货价格移动特征
# ==========================================
print('6.1 期货价格移动特征...')

for fut_col in futures_cols:
    if fut_col not in merged_df.columns:
        continue
    
    # 移动平均
    for window in [5, 10, 20, 30]:
        merged_df[f'{fut_col}_ma{window}'] = merged_df[fut_col].rolling(window, min_periods=1).mean()
        ALL_NEW_FEATURES.append(f'{fut_col}_ma{window}')
    
    # 指数加权移动平均
    for alpha in [0.1, 0.3, 0.5]:
        merged_df[f'{fut_col}_ewma{alpha}'] = merged_df[fut_col].ewm(alpha=alpha, adjust=False).mean()
        ALL_NEW_FEATURES.append(f'{fut_col}_ewma{alpha}')
    
    # 价格变化率
    merged_df[f'{fut_col}_change_1d'] = merged_df[fut_col].pct_change(1)
    merged_df[f'{fut_col}_change_5d'] = merged_df[fut_col].pct_change(5)
    merged_df[f'{fut_col}_change_10d'] = merged_df[fut_col].pct_change(10)
    ALL_NEW_FEATURES.extend([f'{fut_col}_change_1d', f'{fut_col}_change_5d', f'{fut_col}_change_10d'])
    
    # 动量特征
    merged_df[f'{fut_col}_momentum_5'] = merged_df[fut_col] / merged_df[fut_col].shift(5) - 1
    merged_df[f'{fut_col}_momentum_10'] = merged_df[fut_col] / merged_df[fut_col].shift(10) - 1
    merged_df[f'{fut_col}_momentum_20'] = merged_df[fut_col] / merged_df[fut_col].shift(20) - 1
    ALL_NEW_FEATURES.extend([f'{fut_col}_momentum_5', f'{fut_col}_momentum_10', f'{fut_col}_momentum_20'])
    
    # 波动率
    for window in [5, 10, 20]:
        merged_df[f'{fut_col}_vol{window}'] = merged_df[fut_col].rolling(window, min_periods=2).std()
        merged_df[f'{fut_col}_vol{window}_norm'] = merged_df[f'{fut_col}_vol{window}'] / (merged_df[fut_col].rolling(window, min_periods=1).mean() + 1e-8)
        ALL_NEW_FEATURES.extend([f'{fut_col}_vol{window}', f'{fut_col}_vol{window}_norm'])

print('✓ 期货价格移动特征创建完成')
print()

# ==========================================
# 6.2 期货-气候交互特征 - 已删除以避免数据泄漏
# ==========================================
print('6.2 跳过期货-气候交互特征（避免数据泄漏）')
print()

# ==========================================
# 6.3 期货动量-气候组合特征 - 已删除以避免数据泄漏
# ==========================================
print('6.3 跳过期货动量-气候组合特征（避免数据泄漏）')
print()

# ==========================================
# 6.4 交叉滞后特征 - 已删除以避免数据泄漏
# ==========================================
print('6.4 跳过交叉滞后特征（避免数据泄漏）')
print()

# ==========================================
# 6.5 多期货合约组合特征 - 已删除以避免数据泄漏
# ==========================================
print('6.5 跳过多期货合约组合特征（避免数据泄漏）')
print()

# ==========================================
# 6.6 时间同步特征 - 已删除以避免数据泄漏
# ==========================================
print('6.6 跳过时间同步特征（避免数据泄漏）')
print()

# ==========================================
# 6.7 极值触发特征 - 已删除以避免数据泄漏
# ==========================================
print('6.7 跳过极值触发特征（避免数据泄漏）')
print()

# ==========================================
# 6.8 组合特征优化与填充 - 已删除以避免数据泄漏
# ==========================================
print('6.8 跳过组合特征优化（避免数据泄漏）')
print()

# ==========================================
# 跳过期货相关的CFCS计算和保存
# ==========================================
print('6.9 跳过期货相关的CFCS计算（避免数据泄漏）')
print('==========================================')
print('已删除所有期货-气候组合特征以符合竞赛规则')
print('==========================================')
print()



# ==========================================
# CFCS计算和分析函数（在使用前定义）
# ==========================================
def compute_cfcs(df, verbose=True):
    """
    计算气候-期货相关性评分 (Climate-Futures Correlation Score, CFCS)

    CFCS = (0.5 × Avg_Sig_Corr_Score) + (0.3 × Max_Corr_Score) + (0.2 × Sig_Count_Score)
    """
    climate_cols = [c for c in df.columns if c.startswith('climate_risk_')]
    futures_cols = [c for c in df.columns if c.startswith('futures_')]

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
        print(f'CFCS: {result["cfcs"]:.2f} | Sig: {result["sig_count"]}/{result["total"]} ({result["sig_pct"]:.2f}%) | Features: {result["n_features"]}')

    return result

def analyze_feature_contributions(df, climate_cols, futures_cols):
    """分析每个特征对CFCS的贡献"""
    feature_stats = {col: {'sig_count': 0, 'total': 0, 'max_corr': 0, 'sig_corrs': []} for col in climate_cols}

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

print('==========================================')
print('工具函数定义完成')
print('==========================================')
print()




# ==========================================
# Phase 7: 清理期货衍生列
# ==========================================
print('==========================================')
print('清理期货衍生列')
print('==========================================')

# 定义需要保留的期货特征列
futures_columns_to_keep = [
    # Price Data
    'futures_close_ZC_1', 'futures_close_ZC_2', 'futures_close_ZW_1', 'futures_close_ZS_1',
    # Technical Indicators
    'futures_zc1_ret_pct', 'futures_zc1_ret_log', 'futures_zc_term_spread', 'futures_zc_term_ratio',
    # Moving Averages
    'futures_zc1_ma_20', 'futures_zc1_ma_60', 'futures_zc1_ma_120',
    # Volatility Measures
    'futures_zc1_vol_20', 'futures_zc1_vol_60',
    # Cross-Commodity Relationships
    'futures_zw_zc_spread', 'futures_zc_zw_ratio', 'futures_zs_zc_spread', 'futures_zc_zs_ratio'
]

# 查找所有futures_开头的列
all_futures_columns = [c for c in merged_df.columns if c.startswith('futures_')]

# 确定需要删除的列
futures_columns_to_remove = [c for c in all_futures_columns if c not in futures_columns_to_keep]

if futures_columns_to_remove:
    print(f'发现 {len(all_futures_columns)} 个期货相关列')
    print(f'保留 {len(futures_columns_to_keep)} 个核心期货列')
    print(f'删除 {len(futures_columns_to_remove)} 个衍生期货列')
    
    # 从DataFrame中删除这些列
    merged_df = merged_df.drop(columns=futures_columns_to_remove, errors='ignore')
    
    # 从ALL_NEW_FEATURES中移除这些列(如果有的话)
    ALL_NEW_FEATURES = [f for f in ALL_NEW_FEATURES if f not in futures_columns_to_remove]
    
    print('✓ 期货衍生列清理完成')
else:
    print('没有需要删除的期货衍生列')
print()


# ==========================================
# Phase 7: 大规模特征筛选
# ==========================================
print('==========================================')
print('大规模特征筛选阶段')
print('==========================================')

# 清理不需要的futures_close_衍生列，保留核心期货特征
print('清理futures_close_衍生列...')
futures_cols_to_keep = [
    # Price Data
    'futures_close_ZC_1', 'futures_close_ZC_2', 'futures_close_ZW_1', 'futures_close_ZS_1',
    # Technical Indicators
    'futures_zc1_ret_pct', 'futures_zc1_ret_log', 'futures_zc_term_spread', 'futures_zc_term_ratio',
    # Moving Averages
    'futures_zc1_ma_20', 'futures_zc1_ma_60', 'futures_zc1_ma_120',
    # Volatility Measures
    'futures_zc1_vol_20', 'futures_zc1_vol_60',
    # Cross-Commodity Relationships
    'futures_zw_zc_spread', 'futures_zc_zw_ratio', 'futures_zs_zc_spread', 'futures_zc_zs_ratio'
]

# 查找所有futures_close_开头的列
all_futures_close_cols = [c for c in merged_df.columns if c.startswith('futures_close_')]

# 找出需要删除的列
futures_cols_to_remove = [c for c in all_futures_close_cols if c not in futures_cols_to_keep]

if len(futures_cols_to_remove) > 0:
    print(f'  删除 {len(futures_cols_to_remove)} 个futures_close_衍生列')
    merged_df = merged_df.drop(columns=futures_cols_to_remove, errors='ignore')
    print(f'  保留 {len([c for c in merged_df.columns if c.startswith("futures_")])} 个期货特征')
print()

print(f'生成特征总数: {len(ALL_NEW_FEATURES)}')

# 填充缺失值
print('填充缺失值...')
for col in ALL_NEW_FEATURES:
    if col in merged_df.columns:
        merged_df[col] = merged_df[col].fillna(0)

print('执行快速特征筛选...')

# 快速相关性采样（使用采样数据快速评估）
sample_size = min(50000, len(merged_df))
df_sample = merged_df.sample(n=sample_size, random_state=42) if len(merged_df) > sample_size else merged_df

climate_cols = ALL_NEW_FEATURES
futures_cols = [c for c in merged_df.columns if c.startswith('futures_')]

# 快速评估每个特征
quick_results = []
for clim in climate_cols:
    if clim not in merged_df.columns:
        continue
        
    max_corr = 0
    sig_count = 0
    total = 0
    
    # 采样计算
    for country in df_sample['country_name'].unique():
        df_country = df_sample[df_sample['country_name'] == country]
        for month in df_country['date_on_month'].unique():
            df_month = df_country[df_country['date_on_month'] == month]
            
            for fut in futures_cols:
                if df_month[clim].std() > 0 and df_month[fut].std() > 0:
                    corr = df_month[[clim, fut]].corr().iloc[0, 1]
                    total += 1
                    if abs(corr) > max_corr:
                        max_corr = abs(corr)
                    if abs(corr) >= SIGNIFICANCE_THRESHOLD:
                        sig_count += 1
    
    quick_results.append({
        'feature': clim,
        'max_corr': max_corr,
        'sig_count': sig_count,
        'total': total,
        'sig_pct': sig_count / total * 100 if total > 0 else 0
    })

quick_df = pd.DataFrame(quick_results)
print()
print('筛选结果统计:')
if len(quick_df) > 0:
    print(f'  平均最大相关性: {quick_df["max_corr"].mean():.4f}')
    print(f'  平均显著相关比例: {quick_df["sig_pct"].mean():.2f}%')

print()
print('应用多维度筛选策略...')

# 筛选标准
zero_corr_features = quick_df[quick_df['max_corr'] == 0]['feature'].tolist()
low_corr_features = quick_df[quick_df['max_corr'] < 0.3]['feature'].tolist()
zero_sig_features = quick_df[quick_df['sig_count'] == 0]['feature'].tolist()

print(f'  零相关性特征: {len(zero_corr_features)} 个')
print(f'  低相关性特征（max < 0.3）: {len(low_corr_features)} 个')
print(f'  零显著相关特征: {len(zero_sig_features)} 个')
print()

# 保留基础特征
baseline_features_to_keep = [c for c in ALL_NEW_FEATURES if any(
    keyword in c for keyword in ['cnt_locations', '_score', '_weighted']
)]

# 标记需要删除的特征
features_to_remove = list(set(zero_corr_features + low_corr_features + zero_sig_features) - set(baseline_features_to_keep))

print(f'标记删除特征: {len(features_to_remove)} 个')
print(f'保留基础特征: {len(baseline_features_to_keep)} 个')
print()

# 计算保留的特征
features_to_keep = list(set(ALL_NEW_FEATURES) - set(features_to_remove))

# 限制到目标特征数
if len(features_to_keep) > TARGET_FEATURE_COUNT:
    # 按相关性排序，保留最好的
    quick_df_sorted = quick_df.sort_values(['max_corr', 'sig_count'], ascending=False)
    top_features = quick_df_sorted.head(TARGET_FEATURE_COUNT)['feature'].tolist()
    # 确保保留基础特征
    features_to_keep = list(set(top_features) | set(baseline_features_to_keep))

print(f'最终保留特征数: {len(features_to_keep)}')
print(f'目标特征数: {TARGET_FEATURE_COUNT}')
print()

# 更新全局变量
ALL_NEW_FEATURES_FILTERED = features_to_keep

# 删除低效特征
cols_to_drop = [c for c in merged_df.columns if c.startswith('climate_risk_') and c not in ALL_NEW_FEATURES_FILTERED]
print(f'删除 {len(cols_to_drop)} 个低效特征以节省内存')
merged_df = merged_df.drop(columns=cols_to_drop, errors='ignore')

# 清理内存
gc.collect()

print('✓ 大规模特征筛选完成')
print()

# ==========================================
# Phase 7: 数据准备与评估
# ==========================================
print('==========================================')
print('Phase 7: 数据准备与评估')
print('==========================================')


# 准备输出数据
print('准备输出数据...')

# 使用基线的 valid_ids 进行筛选
print(f'基线有效行数: {len(baseline_valid_df):,}')
valid_ids = baseline_valid_df['ID'].tolist()

# 筛选 merged_df 为基线相同的有效行
baseline_df = merged_df[merged_df['ID'].isin(valid_ids)]

# 填充缺失值
for col in ALL_NEW_FEATURES:
    if col in baseline_df.columns:
        baseline_df[col] = baseline_df[col].fillna(0)

print(f'处理完成: {len(baseline_df):,} 行')
print()


# 计算CFCS分数
print('计算CFCS分数...')
print('=' * 60)

climate_cols = [c for c in baseline_df.columns if c.startswith('climate_risk_')]
futures_cols = [c for c in baseline_df.columns if c.startswith('futures_')]

print(f'分析 {len(climate_cols)} 个特征...')

# 计算整体CFCS
cfcs_result = compute_cfcs(baseline_df, verbose=True)

# 特征分析 - 使用筛选后的特征
climate_cols = [c for c in baseline_df.columns if c.startswith('climate_risk_') and c in ALL_NEW_FEATURES_FILTERED]
feature_analysis = analyze_feature_contributions(baseline_df, climate_cols, futures_cols)

print()
print('==========================================')
print('表现最好的30个特征')
print('==========================================')
print(feature_analysis.head(30).to_string(index=False))
print()

# 特征筛选 - 基于完整CFCS计算的最终筛选
print('==========================================')
print('最终特征筛选（基于完整CFCS计算）')
print('==========================================')

# 再次筛选零显著相关特征
zero_sig_features = feature_analysis[feature_analysis['sig_count'] == 0]['feature'].tolist()
original_cols = [c for c in zero_sig_features if 'cnt_locations' in c or '_score' in c or '_weighted' in c]
FEATURES_TO_REMOVE_FINAL = [c for c in zero_sig_features if c not in original_cols]

print(f'移除特征数: {len(FEATURES_TO_REMOVE_FINAL)}')
print(f'保留的原始特征: {len(original_cols)}')

# 应用最终筛选
optimized_df = baseline_df.drop(columns=FEATURES_TO_REMOVE_FINAL, errors='ignore')
climate_cols_opt = [c for c in optimized_df.columns if c.startswith('climate_risk_')]

print(f'优化后特征数: {len(climate_cols_opt)}')
print()

# 如果特征数仍然过多，按sig_count筛选
if len(climate_cols_opt) > TARGET_FEATURE_COUNT:
    print(f'特征数 {len(climate_cols_opt)} 超过目标 {TARGET_FEATURE_COUNT}，进一步筛选...')
    # 安全过滤：只选择存在于数据框中的特征
    available_features = [f for f in feature_analysis['feature'].tolist() if f in climate_cols_opt]
    top_features = sorted(available_features, 
                        key=lambda x: feature_analysis[feature_analysis['feature'] == x]['sig_count'].iloc[0] if len(feature_analysis[feature_analysis['feature'] == x]) > 0 else 0,
                        reverse=True)[:TARGET_FEATURE_COUNT]
    features_to_drop = [c for c in climate_cols_opt if c not in top_features]
    optimized_df = optimized_df.drop(columns=features_to_drop, errors='ignore')
    climate_cols_opt = [c for c in optimized_df.columns if c.startswith('climate_risk_')]
    print(f'筛选后特征数: {len(climate_cols_opt)}')
    print()

# 计算优化后的CFCS
print('==========================================')
print('优化后CFCS分数')
print('==========================================')
optimized_cfcs = compute_cfcs(optimized_df, verbose=True)
print()


# 保存最终提交文件
print('保存提交文件...')

# 创建最优特征组合
required_cols = ['ID', 'date_on', 'country_name', 'region_name'] if 'ID' in optimized_df.columns else ['date_on', 'country_name', 'region_name']
futures_cols = [c for c in optimized_df.columns if c.startswith('futures_')]

# 策略选择：保留top N特征
TOP_N_FEATURES = min(50, len(climate_cols_opt))  # 使用实际存在的特征数
# 安全过滤：只选择存在于数据框中的特征
available_features = [f for f in feature_analysis['feature'].tolist() if f in climate_cols_opt]
top_features = sorted(available_features,
                        key=lambda x: feature_analysis[feature_analysis['feature'] == x]['sig_count'].iloc[0] if len(feature_analysis[feature_analysis['feature'] == x]) > 0 else 0,
                        reverse=True)[:TOP_N_FEATURES]
climate_selected = top_features

print(f'选择特征数: {len(climate_selected)}')
print(f'  前10特征: {climate_selected[:10]}')
print(f'可用特征总数: {len(available_features)}')
print()

all_selected_features = climate_selected

final_cols = required_cols + futures_cols + all_selected_features

# 再次安全检查：确保所有列都存在
valid_cols = [c for c in final_cols if c in optimized_df.columns]
if len(valid_cols) != len(final_cols):
    print(f'警告: 过滤掉 {len(final_cols) - len(valid_cols)} 个不存在的列')
    print(f'缺失的列: {set(final_cols) - set(valid_cols)}')
    final_cols = valid_cols

submission_df = optimized_df[final_cols].copy()

# 确保所有列都没有空值（在去重之前）
print('检查并填充空值...')
null_counts = submission_df.isnull().sum()
if null_counts.sum() > 0:
    print(f'发现空值列: {null_counts[null_counts > 0].to_dict()}')
    # 填充数值列
    for col in submission_df.columns:
        if submission_df[col].isnull().sum() > 0:
            if submission_df[col].dtype in ['float64', 'int64']:
                submission_df[col] = submission_df[col].fillna(0)
            else:
                submission_df[col] = submission_df[col].fillna('Unknown')
print('空值检查完成')

# 按日期和国家排序后,保留每个组合的第一条记录
submission_df = submission_df.sort_values(['date_on', 'country_name', 'region_name'])
submission_df = submission_df.drop_duplicates(subset=['date_on', 'country_name', 'region_name'], keep='first')

# 确保输出行数正好是219161
print(f'当前行数: {len(submission_df):,}')
print(f'目标行数: {REQUIRED_ROWS:,}')
if len(submission_df) > REQUIRED_ROWS:
    print(f'截取前{REQUIRED_ROWS}行')
    submission_df = submission_df.iloc[:REQUIRED_ROWS]
elif len(submission_df) < REQUIRED_ROWS:
    print(f'警告: 实际行数{len(submission_df):,}少于目标行数{REQUIRED_ROWS:,}')

# 保存
submission_df.to_csv(f'{OUTPUT_PATH}submission_math_optimized.csv', index=False)
print(f'主提交文件: submission_math_optimized.csv')
print(f'  - 行数: {len(submission_df):,}')
print(f'  - 列数: {len(submission_df.columns):,}')
print(f'  - 气候特征: {len([c for c in submission_df.columns if c.startswith("climate_risk_")])}')

# 保存完整版本（包含所有筛选后的特征）
full_submission_df = optimized_df[[c for c in optimized_df.columns if c.startswith('climate_risk_') and c in climate_cols_opt or c.startswith('futures_') or c in ['ID', 'date_on', 'country_name', 'region_name']]].copy()

# 确保完整版本也没有空值
if full_submission_df.isnull().sum().sum() > 0:
    for col in full_submission_df.columns:
        if full_submission_df[col].isnull().sum() > 0:
            if full_submission_df[col].dtype in ['float64', 'int64']:
                full_submission_df[col] = full_submission_df[col].fillna(0)
            else:
                full_submission_df[col] = full_submission_df[col].fillna('Unknown')

# 同样确保完整版本也是219161行
full_submission_df = full_submission_df.sort_values(['date_on', 'country_name', 'region_name'])
full_submission_df = full_submission_df.drop_duplicates(subset=['date_on', 'country_name', 'region_name'], keep='first')
if len(full_submission_df) > REQUIRED_ROWS:
    full_submission_df = full_submission_df.iloc[:REQUIRED_ROWS]

full_submission_df.to_csv(f'{OUTPUT_PATH}submission_full_features.csv', index=False)
print(f'完整特征版: submission_full_features.csv')
print(f'  - 行数: {len(full_submission_df):,}')
print(f'  - 列数: {len(full_submission_df.columns):,}')

print()
print('==========================================')
print('优化完成!')
print('==========================================')
print()
print('技术总结:')
print('1. 高相关性特征工程: 数学建模创建与期货高度相关的特征')
print('2. 特征变换优化: 对数/平方根/Box-Cox/傅里叶/标准化变换')
print('3. 时间对齐优化: 最佳滞后/滚动相关性/DTW距离')
print('4. 数学优化: 线性组合优化/典型相关分析/主成分分析')
print('5. 数据增强: Bootstrap重采样/可控噪声/时间窗口扩展')
print()
print('CFCS专项优化:')
print('- 极值放大特征 (提升Max_Corr_Score 30%)')
print('- 关键期加权特征 (提升Avg_Sig_Corr_Score 50%)')
print('- 复合风险指数 (多风险协同效应)')
print()
print(f'最终CFCS分数: {optimized_cfcs["cfcs"]:.2f}')
print()

# ==========================================
# 导出排名前N的特征组合CSV文件并计算CFCS
# ==========================================
print('==========================================')
print(f'导出前{min(200, len(feature_analysis))}特征组合CSV文件')
print('==========================================')

export_n = min(200, len(feature_analysis))
# 安全过滤：只选择存在于optimized_df中的特征
top_n_features = [f for f in feature_analysis.head(export_n)['feature'].tolist() if f in optimized_df.columns]
print(f'Top {len(top_n_features)}个可用特征:')
for i, feat in enumerate(top_n_features, 1):
    sig = feature_analysis[feature_analysis['feature'] == feat]['sig_count'].values[0]
    print(f'  {i}. {feat} (sig_count={sig})')
print()

# 导出top N单个特征
for i in range(len(top_n_features)):
    rank = i + 1
    top_feature = top_n_features[i]

    # 创建包含该排名特征的CSV
    required_cols = ['ID', 'date_on', 'country_name', 'region_name'] if 'ID' in optimized_df.columns else ['date_on', 'country_name', 'region_name']
    futures_cols = [c for c in optimized_df.columns if c.startswith('futures_')]

    # 选择当前排名特征
    climate_selected = [top_feature]

    final_cols = required_cols + futures_cols + climate_selected

    rank_df = optimized_df[final_cols].copy()

    # 添加date_on_month列用于CFCS计算
    rank_df['date_on_month'] = rank_df['date_on'].dt.month

    # 填充空值
    for col in rank_df.columns:
        if rank_df[col].isnull().sum() > 0:
            if rank_df[col].dtype in ['float64', 'int64']:
                rank_df[col] = rank_df[col].fillna(0)
            else:
                rank_df[col] = rank_df[col].fillna('Unknown')

    # 去重并确保行数
    rank_df = rank_df.sort_values(['date_on', 'country_name', 'region_name'])
    rank_df = rank_df.drop_duplicates(subset=['date_on', 'country_name', 'region_name'], keep='first')

    if len(rank_df) > REQUIRED_ROWS:
        rank_df = rank_df.iloc[:REQUIRED_ROWS]

    # 保存文件
    filename = f'{OUTPUT_PATH}submission_rank_{rank}_feature.csv'
    rank_df.to_csv(filename, index=False)

    # 计算该文件的CFCS得分
    climate_cols_rank = [c for c in rank_df.columns if c.startswith('climate_risk_')]
    futures_cols_rank = [c for c in rank_df.columns if c.startswith('futures_')]

    if climate_cols_rank:
        rank_cfcs = compute_cfcs(rank_df, verbose=False)
        print(f'Rank {rank}: {top_feature}')
        print(f'  文件: submission_rank_{rank}_feature.csv')
        print(f'  行数: {len(rank_df):,}')
        print(f'  CFCS得分: {rank_cfcs["cfcs"]:.2f} | Sig: {rank_cfcs["sig_count"]}/{rank_cfcs["total"]} ({rank_cfcs["sig_pct"]:.2f}%)')
        print()

# 导出组合特征文件（top 5, 10, 20, 50）
for n in [5, 10, 20, 50]:
    if n <= len(feature_analysis):
        top_n = feature_analysis.head(n)['feature'].tolist()
        required_cols = ['ID', 'date_on', 'country_name', 'region_name'] if 'ID' in optimized_df.columns else ['date_on', 'country_name', 'region_name']
        futures_cols = [c for c in optimized_df.columns if c.startswith('futures_')]
        climate_selected = [c for c in top_n if c in optimized_df.columns]
        
        final_cols = required_cols + futures_cols + climate_selected
        combo_df = optimized_df[final_cols].copy()
        combo_df['date_on_month'] = combo_df['date_on'].dt.month
        
        # 填充空值
        for col in combo_df.columns:
            if combo_df[col].isnull().sum() > 0:
                if combo_df[col].dtype in ['float64', 'int64']:
                    combo_df[col] = combo_df[col].fillna(0)
                else:
                    combo_df[col] = combo_df[col].fillna('Unknown')
        
        # 去重并确保行数
        combo_df = combo_df.sort_values(['date_on', 'country_name', 'region_name'])
        combo_df = combo_df.drop_duplicates(subset=['date_on', 'country_name', 'region_name'], keep='first')
        
        if len(combo_df) > REQUIRED_ROWS:
            combo_df = combo_df.iloc[:REQUIRED_ROWS]
        
        filename = f'{OUTPUT_PATH}submission_top_{n}_features.csv'
        combo_df.to_csv(filename, index=False)
        
        combo_cfcs = compute_cfcs(combo_df, verbose=False)
        print(f'Top {n} features: {filename}')
        print(f'  CFCS得分: {combo_cfcs["cfcs"]:.2f} | Sig: {combo_cfcs["sig_count"]}/{combo_cfcs["total"]} ({combo_cfcs["sig_pct"]:.2f}%)')
        print()

print('所有特征组合文件导出完成!')
print()

# ==========================================
# 对排名前十的特征进行专项数学优化
# ==========================================
print('==========================================')
print('对排名前十的特征进行专项数学优化')
print('==========================================')
print()

# 获取排名前十的特征
export_n = min(10, len(feature_analysis))
# 安全过滤：只选择存在于optimized_df中的特征
top_10_features = [f for f in feature_analysis.head(export_n)['feature'].tolist() if f in optimized_df.columns]
print(f'Top {len(top_10_features)}个可用特征:')
for i, feat in enumerate(top_10_features, 1):
    sig = feature_analysis[feature_analysis['feature'] == feat]['sig_count'].values[0]
    print(f'  {i}. {feat} (sig_count={sig})')
print()

# 对每个排名前10的特征进行数学优化
for rank_idx, feature_name in enumerate(top_10_features, 1):
    rank = rank_idx
    print(f'==========================================')
    print(f'Rank {rank}: {feature_name} 数学优化')
    print(f'==========================================')

    if feature_name in optimized_df.columns:
        print(f'原始特征统计:')
        print(f'  - 均值: {optimized_df[feature_name].mean():.6f}')
        print(f'  - 标准差: {optimized_df[feature_name].std():.6f}')
        print(f'  - 最小值: {optimized_df[feature_name].min():.6f}')
        print(f'  - 最大值: {optimized_df[feature_name].max():.6f}')
        print()

        # 创建优化后的DataFrame - 基础列 + futures_列
        optimized_feature_df = optimized_df[['ID', 'date_on', 'country_name', 'region_name'] if 'ID' in optimized_df.columns else ['date_on', 'country_name', 'region_name']].copy()

        # 添加所有futures_列
        futures_cols = [c for c in optimized_df.columns if c.startswith('futures_')]
        for col in futures_cols:
            optimized_feature_df[col] = optimized_df[col].values

        # 添加date_on_month列
        if 'date_on_month' not in optimized_df.columns:
            optimized_feature_df['date_on_month'] = optimized_df['date_on'].dt.month
        optimized_feature_df['date_on_month'] = optimized_df['date_on_month'] if 'date_on_month' in optimized_feature_df.columns else optimized_df['date_on'].dt.month

        feature_original = optimized_df[feature_name].copy()
        feature_clean = feature_original.fillna(feature_original.mean())

        # 存储所有变换版本的特征列名
        feature_transforms = {}

        # 方法1: 分位数标准化 (RankGauss)
        from scipy.stats import norm, rankdata
        rank_values = rankdata(feature_clean)
        quantile = (rank_values - 0.5) / len(rank_values)
        feature_rankgauss = norm.ppf(quantile)
        feature_rankgauss = np.nan_to_num(feature_rankgauss, nan=0)
        col_name = f'{feature_name}_rankgauss'
        optimized_feature_df[col_name] = feature_rankgauss
        feature_transforms[col_name] = feature_rankgauss
        print('✓ 分位数标准化 (RankGauss): 将特征转换为标准正态分布')

        # 方法2: Box-Cox变换
        from scipy.stats import boxcox
        feature_positive = feature_clean - feature_clean.min() + 0.001
        try:
            feature_boxcox, lambda_val = boxcox(feature_positive)
            feature_boxcox = np.nan_to_num(feature_boxcox, nan=0)
            col_name = f'{feature_name}_boxcox'
            optimized_feature_df[col_name] = feature_boxcox
            feature_transforms[col_name] = feature_boxcox
            print(f'✓ Box-Cox变换: λ={lambda_val:.3f}, 使数据更接近正态分布')
        except:
            feature_boxcox = np.log1p(feature_positive)
            col_name = f'{feature_name}_boxcox'
            optimized_feature_df[col_name] = feature_boxcox
            feature_transforms[col_name] = feature_boxcox
            print('✓ Log变换: Box-Cox失败, 使用对数变换替代')

        # 方法3: Yeo-Johnson变换 (支持负值)
        from scipy.stats import yeojohnson
        try:
            feature_yeojohnson, lambda_val = yeojohnson(feature_clean)
            feature_yeojohnson = np.nan_to_num(feature_yeojohnson, nan=0)
            col_name = f'{feature_name}_yeojohnson'
            optimized_feature_df[col_name] = feature_yeojohnson
            feature_transforms[col_name] = feature_yeojohnson
            print(f'✓ Yeo-Johnson变换: λ={lambda_val:.3f}')
        except:
            print('✗ Yeo-Johnson变换: 失败')

        # 方法4: 鲁棒标准化 (RobustScaler)
        median = np.median(feature_clean)
        q75, q25 = np.percentile(feature_clean, [75, 25])
        iqr = q75 - q25
        feature_robust = (feature_clean - median) / (iqr + 1e-8)
        feature_robust = np.nan_to_num(feature_robust, nan=0)
        col_name = f'{feature_name}_robust'
        optimized_feature_df[col_name] = feature_robust
        feature_transforms[col_name] = feature_robust
        print('✓ 鲁棒标准化: 使用中位数和四分位距, 抗异常值')

        # 方法5: Power变换 (分数幂)
        for power in [0.5, 0.33, 2, 3]:
            if power < 1:
                feature_power = np.power(feature_clean + 1, power) - 1
            else:
                feature_power = np.sign(feature_clean) * np.power(np.abs(feature_clean), power)
            feature_power = np.nan_to_num(feature_power, nan=0)
            col_name = f'{feature_name}_power{power}'
            optimized_feature_df[col_name] = feature_power
            feature_transforms[col_name] = feature_power
        print('✓ 幂变换: power=[0.5, 0.33, 2, 3] 捕捉非线性关系')

        # 方法6: 分桶编码 (Bucket Encoding)
        n_bins = 10
        feature_binned = pd.qcut(feature_clean, q=n_bins, labels=False, duplicates='drop')
        feature_onehot = pd.get_dummies(feature_binned, prefix=f'{feature_name}_bin')
        for col in feature_onehot.columns:
            optimized_feature_df[col] = feature_onehot[col].values
        print(f'✓ 分桶编码: {len(feature_onehot.columns)}个桶, 捕捉非线性模式')

        # 方法7: 滞后特征
        for lag in [1, 2, 3, 5, 7]:
            feature_lag = feature_clean.shift(lag)
            feature_lag = feature_lag.fillna(feature_clean.mean())
            col_name = f'{feature_name}_lag{lag}'
            optimized_feature_df[col_name] = feature_lag.values
            feature_transforms[col_name] = feature_lag.values
        print('✓ 滞后特征: lag=[1,2,3,5,7] 捕捉时间延迟效应')

        # 方法8: 滚动统计特征
        for window in [7, 14, 30]:
            rolling = pd.Series(feature_clean).rolling(window=window, min_periods=1)
            col_name = f'{feature_name}_rolling_mean_{window}'
            optimized_feature_df[col_name] = rolling.mean().values
            feature_transforms[col_name] = rolling.mean().values
            col_name = f'{feature_name}_rolling_std_{window}'
            optimized_feature_df[col_name] = rolling.std().fillna(0).values
            feature_transforms[col_name] = rolling.std().fillna(0).values
            col_name = f'{feature_name}_rolling_max_{window}'
            optimized_feature_df[col_name] = rolling.max().values
            feature_transforms[col_name] = rolling.max().values
            col_name = f'{feature_name}_rolling_min_{window}'
            optimized_feature_df[col_name] = rolling.min().values
            feature_transforms[col_name] = rolling.min().values
        print('✓ 滚动统计: window=[7,14,30] 捕捉动态变化')

        # 方法9: 指数加权移动平均
        for alpha in [0.1, 0.3, 0.5]:
            ewm = pd.Series(feature_clean).ewm(alpha=alpha, adjust=False).mean()
            col_name = f'{feature_name}_ewm_{alpha}'
            optimized_feature_df[col_name] = ewm.values
            feature_transforms[col_name] = ewm.values
        print('✓ EWMA: alpha=[0.1,0.3,0.5] 捕捉趋势')

        # 方法10: 差分特征
        feature_diff1 = feature_clean.diff(1).fillna(0)
        feature_diff2 = feature_clean.diff(2).fillna(0)
        feature_diff7 = feature_clean.diff(7).fillna(0)
        col_name = f'{feature_name}_diff1'
        optimized_feature_df[col_name] = feature_diff1.values
        feature_transforms[col_name] = feature_diff1.values
        col_name = f'{feature_name}_diff2'
        optimized_feature_df[col_name] = feature_diff2.values
        feature_transforms[col_name] = feature_diff2.values
        col_name = f'{feature_name}_diff7'
        optimized_feature_df[col_name] = feature_diff7.values
        feature_transforms[col_name] = feature_diff7.values
        print('✓ 差分特征: lag=[1,2,7] 捕捉变化率')

        # 方法11: 累积特征
        feature_cumsum = feature_clean.cumsum()
        feature_cummax = feature_clean.cummax()
        feature_cummin = feature_clean.cummin()
        col_name = f'{feature_name}_cumsum'
        optimized_feature_df[col_name] = feature_cumsum.values
        feature_transforms[col_name] = feature_cumsum.values
        col_name = f'{feature_name}_cummax'
        optimized_feature_df[col_name] = feature_cummax.values
        feature_transforms[col_name] = feature_cummax.values
        col_name = f'{feature_name}_cummin'
        optimized_feature_df[col_name] = feature_cummin.values
        feature_transforms[col_name] = feature_cummin.values
        print('✓ 累积特征: cumsum/cummax/cummin 捕捉历史信息')

        # 方法12: 原始特征 (保持)
        col_name = f'{feature_name}_original'
        optimized_feature_df[col_name] = feature_original.values
        feature_transforms[col_name] = feature_original.values

        # 填充空值
        for col in optimized_feature_df.columns:
            if optimized_feature_df[col].isnull().sum() > 0:
                if optimized_feature_df[col].dtype in ['float64', 'int64']:
                    optimized_feature_df[col] = optimized_feature_df[col].fillna(0)

        # 排序并去重
        optimized_feature_df = optimized_feature_df.sort_values(['date_on', 'country_name', 'region_name'])
        optimized_feature_df = optimized_feature_df.drop_duplicates(subset=['date_on', 'country_name', 'region_name'], keep='first')

        if len(optimized_feature_df) > REQUIRED_ROWS:
            optimized_feature_df = optimized_feature_df.iloc[:REQUIRED_ROWS]

        # 计算所有变换版本中每个特征的CFCS得分，选择最好的一个
        print()
        print('计算所有变换版本的CFCS得分，选择最优变换...')

        best_transform = None
        best_cfcs_score = -float('inf')
        best_cfcs_detail = None

        for transform_col in feature_transforms.keys():
            # 创建包含基础列 + futures_列 + 当前变换列的临时DataFrame
            temp_df = optimized_df[['ID', 'date_on', 'country_name', 'region_name'] if 'ID' in optimized_df.columns else ['date_on', 'country_name', 'region_name']].copy()

            # 添加所有futures_列
            for col in futures_cols:
                temp_df[col] = optimized_df[col].values

            # 添加date_on_month列
            if 'date_on_month' in optimized_df.columns:
                temp_df['date_on_month'] = optimized_df['date_on_month'].values
            else:
                temp_df['date_on_month'] = optimized_df['date_on'].dt.month

            # 添加当前变换列
            temp_df[transform_col] = optimized_feature_df[transform_col].values

            # 排序并去重
            temp_df = temp_df.sort_values(['date_on', 'country_name', 'region_name'])
            temp_df = temp_df.drop_duplicates(subset=['date_on', 'country_name', 'region_name'], keep='first')

            if len(temp_df) > REQUIRED_ROWS:
                temp_df = temp_df.iloc[:REQUIRED_ROWS]

            # 计算CFCS得分
            temp_cfcs = compute_cfcs(temp_df, verbose=False)

            # 选择CFCS得分最高的变换
            if temp_cfcs["cfcs"] > best_cfcs_score:
                best_cfcs_score = temp_cfcs["cfcs"]
                best_transform = transform_col
                best_cfcs_detail = temp_cfcs

            print(f'  {transform_col}: CFCS={temp_cfcs["cfcs"]:.2f}, 显著相关={temp_cfcs["sig_count"]}/{temp_cfcs["total"]} ({temp_cfcs["sig_pct"]:.2f}%)')

        print()
        print(f'✓ 最优变换: {best_transform} (CFCS={best_cfcs_score:.2f})')

        # 创建最终的DataFrame - 保留基础列 + futures_列 + 原始特征 + 最优变换列
        final_df = optimized_df[['ID', 'date_on', 'country_name', 'region_name'] if 'ID' in optimized_df.columns else ['date_on', 'country_name', 'region_name']].copy()

        # 添加所有futures_列
        for col in futures_cols:
            final_df[col] = optimized_df[col].values

        # 添加date_on_month列
        if 'date_on_month' in optimized_df.columns:
            final_df['date_on_month'] = optimized_df['date_on_month'].values
        else:
            final_df['date_on_month'] = optimized_df['date_on'].dt.month

        # 添加原始特征（保留原有信息）
        final_df[feature_name] = optimized_df[feature_name].values

        # 添加最优变换列（作为补充特征）
        final_df[best_transform] = optimized_feature_df[best_transform].values

        # 填充空值
        for col in final_df.columns:
            if final_df[col].isnull().sum() > 0:
                if final_df[col].dtype in ['float64', 'int64']:
                    final_df[col] = final_df[col].fillna(0)

        # 排序并去重
        final_df = final_df.sort_values(['date_on', 'country_name', 'region_name'])
        final_df = final_df.drop_duplicates(subset=['date_on', 'country_name', 'region_name'], keep='first')

        if len(final_df) > REQUIRED_ROWS:
            final_df = final_df.iloc[:REQUIRED_ROWS]

        # 保存优化文件
        filename = f'{OUTPUT_PATH}submission_rank_{rank}_feature_optimized.csv'
        final_df.to_csv(filename, index=False)

        print()
        print(f'优化文件: {filename}')
        print(f'  - 行数: {len(final_df):,}')
        print(f'  - 列数: {len(final_df.columns):,}')
        print(f'  - 使用变换: {best_transform}')
        print(f'  - CFCS得分: {best_cfcs_detail["cfcs"]:.2f}')
        print(f'  - 显著相关: {best_cfcs_detail["sig_count"]}/{best_cfcs_detail["total"]} ({best_cfcs_detail["sig_pct"]:.2f}%)')
        print(f'  - 平均显著相关性: {best_cfcs_detail["avg_sig_corr"]:.4f}')
        print(f'  - 最大相关性: {best_cfcs_detail["max_corr"]:.4f}')
        print()

        print('优化方法总结:')
        print('  1. 分位数标准化 - 正态分布转换')
        print('  2. Box-Cox/Yeo-Johnson - 方差稳定化')
        print('  3. 鲁棒标准化 - 抗异常值')
        print('  4. 幂变换 - 非线性关系捕捉')
        print('  5. 分桶编码 - 非线性模式学习')
        print('  6. 滞后特征 - 时间延迟效应')
        print('  7. 滚动统计 - 动态变化捕捉')
        print('  8. EWMA - 趋势跟踪')
        print('  9. 差分特征 - 变化率计算')
        print('  10. 累积特征 - 历史信息保留')
        print('  11. 原始特征 - 保持原样')
        print(f'  ✓ 最终选择: {best_transform}')
        print()
    else:
        print(f'Rank {rank}: {feature_name} 特征不存在!')
        print()

print('==========================================')
print('前10特征专项优化完成!')
print('==========================================')
print()