# Helios Corn Futures Climate Challenge

**Turn weather wisdom into trading gold! Use Helios AI's climate data to decode the weather signals behind corn futures & outsmart the markets**

## Competition Overview

Welcome to the Helios Corn Futures Climate Challenge! This competition challenges data scientists and quantitative analysts to discover novel ways to leverage climate risk data for predicting corn futures price movements. Your mission: find innovative methods to strengthen the correlation between weather patterns and commodity markets.

## The Challenge

Agricultural markets are fundamentally driven by weather, but the relationship between climate conditions and futures prices is complex and often non-linear. Traditional approaches may miss subtle patterns or fail to capture the full economic impact of weather events across different regions and seasons.

**Your goal**: Develop creative approaches to transform Helios's proprietary climate risk data into signals that show stronger correlations with corn futures prices than baseline methods.

## What Makes This Unique

- **Proprietary Climate Intelligence**: Access to Helios's advanced climate risk model that goes beyond simple weather data
- **Real Economic Impact**: Climate risks are pre-classified based on actual crop tolerance thresholds
- **Global Scale**: Data spans major corn-producing regions worldwide
- **Production Weighting**: Regional market share data enables economic impact modeling
- **Multi-dimensional Risk**: Four distinct climate risk categories (heat, cold, drought, excess precipitation)

## Dataset Highlights

### Climate Risk Data
- **Daily assessments** across global corn-growing regions
- **Risk classifications**: Low, Medium, High based on crop-specific thresholds
- **Four risk categories**: Heat stress, cold stress, drought, excess precipitation
- **Regional aggregation**: Location counts by risk level for each region/day
- **Economic context**: Production share data for weighting regional importance

### Futures Market Data
- **Comprehensive pricing**: Corn (ZC), wheat (ZW), soybean (ZS) futures
- **Technical indicators**: Returns, volatility, moving averages
- **Market structure**: Term spreads and cross-commodity relationships
- **Daily frequency**: Aligned with climate risk assessments

## Evaluation Methodology

Your submissions will be ranked on the **Climate-Futures Correlation Score (CFCS)** - a single composite metric that measures how well your engineered features correlate with futures market variables.

### 1. Correlation Computation
- **Granularity**: Monthly correlations by commodity, country, and month
- **Scope**: All climate risk variables vs. all futures variables
- **Filtering**: Only correlations with sufficient variance are computed
- **Precision**: Results rounded to 5 decimal places

### 2. Composite Score Calculation

Your leaderboard position will be determined by a single **Climate-Futures Correlation Score (CFCS)** that combines multiple correlation metrics:

**CFCS = (0.5 × Avg_Sig_Corr_Score) + (0.3 × Max_Corr_Score) + (0.2 × Sig_Count_Score)**

Where:
- **Avg_Sig_Corr_Score**: Average of ONLY significant correlations (|corr| ≥ 0.5), normalized to 0-100 scale
- **Max_Corr_Score**: Normalized maximum absolute correlation (0-100 scale)  
- **Sig_Count_Score**: Percentage of correlations that are significant (≥ |0.5|)

### 3. Score Components Explained

**Average Significant Correlation Score (50% weight)**
- Measures quality of meaningful relationships (only correlations ≥ |0.5|)
- Rewards consistently strong signals, ignoring weak noise
- Formula: `min(100, |mean_significant_correlation| × 100)`
- Caps at 100 when average significant correlation reaches 1.0
- Returns 0 if no significant correlations found

**Maximum Correlation Score (30% weight)**
- Rewards discovery of exceptionally strong climate-market relationships
- Incentivizes breakthrough insights and novel feature engineering
- Formula: `min(100, |max_correlation| × 100)`
- Caps at 100 when maximum absolute correlation reaches 1.0

**Significant Count Score (20% weight)**
- Measures breadth of meaningful relationships discovered
- Rewards approaches that find multiple strong signals
- Formula: `(count_significant_correlations / total_correlations) × 100`
- Percentage of correlations with absolute value ≥ 0.5

### 4. Example Scoring
**Baseline Performance:**
- Average significant |correlation|: 0.55 → Avg_Sig_Score = 55
- Maximum |correlation|: 0.60 → Max_Score = 60  
- Significant correlations: 8% → Count_Score = 8
- **CFCS = (0.5×55) + (0.3×60) + (0.2×8) = 46.1**

**Target Performance:**
- Average significant |correlation|: 0.65 → Avg_Sig_Score = 65
- Maximum |correlation|: 0.80 → Max_Score = 80
- Significant correlations: 20% → Count_Score = 20
- **CFCS = (0.5×65) + (0.3×80) + (0.2×20) = 60.5**

## Potential Approaches

While we encourage creativity, here are some directions to consider:

### Feature Engineering
- **Production-weighted risk scores**: Combine regional risks with market share data
- **Temporal aggregations**: Weekly, monthly, or seasonal risk summaries
- **Cross-regional patterns**: Correlations between geographically distant regions
- **Risk momentum**: Changes in risk levels over time
- **Composite indices**: Multi-factor climate stress indicators

### Advanced Techniques
- **Non-linear transformations**: Capture threshold effects in climate-market relationships
- **Lag analysis**: Account for delayed market responses to weather events
- **Seasonal adjustments**: Normalize for typical seasonal patterns
- **Volatility modeling**: Link climate uncertainty to market volatility
- **Regime detection**: Identify different climate-market relationship periods

### Domain Knowledge Integration
- **Growing season alignment**: Match climate risks to crop development stages
- **Supply chain modeling**: Consider storage, transportation, and processing impacts
- **Market psychology**: Model how weather news affects trader behavior
- **Cross-commodity effects**: Leverage substitution relationships between crops

## Submission Requirements

Your submission should include:

1. **Engineered Dataset**: Enhanced version of the provided data with your novel features
2. **Feature Documentation**: Clear explanation of your feature engineering approach
3. **Code**: Reproducible pipeline for generating your features
4. **Analysis**: Demonstration of improved correlations vs. baseline methods

### **Critical Naming Requirements**

**⚠️ IMPORTANT**: Your submission must follow these naming conventions:

- **Climate Features**: All engineered climate features must start with `climate_risk_`
  - ✅ Good: `climate_risk_heat_stress_weighted`, `climate_risk_drought_ma_30d`
  - ❌ Bad: `heat_stress_risk`, `my_climate_feature`, `weather_index`

- **Required Columns**: Your submission must include:
  - `date_on`: Date column (YYYY-MM-DD format)
  - `country_name`: Country name matching the original data
  - `region_name`: Region name (optional but recommended)

- **Futures Data**: The evaluation system will provide `futures_*` columns - do not modify these names

**The evaluation metric automatically detects features by their prefixes. Incorrect naming will result in zero score.**

## Evaluation Timeline

- **Development Phase**: Use historical data to develop and validate approaches
- **Final Evaluation**: Submissions tested on held-out time periods
- **Correlation Analysis**: Comprehensive correlation computation across all dimensions

## Why This Matters

Success in this challenge has real-world applications:
- **Risk Management**: Better weather-based hedging strategies
- **Trading Alpha**: Novel signals for commodity trading
- **Agricultural Finance**: Improved crop insurance and lending models
- **Supply Chain**: Enhanced forecasting for food companies
- **Climate Adaptation**: Better understanding of weather-market linkages

## Getting Started

1. **Explore the Data**: Understand the climate risk classifications and regional patterns
2. **Baseline Analysis**: Compute initial correlations using raw climate data
3. **Feature Engineering**: Develop novel transformations and aggregations
4. **Validation**: Test your approaches across different time periods and regions
5. **Optimization**: Iterate to maximize correlation strength and significance

## Resources

- `dataset_description.md`: Comprehensive data documentation
- `submission_sample.ipynb`: Example evaluation methodology
- `corn_climate_risk_futures_daily_master.csv`: Main dataset
- `corn_regional_market_share.csv`: Production weighting data

---

**Ready to turn weather data into market insights? Let's see how creative you can get with climate intelligence!**

*Good luck, and may the correlations be with you!* 🌽📈