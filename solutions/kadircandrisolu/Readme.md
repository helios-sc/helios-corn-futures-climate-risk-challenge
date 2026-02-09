# Helios Corn Futures Climate Challenge - Solution

## Team Information
- Kaggle Username: kadircandrisolu
- Submission Private Score: 79.26000

## Solution Overview
This solution uses production-weighted climate risk aggregations with long-term rolling windows (400-430 days) to capture climate-futures correlations.

### Key Features
- Production-weighted risk scoring by region
- Country-level aggregations (mean, max, std, weighted sum)
- Long-term rolling windows (400-430 days)
- Cumulative drought stress tracking
- Final selected feature: `climate_risk_drought_weighted_country_cumsum`

## Data Sources
- Competition-provided climate risk data
- Competition-provided futures data
- Competition-provided regional market share data
- **No external data used**

## Reproducibility Instructions

Python version: 3.11.13

### Environment Setup
```bash
pip install -r requirements.txt
```

### Running the Solution
1. Place data files in appropriate directory:
   - `corn_climate_risk_futures_daily_master.csv`
   - `corn_regional_market_share.csv`

2. Run the notebook:
```bash
jupyter notebook solution.ipynb
```

3. Output will be saved as `submission.csv`

## Methodology

### Feature Engineering Pipeline
1. **Production Weighting**: Merged regional production shares
2. **Risk Scores**: Calculated weighted risk scores for 4 climate categories
3. **Temporal Aggregations**: 400-430 day rolling windows
4. **Country Aggregates**: Mean, max, std, weighted sum by country
5. **Cumulative Tracking**: Cumulative drought stress over time

### Feature Selection
- Used beam search on 2025 validation set
- Optimized for CFCS score
- Selected: `climate_risk_drought_weighted_country_cumsum`

## Anti-Gaming Compliance
✅ All `climate_risk_*` features created ONLY from climate data and production shares
✅ `futures_*` columns used ONLY for correlation evaluation
✅ No futures data used in feature engineering
✅ No data leakage from futures to climate features

## Python Environment
- Python 3.x
- See requirements.txt for dependencies