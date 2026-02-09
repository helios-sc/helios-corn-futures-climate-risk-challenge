# Helios Corn Futures Climate Challenge - Winner Verification Package

## Project Overview

This repository contains complete, reproducible code for generating submissions to the Helios Corn Futures Climate Challenge on Kaggle.

## ⚠️ Important Compliance Notice

**Highest Compliant Submission**: 75.30 CFCS
- File: `submission_rank_3_feature_optimized.csv`
- Code: `75.30-submission_rank_3_feature-csv.py`
- Status: ✅ **Fully Compliant**

**86.85 CFCS Submission - Violation**:
- This submission contains features generated using futures data, which seriously violates the anti-gaming rules
- Violation features include: DTW features, futures-climate interaction features, high-correlation optimization
- Feature names directly contain `futures_close_ZC_2`
- This submission has been disqualified

**Competition Goal**: Discover novel methods to transform Helios's proprietary climate risk data into signals that show stronger correlations with corn futures prices.

## Competition Details

- **Competition Name**: Forecasting the Future - The Helios Corn Climate Challenge
- **Platform**: Kaggle
- **Dataset**: Helios Proprietary Climate Risk Model + Barchart Futures Market Data
- **Evaluation Metric**: Climate-Futures Correlation Score (CFCS)
  - CFCS = (0.5 × Avg_Sig_Corr_Score) + (0.3 × Max_Corr_Score) + (0.2 × Sig_Count_Score)

## Environment Requirements

### Python Version
- Python 3.8 or later

### Dependencies
See `requirements.txt` for complete list.

## Project Structure

```
.
├── README.md                           # This file
├── METHODOLOGY.md                      # Detailed methodology and feature engineering
├── ANTI-GAMING_COMPLIANCE.md          # Anti-gaming compliance verification
├── SUBMISSION_REFERENCE.md            # Submission details and timestamps
├── COMPLIANCE_SUMMARY.md            # Compliance summary
├── requirements.txt                   # Python dependencies
├── 75.30-submission_rank_3_feature-csv.py    # Compliant submission: Rank 3 features (CFCS: 75.30)
└── 86.85-submission_top_5_features-csv.py    # Violation: Top 5 features optimized (CFCS: 86.85) - Reference only
```

## How to Reproduce Submissions

### Step 1: Set Up Environment
```bash
# Install required packages
pip install -r requirements.txt
```

### Step 2: Prepare Data
Place the competition data files in a directory structure:
```
/data/
├── corn_climate_risk_futures_daily_master.csv
└── corn_regional_market_share.csv
```

Or use Kaggle dataset path:
```python
DATA_PATH = '/kaggle/input/forecasting-the-future-the-helios-corn-climate-challenge/'
```

### Step 3: Run Submission Scripts

#### For 75.30 CFCS Score (Rank 3 Features) - ✅ Compliant Submission
```bash
python 75.30-submission_rank_3_feature-csv.py
```

This script will:
- Generate baseline features (risk scores, composites, rolling statistics)
- Create feature transformations
- Apply statistical feature selection
- Select top 3 features based on CFCS score optimization
- **Output: `submission_rank_3_feature_optimized.csv`**

#### For 86.85 CFCS Score (Top 5 Features) - ⚠️ Violation (Reference Only)
```bash
python 86.85-submission_top_5_features-csv.py
```

**⚠️ WARNING**: This script generates violation features and should only be used for research purposes, not for official submission.

Violation features include:
- DTW features (using futures price series)
- Futures-climate interaction features (direct multiplication)
- High-correlation feature engineering (using futures as target)

**Output: `submission_top_5_features_optimized.csv`** - Violation submission, disqualified

### Step 4: Verify Output

Check that output files have:
- **Required rows**: 219,161 records
- **Required columns**: `date_on`, `country_name`, `region_name`
- **Climate features**: All must start with `climate_risk_` prefix
- **Futures columns**: Original futures_* columns (provided by evaluation system)

## Key Features of the Approach

### 1. Baseline Feature Engineering
- **Risk Score Calculation**: Normalized weighted scores for 4 risk types (heat, cold, drought, excess precipitation)
- **Composite Indices**: Temperature stress, precipitation stress, overall stress, combined stress
- **Temporal Features**: Rolling means/max (7, 14, 30 day windows), momentum, acceleration
- **Country Aggregations**: Production-weighted national summaries

### 2. Advanced Transformations
For each baseline feature, 12 transformation methods are applied:
1. Quantile normalization
2. Box-Cox/Yeo-Johnson transformations
3. Robust scaling
4. Power transformations
5. Binning/encoding
6. Lag features
7. Rolling statistics
8. Exponential Weighted Moving Average (EWMA)
9. Difference features
10. Cumulative features
11. Seasonal decomposition
12. Original feature preservation

### 3. Feature Selection
- Statistical significance filtering (≥ 5% significant correlations)
- Variance-based filtering
- Correlation-based ranking
- CFCS score maximization

### 4. Transformation Optimization
For each selected feature, all transformation versions are evaluated:
- CFCS score computed for each transform
- Best-performing transform automatically selected
- Ensures optimal feature-futures correlation

## Anti-Gaming Compliance

### ✅ 75.30 CFCS Submission - Fully Compliant

**IMPORTANT**: The 75.30 CFCS submission strictly complies with competition anti-gaming rules:

✅ **No data leakage**: All `climate_risk_*` features are derived SOLELY from climate risk data columns
✅ **No futures input**: No `futures_*` columns were used to generate `climate_risk_*` features
✅ **Only correlation for selection**: Futures columns used ONLY for feature selection and evaluation
✅ **Proven in code**: All violation features have been explicitly disabled

See `ANTI-GAMING_COMPLIANCE.md` for detailed verification.

### ⚠️ 86.85 CFCS Submission - Violation

This submission contains serious violation features:

❌ **Data leakage**: DTW features directly use futures price series
❌ **Futures input**: Feature names directly contain `futures_close_ZC_2`
❌ **Direct manipulation**: High-correlation feature engineering uses futures as target

See `COMPLIANCE_SUMMARY.md` for detailed explanation.

## Contact & Team Information

**Kaggle Username/Team Name**: DragonAJA
**Compliant Submission**: 75.30-submission_rank_3_feature-csv.py
- 75.30 CFCS Submission:https://www.kaggle.com/code/dragonaja/helios-corn-futures-enhanced-feature-engineering/notebook?scriptVersionId=294846228
- 86.85 CFCS Submission:https://www.kaggle.com/code/dragonaja/helios-corn-futures-enhanced-feature-engineering/notebook?scriptVersionId=294465402
## Additional Notes

- Code is fully reproducible with provided data
- No external data sources beyond competition files
- No custom-trained models or external tools
- All computation done using standard Python scientific computing stack

## Contact for Verification

**Note**: Only the 75.30 CFCS submission is compliant and eligible for prize consideration. The 86.85 CFCS submission has been disqualified due to violation of anti-gaming rules.

For any questions or clarification regarding these submissions, please contact:
- **Email**: [To be filled by participant]
- **Kaggle Username**: [To be filled by participant]

## License

This code is submitted for competition verification purposes only.
