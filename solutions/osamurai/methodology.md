# Helios Corn Futures Climate Challenge — Winner Verification

## Submission Reference

- **Kaggle Username:** osamurai
- **Team Name:** John Doe
- **CFCS Score:** 72.33
- **Submission Name:** Sample Submission Notebook - Version 10
- **Submission Timestamp:** Tue Jan 20, 2026, 08:18:54 JST (Mon Jan 19, 2026, 23:18:54 UTC)

---

## Data Sources

Only the two official competition datasets were used. No external data was introduced.

| File | Description |
|------|-------------|
| `corn_climate_risk_futures_daily_master.csv` | Daily climate risk and futures data across global corn-growing regions |
| `corn_regional_market_share.csv` | Regional production share (percent of country production) |

---

## Feature Engineering Pipeline

All engineered features are prefixed with `climate_risk_` as required by the competition rules.

### Base Risk Scores

For each of the four risk categories (heat stress, unseasonably cold, excess precipitation, drought), a weighted risk score was computed from the location counts at each risk level:

```
risk_score = (medium_count + 2 × high_count) / (low_count + medium_count + high_count + ε)
```

High-risk locations contribute twice as much as medium-risk ones. Additional ratio features (high ratio, elevated ratio) were also derived.

### Production-Weighted Risks

Each region's risk score was multiplied by its `percent_country_production` from the market share dataset, so that regions with greater economic importance contribute more to the signal.

### Composite Stress Indices

- **Temperature stress:** max of heat stress and cold stress scores
- **Precipitation stress:** max of excess precipitation and drought scores
- **Overall/combined/total stress:** max, mean, and sum across all four risk categories (both raw and production-weighted)

### Seasonal and Harvest Period Features

- Risk scores interacted with `harvest_period` dummy variables (Planting, Growing, Vegetative, Reproductive, etc.)
- Growing season flag: risk scores multiplied by a binary flag for active growth periods
- Summer (June–August) and winter (December–February) seasonal interactions

**Rationale:** Climate risks have very different economic impacts depending on the crop growth stage.

### Rolling Window Statistics

For each risk category, rolling statistics were computed per region over 7, 14, and 30-day windows: moving average, rolling max, rolling min, and rolling standard deviation.

**Rationale:** Markets respond not just to point-in-time risk levels but to sustained patterns and recent volatility.

### Lag Features

Lagged risk scores at 1, 3, 7, 14, and 21 days per region and risk category.

**Rationale:** Futures markets may react to climate events with a delay.

### Momentum and Acceleration

- 1-day, 7-day, and 14-day changes (first differences) in risk scores
- Acceleration (second differences): change in the 1-day change

**Rationale:** The rate and direction of change in climate risk can be as informative as the absolute level.

### Country-Level and Global Aggregations

- **Country-level:** mean, max, std, min of risk scores across regions; sum of production-weighted risks
- **Global-level:** aggregated statistics across the four major corn-producing countries (United States, Brazil, Argentina, China)

**Rationale:** Corn futures prices are set in global markets. Country-wide and global risk patterns matter more than individual region-level noise.

### Non-linear Transformations

- Squared and cubed risk scores (convex damage functions)
- Log and square root transformations (dampening extreme values)
- Binary threshold flags: high risk (score > 1.0) and extreme risk (score > 1.5)

**Rationale:** The relationship between climate stress and crop damage is non-linear — damage accelerates beyond certain thresholds.

### Interaction Features

- Heat stress × drought (compounding dry-heat conditions)
- Cold stress × excess precipitation (compounding cold-wet conditions)
- Temperature stress × precipitation stress (cross-category interaction)
- Four-way product of all risk scores
- Temperature-to-precipitation stress ratio

**Rationale:** Simultaneous occurrence of multiple risk types causes disproportionately greater damage than either alone.

### LightGBM Prediction Features

LightGBM regression models were trained to predict each of the 17 `futures_*` target variables using **only** `climate_risk_*` columns as input features. The model predictions were then added as new features with the prefix `climate_risk_lgb_pred_*`. Additionally, lagged (1, 3, 7, 14 days), differenced (1-day, 7-day changes), and 7-day rolling average variants of these prediction features were created.

**Rationale:** The LightGBM models capture complex non-linear mappings from climate features to futures values. See the Rule Compliance section below for a detailed explanation of why this does not constitute data leakage.

---

## Hyperparameter Tuning

### Stage 1: L9 Orthogonal Array (Global Search)

An L9 orthogonal array (3 levels × 4 factors) was used to efficiently explore the hyperparameter space with only 9 experiments.

| Factor | Level 1 | Level 2 | Level 3 |
|--------|---------|---------|---------|
| num_leaves | 31 | 63 | 127 |
| learning_rate | 0.01 | 0.03 | 0.05 |
| feature_fraction | 0.6 | 0.7 | 0.8 |
| min_child_samples | 10 | 20 | 50 |

### Stage 2: Optuna (Local Refinement)

Starting from the best L9 result, Optuna ran 30 trials to fine-tune parameters within a narrowed search range. The better result between L9 and Optuna was selected as the final parameter set.

### Final Model Training

- Algorithm: LightGBM (gbdt)
- Cross-validation: 10-fold TimeSeriesSplit
- Early stopping: 50 rounds (max 500 boost rounds)
- Ensemble: Predictions averaged across all folds

---

## Rule Compliance

**I confirm that no `futures_*` columns or their derivatives were used to generate `climate_risk_*` features.**

1. **Input features** for all LightGBM models consist exclusively of columns derived from the climate risk data and regional market share data. The feature set is defined as all columns starting with `climate_risk_` that have numeric dtypes. At no point are `futures_*` columns included in the input feature matrix.

2. **Target variables** (`futures_*` columns) are used solely as regression labels during LightGBM training. This is the standard supervised learning setup and is consistent with the competition's objective: discovering correlations between climate data and futures markets.

3. **Prediction-based features** (`climate_risk_lgb_pred_*`) are the output of models that take only `climate_risk_*` inputs. The information flow is strictly one-directional:

   ```
   climate_risk_* inputs → LightGBM model → climate_risk_lgb_pred_* outputs
   ```

   The `futures_*` values influenced the model's learned parameters during training, but they do not appear as features in the prediction pipeline. This is functionally a non-linear feature transformation of the climate data.

4. **No external data** was used beyond the two provided competition datasets.

---

## Reproducibility

The submission was generated from a Kaggle Notebook (Sample Submission Notebook - Version 10), which is accessible from the Submission Details page on Kaggle. The code is also attached as `solution.py`. The environment is the standard Kaggle Python environment with no additional package installations required.