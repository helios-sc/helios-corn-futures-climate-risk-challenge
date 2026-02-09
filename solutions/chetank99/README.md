# Helios Corn Futures Climate Challenge

## Overview

**Only the competition data used, which were the two files!**

## Anti-Gaming Compliance Statement

```
a quick tl;dr of proof of no future stuff used:

1. no futures_* columns were used to generate climate_risk_* features
   - All 11 features are derived ONLY from climate_risk_cnt_locations_* columns
   - futures_* columns are included in submission for evaluation but NOT used in feature engineering

2. All features use backward-looking rolling windows only
   - 120-day rolling mean uses min_periods=30, looking at past data only
   - No future data leakage

3. No external data sources
   - Only competition-provided CSV files are used

4. No circular dependencies
   - Features are calculated from raw climate risk counts
   - No derivatives of futures prices used
```
TL;DR of why and how the model's features rationale was arrived at can be better seen in [the jupyter notebook showcasing the _journey_](model_rationale.ipynb)
## Now Let's see the fun stuff!!!!!

## what pipeline did my detective work result in?

### Step 1: drought score calculation
**what i looked at:** `climate_risk_cnt_locations_drought_risk_low/medium/high`
**what i got:** `drought_score`

**formula:** `(medium + 2 * high) / (low + medium + high)`

### Step 2: country-level aggregation
**what i looked at:** `drought_score` grouped by country and date

**what i got:** `arg_mean`, `brazil_mean`, `global_mean`
(took these because of their geo influence which i saw from the data wrt others and reading up online showcased how beautiful (yet complicated) their influence is on the logistics)


### Step 3: temporal smoothing (backward-looking)
**what i looked at:** Daily country means

**what i got:** `arg_post`, `brazil_post`, `global_post`

**method:** 120-day rolling mean with min_periods=30

### Step 4: ratio calculation
**what i looked at:** `arg_post`, `global_post`

**what i got:** `ratio = arg_post / global_post`

**interpretation:** Argentina's relative drought stress vs global average

### Step 5: month-specific feature creation
**what i looked at:** `ratio`, `ratio_sq`, `ratio_cube`, `sync`

**what i got:** 11 climate_risk_* features

**method:** Mask by Argentina harvest months (Jan, Feb, Nov, Dec)

## final features used for the model!!!! (11 )

| feature | description | months active |
|---------|-------------|---------------|
| `climate_risk_ratio_M01` | Argentina/Global ratio | January |
| `climate_risk_ratio_sq_M01` | Ratio squared | January |
| `climate_risk_ratio_M02` | Argentina/Global ratio | February |
| `climate_risk_ratio_sq_M02` | Ratio squared | February |
| `climate_risk_ratio_M11` | Argentina/Global ratio | November |
| `climate_risk_ratio_sq_M11` | Ratio squared | November |
| `climate_risk_ratio_M12` | Argentina/Global ratio | December |
| `climate_risk_ratio_sq_M12` | Ratio squared | December |
| `climate_risk_ratio_cube_M01` | Ratio cubed | January |
| `climate_risk_sync_M01` | Argentina x Brazil sync | January |
| `climate_risk_sync_M02` | Argentina x Brazil sync | February |

## why through trial and error I understood that these work!!!

1. **relative vs absolute:** The ratio (Argentina/Global) captures relative scarcity, essentially, when Argentina is disproportionately affected, it impacts global corn prices more than uniform global drought.

2. **harvest season focus:** Argentina's corn harvest is Nov-Feb. Climate stress during these months has the strongest price impact.

3. **non-linear effects:** Squared and cubed terms capture diminishing/accelerating returns of drought severity.

4. **regional synchronization:** When both Argentina & Brazil (major exporters) experience drought simultaneously, it signals severe supply disruption.

## if you want to reproduce these results 

### prereq
```bash
pip install pandas numpy
```

### Steps
1. clone repo
2. place competition data files in `data/` folder:
   - `corn_climate_risk_futures_daily_master.csv`
   - `corn_regional_market_share.csv`
3. run:
   ```bash
   python generate_submission.py
   ```
4. output will be saved to `output/submission.csv`

### Expected Output
```
features: 11
rows: 219,161
file saved: output\submission.csv
```

## env used

- Python 3.8+
- pandas >= 1.5.0
- numpy >= 1.21.0
