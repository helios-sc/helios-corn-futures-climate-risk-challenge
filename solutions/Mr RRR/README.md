# Mr RRR - Helios Corn Futures Climate Challenge Reproducibility

## Reproducibility Package
- `1_Data_Processing.ipynb`: Step 1 data processing and alignment
- `2_Feature_Engineering.ipynb`: Step 2 feature engineering
- `3_Feature_Selection.ipynb`: Step 3 feature selection (CFCS Top4)
- `4_Feature_AE.ipynb`: Step 4 AE training
- `5_submission.ipynb`: Step 5 upload required files, modify the code like path, then submit
- `requirements.txt`: Python dependencies
- `forecasting-the-future-the-helios-corn-climate-challenge/`: competition data placeholder (download separately)

## Environment
- Check your python version: 'python --version'
- Python: `3.10+` recommended (Personally I use Python 3.13.6 for this project, remember to install ipykernel recommended according to your python version if you want to run it on Jupyter notebook)
- Dependencies: see `requirements.txt`

## Reproduction Steps (run 0 → 4 in order)
### Step 0: Create a virtual environment and install dependencies
PowerShell example:
```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 1: Data processing
Run `1_Data_Processing.ipynb` to generate:
- `Data/Processing/processed.parquet`
- `Data/Processing/valid_ids.parquet`

### Step 2: Feature engineering
Run `2_Feature_Engineering.ipynb` (reads Step 1 output) to generate:
- `Data/Feature/train.parquet`
- `Data/Feature/test.parquet`
- `Data/Feature/climate_feature_columns.parquet`

### Step 3: Feature selection
Run `3_Feature_Selection.ipynb` (reads Step 2 output) to generate:
- `Data/Selection/feature_cfcs_*.parquet`
- `Data/Selection/selected_features_*.parquet`
- `Data/Selection/train_*.parquet` / `Data/Selection/test_*.parquet`

### Step 4: AE
Run `4_Feature_AE.ipynb` (reads Step 3 output and aligns with Step 1 `valid_ids`) to generate:
- `Data/Selection/ae_model_*.pth`
- `Data/Selection/ae_scaler_*.joblib`
- `Data/Selection/ae_input_features_*.parquet`
- `Data/Selection/selected_features_ae_*.parquet`
- `submission.csv` (final upload file)

### Step 5: Submission
Upload the files required in `5_submission.ipynb` then import the notebook and submit
(It includes one dataset for previous reproduction, you may add new one)

> For local runs, ensure `DATA_DIR` points to the competition data folder; on Kaggle use `/kaggle/input/forecasting-the-future-the-helios-corn-climate-challenge`.

## Methodology Summary
### Data sources
- Official competition data only:
  - `corn_climate_risk_futures_daily_master.csv`
  - `corn_regional_market_share.csv`
- No external data or tools were used

### Key feature engineering (Step 2)
- Seasonal/time features: `day_of_year`, `quarter`, seasonal `sin/cos`, hemispheric shift
- Risk intensity metrics: `score / high_share / balance / entropy`, etc.
- Time-series stats: rolling mean/max/volatility, lags, EMA, momentum/acceleration
- Event features: threshold persistence, event AUC, spike detection
- Combinations/interactions: temperature/precip stress, diffs, ratios, interactions
- Country-level aggregation: weighted stats and concentration by `country_name × date_on`

### Anti-gaming statement
- All `climate_risk_*` features are derived only from original `climate_risk_*` fields, time fields, and market share.
- **No `futures_*` columns or derivatives were used to generate `climate_risk_*` features.**
- `futures_*` are used only for correlation evaluation (Step 3) and submission format requirements.

## Submission Reference
- Team name: `Mr RRR`
- Submission name: `55 AE - More Self Agriculture 0.5 - top4` (included in this repo, you may just upload this then submit to test)
- Kaggle username(s): `Mr RRR`
