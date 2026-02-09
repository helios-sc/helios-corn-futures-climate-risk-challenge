# Helios Corn Futures Climate Challenge

---

## Submission Scores

| Score Type | Value |
|------------|-------|
| **Public CFCS Score** | 85.67 |
| **Private CFCS Score** | 78.17 |
| **Notebook CFCS Score (0-1 scale)** | 0.73272 |
| **Notebook CFCS Score (0-100 scale)** | 73.27 |

**Submission File:** https://drive.google.com/drive/folders/1xLbg46CfF21zdGEt3b5EJnxXGCPQUIeA?usp=sharing

---

## Table of Contents

1. [Reproducibility Package](#1-reproducibility-package)
   - 1.1 [Complete Code](#11-complete-code)
   - 1.2 [Feature Engineering Pipeline](#12-feature-engineering-pipeline)
   - 1.3 [Reproduction Instructions](#13-reproduction-instructions)
   - 1.4 [Environment Details](#14-environment-details)
2. [Methodology Summary](#2-methodology-summary)
   - 2.1 [Data Sources Used](#21-data-sources-used)
   - 2.2 [Key Feature Engineering Steps and Rationale](#22-key-feature-engineering-steps-and-rationale)
   - 2.3 [Anti-Gaming Compliance Confirmation](#23-anti-gaming-compliance-confirmation)
3. [Submission Reference](#3-submission-reference)

---

## 1. Reproducibility Package

### 1.1 Complete Code

The complete solution is contained in a single Jupyter notebook:

| Property | Value |
|----------|-------|
| **Filename** | `helios_solution.ipynb` |
| **Language** | Python 3.12.12 |
| **Platform** | Kaggle Notebooks |
| **Environment** | Pin to original environment (2025-12-17) |
| **Accelerator** | None (CPU only) |

#### Code Structure Overview

```
helios_solution.ipynb
├── Configuration & Setup
│   ├── imports
│   ├── Configuration parameters
│   └── Data paths
├── Feature Engineering Helper Functions
├── Data Loading & Preprocessing
├── Feature Engineering (1,494 features)
│   ├── Initialization & time bin config
│   ├── Time bin creation
│   ├── Groupby aggregation features
│   ├── Categorical, risk scores, interactions
│   ├── Quantile & ratio features
│   ├── Compound stress features
│   ├── Expanded aggregations
│   └── Final aggregations
├── Feature Summary & Correlation Filtering
├── Feature Analysis & Ranking
├── Scoring with Top Features
└── Submission File Creation
```

---

### 1.2 Feature Engineering Pipeline

#### 1.2.1 Pipeline Overview

The pipeline transforms raw climate risk counts into 1,494 engineered features, then selects the top 5 based on correlation analysis with futures data.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        FEATURE ENGINEERING PIPELINE                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  [Raw Data] ──► [Temporal Features] ──► [Time Binning] ──► [Aggregations]   │
│                                                                              │
│       │              │                      │                   │            │
│       ▼              ▼                      ▼                   ▼            │
│  ┌─────────┐   ┌──────────┐          ┌───────────┐      ┌────────────┐      │
│  │ Climate │   │ year     │          │ tertile   │      │ mean, sum  │      │
│  │ Risk    │   │ month    │          │ quartile  │      │ std, var   │      │
│  │ Counts  │   │ quarter  │          │ quintile  │      │ max, min   │      │
│  │ (12)    │   │ week     │          │ sextile   │      │ of Climate │      │
│  └─────────┘   │ day_of_  │          │ octile    │      │ Risk Counts│      │
│                │ year     │          │ decile    │      │ grouped by │      │
│                └──────────┘          │ tredecile │      │ Time Bins  │      │
│                                      │ vigintile │      └────────────┘      │
│                                      └───────────┘                          │
│                                                                              │
│  [Compound Features] ──► [Weighted Features] ──► [Cross-Risk Features]      │
│                                                                              │
│       │                       │                        │                     │
│       ▼                       ▼                        ▼                     │
│  ┌───────────┐         ┌───────────┐           ┌────────────┐               │
│  │ drought × │         │ risk ×    │           │ heat +     │               │
│  │ heat      │         │ production│           │ drought    │               │
│  │ drought × │         │ weight    │           │ cold +     │               │
│  │ excess    │         │           │           │ excess     │               │
│  └───────────┘         └───────────┘           └────────────┘               │
│                                                                              │
│  [1,494 Features] ──► [Correlation Filter] ──► [Top 5 Selection]            │
│                              │                        │                      │
│                              ▼                        ▼                      │
│                       Remove ≥98%              Select by                     │
│                       correlated               sig_count                     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Pipeline Flow:**

1. **Load Raw Data** → Load climate risk location counts and market share data
2. **Create Temporal Features** → Extract year, month, quarter, week from dates
3. **Create Time Bins** → Group data into tertile, quartile, quintile, etc.
4. **Create Aggregation Features** → Compute mean, sum, std, var, max, min
5. **Create Compound Features** → Combine risk types (drought × heat, etc.)
6. **Create Weighted Features** → Multiply risk counts by production weight
7. **Create Cross-Risk Features** → Sum and interact different risk categories
8. **Apply Correlation Filter** → Remove features with ≥98% correlation
9. **Select Top 5 Features** → Rank by significant correlation count

**Result:** 1,494 features created → Top 5 selected for final submission

---

**How Compound Features are Created:**

Compound features combine two or more risk categories to capture simultaneous stress conditions:

| Operation | Formula | Example |
|-----------|---------|---------|
| **Product** | risk_A × risk_B | `drought_risk_medium × excess_precip_risk_medium` |
| **Sum** | risk_A + risk_B | `drought_risk_high + heat_stress_risk_high` |
| **Geometric Mean** | √(risk_A × risk_B) | `sqrt(drought_risk × heat_stress_risk)` |
| **Max** | max(risk_A, risk_B) | `max(drought_risk_high, excess_precip_risk_high)` |
| **Min** | min(risk_A, risk_B) | `min(drought_risk_medium, heat_stress_risk_medium)` |

These compound features are then aggregated (mean, sum, std, etc.) across time bins and country groupings to create the final engineered features.

#### 1.2.2 Feature Engineering Stages

| Stage | Description |
|-------|-------------|
| **Data Loading** | Load CSVs, create temporal columns, merge market share |
| **Time Binning** | Create 8 time bin granularities |
| **Date Decile Groupby** | High-impact risk aggregations by decile |
| **Low-Risk Features** | Cold stress and low-severity features |
| **Tredecile Features** | Medium/high severity tredecile aggs |
| **Categorical Encoding** | Harvest period encoding |
| **Risk Scores** | Weighted risk scores per category |
| **Cross-Risk Interactions** | Risk pair products, sums, maxes |
| **Country Aggregations** | Country-level mean, max, sum, std |
| **Quantile Features** | Tertile, quartile, quintile bins |
| **Country Time-Bin Std** | Std deviation features |
| **Region-Weighted** | Production-weighted risks |
| **Seasonal Aggregations** | Monthly/quarterly patterns |
| **Risk Ratios** | Relative intensity features |
| **Drought-Heat Compound** | Drought + heat interactions |
| **Weighted Compounds** | Various weighted compounds |
| **Drought-Excess & All-Risk** | Expanded compounds |
| **Geomean Focus** | Geometric mean features |
| **Expanded Aggregations** | Tertile, octile, decile aggs |
| **Final Aggregations** | Quartile-country, sextile-country |
| **Correlation Filter** | Remove ≥98% correlated features |

**Total Features Created:** 1,494

#### 1.2.3 Helper Functions

The following reusable functions are defined:

```python
def quarter_to_date(year, quarter)
    # Convert year-quarter to timestamp

def create_time_bins(df, date_col, quarters_per_bin, bin_name, ...)
    # Create temporal bins for grouping

def create_groupby_agg_features(df, source_cols, groupby_cols, agg_funcs, ...)
    # Create aggregation features (mean, sum, std, etc.)

def create_spatial_std_features(df, source_cols, date_col, ...)
    # Create spatial standard deviation features

def create_groupby_std_features(df, source_cols, groupby_cols, ...)
    # Create std features within groups

def create_categorical_encoding(df, source_col, feature_name, ...)
    # Encode categorical columns

def create_risk_score_features(df, risk_categories, weights=(1,2,3), ...)
    # Create weighted risk scores

def create_cross_risk_features(df, risk_pairs, ...)
    # Create cross-risk interaction features

def create_country_agg_features(df, source_cols, ...)
    # Create country-level aggregations
```

---

### 1.3 Reproduction Instructions

#### Option A: Kaggle Notebooks (Recommended)

1. **Create New Notebook**
   - Go to https://www.kaggle.com/code
   - Click "New Notebook"

2. **Upload Solution**
   - Click "File" → "Import Notebook"
   - Upload `helios_solution.ipynb`

3. **Add Competition Data**
   - Click "Add Data" in the right panel
   - Search: "forecasting-the-future-the-helios-corn-climate-challenge"
   - Click "Add" to attach the dataset

4. **Configure Notebook Settings** (Right Panel → Settings)
   | Setting | Value |
   |---------|-------|
   | **Environment** | Pin to original environment (2025-12-17) |
   | **Accelerator** | None |
   | **Internet** | On (optional, not used) |

5. **Verify Data Path**
   - Ensure the data paths are set to:
   ```python
   DATA_PATH = '/kaggle/input/forecasting-the-future-the-helios-corn-climate-challenge/'
   OUTPUT_PATH = '/kaggle/working/'
   ```

6. **Execute All Cells**
   - Click "Run All"
   - Estimated runtime: 7 to 10 minutes (CPU only)

7. **Retrieve Output**
   - Output file: `/kaggle/working/submission.csv`
   - Download from the "Output" tab

#### Option B: Local Execution (Windows)

> **Note:** The code was originally developed and run on Kaggle Notebooks during the competition. Local execution was only tested after the competition ended to verify reproducibility.

**Verified Local Environment (Windows):**

The notebook has been successfully tested on a local Windows machine with the following configuration, producing the **same CFCS score (0.73272 on 0-1 scale, or 73.27 on 0-100 scale)** as the Kaggle environment:

| Setting | Value |
|---------|-------|
| **Operating System** | Windows |
| **Python Version** | 3.12.2 |
| **pandas** | 2.2.2 |
| **numpy** | 2.0.2 |
| **scipy** | 1.15.3 |

**Steps to Reproduce Locally on Windows:**

1. **Ensure Python 3.12.x is installed**

   Open Command Prompt or PowerShell and run:
   ```bash
   python --version
   ```
   Should show Python 3.12.x

2. **Download Competition Data**

   Download from Kaggle competition page:
   - `corn_climate_risk_futures_daily_master.csv`
   - `corn_regional_market_share.csv`

3. **Set Up Directory Structure**

   Create the following folder structure:
   ```
   your_project_folder/
   ├── helios_solution.ipynb
   ├── requirements.txt
   ├── data/
   │   ├── corn_climate_risk_futures_daily_master.csv
   │   └── corn_regional_market_share.csv
   └── output/
       └── (submission.csv will be generated here)
   ```

4. **Create Virtual Environment**

   Open Command Prompt or PowerShell, navigate to your project folder, and run:
   ```bash
   # Navigate to your project folder
   cd path/to/your_project_folder

   # Create virtual environment
   python -m venv venv

   # Activate virtual environment (Windows)
   venv\Scripts\activate

   # Activate virtual environment (macOS/Linux)
   source venv/bin/activate
   ```

   After activation, you should see `(venv)` at the beginning of your command prompt or terminal.

5. **Install Dependencies**

   With the virtual environment activated, install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

6. **Modify Data Paths in Notebook - IMPORTANT**

   The notebook is configured for Kaggle paths by default. For local execution, you **must** change the data paths from:
   ```python
   # Original Kaggle paths (will not work locally)
   DATA_PATH = '/kaggle/input/forecasting-the-future-the-helios-corn-climate-challenge/'
   OUTPUT_PATH = '/kaggle/working/'
   ```

   To your local paths (relative to the working directory where the notebook is located):
   ```python
   # Local paths - relative to the notebook's working directory
   # If notebook is in: C:/Projects/helios/helios_solution.ipynb
   # Then 'data/' refers to: C:/Projects/helios/data/
   # And 'output/' refers to: C:/Projects/helios/output/
   DATA_PATH = 'data/'
   OUTPUT_PATH = 'output/'
   ```

   Or use absolute paths if preferred:
   ```python
   # Absolute paths example (Windows)
   DATA_PATH = 'C:/Users/YourName/Projects/helios/data/'
   OUTPUT_PATH = 'C:/Users/YourName/Projects/helios/output/'
   ```

7. **Launch Jupyter Notebook**

   With the virtual environment still activated, run:
   ```bash
   jupyter notebook
   ```

   This will:
   - Start the Jupyter server
   - Open your default web browser automatically
   - Display a URL like: `http://localhost:8888/?token=...`

   If the browser does not open automatically, copy the URL from the terminal and paste it in your browser.

8. **Open and Run the Notebook**

   - In the Jupyter file browser, click on `helios_solution.ipynb` to open it
   - Run all cells: Click **Kernel** → **Restart & Run All**
   - Wait for execution to complete (15 to 20 minutes)
   - Output will be saved to: `output/submission.csv`

> **Note:** Execution time can be reduced by eliminating custom features that do not contribute to the final selection.

9. **Deactivate Virtual Environment (when done)**
   ```bash
   deactivate
   ```

**Result:** Running on local Windows environment produces the same CFCS score (0.73272 on 0-1 scale, or 73.27 on 0-100 scale) as Kaggle notebook, confirming reproducibility.

---

### 1.4 Environment Details

#### Kaggle Notebook Settings (Exact Configuration Used)

| Setting | Value |
|---------|-------|
| **Python Version** | 3.12.12 |
| **Environment** | Pin to original environment (2025-12-17) |
| **Accelerator** | None (CPU only, no GPU required) |
| **Internet** | On (not used by code) |
| **Persistence** | Files only |

#### Python Version
```
Python 3.12.12
```

#### requirements.txt
```txt
pandas>=2.0.0
numpy>=1.24.0
scipy>=1.11.0
```

#### Notes on Environment
- **No GPU required:** The solution uses only CPU-based operations
- **No internet access required:** All data is loaded from local competition files. Internet setting can be turned off
- **Memory:** Successfully tested on Kaggle (30 GB RAM) and local machine (32 GB RAM). Minimum RAM requirement is unknown

---

## 2. Methodology Summary

### 2.1 Data Sources Used

#### 2.1.1 Competition Data (Official - Used)

**Data Source:** https://www.kaggle.com/competitions/forecasting-the-future-the-helios-corn-climate-challenge/data

| File | Records | Description |
|------|---------|-------------|
| `corn_climate_risk_futures_daily_master.csv` | 320,661 | Main dataset with climate risks and futures |
| `corn_regional_market_share.csv` | 86 | Regional production percentages |

#### 2.1.2 Data Structure Details

**Main Dataset Columns Used:**

| Category | Columns | Usage in Solution |
|----------|---------|-------------------|
| **Geographic IDs** | `country_name`, `region_name`, `region_id` | Groupby keys |
| **Temporal** | `date_on` | Time binning, groupby |
| **Climate Risk Counts** | `climate_risk_cnt_locations_*` (12 columns) | Primary feature source |

**Climate Risk Columns (Input Features):**

| Risk Category | Low | Medium | High |
|--------------|-----|--------|------|
| Heat Stress | `climate_risk_cnt_locations_heat_stress_risk_low` | `..._medium` | `..._high` |
| Cold Stress | `climate_risk_cnt_locations_unseasonably_cold_risk_low` | `..._medium` | `..._high` |
| Excess Precip | `climate_risk_cnt_locations_excess_precip_risk_low` | `..._medium` | `..._high` |
| Drought | `climate_risk_cnt_locations_drought_risk_low` | `..._medium` | `..._high` |

**Market Share Data:**

| Column | Description | Usage |
|--------|-------------|-------|
| `region_id` | Unique region identifier | Merge key |
| `percent_country_production` | % of national corn production | Production weighting |

#### 2.1.3 External Data/Tools

| External Source | Used? | Details |
|-----------------|-------|---------|
| External APIs | **NO** | No external APIs called |
| External Datasets | **NO** | Only competition data used |
| Pre-trained Models | **NO** | No ML models used |
| Web Scraping | **NO** | No web data collected |
| Third-party Libraries | **YES** | pandas, numpy, scipy |

---

### 2.2 Key Feature Engineering Steps and Rationale

> **Note:** The climate risk data comes from Helios's proprietary climate risk model, which counts the number of locations (POIs) experiencing each risk level daily. Understanding the risk level definitions is crucial for interpreting the features.

#### Risk Level Definitions

| Risk Level | Definition |
|------------|------------|
| **High Risk** | Conditions are both **anomalous** (statistically unusual) AND **outside the crop's comfort bounds** |
| **Medium Risk** | Conditions are **outside the crop's comfort bounds** AND **above normal historical patterns** (similar to 1.5-2.5 standard deviations) but not yet anomalous |
| **Low Risk** | Normal conditions within acceptable parameters for corn growth |

#### Risk Categories

| Category | Internal Name | Description |
|----------|---------------|-------------|
| **Heat Stress** | daily_too_hot_risk | Maximum temperature exceeding corn tolerance |
| **Cold Stress** | daily_too_cold_risk | Minimum temperature below corn requirements |
| **Excess Precipitation** | daily_too_wet_risk | Rainfall exceeding optimal levels |
| **Drought** | daily_too_dry_risk | Insufficient precipitation for corn needs |

---

#### Top 5 Key Features Selected

These are the final features selected based on significant correlation count (sig_count) with futures data:

| Rank | Feature Name | Sig Count |
|------|--------------|-----------|
| 1 | `climate_risk_wsum_quartile_agg_climate_risk_wsum_w_non_drought_med_sum_mean` | 904 |
| 2 | `climate_risk_wsum_quartile_agg_climate_risk_wsum_w_non_drought_med_sum_sum` | 848 |
| 3 | `climate_risk_wsum_quartile_agg_climate_risk_wsum_w2_all_med_sum_mean` | 829 |
| 4 | `climate_risk_compound_med_sextile_agg_climate_risk_compound_med_drought_excess_med_product_sum` | 801 |
| 5 | `climate_risk_de_compound_quartile_agg_climate_risk_de_compound_w_drought_excess_med_min_max` | 800 |

#### Why These Key Features Work

| Pattern | Explanation |
|---------|-------------|
| **Quartile/Sextile Time Bins** | 3 to 6 month aggregation periods match the futures market's forward-looking horizon |
| **Weighted (wsum) Features** | Production weighting using `percent_country_production` focuses on economically significant regions |
| **Medium Risk Level** | Medium severity (outside comfort bounds, 1.5-2.5 std dev) is most predictive - High risks are rare/anomalous, Low risks indicate normal conditions |
| **Compound Features** | Drought × Excess Precipitation captures alternating too_dry/too_wet stress patterns |
| **Sum/Mean Aggregations** | Total and average location counts within time bins capture cumulative market-relevant signals |

#### Understanding the Key Feature Names

Breaking down the top 5 feature names using terminology from the dataset:

**Feature 1:** `climate_risk_wsum_quartile_agg_climate_risk_wsum_w_non_drought_med_sum_mean`

| Component | Meaning |
|-----------|---------|
| `climate_risk_` | Required prefix for all climate features |
| `wsum` | Weighted sum (location counts × `percent_country_production`) |
| `quartile_agg` | Aggregated by quartile time bins (4 bins over the date range) |
| `w_non_drought_med` | Weighted sum of non-drought medium risks: `heat_stress_risk_medium` + `unseasonably_cold_risk_medium` + `excess_precip_risk_medium` |
| `sum` | Sum of weighted location counts within each group |
| `mean` | Mean of those sums across the quartile |

**Feature 4:** `climate_risk_compound_med_sextile_agg_climate_risk_compound_med_drought_excess_med_product_sum`

| Component | Meaning |
|-----------|---------|
| `compound_med` | Compound of medium-level risks |
| `sextile_agg` | Aggregated by sextile time bins (6 bins) |
| `drought_excess_med` | Combination of `drought_risk_medium` and `excess_precip_risk_medium` |
| `product` | Multiplication: `drought_risk_medium` × `excess_precip_risk_medium` (captures co-occurrence) |
| `sum` | Sum of products within each group |

**Feature 5:** `climate_risk_de_compound_quartile_agg_climate_risk_de_compound_w_drought_excess_med_min_max`

| Component | Meaning |
|-----------|---------|
| `de_compound` | Drought-Excess compound feature |
| `quartile_agg` | Aggregated by quartile time bins |
| `w_drought_excess_med` | Weighted drought × excess_precip at medium level |
| `min` | Minimum value within the group |
| `max` | Maximum of those minimums across the quartile |

---

### Feature Engineering Process

The following steps describe how the 1,494 features were created, from which the top 5 key features were selected.

#### Step 1: Temporal Feature Extraction

**What:** Extract year, month, quarter, week, day_of_year from `date_on`

**Code:**
```python
merged_df['year'] = merged_df['date_on'].dt.year
merged_df['month'] = merged_df['date_on'].dt.month
merged_df['day_of_year'] = merged_df['date_on'].dt.dayofyear
merged_df['quarter'] = merged_df['date_on'].dt.quarter
merged_df['week_of_year'] = merged_df['date_on'].dt.isocalendar().week
```

**Rationale:**
- **Seasonality in Agriculture:** Corn futures prices (ZC*1, ZC*2 from Barchart API) are highly seasonal, tied to planting, growing, and harvest cycles
- **Climate Risk Timing:** The dataset includes `harvest_period` (e.g., "Harvest", "Planting") and `growing_season_year` - climate impacts vary by growth stage
- **Grouping Foundation:** These temporal features enable aggregating location counts within meaningful time periods (e.g., monthly country-level risk location counts)
- **Pattern Discovery:** Allows identification of recurring seasonal patterns in climate-futures relationships across the 2015-2025 date range

---

#### Step 2: Time Bin Creation

**What:** Create 8 different temporal granularities (tertile through vigintile)

**Configuration:**
```python
TIME_BIN_CONFIG = {
    'quarters_per_tertile': ceil(total_quarters / 3),    # 3 bins
    'quarters_per_quartile': ceil(total_quarters / 4),   # 4 bins
    'quarters_per_quintile': ceil(total_quarters / 5),   # 5 bins
    'quarters_per_sextile': ceil(total_quarters / 6),    # 6 bins
    'quarters_per_octile': ceil(total_quarters / 8),     # 8 bins
    'quarters_per_decile': ceil(total_quarters / 10),    # 10 bins
    'quarters_per_tredecile': ceil(total_quarters / 13), # 13 bins
    'quarters_per_vigintile': ceil(total_quarters / 20), # 20 bins
}
```

**Rationale:**
- **Multi-Scale Analysis:** Climate impacts manifest at different time scales - sudden weather events (days), seasonal patterns (months), and long-term trends (years)
- **Optimal Time Period Discovery:** By creating multiple bin sizes, we let the correlation analysis identify which time period size best captures the climate-futures relationship
- **Noise Reduction:** Coarser bins (tertile, quartile) smooth out daily fluctuations in location counts while preserving meaningful trends
- **Fine-Grained Patterns:** Finer bins (decile, vigintile) capture shorter-term market reactions to climate events
- **Why Quartile and Sextile Performed Best:** The top features use quartile and sextile bins, suggesting futures markets respond to climate patterns aggregated over 2 to 4 quarter periods

---

#### Step 3: Production-Weighted Aggregations

**What:** Weight climate risk location counts by regional production importance using `percent_country_production`

**Code:**
```python
# Merge production weights from corn_regional_market_share.csv
merged_df = merged_df.merge(
    market_share_df[['region_id', 'percent_country_production']],
    on='region_id', how='left'
)

# Create weighted features: location_count × production_percentage
df[f'climate_risk_weighted_{risk_col}'] = df[risk_col] * df['percent_country_production']
```

**Rationale:**
- **Economic Significance:** The `percent_country_production` column (from `corn_regional_market_share.csv`) quantifies each region's contribution to national corn production
- **Supply-Side Reality:** Corn futures prices respond to expected supply changes - major producing regions dominate supply expectations
- **Signal Amplification:** Weighting amplifies climate signals from regions that actually move markets
- **Noise Reduction:** Reduces influence of climate events in regions with minimal or zero production percentages

---

#### Step 4: Compound Stress Features

**What:** Combine multiple risk category location counts through multiplication, addition, geometric mean

**Code:**
```python
# Product (multiplicative compound) - counts locations with BOTH stresses
df['compound_drought_heat'] = df['climate_risk_cnt_locations_drought_risk_high'] * \
                              df['climate_risk_cnt_locations_heat_stress_risk_high']

# Geometric mean (balanced compound)
df['geomean_drought_heat'] = np.sqrt(df['drought_risk'] * df['heat_stress_risk'])

# Sum (additive compound) - total locations under either stress
df['sum_drought_heat'] = df['climate_risk_cnt_locations_drought_risk_high'] + \
                         df['climate_risk_cnt_locations_heat_stress_risk_high']
```

**Rationale:**
- **Synergistic Stress:** Agricultural damage is often non-linear - drought (daily_too_dry_risk) combined with heat stress (daily_too_hot_risk) causes disproportionately more damage than either alone
- **Biological Reality:** Corn under drought stress has reduced ability to cope with excessive temperatures. The combination can cause total crop failure
- **Product Captures Co-occurrence:** Multiplication creates high values only when BOTH stresses have high location counts simultaneously
- **Geometric Mean for Balance:** Prevents extreme location counts from dominating. Captures balanced compound stress
- **Key Combinations Identified:**
  - **Drought + Heat:** Classic compound stress - insufficient precipitation + excessive temperature
  - **Drought + Excess Precipitation:** Alternating too_dry and too_wet conditions damage root systems
  - **Non-drought Medium Risks:** Combination of heat_stress, unseasonably_cold, and excess_precip at medium severity level (outside comfort bounds but not anomalous)

---

#### Step 5: Groupby Aggregations

**What:** Compute statistics (mean, sum, std, var, max, min) of location counts across different groupings

**Code:**
```python
df[feat_name] = df.groupby(groupby_cols)[source_col].transform(agg_func)
```

**Grouping Dimensions:**
- Time bins (quartile, sextile, etc.)
- `country_name` (from geographic identifiers)
- Country + `date_on_month`
- Country + Time bin

**Aggregation Functions:**
| Function | What It Captures | Market Relevance |
|----------|------------------|------------------|
| `mean` | Average location count at risk level | Baseline market expectations |
| `sum` | Total location counts (cumulative exposure) | Cumulative stress across regions |
| `std` | Volatility in location counts | Uncertainty premium in futures pricing |
| `var` | Variance in location counts | Squared volatility for sensitivity analysis |
| `max` | Peak location count (worst day) | Extreme event impact on prices |
| `min` | Minimum location count | Best-case scenario reference |

**Rationale:**
- **Different Market Signals:** Futures traders (using ZC*1, ZC*2 prices) respond to different aspects of climate risk:
  - **Mean:** "What's the typical number of locations at risk?" - sets baseline expectations
  - **Sum:** "How many total location-days of stress?" - cumulative damage assessment
  - **Std/Var:** "How uncertain is the climate outlook?" - drives risk premium in futures
  - **Max:** "What was the peak stress?" - extreme events move corn and related commodity prices sharply
- **Grouping Logic:**
  - Country-level captures national supply impacts across the 12 countries in the dataset
  - Time-bin level captures temporal patterns across the 2015-2025 date range
  - Country + Month captures seasonal patterns by region (aligned with harvest_period)

---

#### Step 6: Feature Selection

**What:** Rank features by significant correlation count with futures columns

**Code:**
```python
SIGNIFICANCE_THRESHOLD = 0.6
FEATURE_SELECTION_STRATEGY = 'sig_count'
TOP_N_FEATURES = 5

# Count correlations >= 0.6 with all 17 futures columns
for climate_col in climate_cols:
    for futures_col in futures_cols:  # futures_close_ZC_1, futures_zc1_ret_pct, etc.
        corr = np.corrcoef(climate_data, futures_data)[0, 1]
        if abs(corr) >= SIGNIFICANCE_THRESHOLD:
            sig_count += 1

# Select top 5 by sig_count
top_features = feature_analysis.nlargest(TOP_N_FEATURES, 'sig_count')
```

**Futures Columns Used for Correlation (17 total):**
- **Price Data:** futures_close_ZC_1, futures_close_ZC_2, futures_close_ZW_1, futures_close_ZS_1
- **Technical Indicators:** futures_zc1_ret_pct, futures_zc1_ret_log, futures_zc_term_spread, futures_zc_term_ratio
- **Moving Averages:** futures_zc1_ma_20, futures_zc1_ma_60, futures_zc1_ma_120
- **Volatility:** futures_zc1_vol_20, futures_zc1_vol_60
- **Cross-Commodity:** futures_zw_zc_spread, futures_zc_zw_ratio, futures_zs_zc_spread, futures_zc_zs_ratio

**Rationale:**
- **Robustness Over Single Correlation:** A feature that correlates strongly with multiple futures columns (prices, returns, spreads, volatility) across multiple time periods is more robust
- **Significance Threshold (0.6):** Filters out weak correlations that may be noise. Focuses on economically meaningful relationships
- **sig_count Strategy:** Counts how many (country, month, futures_column) combinations show |correlation| >= 0.6 - higher count = more consistent predictor
- **Why Top 5:** Balances signal strength with parsimony. Avoids overfitting while capturing main climate-futures relationships
- **Alternative Strategies Available:** max_corr (highest single correlation), avg_sig_corr (average of significant correlations), weighted combination

---

#### Rationale for Other Engineered Features (Not in Top 5)

The following feature types were created during exploration. While they did not make the final top 5, understanding their rationale provides insight into the feature engineering approach:

| Feature Type | What It Captures | Rationale |
|--------------|------------------|-----------|
| **Date Decile Groupby** | `climate_risk_cnt_locations_*` aggregated by decile time bins | Finer time resolution (10 bins) to capture shorter-term patterns |
| **Low-Risk Level Features** | Aggregations of `*_risk_low` columns | Count locations with normal conditions (within acceptable parameters for corn growth) |
| **High-Risk Level Features** | Aggregations of `*_risk_high` columns | Capture anomalous extreme events |
| **Tredecile/Vigintile Features** | Very fine time bins (13 to 20 bins) | Capture week-to-week or month-to-month patterns |
| **Categorical Encoded Features** | `harvest_period` encoded numerically | Capture growing season phase (Planting, Harvest, etc.) |
| **Risk Score Features** | Weighted combination: `(low×1 + medium×2 + high×3) / total` | Single score representing overall risk severity |
| **Cross-Risk Interaction (High)** | `drought_risk_high` × `heat_stress_risk_high` | Compound stress at high severity |
| **Country Daily Aggregations** | Mean/max/sum of location counts per country per day | Country-level daily risk summary |
| **Spatial Std Features** | Std deviation of location counts across regions on same date | Geographic dispersion of risk |
| **Monthly/Quarterly Aggregations** | Groupby `date_on_month` or `quarter` | Calendar-based seasonality |
| **Risk Ratio Features** | `high / (low + medium + high)` - proportion at high risk | Relative severity regardless of total locations |
| **Risk Dominance Features** | Which risk level dominates (high > medium > low) | Categorical risk state |
| **Drought-Heat High Compound** | `drought_risk_high` × `heat_stress_risk_high` | Extreme compound stress |
| **Geomean Features** | `sqrt(risk1 × risk2)` - geometric mean | Balanced compound that does not favor extremes |
| **Cross-Level Compounds** | `drought_risk_high` × `excess_precip_risk_medium` | Mixed severity interactions |
| **Cold Stress Features** | Aggregations of `unseasonably_cold_risk_*` | Temperature below corn requirements |
| **Single Risk Aggregations** | Individual risk categories without compounding | Isolated risk signals |

#### Key Insights from Feature Exploration

1. **Medium risk level is optimal:** Features using `_risk_medium` consistently outperformed `_risk_high` (too rare) and `_risk_low` (no signal)

2. **Quartile and sextile time bins are optimal:** 4 to 6 bins over the data range captured the right level of detail.

3. **Compound features outperform single risks:** Combinations like `drought × excess_precip` capture market-relevant stress patterns better than individual risks

4. **Production weighting is essential:** Weighted features (`wsum`) consistently ranked higher than unweighted counterparts

5. **Sum and mean aggregations dominate:** These captured cumulative and average risk better than std/var/max/min for top features

6. **Non-drought combinations are powerful:** The top feature uses `non_drought_med` (heat + cold + excess precipitation), suggesting the market responds to combined non-drought stresses

---

### 2.3 Anti-Gaming Compliance Confirmation

#### Statement of Compliance

**I confirm that no `futures_*` columns or their derivatives were used to generate `climate_risk_*` features, per the anti-gaming rules.**

##### 1. Feature Engineering

All 1,494 engineered features use ONLY these inputs:
- `climate_risk_cnt_locations_*` (12 original climate columns)
- `date_on` and derived temporal columns
- `country_name`, `region_name` (for groupby)
- `percent_country_production` (for weighting)

**Example feature creation (no futures involved):**
```python
# Groupby aggregation - uses only climate data and grouping columns
df[feat_name] = df.groupby(['country_name', 'climate_risk_time_bin_quartile'])[
    'climate_risk_cnt_locations_drought_risk_high'
].transform('mean')

# Weighted feature - uses only climate data and production weight
df[feat_name] = df['climate_risk_cnt_locations_heat_stress_risk_high'] * \
                df['percent_country_production']

# Compound feature - uses only climate columns
df[feat_name] = df['climate_risk_cnt_locations_drought_risk_medium'] * \
                df['climate_risk_cnt_locations_excess_precip_risk_medium']
```

##### 2. Futures Usage

Futures columns are accessed ONLY for:
1. **Evaluation/Scoring** - Computing CFCS score AFTER feature engineering
2. **Submission Assembly** - Including futures columns in output file

```python
# Futures used only for scoring (post-feature-engineering)
futures_cols = [c for c in df.columns if c.startswith('futures_')]
# ... correlation computation for evaluation only ...

# Futures included in submission file (not used for feature creation)
required_cols = ['ID', 'date_on', 'country_name', 'region_name'] + futures_cols + top_features
submission = baseline_df[required_cols].copy()
```

##### 3. Verification Checklist

| Requirement | Status | Evidence |
|-------------|--------|----------|
| No futures columns used in feature creation | ✅ COMPLIANT | Feature engineering only references `climate_risk_*` columns |
| No futures derivatives used | ✅ COMPLIANT | No calculations involving futures data |
| Futures used only for evaluation | ✅ COMPLIANT | Correlations computed post-engineering |
| All climate features prefixed correctly | ✅ COMPLIANT | All start with `climate_risk_` |

---

## 3. Submission Reference

**Submission File:** https://drive.google.com/drive/folders/1xLbg46CfF21zdGEt3b5EJnxXGCPQUIeA?usp=sharing

### 3.1 Account Information

| Field | Value |
|-------|-------|
| **Kaggle Username** | ganeshstemx |
| **Kaggle Profile** | https://www.kaggle.com/ganeshstemx |

### 3.2 Submission Details

| Field | Value |
|-------|-------|
| **Submission File** | `submission.csv` |
| **Submission Name** | Helios \| 231 - Version 2 |
| **Submission Timestamp (UTC)** | Fri Jan 30, 2026 ,17:00:07 UTC |
| **Submission Timestamp (EST)** | Fri Jan 30, 2026 ,12:00:07 PM EST |
| **Submission Timestamp (IST)** | Fri Jan 30, 2026 ,22:30:07 IST |

### 3.3 Final Scores

#### Kaggle Leaderboard Scores

| Score Type | Value |
|------------|-------|
| **Public CFCS Score** | 85.67000 |
| **Private CFCS Score** | 78.17000 |

#### Notebook Output Scores

| Metric | Value |
|--------|-------|
| **CFCS Score (0-1 scale)** | 0.73272 |
| **CFCS Score (0-100 scale)** | 73.27 |
| **Average Significant Correlation** | 0.733689 |
| **Maximum Correlation** | 0.971102 |
| **Significant Correlation Count** | 4,182 / 11,220 (37.27%) |

#### CFCS Formula Difference

The notebook uses a simplified CFCS formula for feature evaluation that produces a **0-1 scale**:

```python
# Notebook formula (0-1 scale)
cfcs = (0.5 * avg_sig_corr) + (0.3 * max_corr) + (0.2 * sig_pct / 100)
```

The official Kaggle evaluation uses the full formula that produces a **0-100 scale**:

```python
# Official Kaggle formula (0-100 scale)
avg_sig_score = min(100, avg_sig_corr * 100)
max_score = min(100, max_corr * 100)
cfcs = (0.5 * avg_sig_score) + (0.3 * max_score) + (0.2 * sig_pct)
```
- **Notebook CFCS:** 0.73272 (0-1 scale) = 73.27 (0-100 scale)
- **Kaggle Public CFCS:** 85.67 (0-100 scale)
- **Kaggle Private CFCS:** 78.17 (0-100 scale)

### 3.4 Top 5 Selected Features

| Rank | Feature Name | Sig Count | Max Corr | Avg Sig Corr |
|------|--------------|-----------|----------|--------------|
| 1 | `climate_risk_wsum_quartile_agg_climate_risk_wsum_w_non_drought_med_sum_mean` | 904 | 0.9711 | 0.7414 |
| 2 | `climate_risk_wsum_quartile_agg_climate_risk_wsum_w_non_drought_med_sum_sum` | 848 | 0.9433 | 0.7337 |
| 3 | `climate_risk_wsum_quartile_agg_climate_risk_wsum_w2_all_med_sum_mean` | 829 | 0.9590 | 0.7183 |
| 4 | `climate_risk_compound_med_sextile_agg_climate_risk_compound_med_drought_excess_med_product_sum` | 801 | 0.9404 | 0.7164 |
| 5 | `climate_risk_de_compound_quartile_agg_climate_risk_de_compound_w_drought_excess_med_min_max` | 800 | 0.9694 | 0.7587 |

### 3.5 Submission File Statistics

| Metric | Value |
|--------|-------|
| **Total Rows** | 219,161 |
| **Total Columns** | 26 |
| **Climate Features** | 5 |
| **Futures Columns** | 17 |
| **Metadata Columns** | 4 (ID, date_on, country_name, region_name) |

---
