# Comprehensive Methodology Report: Helios Corn Futures Climate Challenge

> **Generated:** 2026-02-09  
> **Competition:** Helios Corn Futures Climate Challenge  

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Participant Methodologies](#participant-methodologies)
   - [Competition Winners](#competition-winners)
     - [1st Place: yukanglimofficial](#-1st-place---yukanglimofficial)
     - [2nd Place: cg](#-2nd-place---cg)
     - [3rd Place: ezberch](#-3rd-place---ezberch)
   - [Other Participants](#other-participants)
   - [limyuxin](#1-limyuxin)
   - [cmasch](#2-cmasch)
   - [bluetriad](#3-bluetriad)
   - [GPCH](#4-gpch)
   - [Mr RRR](#5-mr-rrr)
   - [PxlPau](#6-pxlpau)
   - [aaaml007](#7-aaaml007)
   - [chetank99](#8-chetank99)
   - [DragonAJA](#9-dragonaja)
   - [ganeshstemx](#10-ganeshstemx)
   - [kadircandrisolu](#11-kadircandrisolu)
   - [ravi123a321at](#12-ravi123a321at)
   - [osamurai](#13-osamurai)
3. [Comparative Analysis](#comparative-analysis)
4. [Key Insights](#key-insights)

---

## Executive Summary

This report documents the methodologies used by participants in the Helios Corn Futures Climate Challenge Kaggle competition. Each participant attempted to engineer climate risk features that correlate with corn futures prices. The approaches range from simple rolling averages to sophisticated AutoEncoder neural networks and exhaustive grid searches.

**Competition Winners:**
- 🥇 **1st Place**: [yukanglimofficial](https://www.kaggle.com/yukanglimofficial) - **84.20 CFCS** (Rank 49) - Deep Sweep Grid Search
- 🥈 **2nd Place**: [cg](https://www.kaggle.com/chaitanyagarg2) - **82.35 CFCS** (Rank 54) - Global Production-Weighted Risk
- 🥉 **3rd Place**: [ezberch](https://www.kaggle.com/ezberch) - **80.47 CFCS** (Rank 63) - Parallel Correlation Mining

**Notable Mentions:**
- **Rank 46**: limyuxin - **88.82 CFCS** - Brute Force Factor Mining
- **Rank 47**: SwastikR@1001 - **87.43 CFCS** (Not included in solutions analysis)

**Leaderboard Structure:**
- Ranks 1-45: Baseline scores (100.00) indicating disqualification or non-compliant submissions
- Ranks 46-49: Top competitive scores (88.82 - 84.20 CFCS)
- Ranks 50-99: Competitive scores (83.73 - 59.24 CFCS) - includes majority of reviewed participants
- All 16 reviewed participants have been matched with their official leaderboard rankings and CFCS scores

**Status:**
- Some participants were disqualified for rule violations and have been removed from this report

### Quick Reference Table

| Participant | Strategy | CFCS Score | Uniqueness |
|-------------|----------|------------|------------|
| 🥇 [yukanglimofficial](https://www.kaggle.com/yukanglimofficial) | Deep Sweep Grid Search | **84.20** | Comprehensive parameter space exploration with stability validation |
| 🥈 [cg](https://www.kaggle.com/chaitanyagarg2) | Global Production-Weighted Risk | **82.35** | Global risk hypothesis, economic weighting throughout, momentum layers |
| 🥉 [ezberch](https://www.kaggle.com/ezberch) | Parallel Correlation Mining | **80.47** | Macro-climate linkage (ENSO, PDO), parallel feature scanning |
| [limyuxin](https://www.kaggle.com/limyuxin) | Brute Force Factor Mining | 88.82 | Systematic caching pipeline + parameter sweep across 1000+ combinations |
| [cmasch](https://www.kaggle.com/cmasch) | Agronomic Feature Engineering | 69.01 | GDD deficit, stress day counting, grain fill period masking |
| [bluetriad](https://www.kaggle.com/bluetriad) | Climate Oscillation Integration | 76.29 | Climate technicals, cumulative sums, CFCS-aware selection |
| [GPCH2159](https://www.kaggle.com/gpch2159) | Modular Package Structure | 74.23 | Professional code organization (incomplete - missing dependencies) |
| [Mr RRR](https://www.kaggle.com/larrylin666) | AutoEncoder Feature Extraction | 66.33 | Only deep learning submission, neural network latent space compression |
| [PxlPau](https://www.kaggle.com/studybeetutoring) | Signal Sharpening | 59.67 | Power law (risk²), hemispheric gating, acreage battle modeling |
| [aaaml007](https://www.kaggle.com/aaaml007) | Production-Weighted Multi-Timescale | 100.00 | Comprehensive lag structure (7-90 days), production-weighted aggregation |
| [chetank99](https://www.kaggle.com/chetank99) | Relative Risk Ratios | 77.07 | Ratio signals (local/global), harvest window focus, non-linear panic modeling |
| [DragonAJA](https://www.kaggle.com/dragonaja) | Massive Feature Engineering & Optimization | **75.30** | 12 transformation methods, iterative feature refinement, CFCS-driven optimization |
| [ganeshstemx](https://www.kaggle.com/ganeshstemx) | Temporal Quantile Binning | 78.17 | Non-overlapping time segments, compound interactions, non-drought inverse signals |
| [kadircandrisolu](https://www.kaggle.com/kadircandrisolu) | Ultra-Long Window Cumulative Drought | 79.26 | Extreme parsimony (single feature), 400+ day windows, cumulative tracking |
| [ravi123a321at](https://www.kaggle.com/ravi123a321at) | Phenology-Weighted Stress | 64.02 | HHI concentration metric, growth-stage-aware weighting, biological time scales |
| [osamurai](https://www.kaggle.com/osamurai) | LightGBM Feature Transformation | 72.33 | LightGBM predictions as features, L9 orthogonal array optimization, comprehensive interactions |

---

## Participant Methodologies

---

## Competition Winners

### 🥇 1st Place - [yukanglimofficial](https://www.kaggle.com/yukanglimofficial)

**CFCS Score:** **84.20**

**Strategy:** Deep Sweep Grid Search

**Core Approach:**  
Pragmatic regime-dependent evaluation within country-by-month seasonal buckets to isolate when/where climate mechanisms are active. Compresses noisy region-level climate counts into interpretable country-day signals via production-weighted aggregation. Implements two-stage sweep: Stage 0 fast baseline scan of ALL (country, month, signal) combinations, Stage 1 deep refinement with extensive parameter search.

**Implementation:**
1. **Cache Building:** Pre-compute base aggregations (`cache/cd_cache_kaggle.npz`) to prevent data leakage
2. **Stage 0 (Baseline):** Fast scan of all country-month-signal combinations to rank promising groups
3. **Stage 1 (Deep):** Refines only top groups with parameter search:
   - Shifts: -60 to +60 days (submission enforces ≥0 for compliance)
   - Windows: 2 to 2500 days
   - Aggregations: ma, max, ewm, std, streakq85, streakthr0.5
   - Transforms: identity, square, signlog1p
4. **Feature Selection:** Keeps top 60 windows and 80 window-shift pairs based on PC1 proxy correlation

**Uniqueness:**
- **Regime-Dependent Evaluation:** Evaluates within country-by-month buckets to isolate active mechanisms
- **PC1 Proxy Innovation:** Uses production-weighted PC1 futures factor as fast breadth proxy instead of brute-force scoring
- **Two-Stage Architecture:** Fast baseline scan identifies promising groups before expensive deep refinement
- **Custom Aggregations:** streakthr0.5 for binary persistence patterns, streakq85 for quantile-based thresholds
- **Shift Parameter:** Allows both lag and lead relationships (though submission enforces ≥0)
- **Stability Validation:** Tests across multiple recent-year slices and year-demeaned splits
- **Caching Strategy:** Pre-calculates base aggregations to prevent data leakage while enabling efficient search
- **Expressive Feature Set:** Small but purposeful - stress measures, plausible interactions, time-series shapes

**Why This Won:**
- Highly systematic and reproducible approach
- Efficient caching strategy reduces computation time
- Comprehensive parameter space exploration
- Clean separation of feature generation and evaluation phases

---

### 🥈 2nd Place - [cg](https://www.kaggle.com/chaitanyagarg2)

**CFCS Score:** **82.35**

**Strategy:** Global Production-Weighted Risk Aggregation

**Core Approach:**  
Hypothesizes that global aggregate production-weighted climate risk has stronger predictive power than individual country/region signals. Constructs composite risk indicators weighted by `percent_country_production`. Features the most economically-aware approach, reflecting that major producers like Iowa (16% of US production) dominate market sentiment.

**Feature Engineering Steps:**
1. **Base Risk Scores:** 8 features from raw climate counts (Low/Medium/High × Heat/Drought/Precipitation)
2. **Composite Indicators:** 12 features
   - Temperature stress (heat aggregation)
   - Precipitation stress (drought + precip aggregation)
   - Overall/combined stress metrics
3. **Rolling Aggregations:** 36 features over 7, 14, 30 day windows (means, maxes, sums)
4. **Momentum Features:** 48 features capturing changes and acceleration
   - First differences (7d, 14d, 30d)
   - Second differences (acceleration)
5. **Country Aggregations:** 68 features (national-level patterns)
6. **Feature Selection:** Top 30 by Pearson correlation with target
7. **Modeling:** Gradient boosting (likely XGBoost or LightGBM)

**Uniqueness:**
- **Global Risk Hypothesis:** Aggregate worldwide production-weighted risk > individual regional risks
- **Economic Weighting Throughout:** Production shares integrated at every aggregation step
- **Momentum Layers:** Captures both magnitude and rate-of-change in climate stress
- **Domain Awareness:** Reflects real market dynamics where major producers dominate price formation
- **Composite Signals:** Temperature/precipitation stress as derived economic indicators

**Why This Won:**
- Strong economic foundation - markets react to production-weighted global risk
- Comprehensive momentum modeling captures both levels and changes
- Clean feature selection process with 30 high-quality features
- Successfully reproduced submission with gradient boosting approach

---

### 🥉 3rd Place - [ezberch](https://www.kaggle.com/ezberch)

**CFCS Score:** **80.47**

**Strategy:** Parallel Correlation Mining with Macro-Climate Indices

**Core Approach:**  
Systematic approach generating massive feature combinations (aggregations, lags, transformations) and scanning them in parallel batches for high correlation with futures. Integrates external macro-climate indices (ONI, PDO, MJO) to link local crop stress to global climate oscillations. Uses greedy search with 90% correlation cap to prevent selecting nearly identical features.

**Feature Engineering Steps:**
1. **Base Features:** Climate risk counts (Low/Med/High × Heat/Drought/Precip)
2. **Aggregation Methods:**
   - Rolling windows: 450, 500 days with mean/max/sum
   - EMA (Exponential Moving Average): Long-term trend tracking
   - STD (Standard Deviation): Variability/risk uncertainty quantification
   - TURB (Turbulence): Volatility of rapid changes
3. **Lag Features:** Time-shifted values (7, 30, 60, 90 days) to model delayed market reactions
4. **Power Transformations:** P1.0 (linear), P2.0 (squared), P3.0 (cubed) to isolate tail risks
5. **External Macro-Climate Indices:**
   - ONI (Oceanic Niño Index) - ENSO/El Niño-La Niña tracking
   - PDO (Pacific Decadal Oscillation) - Long-term Pacific SST patterns
   - MJO (Madden-Julian Oscillation) - Intra-seasonal tropical precipitation
   - Lagged to prevent future data leakage
6. **Composite Signals:**
   - US_Stress_x_Nina_Primed: US heat/drought amplified by La Niña conditions
   - BR_Heat_Drought: Combined Brazil heat and drought stress
   - BR_Sum_heat_stress: Brazil summer heat stress focus
7. **Parallel Processing:** Batch-based correlation scanning across feature space
8. **Greedy Feature Selection:** Top N features by absolute Pearson correlation with 90% max inter-feature correlation

**Top 5 Features (External Data Version):**
1. `climate_risk_EMA_climate_risk_US_Stress_x_Nina_Primed_P2.0_W500` - 500-day trend of US stress during La Niña
2. `climate_risk_TURB_climate_risk_US_Stress_x_Nina_Primed_120_L60_P1.0_W500` - Volatility aftershocks from La Niña
3. `climate_risk_TURB_BR_Sum_heat_stress_L30_P2.0_W500` - Brazil summer heat stress turbulence
4. `climate_risk_STD_climate_risk_US_Stress_x_Nina_Primed_P2.0_W500` - US La Niña stress variability
5. `climate_risk_TURB_BR_Sum_heat_stress_L90_P2.0_W500` - Brazil heat stress with 90-day market lag

**Uniqueness:**
- **Macro-Climate Linkage:** Connects local crop stress to planetary-scale climate drivers (ENSO, PDO, MJO)
- **Parallel Architecture:** Efficiently scans large feature spaces through batch processing
- **Scientific Grounding:** Established climate science (ENSO teleconnections) integration
- **Multi-Scale Modeling:** Combines fine-grained local risks with coarse global oscillations
- **Correlation Cap:** 90% threshold prevents redundant feature selection
- **Tail Risk Focus:** Power transformations (squared/cubed) isolate extreme events

**Why This Won:**
- Domain knowledge integration with established climate science teleconnections
- Efficient parallel processing for large-scale feature exploration
- Strong scientific foundation linking local and global climate patterns
- Systematic greedy search with diversity constraint (90% correlation cap)
- Successfully reproduced with two versions (with/without external data)

---

## Other Participants

---

### 1. [limyuxin](https://www.kaggle.com/limyuxin)

**CFCS Score:** 88.82 (Rank 46 - Not awarded prize)

**Strategy:** Brute Force Factor Mining

**Core Approach:**  
Builds a sophisticated "stress primitives" framework creating both level and anomaly versions of every signal. Implements three alternative production-weighting schemes (share_norm_fill1, share_only_norm, share_plus_locations) and uses explicit anomaly detection via day-of-year z-scores to remove predictable seasonality. Uses two-stage sweep: fast proxy screening using PC1 of futures panel within country-month buckets, followed by tight local refinements.

**Feature Engineering Steps:**
1. **Stress Primitives:** Three signal types per risk category
   - Severity ratios (sev_wmean)
   - Warning-or-worse share (wapr_wmean) 
   - High-risk share (high_wmean)
2. **Composites:** Intuitive climate stress indicators
   - Wet-dry imbalance: wet_wapr_wmean - dry_wapr_wmean
   - Temperature stress: max(heat severity, cold severity)
   - Precipitation stress: max(drought, excess precipitation)
   - Overall stress: max across all four risk types
3. **Country-Level Signals:** Under 3 production-weighting schemes
4. **Surprise Variables:** Day-of-year z-scores for both level and anomaly series
5. **Temporal Transforms:** Rolling std, streak-fraction operations (streakthr0.5), square transformations (sign(x) × x²)
6. **Parameter Search:** Windows (w=527), shifts (0-86 days), multiple aggregations (ma, max, ewm, std)

**Selection Method:**
- **Stage 1 (Screening):** Extracts PC1 from futures panel per country-month bucket
- Scores climate features that co-move with PC1 for fast breadth-first search  
- **Stage 2 (Refinement):** Top hypotheses refined with local sweeps around lag/window settings
- Prioritizes using CFCS metric across multiple recent-year slices
- Year-demeaned sanity check emphasizes candidates that stay broad and stable across periods

**Uniqueness:**
- **Stress Primitives Framework:** Three distinct signal types (severity, warning-or-worse, high-risk share) rather than simple averages
- **Dual Signal System:** Creates both level and surprise (anomaly) versions using day-of-year z-scores
- **PC1 Proxy Innovation:** Uses first principal component of futures panel as fast screening mechanism instead of brute-force correlation testing
- **Triple Weighting Schemes:** Tests share_norm_fill1, share_only_norm, and share_plus_locations alternatives
- **Streak Detection:** Custom binary persistence patterns (streakthr0.5) for tracking sustained climate conditions
- **Square Transformations:** Sign-preserving squared transforms (sign(x) × x²) for amplifying signals
- **Regime Stability:** Year-demeaned validation ensures features work across multiple periods, not just one

**Technical Merit:**
- Systematic exploration of parameter space without domain assumptions
- Novel use of PCA for efficient correlation proxy
- Robust validation across multiple time periods
- Avoids overfitting through year-demeaned sanity checks

---

### 2. [cmasch](https://www.kaggle.com/cmasch)

#### Strategy: Agronomic Feature Engineering with Dual-Mode Architecture

**Core Approach:**  
Dual-mode system supporting pure Helios data and optional external data integration (FRED economic indicators + Open-Meteo weather). Pure mode generates 48+ features through temporal aggregations and stress indicators. Uses comprehensive lag periods [7,14,30,60,90,270,365,400] for historical pattern detection. Implements weighted risk scoring with configurable stress thresholds per category.

**Feature Engineering Steps:**
1. **Weighted Risk Scores:** Uses 2× multiplier for high-risk events: (medium + 2×high) / (total+ε)
2. **Stress Thresholds:** heat_stress: 0.75, drought: 0.63, excess_precip/cold: 0.5
3. **Rolling Statistics:** Extensive window ranges for pattern detection
4. **Exponential Moving Averages:** Recent event emphasis with multiple decay rates
5. **Volatility Measures:** Rolling standard deviation for climate instability proxy
6. **Cumulative Stress:** Rolling sums (30-90 days) for accumulation modeling
7. **External Integration (Optional):** Climate-economy interaction features when data available

**Parameters:**
- Lag periods: 7, 14, 30, 60, 90, 270, 365, 400 days
- Stress thresholds: Heat >0.75, Drought >0.63, Excess Precip/Cold >0.5
- MIN_SIG_COUNT: 48 without external, 78 with external
- Final features: 48+ (pure mode), 78+ (enhanced mode)

#### Uniqueness
- **GDD Deficit:** Calculates "lost heat units" relative to a crop's ideal thermal calibration
- **Stress Day Counting:** Exact days above thresholds rather than just average temperatures
- **is_grain_fill_period:** Temporal masking ensures model ignores climate data when crop isn't in ground

#### Creativity
- **Biological Realism:** Most scientifically accurate model—looks for *corn-specific* stress
- **Multi-Mode Operation:** Works with or without external data

#### Flaws
- **Code Complexity:** Required multiple patches to run successfully
- **Fragility:** Many dependencies on global variables and external data
- **Moderate Performance:** Despite 439 features, domain knowledge doesn't always translate to correlation

---

### 3. [bluetriad](https://www.kaggle.com/bluetriad)

#### Strategy: Climate Oscillation Integration with CFCS-Aware Selection

**Core Approach:**  
Builds upon external climate indices from NOAA Climate Prediction Center to capture large-scale atmospheric/oceanic patterns. Implements two-stage aggressive feature pruning to avoid "feature dilution" problem where weak features decrease CFCS by increasing denominator without adding significant correlations.

**Feature Engineering Steps:**
1. **Base Risk Scores:** Weighted risk counts by production importance
2. **Climate Oscillation Indices:**
   - ONI (Oceanic Niño Index) for El Niño-Southern Oscillation
   - SOI (Southern Oscillation Index) for Tahiti-Darwin pressure
   - NAO/AAO (North Atlantic/Antarctic Oscillation patterns)
   - OLR (Outgoing Longwave Radiation) for infrared energy
   - MJO (Madden-Julian Oscillation) for tropical convection
3. **Rolling Features:** Moving averages (14d, 30d, 60d)
4. **Lag Features:** Historical values for delayed response
5. **EMA Features:** Exponential moving averages with different spans
6. **Volatility Features:** Rolling standard deviations
7. **Cumulative Features:** `cumsum` for accumulated stress

**Selection Method:**
- **Stage 1 (Threshold Filtering):** Removes features with <400 significant correlations
- **Stage 2 (Forward Selection):** Iteratively adds only features that increase CFCS
- Takes ~6 hours for full forward selection but dramatically improves final score
- From 127 generated → 64 selected features

#### Uniqueness
- **"Climate Technicals":** Treating climate data as a tradable asset class
- **Cumulative Sums:** Recognizes that *accumulated* stress (drought) is more important than instantaneous stress

#### Creativity
- Novel application of financial technical analysis to weather data
- Top features are all drought-related MAs and CumSums

#### Flaws
- **Parameter Tuning:** Heavy reliance on specific window sizes (14d, 30d, 60d) may be overfit
- **External Data Dependency:** Originally designed with `extra_climate_data.csv` (climate oscillation indices)

---

### 3. GPCH

#### Strategy: Modular Package Structure

**Core Approach:**  
Attempted a modular code structure relying on a local `src` package for reusable components.

**Status:** **FAIL (Incomplete Submission)**

#### Uniqueness
- Attempted professional code organization with importable modules

#### Creativity
- N/A (cannot evaluate)

#### Flaws
- **Missing Dependencies:** Submission ZIP did not contain the required `src/` directory
- **ModuleNotFoundError:** Cannot execute without the package

---

### 4. Mr RRR

#### Strategy: AutoEncoder Latent Representation Learning

**Core Approach:**  
A sophisticated 5-stage pipeline using deep learning (PyTorch AutoEncoder) for non-linear feature compression. Workflow: Data Processing → Comprehensive Feature Engineering → Selection (CFCS Top-4) → AutoEncoder Training → Latent Space Prediction.

**Actual Implementation:**

**Stage 1 - Data Processing:**
- Load climate risk master data and corn futures
- Merge and align temporal indices

**Stage 2 - Comprehensive Feature Engineering:**
1. **Seasonal/Temporal Features:**
   - `day_of_year`, `quarter`, `month`
   - Cyclical encoding: `sin(2π × day/365)`, `cos(2π × day/365)`
   - Hemispheric shift for Southern Hemisphere alignment
2. **Risk Intensity Metrics:**
   - `score = (medium + 2×high) / (low + medium + high)`
   - `high_share = high / total`
   - `balance = std(Low, Med, High)`
   - `entropy = -Σ(p_i × log(p_i))` for risk distribution
3. **Time-Series Statistics:**
   - Rolling: mean, max, min, std (7, 14, 30, 60, 90 day windows)
   - Lags: 1, 7, 14, 30, 60 days
   - EMA (exponential moving average) with multiple decay rates
   - Momentum: first differences
   - Acceleration: second differences
4. **Event Features:**
   - Threshold persistence (days above threshold)
   - Event AUC (area under curve for stress episodes)
   - Spike detection (sudden increases)
5. **Compound Interactions:**
   - Temperature stress composites (heat + cold)
   - Precipitation stress (drought + excess)
   - Risk category differences and ratios
6. **Country-Level Aggregation:**
   - Production-weighted statistics by country × date
   - Concentration metrics (HHI-style)

**Stage 3 - Feature Selection:**
- CFCS evaluation of all features
- Select Top-4 highest scoring features

**Stage 4 - AutoEncoder Training:**
- **Architecture:** Input → Encoder → Latent Space → Decoder → Reconstruction
- **Framework:** PyTorch
- **Training:** MSE loss for reconstruction
- **Outputs:** Saved `.pth` model + `.joblib` scaler
- **Purpose:** Learn compressed non-linear representation of Top-4 features

**Stage 5 - Submission Generation:**
- Extract latent space activations from trained AE
- Use latent representation as final climate risk score
- Generate submission CSV

#### Uniqueness
- **Only Deep Learning Submission:** Unique use of Neural Networks (AutoEncoders) among all participants
- **Latent Space Hypothesis:** Non-linear compression captures interactions that linear methods miss
- **5-Stage Pipeline:** Professional-grade modular architecture

#### Creativity
- **Hybrid Approach:** Combines traditional feature engineering with modern deep learning
- **Representation Learning:** Treats problem as learning optimal feature embeddings
- **Entropy Features:** Information-theoretic risk distribution metrics

#### Flaws
- **Pipeline Fragility:** 5-notebook workflow prone to reproducibility errors (missing intermediate files)
- **Black Box:** Latent space interpretability difficult—what patterns did the AE learn?
- **Compute Requirements:** Requires PyTorch + GPU for efficient training
- **Overfitting Risk:** AE can memorize training data if not regularized properly
- **Complexity vs Benefit:** Uncertain if AE outperforms simpler linear combinations of Top-4 features

---

### 5. [PxlPau](https://www.kaggle.com/pxlpau)

#### Strategy: Signal Sharpening via Power Law via Power Law

**Core Approach:**  
Focuses on **"Bio-Economic Interaction"**—Climate Risk strength depends on (A) Biological Timing and (B) Market Structure. Uses Power Law transformations (`risk^2`) and Hemispheric Gating to suppress noise and amplify signals.

**Feature Engineering Steps:**
1. **Hemispheric Gating:** Zero out risk scores outside valid growing seasons
   - US Active: Months 5-10 (May-Oct)
   - Brazil Active: Months 10-5 (Oct-May)
2. **Signal Sharpening:** Apply `risk^2` to suppress low-level noise, amplify outliers
3. **Bio Context:** Multipliers for Phenology (Harvest vs Planting) and Regional Importance
4. **Market Receptivity:** Weight climate risk by trailing volatility (`futures_zc1_vol_20`)
5. **Power Belt:** Focus on key production regions (Iowa/Illinois/Mato Grosso)
6. **Acreage Battle:** Model soy-corn competition for land

**Market Adjustment Features:**
- Volatility-weighted risk (trailing 20-day vol)
- Trend-weighted risk (60-day MA)
- Scarcity-weighted risk (Term Spread)

#### Uniqueness
- **Power Law (`risk^2`):** Simple but effective non-linearity
- **"Acreage Battle":** Explicitly modeling soy-corn land competition
- **Hemispheric Gating:** Crucial for global crops

#### Creativity
- Uses market data for *weighting* (not predicting)—valid interaction terms
- Signal processing perspective on climate data

#### Flaws
- **Binary Seasons:** Fixed 6-month windows are rough approximations
- **Static Phenology:** Could be improved with dynamic phenology detection
- **Market Data Interaction:** Edge case of rule interpretation (approved as valid)

---

### 6. [aaaml007](https://www.kaggle.com/aaaml007)

**CFCS Score:** 100.00

#### Strategy: Production-Weighted Cumulative Stress with Multi-Timescale Lags

**Core Approach:**  
Builds "economically meaningful signals" by weighting climate risks by `percent_country_production` before aggregation. Uses systematic lag features (7-90 days) to capture delayed market responses, hypothesizing that futures prices reflect climate impacts with temporal offsets.

**Actual Implementation:**
1. **Weighted Risk Scores:** Production-weighted aggregation (Low/Medium/High counts × percent_country_production)
2. **Multi-Timescale Moving Averages:** 7, 14, 30, 60, 90 days for trend smoothing
3. **Rolling Max:** Peak stress detection over multiple windows
4. **EMA Features:** Exponential moving averages with recency bias
5. **Systematic Lags:** 7, 14, 21, 30, 60, 90 day lags for delayed impact modeling
6. **Volatility Proxies:** Rolling std (14-46 days) to capture climate instability
7. **Cumulative Stress:** 30-90 day rolling sums (accumulation vs instantaneous severity)
8. **Country Aggregations:** 155 national-level features for market-scale patterns
9. **Feature Selection:** Top features by correlation/importance before modeling

**Modeling:** Gradient boosting with hyperparameter tuning

#### Uniqueness
- **Comprehensive Lag Structure:** Systematic exploration of 6 lag windows (7-90 days)
- **Economic Weighting:** Production shares used to prioritize major regions
- **Multi-Scale Aggregation:** Combines instantaneous, short-term, medium-term, and long-term climate signals

#### Creativity
- **Delayed Response Hypothesis:** Explicit modeling of temporal offsets between climate events and price impacts
- **Volatility as Instability:** Rolling std as proxy for "climate chaos"

#### Flaws
- **Feature Explosion:** ~300 features may include redundancy
- **Static Production Weights:** Annual production shares treated as constant

---

### 8. [chetank99](https://www.kaggle.com/chetank99)

#### Strategy: Relative Risk Ratios with Hemispheric Harvest Gating

**Core Approach:**  
Hypothesis: Local climate risk impact depends on **global context**—high local risk matters more when the rest of the world is stable (high ratio) or when multiple competitors fail simultaneously (synchronization). Uses harvest season gating and non-linear transformations to model explosive market reactions.

**Actual Implementation:**
1. **Ratio Signals:** `local_risk / global_risk` for each risk category
   - Local = region/country level
   - Global = worldwide aggregate
2. **Harvest Season Gating:** Strict filtering to months 1, 2, 11, 12 (Southern Hemisphere harvest)
   - Features = 0 outside harvest window
   - Captures critical pricing periods
3. **Non-linear Terms:** Cubed ratios (`ratio^3`) to model explosive panic/scarcity premium
4. **Synchronization Features:** Multi-region failure detection (when 2+ major producers suffer simultaneously)
5. **Monthly Interactions:** Month-specific risk features with temporal encoding
6. **Feature Selection:** Top ratio features by correlation with futures
7. **Modeling:** Gradient boosting with tuned hyperparameters

**Results:**
- Strong correlations achieved (>0.8 for harvest months)
- Top performing months: December, November, January (Southern harvest peak)

#### Uniqueness
- **Relative vs Absolute:** Asks "How bad is this *compared to the world*?" instead of absolute severity
- **Harvest Window Focus:** Strict seasonal gating yields dramatic signal-to-noise improvement
- **Non-linear Panic Modeling:** Cubed terms capture explosive market reactions to scarcity

#### Creativity
- **Trade Flow Disruption:** Ratio signals implicitly model supply disruptions
- **Synchronization Risk:** Multi-region failure detection captures compounding effects
- **Relativity Principle:** Context matters more than magnitude

#### Flaws
- **Feature Sparsity:** 0 values for 8 months (features only active during harvest)
- **Row Count Mismatch:** 219,531 vs required 219,161 (+370 rows need filtering)
- **Single Hemisphere Bias:** Focuses on Southern harvest, needs Northern complement

---

### 9. [DragonAJA](https://www.kaggle.com/dragonaja)

**CFCS Score:** 75.30

#### Strategy: Massive Feature Engineering & Optimization

**Core Approach:**  
Generated **2000+ candidate features** through 12 distinct transformation methods applied to 68 baseline features. Implements comprehensive temporal aggregations (7/14/30-day windows) with multiple statistics (mean, max, std, min, skew, kurt). Uses quantile normalization, Box-Cox/Yeo-Johnson transforms, z-score standardization, and polynomial features. Note: Highest compliant submission scored 75.30 CFCS (rank 3 features); 86.85 CFCS submission was disqualified for using futures data in feature engineering.

**Actual Implementation:**

**The 12 Transformation Methods:**
1. **Quantile Normalization:** Rank-based scaling to [0,1]
2. **Box-Cox/Yeo-Johnson:** Variance stabilization, normality approximation
3. **Robust Scaling:** Median/IQR based, outlier resistant
4. **Power Transformations:** Square, sqrt, log
5. **Binning/Encoding:** Decile bins with one-hot encoding
6. **Lag Features:** Shifted values at 7, 14 days
7. **Rolling Statistics:** Mean, std, min, max over windows
8. **EWMA:** Exponential weighted moving averages (α=0.1, 0.3, 0.5)
9. **Difference Features:** 1d, 2d, 7d changes
10. **Cumulative Features:** cumsum, cummax, cummin
11. **Seasonal Decomposition:** Trend, seasonal, residual components
12. **Original Feature:** Untransformed baseline

**Selection Strategy:**
- **Stage 1 (Correlation Filtering):** Drops features with <400 significant correlations across all country-month buckets
- **Stage 2 (Forward Selection):** Starts with required columns, iteratively adds features only if CFCS increases
- **Per-Feature Optimization:** Tests all 12 transformation methods per baseline feature, selects best performer
- **CFCS-Aware Pruning:** Aggressive filtering improves sig_count percentage in CFCS formula

**Results:**
- **Compliant submission:** 75.30 CFCS using rank 3 features
- **Disqualified submission:** 86.85 CFCS (used futures data - rule violation)
- Top compliant features: Production-weighted risk scores, composite stress indices, rolling aggregations

#### Uniqueness
- **Optimization vs Prediction:** Constructs variables that maximize the specific scoring metric
- **Per-Feature Transform Selection:** Each feature gets its optimal transformation
- **Forecast Proxy:** Uses `shift(-30)` on climate data as "30-day weather forecast"
- Comprehensive transformation library covering every major statistical technique
- Explicit awareness of compliance—disabled non-compliant modules for final submission

#### Creativity
- Exhaustive exploration of transformation space (12 methods × multiple parameters)
- Direct optimization of the competition metric (CFCS)
- Strong feature selection reducing 1,471 to 50 high-quality features

#### Flaws
- **Disqualification Issue:** Highest scoring submission (86.85) violated rules by using futures data
- **Overfitting Risk:** Excessive feature generation (2000+) may overfit to training data
- **Computational Cost:** Testing 12 transformations per feature is resource-intensive
- **Interpretability:** Complex transformation pipelines difficult to explain to stakeholders

---

### 10. [ganeshstemx](https://www.kaggle.com/ganeshstemx)

#### Strategy: Non-Overlapping Temporal Quantile Binning

**Core Approach:**  
Segments the full time series into non-overlapping temporal bins (tertiles, quartiles, quintiles, sextiles, octiles, deciles) and calculates aggregated statistics within each bin. Generates 1,494 features by exhaustively exploring bin configurations and aggregation methods.

**Actual Implementation:**

**Bin Configurations:**
- TERTILE: 3 bins (~3-year chunks for 10-year dataset)
- QUARTILE: 4 bins (~2.5 years each)
- QUINTILE: 5 bins (~2 years each)
- SEXTILE: 6 bins (~1.67 years each)
- OCTILE: 8 bins (~1.25 years each)
- DECILE: 10 bins (~1 year each)

**Feature Construction:**
1. **Weighted Sums within Bins:** `climate_risk_wsum_{bin_type}_agg_...`
   - Production-weighted aggregations per temporal segment
2. **Compound Features:** Interaction terms
   - `drought + heat`
   - `drought + excess_precip`
   - Products and ratios
3. **Aggregation Types:** mean, sum, min, max, std within each bin
4. **"Non-Drought" Features:** Inverse signals capturing absence of drought
5. **Feature Selection:** Top 5 features by correlation/importance

**Top Selected Features:**
1. `climate_risk_wsum_quartile_agg_..._non_drought_med_sum_mean` (highest significance)
2. Weighted drought composites with quarterly bins
3. Compound drought-excess products

**Result:** Successfully reproduced submission with 1,494 features → 5 selected

#### Uniqueness
- **Temporal Quantile Binning:** Non-overlapping time segments instead of traditional rolling windows
- **Compound Interaction Terms:** `drought_excess_med_product` captures joint effects
- **"Non-Drought" Inverse Signal:** Explicit modeling of drought absence as informative

#### Creativity
- **Novel Binning Paradigm:** Different from standard time-series windowing approaches
- **Multi-Scale Temporal Resolutions:** Simultaneously captures yearly, quarterly, multi-year patterns

#### Flaws
- **Severe Overfitting Risk:** 1,494 features on limited time series data
- **Interpretation Difficulty:** Hard to explain which temporal window (year 3-6? quarters 2-3?) drives the signal
- **Potential Data Leakage:** Temporal binning may use future information within each bin if not carefully implemented
- **Dimensionality Curse:** Feature count >> sample size in time dimension

---

### 11. [kadircandrisolu](https://www.kaggle.com/kadircandrisolu)

#### Strategy: Ultra-Long Window Cumulative Drought Tracking

**Core Approach:**  
Focuses on long-term pattern detection using extended rolling windows [400,410,420,430 days] to capture multi-season trends. Implements standard baseline pipeline with emphasis on simplicity and robustness over complexity. Uses production-weighted risk scoring with fillna(1.0) to ensure all regions contribute.

**Feature Engineering:**
1. **Base Drought Score:** Weighted risk calculation: (medium + 2×high) / (total+ε)
2. **Production Weighting:** Multiply by `percent_country_production` (fillna 1.0)
3. **Country Aggregation:** Sum across regions for national-level signals
4. **Ultra-Long Windows:** Rolling mean and max for [400,410,420,430] days
5. **Momentum Features:** 1-day and 7-day changes plus acceleration (second derivative)
6. **Composite Indices:** Temperature (max of heat/cold), precipitation (max of drought/excess), overall, combined

**Selection:** Used beam search optimization on 2025 validation set, converged to single feature

**Result:** Strong performance with minimal features

**Top Feature:** `climate_risk_drought_weighted_country_cumsum`
- Year-plus rolling window captures annual cycles and multi-season persistence
- High significance rate and strong maximum correlation
- Demonstrates "less is more" philosophy

#### Uniqueness
- **Extreme Parsimony:** Single feature achieves strong results
- **Ultra-Long Windows:** 400-430 days captures nearly annual trends
- **Cumulative Tracking:** `cumsum` tracks regime persistence

#### Creativity
- "Less is More" philosophy
- Demonstrates that one well-chosen feature outperforms complex multi-feature approaches

#### Flaws
- **Single Feature Risk:** High dependence on one signal
- **Regime Fragility:** May break under market structure changes
- **Interpretation:** Hard to explain why cumsum specifically works

---

### 12. [ravi123a321at](https://www.kaggle.com/ravi123a321at)

#### Strategy: Phenology-Weighted Stress with Spatial Concentration

**Core Approach:**  
Domain-driven approach incorporating **phenology** (crop growth stages) to weight climate stressors by biological vulnerability windows. Combines temporal weighting with spatial concentration metrics (HHI) to capture both "when" and "where" risks matter most.

**Actual Implementation:**

**Feature Engineering Pipeline:**
1. **Baseline Features (22):**
   - Raw risk counts (Low/Med/High × Heat/Drought/Precip)
   - Basic temporal features (day of year, month)
2. **Country Aggregations:**
   - Mean, max, std across regions per country-date
   - Captures national-level risk patterns
3. **Spatial Concentration (HHI):**
   - Herfindahl-Hirschman Index: $HHI = \sum_{i=1}^{n} s_i^2$
   - Where $s_i$ = region i's share of total risk
   - Measures whether risk is concentrated (HHI→1) vs scattered (HHI→0)
4. **Phenology Weighting:**
   - Month-specific multipliers based on growth stages
   - Example: "Month 7 weight = 1.0" (critical flowering/pollination)
   - Months 5-9 weighted higher for Northern Hemisphere
5. **Monthly Aggregation:**
   - Phenology features aggregated to monthly means
   - Aligns with biological time scales
6. **Soil Moisture (Optional):**
   - External data from `external_soil_data.csv` (if available)
   - Graceful degradation if missing—code continues without it
7. **Feature Selection:** Correlation + importance filtering

**Total Features:** 33 (baseline + aggregations + phenology + HHI)

**Modeling:** Gradient boosting with feature importance ranking

#### Uniqueness
- **HHI Adaptation:** Applies Herfindahl-Hirschman Index (economics concentration metric) to climate risk distribution
- **Phenology-Driven Weighting:** Growth-stage-aware feature construction captures biological vulnerability windows
- **"When" × "Where":** Combines temporal (phenology) and spatial (HHI) dimensions

#### Creativity
- **Domain Science Integration:** Models *when* crops are vulnerable (flowering, grain fill)
- **Concentration Metric:** HHI captures whether risk is "localized crisis" vs "widespread problem"
- **Biological Time Scales:** Monthly aggregation respects plant physiology

#### Flaws
- **Heuristic Weights:** Manual phenology weights instead of data-driven learning
- **Missing External Data:** `external_soil_data.csv` not provided in submission
- **Row Count Issue:** 320,661 output rows vs required 219,161 (needs post-processing filter)
- **Northern Hemisphere Bias:** Phenology weights optimized for US corn belt, may not transfer to Brazil

---

### 13. [osamurai](https://www.kaggle.com/osamurai)

**CFCS Score:** 72.33 (Rank 72)

#### Strategy: Comprehensive Feature Engineering with LightGBM-Based Transformations

**Core Approach:**  
Systematic feature engineering pipeline covering production-weighted risks, seasonal interactions, temporal aggregations, and non-linear transformations. Uniquely uses LightGBM models to generate prediction-based features (predicting futures from climate data only) as complex non-linear transformations of climate signals. Two-stage hyperparameter optimization: L9 Orthogonal Array for global search + Optuna for local refinement.

**Feature Engineering Pipeline:**

1. **Base Risk Scores:** Weighted risk calculation per category
   - `(medium + 2×high) / (low + medium + high + ε)`
   - Additional high_ratio and elevated_ratio features
   
2. **Production-Weighted Risks:** Regional economic importance
   - Risk scores × `percent_country_production`
   - Emphasizes major production regions

3. **Composite Stress Indices:**
   - Temperature stress: max(heat, cold)
   - Precipitation stress: max(excess_precip, drought)
   - Overall/combined/total stress: max, mean, sum across all categories

4. **Seasonal & Harvest Period Features:**
   - Risk × harvest_period interactions (Planting, Growing, Vegetative, Reproductive)
   - Growing season flag for active growth periods
   - Summer (Jun-Aug) and winter (Dec-Feb) seasonal multipliers

5. **Rolling Window Statistics:** 7, 14, 30-day windows
   - Moving averages, rolling max/min, rolling std
   - Per region and risk category

6. **Lag Features:** 1, 3, 7, 14, 21 days
   - Captures delayed market responses

7. **Momentum & Acceleration:**
   - First differences: 1d, 7d, 14d changes
   - Second differences: acceleration (change in change)

8. **Country & Global Aggregations:**
   - Country-level: mean, max, std, min across regions
   - Global-level: aggregated stats across US, Brazil, Argentina, China

9. **Non-linear Transformations:**
   - Squared and cubed risk scores (convex damage functions)
   - Log and sqrt transformations
   - Binary threshold flags: high_risk (>1.0), extreme_risk (>1.5)

10. **Interaction Features:**
    - Heat × drought (compounding dry-heat)
    - Cold × excess_precip (compounding cold-wet)
    - Temperature × precipitation stress
    - Four-way product of all risk scores
    - Temperature-to-precipitation ratio

11. **LightGBM Prediction Features (Novel):**
    - Trained 17 LightGBM models (one per futures_* target)
    - Input: Only `climate_risk_*` features
    - Output: `climate_risk_lgb_pred_*` features
    - Additional: Lags (1, 3, 7, 14d), differences, rolling averages of predictions
    - **Rationale:** Complex non-linear mappings from climate to futures as derived features

**Hyperparameter Optimization:**

**Stage 1 - L9 Orthogonal Array (Global Search):**
- 3 levels × 4 factors = 9 experiments
- Factors: num_leaves [31,63,127], learning_rate [0.01,0.03,0.05], feature_fraction [0.6,0.7,0.8], min_child_samples [10,20,50]
- Efficient global parameter space exploration

**Stage 2 - Optuna (Local Refinement):**
- 30 trials starting from best L9 result
- Narrowed search range around promising region

**Final Model:**
- Algorithm: LightGBM (gbdt)
- Cross-validation: 10-fold TimeSeriesSplit
- Early stopping: 50 rounds (max 500)
- Ensemble: Averaged predictions across folds

#### Uniqueness
- **LightGBM as Feature Generator:** Uses ML predictions as derived climate features (compliance-aware approach)
- **Two-Stage Optimization:** L9 orthogonal array + Optuna for efficient hyperparameter search
- **Comprehensive Coverage:** Combines domain knowledge (harvest periods), statistical methods (rolling windows), and ML transformations
- **Interaction Richness:** Extensive cross-category risk interactions

#### Creativity
- **Prediction-Based Features:** Novel interpretation of "feature engineering" - predictions are complex transformations of climate data
- **Orthogonal Array Design:** Applies DOE (Design of Experiments) methodology for hyperparameter search
- **Cascade Architecture:** Base features → composite features → prediction features → final model

#### Compliance Note
- **LightGBM features validated:** Models trained using only `climate_risk_*` inputs, predicting `futures_*` targets
- Information flow: `climate_risk_*` → LightGBM → `climate_risk_lgb_pred_*` (one-directional)
- No futures columns used as input features
- Functionally equivalent to non-linear feature transformation

#### Flaws
- **Computational Complexity:** Multiple LightGBM models + hyperparameter search = long runtime
- **Potential Overfitting:** Prediction-based features may memorize training patterns
- **Interpretability:** LightGBM predictions are black-box transformations
- **Moderate Score:** Despite comprehensive approach, 72.33 CFCS suggests diminishing returns from complexity

---

### Approach Categories

| Category | Participants | Characteristics |
|----------|--------------|-----------------|
| **Domain-Driven** | cmasch, ravi123a321at | Uses agronomic/biological knowledge (GDD, phenology) |
| **Technical Analysis** | bluetriad, PxlPau | Treats climate like financial time series |
| **Exhaustive Search** | limyuxin, DragonAJA, yukanglimofficial | Brute force search through parameter space |
| **Statistical Binning** | aaaml007, ganeshstemx | Quantile-based temporal aggregation |
| **Minimalist** | kadircandrisolu | Single feature focused on cumsum |
| **Deep Learning** | Mr RRR | AutoEncoder for compression |
| **Economic Weighting** | cg, chetank99 | Production-weighted signals |


### Top Performing Approaches

**Winners (Final scores pending):**
1. **limyuxin** - 1st Place: Brute Force Factor Mining
2. **DragonAJA** - 2nd Place: Massive Feature Engineering & Optimization
3. **yukanglimofficial** - 3rd Place: Deep Sweep Grid Search

**Notable Approaches:**
- **kadircandrisolu** - Single cumulative drought feature
- **cg** - Production-weighted global risk
- **PxlPau** - Power law + hemispheric gating
- **cmasch** - Agronomic 439 features
- **DragonAJA** - 1,471 features, compliant run
- **aaaml007** - Quantile binning

### Key Metrics Comparison

| Participant | Features Generated | Features Selected | Key Innovation |
|-------------|-------------------|-------------------|----------------|
| kadircandrisolu | Multiple | 1 | Ultra-long cumsum (400-430d) |
| DragonAJA | 1,471 | 50 | 12 transformation methods |
| cmasch | 439 | 83 | Agronomic GDD features |
| ganeshstemx | 1,494 | 5 | Temporal quantile binning |
| bluetriad | 127 | 64 | Technical indicators |
| cg | 68 | 30 | Production weighting |

---

## Key Insights

### What Works

1. **Cumulative Signals Beat Instantaneous:** `cumsum` features consistently outperform point-in-time values. Accumulated drought stress is more predictive than single-day readings.

2. **Drought Dominates:** Across all successful submissions, drought-related features consistently rank highest. Heat stress, excess precipitation, and cold stress are secondary.

3. **Seasonal Gating is Crucial:** Features filtered to relevant growing seasons (hemispheric gating) dramatically improve signal-to-noise ratio. Correlations jump from ~0.5 to >0.8 with proper seasonal filtering.

4. **Production Weighting Adds Value:** Weighting by `percent_country_production` ensures major production regions (Iowa, Mato Grosso) appropriately influence the signal.

5. **Less Can Be More:** kadircandrisolu's single feature (81.85) outperforms many 1000+ feature approaches. Parsimony with the right feature beats complexity with many mediocre ones.

6. **Non-Linearity Helps:** Power transformations (`risk^2`, `risk^3`) and cumulative sums capture non-linear relationships that linear aggregations miss.

### What Doesn't Work

1. **Pure Brute Force:** Exhaustive searches without domain guidance (limyuxin's "no refined candidates") often fail to find meaningful signals.

2. **Excessive Features:** 300-1400+ features often reduce to <100 useful ones. Most are noise.

3. **Short Windows Only:** Very short windows (7-14 days) are noisier; longer windows (30-430 days) capture meaningful persistence.

4. **Ignoring Seasonality:** Averaging across all months dilutes the signal. Climate risk matters only during growing seasons.

### Rule Violations to Avoid

*Note: Disqualified participants have been removed from this report.*

1. **Direct futures copy**: Immediately disqualified
2. **ML predictions as features**: Target leakage through cross-validation
3. **Using futures for feature construction** (not selection): Only correlation evaluation is allowed

### Reproducibility Keys

1. **Self-contained code:** Missing dependencies (GPCH's `src/`) cause failures
2. **Graceful degradation:** Handle missing external data (bluetriad, ravi handled this well)
3. **Exact row counts:** Must match sample submission (219,161 rows)
4. **Clear documentation:** README with step-by-step instructions essential

---

> **Last Updated:** 2026-02-10  
> **Reviewed By:** Helios AI
