# Comprehensive Methodology Report: Helios Corn Futures Climate Challenge

> **Generated:** 2026-02-09  
> **Competition:** Helios Corn Futures Climate Challenge  
> **Participants Reviewed:** 16 (2 disqualified participants removed)

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Participant Methodologies](#participant-methodologies)
   - [Competition Winners](#competition-winners)
     - [1st Place: limyuxin](#-1st-place---limyuxin)
     - [2nd Place: DragonAJA](#-2nd-place---dragonaja)
     - [3rd Place: yukanglimofficial](#-3rd-place---yukanglimofficial)
   - [Other Participants](#other-participants)
   - [cmasch](#1-cmasch)
   - [bluetriad](#2-bluetriad)
   - [GPCH](#3-gpch)
   - [Mr RRR](#4-mr-rrr)
   - [PxlPau](#5-pxlpau)
   - [aaaml007](#6-aaaml007)
   - [cg](#7-cg)
   - [chetank99](#8-chetank99)
   - [ezberch](#9-ezberch)
   - [ganeshstemx](#10-ganeshstemx)
   - [kadircandrisolu](#11-kadircandrisolu)
   - [ravi123a321at](#12-ravi123a321at)
3. [Comparative Analysis](#comparative-analysis)
4. [Key Insights](#key-insights)

---

## Executive Summary

This report documents the methodologies used by participants in the Helios Corn Futures Climate Challenge Kaggle competition. Each participant attempted to engineer climate risk features that correlate with corn futures prices. The approaches range from simple rolling averages to sophisticated AutoEncoder neural networks and exhaustive grid searches.

**Competition Winners:**
- 🥇 **1st Place**: limyuxin - Brute Force Factor Mining
- 🥈 **2nd Place**: DragonAJA - Massive Feature Engineering & Optimization
- 🥉 **3rd Place**: yukanglimofficial - Deep Sweep Grid Search

Note: Some participants were disqualified for rule violations and have been removed from this report.

### Quick Reference Table

| Rank | Participant | Strategy | CFCS Score | Status | Uniqueness |
|------|-------------|----------|------------|--------|------------|
| 🥇 1st | limyuxin | Brute Force Factor Mining | TBD | WINNER | Grid search methodology, PCA proxy |
| 🥈 2nd | DragonAJA | Massive Feature Engineering & Optimization | TBD | WINNER | 12 transformation methods, CFCS-specific optimization |
| 🥉 3rd | yukanglimofficial | Deep Sweep Grid Search | TBD | WINNER | Systematic caching + parameter sweep |
| - | limyuxin | Brute Force Factor Mining | - | PASS | Grid search methodology |
| - | cmasch | Massive Agronomic Feature Engineering | - | PASS | GDD, phenology-based |
| - | bluetriad | Technical Analysis of Climate | - | PASS | EMA, RSI-like indicators |
| - | DragonAJA | Massive Feature Engineering & Optimization | - | PASS | 12 transformation methods |
| - | GPCH | Modular Package | - | FAIL | Missing dependencies |
| - | Mr RRR | AutoEncoder Feature Extraction | - | PASS | Neural network compression |
| - | PxlPau | Signal Sharpening | - | PASS | Power law, hemispheric gating |
| - | aaaml007 | Quantile Binning | - | PASS (Warning) | Exhaustive statistical binning |
| - | cg | Production-Weighted Global Risk | - | PASS | Economic weighting |
| - | chetank99 | Relative Risk Ratios | - | PASS | Seasonal gating, ratio signals |
| - | ezberch | Brute Force Correlation Mining | - | PASS | Macro-climate linkage |
| - | ganeshstemx | Quantile Binning | - | PASS | Temporal quantile bins |
| - | kadircandrisolu | Cumulative Drought Stress | - | PASS | Single feature, ultra-long windows |
| - | ravi123a321at | Phenology-Weighted Stress | - | PASS | HHI concentration metric |
| - | yukanglimofficial | Deep Sweep Grid Search | - | PASS | Systematic caching + search |

---

## Participant Methodologies

---

## Competition Winners

### 🥇 1st Place - limyuxin

**CFCS Score:** TBD

**Strategy:** Brute Force Factor Mining

**Core Approach:**  
Treats feature discovery as a search problem rather than a domain modeling problem. Exhaustively searches a massive parameter space (window sizes, shift lags, aggregation methods) to discover climate risk features that correlate with futures prices.

**Feature Engineering Steps:**
1. **Stress Primitives:** Maps regional low/medium/high risk-location counts into severity-style ratios
2. **Composites:** Creates wet–dry imbalance, temperature/precipitation stress maxima, overall stress
3. **Country-Level Signals:** Expresses concepts under multiple production-weighting schemes
4. **Surprise Variables:** Removes predictable seasonality with day-of-year z-scores
5. **Temporal Transforms:** Tests lags (timing), windowed aggregation (short shocks vs persistent regimes), and non-linear transforms

**Selection Method:**
- Uses PCA (PC1) of futures panel within each country-month bucket as screening proxy
- Refines strongest candidates with tight local sweeps around lag/window settings
- Prioritizes using CFCS metric across multiple recent-year slices

**Uniqueness:**
- **"Mining" vs "Engineering":** Treats the problem as a search task rather than a domain modeling task
- **Grid Search Methodology:** Tests thousands of (window, shift, aggregation) combinations systematically
- **Year-Demeaned Sanity Check:** Emphasizes candidates that stay broad and stable
- Uses **first principal component (PC1)** as a fast proxy for futures co-movement
- Explicit "surprise" version of every signal using z-scores

**Why This Won:**
- Systematic exploration of parameter space without domain assumptions
- Novel use of PCA for efficient correlation proxy
- Robust validation across multiple time periods
- Avoids overfitting through year-demeaned sanity checks

---

### 🥈 2nd Place - DragonAJA

**CFCS Score:** TBD

**Strategy:** Massive Feature Engineering & Optimization

**Core Approach:**  
Generated **1,471 features** using 12 systematically applied transformation methods, then used a bespoke "CFCS-Specific Feature Selection" algorithm to maximize the competition metric directly.

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
- **Stage 1 (Statistical Filtering):** Keep features with sig_pct ≥ 5%
- **Stage 2 (CFCS Ranking):** Select top N by CFCS score
- **Per-Feature Optimization:** Test ALL transformations per feature, pick best

**Results:**
- Compliant run completed successfully
- Top features: `excess_precip_cumsum`, `unseasonably_cold_cumsum`, `drought_cumsum`

**Uniqueness:**
- **Optimization vs Prediction:** Constructs variables that maximize the specific scoring metric
- **Per-Feature Transform Selection:** Each feature gets its optimal transformation
- **Forecast Proxy:** Uses `shift(-30)` on climate data as "30-day weather forecast"
- Comprehensive transformation library covering every major statistical technique
- Explicit awareness of compliance—disabled non-compliant modules for final submission

**Why This Won:**
- Exhaustive exploration of transformation space (12 methods × multiple parameters)
- Direct optimization of the competition metric (CFCS)
- Strong feature selection reducing 1,471 to 50 high-quality features
- Compliance-first approach ensuring valid submission

---

### 🥉 3rd Place - yukanglimofficial

**CFCS Score:** TBD

**Strategy:** Deep Sweep Grid Search

**Core Approach:**  
Systematic "Deep Sweep" Grid Search methodology. Caches base aggregations then iteratively sweeps thousands of parameter combinations (windows, shifts, transforms).

**Implementation:**
1. **Cache Building:** Pre-compute base aggregations (`cache/cd_cache_kaggle.npz`)
2. **Baseline Scan:** `baseline_scan_all.csv` generation
3. **Grid Search:** Systematic exploration of parameter space
4. **Feature Selection:** Based on correlation ranking

**Uniqueness:**
- **Systematic Search:** Structured grid search over hyperparameters
- **Caching:** Efficient reuse of expensive aggregations
- **Reproducible:** Clear separation of compute vs evaluation
- Cache mechanism prevents leakage by pre-calculating base features

**Why This Won:**
- Highly systematic and reproducible approach
- Efficient caching strategy reduces computation time
- Comprehensive parameter space exploration
- Clean separation of feature generation and evaluation phases

---

## Other Participants

---

### 1. cmasch

#### Strategy: Massive Agronomic Feature Engineering

**Core Approach:**  
A "Massive Feature Engineering" strategy deeply rooted in agronomy. Generated **439 features** focusing on Growing Degree Days (GDD), specific biological growth stages (Pollination, Grain Fill), and cumulative stress counters.

**Feature Engineering Steps:**
1. **Base Risk Scores:** 20 features from core climate risk columns
2. **Time-Series Features:** 202 features using rolling windows, lags, EMAs
3. **Stress Day Counters:** Counting exact days above biological thresholds
4. **Interaction Features:** Heat × drought, cold × precipitation
5. **GDD Proxy Features:** Temperature-based "heat unit" accumulation
6. **Anomaly Scores:** Deviation from seasonal norms
7. **Z-Score Normalization:** Standardized risk levels
8. **Non-linear Transforms:** Log, sqrt, squared terms
9. **Volatility Features:** Rolling standard deviations
10. **Persistence Features:** Duration of stress conditions
11. **Regional Comparison Features:** Relative risk across regions
12. **Distribution Features:** Skewness, kurtosis of risk distributions

**Parameters:**
- Lag periods: 7, 14, 30, 60, 90, 270, 365, 400 days
- Stress thresholds: Heat >0.75, Drought >0.63, Excess Precip >0.5, Cold >0.5
- Final selection: 83 features

**External Data (Optional):**
- NOAA/ERA5 Weather Data
- FRED Economic Data (Dollar Index, Oil Price, VIX, etc.)

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

### 2. bluetriad

#### Strategy: Technical Analysis of Climate

**Core Approach:**  
Treats climate risk time series like financial assets, applying technical indicators such as EMA, Volatility, Momentum, and RSI-like indicators to climate risk scores.

**Feature Engineering Steps:**
1. **Base Risk Scores:** 8 features from weighted risk counts
2. **Rolling Features:** Moving averages (14d, 30d, 60d)
3. **Lag Features:** Historical values at various intervals
4. **EMA Features:** Exponential moving averages with different spans
5. **Volatility Features:** Rolling standard deviations
6. **Cumulative Features:** `cumsum` for accumulated stress
7. **Non-linear Features:** Squared and cubed terms
8. **Interaction Features:** Cross-risk type interactions
9. **Seasonal Features:** Month-based dummies
10. **Momentum Features:** Rate of change indicators
11. **Country Aggregations:** National-level statistics

**Selection Method:**
- Removes features with 0 significant correlations
- Filters based on "Significant Correlation Count" (sig_count)
- Final: 64 features from 127 generated

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

#### Strategy: AutoEncoder Feature Extraction

**Core Approach:**  
A sophisticated 5-stage pipeline culminating in an **AutoEncoder (AE)** neural network for feature extraction/compression. The workflow: Data Processing → Feature Engineering → Selection (CFCS Top4) → AutoEncoder Training → Submission.

**Feature Engineering Steps:**
1. **Seasonal/Time Features:** `day_of_year`, `quarter`, seasonal `sin/cos`, hemispheric shift
2. **Risk Intensity Metrics:** `score / high_share / balance / entropy`
3. **Time-Series Stats:** Rolling mean/max/volatility, lags, EMA, momentum/acceleration
4. **Event Features:** Threshold persistence, event AUC, spike detection
5. **Combinations/Interactions:** Temperature/precip stress, diffs, ratios
6. **Country-Level Aggregation:** Weighted stats and concentration by country × date

**AutoEncoder Architecture:**
- Input: Selected climate risk features
- Output: Compressed latent representation as final risk score
- Stored as: `.pth` model + `.joblib` scaler

#### Uniqueness
- **Deep Learning for Feature Extraction:** Only submission using Neural Networks (AutoEncoders)
- **Latent Compression:** Finds non-linear representations that linear averages miss

#### Creativity
- Professional-grade 5-notebook pipeline structure
- Uses PyTorch for AE implementation
- Combines traditional feature engineering with deep learning

#### Flaws
- **Complexity:** 5-step pipeline is fragile to reproducibility errors
- **Black Box:** Hard to interpret what the AE learns
- **Compute Requirements:** Requires GPU for efficient training

---

### 5. PxlPau

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

### 6. aaaml007

#### Strategy: Quantile Binning

**Core Approach:**  
Generates massive numbers of features based on tertiles, quartiles, quintiles, deciles, etc., then aggregates risk scores within those bins.

**Feature Engineering Steps:**
1. **Weighted Risk Scores:** Raw counts → single `risk_score` weighted by production
2. **Moving Averages:** 7, 14, 30, 60, 90 days
3. **Rolling Max:** Peak stress detection
4. **EMA:** Recent event emphasis
5. **Lag Features:** 7 to 90 days
6. **Volatility:** Rolling std (14-46 days) for "climate instability"
7. **Cumulative Stress:** Rolling sums (30-90 days)
8. **Country Aggregations:** 155 features at national level

**Result:** Reproduced successfully

#### Uniqueness
- **Quantile Binning:** Exhaustive aggregation over every conceivable time window
- **Statistical Focus:** Pure mathematical approach without domain assumptions

#### Creativity
- Systematic exploration of statistical aggregation methods

#### Flaws
- **Row Count Mismatch:** 214,139 vs required 219,161
- **"More is not always better":** 300+ features barely outperform random baseline
- **Complexity without causality is noise**

---

### 7. cg

#### Strategy: Production-Weighted Global Risk

**Core Approach:**  
Creates a "Global Aggregate Risk" signal using production weights. Hypothesis: global aggregate production-weighted risk has more impact than individual country risks.

**Feature Engineering Steps:**
1. **Base Risk Scores:** 8 features
2. **Composites:** 12 features (temperature stress, precipitation stress, overall/combined)
3. **Rolling Features:** 36 features (7, 14, 30 day windows)
4. **Momentum Features:** 48 features (changes and acceleration)
5. **Country Aggregations:** 68 features total

**Selection:** Top 30 by correlation

**Result:** Reproduced successfully

#### Uniqueness
- **Economic Weighting:** Uses `percent_country_production` for signal construction
- **Global Aggregate Risk:** Focuses on worldwide production impact

#### Creativity
- Domain-aware weighting reflects real market dynamics
- Major producers (Iowa = 16% of US) get proportionate weight

#### Flaws
- **Static Weights:** Production shares hardcoded, not dynamic per year
- **Simple Selection:** Top-30 by correlation may miss non-linear relationships

---

### 8. chetank99

#### Strategy: Relative Risk Ratios

**Core Approach:**  
Focuses on "Relative Risk" (local/global ratios) and "Market Synchronization". Hypothesis: high local risk matters *more* when rest of world is stable (high ratio), or when multiple competitors fail (synchronization).

**Feature Engineering Steps:**
1. **Ratio Signals:** `local_risk / global_risk`
2. **Harvest Season Gating:** Strict filtering to months 1, 2, 11, 12 (Southern Hemisphere)
3. **Non-linear Terms:** Cubed ratios (`ratio^3`) for explosive panic modeling
4. **Synchronization Features:** Multiple competitor failure detection
5. **Monthly Features:** Month-specific risk features

**Results:**
- Strong correlations achieved
- Top months: December, November, January

#### Uniqueness
- **Global Context:** Asks "How bad is this region *compared to the world*?"
- **Harvest Gating:** Strict seasonal windows yield correlations >0.8
- **Non-linear Panic:** Cubed terms model explosive market reactions

#### Creativity
- Ratio signals capture trade flow disruption potential
- Relativity > Absolute values

#### Flaws
- **Feature Sparsity:** Features are 0 outside harvest window (8 months blind)
- **Row Count Warning:** 219,531 vs 219,161 (+370 rows)
- **Single Hemisphere Focus:** Needs complementary Northern Hemisphere features

---

### 9. ezberch

#### Strategy: Brute Force Correlation Mining

**Core Approach:**  
Generates feature combinations (aggregations, lags, external indices like ONI/PDO) and scans them in parallel batches for high correlation with futures.

**Feature Engineering Steps:**
1. **Aggregation Features:** Various window sizes and methods
2. **Lag Features:** Time-shifted values
3. **External Climate Indices:** ONI (El Niño), PDO incorporation
4. **Parallel Processing:** Batch-based correlation scanning

**External Data:**
- `external_oni.csv` (Oceanic Niño Index)
- `external_indices.csv` (Climate oscillation indices)

**Status:** Long-running execution (partial verification)

#### Uniqueness
- **Macro-Climate Linkage:** Links local crop stress to global climate oscillations (ENSO/PDO)

#### Creativity
- Incorporates macro-climate drivers (El Niño/La Niña)—scientifically sound

#### Flaws
- **Brute Force Risk:** High spurious correlation risk without p-value corrections
- **Extremely Long Runtime:** >2 hours for full execution
- **Multiple Testing Problem:** No apparent correction for hypothesis count

---

### 10. ganeshstemx

#### Strategy: Temporal Quantile Binning

**Core Approach:**  
Segments time series into multiple temporal bins (tertile, quartile, quintile, decile, etc.) and calculates aggregated statistics within each bin. Generated **1,494 features**.

**Bin Configurations:**
- TERTILE: 3 bins (~3-year chunks)
- QUARTILE: 4 bins
- QUINTILE: 5 bins
- SEXTILE: 6 bins
- OCTILE: 8 bins
- DECILE: 10 bins

**Feature Types:**
- Weighted sums within bins
- Compound features (drought+heat, drought+excess precip)
- Mean, sum, min, max aggregations

**Top Features (selected 5):**
1. `climate_risk_wsum_quartile_agg_..._non_drought_med_sum_mean` (high significance)
2. Weighted drought composites
3. Compound drought-excess products

**Result:** Reproduced successfully

#### Uniqueness
- **Temporal Quantile Bins:** Non-overlapping time segments instead of rolling windows
- **Compound Features:** Interaction terms like `drought_excess_med_product`
- **"Non-Drought" Feature:** Captures absence of drought as inverse signal

#### Creativity
- Novel binning approach different from traditional rolling windows

#### Flaws
- **Overfitting Risk:** 1,494 features is aggressive
- **Interpretation Difficulty:** Hard to identify which temporal window drives signal
- **Potential Leakage:** Temporal binning may use future information within each bin

---

### 11. kadircandrisolu

#### Strategy: Cumulative Drought Stress

**Core Approach:**  
Ultra-minimalist—focuses on long-term accumulation of drought risk (400-430 day windows) weighted by production share. Uses **only ONE final feature**.

**Feature Engineering:**
1. **Base Drought Score:** Weighted risk calculation
2. **Production Weighting:** Multiply by `percent_country_production`
3. **Country Aggregation:** Sum across regions
4. **Ultra-Long Cumsum:** 400-430 day cumulative sum

**Selection:** Beam search optimization converged to single feature

**Result:** Strong performance with minimal features

**Top Feature:** `climate_risk_drought_weighted_country_cumsum`
- High significance rate
- Strong maximum correlation

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

### 12. ravi123a321at

#### Strategy: Phenology-Weighted Stress

**Core Approach:**  
Domain-driven approach incorporating **phenology** (plant growth stages) to weight climate stressors. Focuses on "drought exposure" during critical growing months and spatial concentration.

**Feature Engineering Steps:**
1. **Baseline Features:** 22 base risk features
2. **Country Aggregation:** Mean, max, std across regions
3. **Spatial Concentration (HHI):** Herfindahl-Hirschman Index for risk distribution
4. **Phenology Weighting:** Month-specific weights (e.g., "Month 7 weight = 1.0")
5. **Monthly Aggregation:** Phenology features aggregated to monthly means
6. **Soil Moisture (Optional):** External data integration (graceful degradation if missing)

**Total Features:** 33

**HHI Formula:**
$$HHI = \sum_{i=1}^{n} s_i^2$$
where $s_i$ is region i's share of total risk

#### Uniqueness
- **HHI (Herfindahl-Hirschman Index):** Adapts economic concentration metric to climate risk
- **Phenology-Driven Weighting:** Timing-aware feature construction

#### Creativity
- Domain science approach—models *when* crops are vulnerable
- HHI captures whether risk is concentrated vs scattered

#### Flaws
- **Heuristic Weights:** Manual phenology weights instead of learned
- **Missing External Data:** `external_soil_data.csv` not included
- **Row Count Issue:** 320,661 output needs filtering

---

## Comparative Analysis

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

> **Document Version:** 2.0  
> **Last Updated:** 2026-02-09  
> **Changes:** Removed CFCS scores (pending final evaluation), removed disqualified participants, added winner placeholders  
> **Reviewed By:** Competition Verification Team
