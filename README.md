# 🌽 Helios Corn Futures Climate Risk Challenge

[![Kaggle Competition](https://img.shields.io/badge/Kaggle-Competition-20BEFF?style=flat&logo=kaggle)](https://www.kaggle.com/competitions/forecasting-the-future-the-helios-corn-climate-challenge)
[![Competition Status](https://img.shields.io/badge/Status-Completed-success)](https://www.kaggle.com/competitions/forecasting-the-future-the-helios-corn-climate-challenge/leaderboard)
[![Prize Pool](https://img.shields.io/badge/Prize%20Pool-$7,500-gold)](https://www.kaggle.com/competitions/forecasting-the-future-the-helios-corn-climate-challenge/overview)

**Turn weather wisdom into trading gold!** This repository contains participant solutions, code, and results from the Helios Corn Futures Climate Risk Challenge - a Kaggle competition that challenged data scientists to leverage climate risk data for predicting corn futures price movements.

## 📊 Competition Overview

**Competition Period:** December 17, 2025 - January 30, 2026  
**Participants:** 612 Entrants | 162 Participants | 153 Teams | 2,880 Submissions

### The Challenge

Agricultural markets are fundamentally driven by weather, but the relationship between climate conditions and futures prices is complex and often non-linear. This competition challenged participants to:

- Develop creative approaches to transform Helios's proprietary climate risk data into signals
- Find innovative methods to strengthen correlations between weather patterns and commodity markets
- Engineer features that show stronger correlations with corn futures prices than baseline methods

### What Made This Unique

- **Proprietary Climate Intelligence**: Access to Helios's advanced climate risk model
- **Real Economic Impact**: Climate risks pre-classified based on actual crop tolerance thresholds
- **Global Scale**: Data spanning major corn-producing regions worldwide
- **Production Weighting**: Regional market share data for economic impact modeling
- **Multi-dimensional Risk**: Four distinct climate risk categories (heat, cold, drought, excess precipitation)

## 🏆 Prizes & Winners

**Total Prize Pool: $7,500**

- 🥇 **1st Place** - $4,000
- 🥈 **2nd Place** - $2,000
- 🥉 **3rd Place** - $1,500

Plus a chance to earn a full-time or internship position at [Helios AI](https://www.helios.sc)!

### 🏅 Competition Winners & Results

**Final Leaderboard (Top 3):**
- 🥇 **1st Place**: [limyuxin](https://www.kaggle.com/limyuxin) - **88.82 CFCS**
- 🥈 **2nd Place**: [DragonAJA](https://www.kaggle.com/dragonaja) - **86.85 CFCS**  
- 🥉 **3rd Place**: [yukanglimofficial](https://www.kaggle.com/yukanglimofficial) - **84.20 CFCS**

**📊 Comprehensive Methodology Report**

Want to understand how winners achieved their results? Check out our detailed analysis:

**[📖 COMPETITION_WINNERS_AND_METHODOLOGIES.md](solutions/COMPETITION_WINNERS_AND_METHODOLOGIES.md)**

This comprehensive report includes:
- **Detailed Winner Strategies**: In-depth breakdowns of 1st, 2nd, and 3rd place approaches
- **16 Participant Solutions**: Complete methodology analysis for all major participants
- **Quick Reference Table**: Easy comparison of strategies, scores, and unique approaches
- **Feature Engineering Techniques**: Specific implementation details for each method
- **Code Analysis**: Actual code patterns and techniques used
- **Comparative Analysis**: What worked, what didn't, and why
- **Key Insights**: Lessons learned from the competition

*From brute-force grid searches to neural network autoencoders, this report documents every major approach attempted in the challenge.*

## 📈 Evaluation Metric

Submissions were ranked using the **Climate-Futures Correlation Score (CFCS)** - a composite metric combining:

```
CFCS = (0.5 × Avg_Sig_Corr_Score) + (0.3 × Max_Corr_Score) + (0.2 × Sig_Count_Score)
```

**Score Components:**
- **Avg_Sig_Corr_Score (50% weight)**: Average of significant correlations (|corr| ≥ 0.5)
- **Max_Corr_Score (30% weight)**: Maximum absolute correlation discovered
- **Sig_Count_Score (20% weight)**: Percentage of correlations that are significant

## 📁 Repository Structure

```
├── docs/                           # Competition documentation
│   ├── dataset_description.md      # Dataset details
│   ├── overview.md                 # Competition overview
│   └── rules.md                    # Competition rules
├── solutions/                      # Participant solutions
│   ├── COMPETITION_WINNERS_AND_METHODOLOGIES.md  # 📊 Comprehensive methodology report
│   ├── aaaml007/                   # Solution by participant
│   ├── ardi/
│   ├── bluetriad/
│   ├── cg/
│   ├── chetank99/
│   ├── cmasch/
│   ├── DragonAJA/
│   ├── ezberch/
│   ├── ganeshstemx/
│   └── ...                         # More participant solutions
├── evaluate.py                     # Evaluation script
└── README.md                       # This file
```

## 🚀 Getting Started

### Download the Dataset

To download the competition dataset, you'll need the Kaggle API installed:

```bash
pip install kaggle
```

Then download the dataset:

```bash
kaggle competitions download -c forecasting-the-future-the-helios-corn-climate-challenge
```

### Evaluating a Submission

You can evaluate any submission CSV file using the provided evaluation script:

```bash
python3 evaluate.py path/to/submission.csv
```

Example:
```bash
python3 evaluate.py solutions/bluetriad/submission.csv
```

This will output:
- CFCS Score
- Component breakdown (Avg Significant Correlation, Max Correlation, Significant Count)
- Top 10 significant correlations

## 🔄 Reproducing Participant Solutions

### Important: Kaggle Environment Differences

⚠️ **All participant solutions were developed for the Kaggle notebook environment** and require modifications to run locally:

#### 1. **Data Loading Differences**

**On Kaggle (original code):**
```python
# Kaggle notebooks use predefined input paths
df = pd.read_csv('/kaggle/input/forecasting-the-future-the-helios-corn-climate-challenge/corn_climate_risk_futures_daily_master.csv')
```

**Local environment (modified):**
```python
# Download dataset first, then use local paths
df = pd.read_csv('./data/corn_climate_risk_futures_daily_master.csv')
```

#### 2. **Evaluation Method Differences**

**On Kaggle:**
- Submissions are uploaded to Kaggle's platform
- Evaluation runs on private test set with hidden futures data
- Instant CFCS score feedback through leaderboard

**Local environment:**
- Use `evaluate.py` script provided in this repository
- Requires merging your submission with the master dataset
- Manual scoring using the CFCS formula

### Step-by-Step Reproduction Guide

#### Prerequisites

```bash
# Install required packages (adjust based on solution)
pip install pandas numpy scikit-learn
pip install matplotlib seaborn  # For visualization
pip install lightgbm xgboost     # If solution uses gradient boosting
pip install torch                # If solution uses neural networks (e.g., Mr RRR)
```

#### Step 1: Download and Extract Dataset

```bash
# Download competition dataset
kaggle competitions download -c forecasting-the-future-the-helios-corn-climate-challenge

# Extract files
unzip forecasting-the-future-the-helios-corn-climate-challenge.zip -d data/

# Verify required files
ls data/
# Should see:
#   - corn_climate_risk_futures_daily_master.csv
#   - corn_regional_market_share.csv
#   - sample_submission.csv
```

#### Step 2: Modify Solution Code for Local Environment

**Required Code Changes:**

1. **Update file paths:**
```python
# Original Kaggle path
'/kaggle/input/forecasting-the-future-the-helios-corn-climate-challenge/...'

# Change to local path
'./data/...'
```

2. **Handle missing external data:**
```python
# Some solutions use external data (e.g., climate indices)
# These may need to be downloaded separately or gracefully skipped
try:
    external_data = pd.read_csv('./data/external_climate_data.csv')
except FileNotFoundError:
    print("External data not found, using base features only")
    external_data = None
```

3. **Adjust output paths:**
```python
# Original: saves to Kaggle's working directory
submission.to_csv('submission.csv', index=False)

# Local: save to solution-specific folder
submission.to_csv('./solutions/username/my_submission.csv', index=False)
```

#### Step 3: Run Solution Code

```bash
# Navigate to solution directory
cd solutions/bluetriad/

# Run the solution script (format varies by participant)
python bluetriad_solution.py

# Or for Jupyter notebooks
jupyter notebook helios-externaldatav3.ipynb
```

#### Step 4: Evaluate Results

```bash
# Use the provided evaluation script
python ../../evaluate.py ./solutions/bluetriad/submission.csv

# Compare with documented CFCS score
# Example output:
# ===========================================
# CFCS Score: 76.29
# ===========================================
# Component Breakdown:
# - Avg Significant Correlation: 0.xxx
# - Max Correlation: 0.xxx
# - Significant Count: xxx
```

### Common Issues and Solutions

#### Issue 1: Missing Dependencies
```bash
# Error: ModuleNotFoundError: No module named 'xxx'
# Solution: Install missing package
pip install xxx
```

#### Issue 2: File Not Found
```bash
# Error: FileNotFoundError: [Errno 2] No such file or directory
# Solution: Check and update all file paths to point to local data directory
```

#### Issue 3: Memory Errors
```bash
# Error: MemoryError or kernel died
# Solution: Reduce batch size, use chunking, or increase system RAM
# Example chunking:
chunks = pd.read_csv('large_file.csv', chunksize=10000)
for chunk in chunks:
    process(chunk)
```

#### Issue 4: Missing External Data
```bash
# Some solutions use external climate data (ONI, PDO, FRED, Open-Meteo)
# These are typically optional enhancements
# Solutions should work with base data only (sometimes at reduced performance)
```

### Solution-Specific Notes

#### Solutions with External Data Requirements:
- **bluetriad**: Uses `extra_climate_data.csv` (climate oscillation indices)
- **cmasch**: Optional FRED economic indicators + Open-Meteo weather
- **ezberch**: External macro-climate indices (ONI, PDO)

#### Solutions with Complex Dependencies:
- **Mr RRR**: Requires PyTorch for AutoEncoder training
- **GPCH**: Requires custom `src/` package (incomplete in submission)

#### Solutions with Minimal Dependencies:
- **kadircandrisolu**: Pure pandas/numpy, easy to reproduce
- **cg**: Standard scikit-learn stack
- **PxlPau**: Minimal dependencies, straightforward pipeline

### Verifying Reproducibility

To verify your reproduction matches the original submission:

```bash
# 1. Generate submission file using modified code
python solution_script.py

# 2. Compare with original submission
diff my_submission.csv solutions/username/submission.csv

# 3. Evaluate both
python evaluate.py my_submission.csv
python evaluate.py solutions/username/submission.csv

# Scores should match (small differences due to floating-point precision are acceptable)
```

### Running Solutions on Kaggle (Recommended)

For the most accurate reproduction, run solutions directly on Kaggle:

1. Fork the [original competition notebook](https://www.kaggle.com/code/edenecanlilar/sample-notebook-eden)
2. Copy participant's code into the notebook
3. Run in Kaggle environment (data paths work automatically)
4. Submit to competition to verify CFCS score

**Benefits:**
- No path modifications needed
- Same environment as original development
- Access to competition's evaluation infrastructure
- Faster iteration with pre-loaded datasets

## 📊 Dataset Highlights

### Climate Risk Data
- Daily assessments across global corn-growing regions
- Risk classifications: Low, Medium, High based on crop-specific thresholds
- Four risk categories: Heat stress, cold stress, drought, excess precipitation
- Regional aggregation: Location counts by risk level for each region/day
- Economic context: Production share data for weighting regional importance

### Futures Market Data
- Comprehensive pricing: Corn (ZC), wheat (ZW), soybean (ZS) futures
- Technical indicators: Returns, volatility, moving averages
- Market structure: Term spreads and cross-commodity relationships
- Daily frequency: Aligned with climate risk assessments

## 🎯 Submission Requirements

Submissions required:

1. **Engineered Dataset**: Enhanced version with novel features
2. **Feature Documentation**: Clear explanation of approach
3. **Code**: Reproducible pipeline for generating features
4. **Analysis**: Demonstration of improved correlations vs. baseline

### Critical Naming Conventions

⚠️ **IMPORTANT**: Features must follow these conventions:
- Climate Features: Must start with `climate_risk_`
  - ✅ Good: `climate_risk_heat_stress_weighted`, `climate_risk_drought_ma_30d`
  - ❌ Bad: `heat_stress_risk`, `my_climate_feature`, `weather_index`
- Required Columns: `date_on`, `country_name`, `region_name`
- Futures Data: Columns starting with `futures_*` (provided by evaluation system)

## 💡 Approaches Used by Participants

Participants explored various approaches including:

### Feature Engineering
- Production-weighted risk scores
- Temporal aggregations (weekly, monthly, seasonal)
- Cross-regional patterns
- Risk momentum and trend indicators
- Composite climate stress indices

### Advanced Techniques
- Non-linear transformations for threshold effects
- Lag analysis for delayed market responses
- Seasonal adjustments
- Volatility modeling
- Regime detection

### Domain Knowledge Integration
- Growing season alignment
- Supply chain modeling
- Market psychology factors
- Cross-commodity effects

## 📚 Resources

- [Competition Page](https://www.kaggle.com/competitions/forecasting-the-future-the-helios-corn-climate-challenge)
- [Competition Overview](https://www.kaggle.com/competitions/forecasting-the-future-the-helios-corn-climate-challenge/overview)
- [Leaderboard](https://www.kaggle.com/competitions/forecasting-the-future-the-helios-corn-climate-challenge/leaderboard)
- [Discussion Forum](https://www.kaggle.com/competitions/forecasting-the-future-the-helios-corn-climate-challenge/discussion)
- [Sample Submission Notebook](https://www.kaggle.com/code/edenecanlilar/sample-notebook-eden)

## 🌐 Why This Matters

Success in this challenge has real-world applications:
- **Risk Management**: Better weather-based hedging strategies
- **Trading Alpha**: Novel signals for commodity trading
- **Agricultural Finance**: Improved crop insurance and lending models
- **Supply Chain**: Enhanced forecasting for food companies
- **Climate Adaptation**: Better understanding of weather-market linkages

## 🤝 Contributing

This repository archives the completed competition. If you participated and would like to add or update your solution:

1. Fork the repository
2. Add/update your solution in `solutions/<your_username>/`
3. Include your code, documentation, and submission file
4. Submit a pull request

## 📄 Citation

```
Eden Canlilar. Helios Corn Futures Climate Challenge.
https://kaggle.com/competitions/forecasting-the-future-the-helios-corn-climate-challenge, 2025. Kaggle.
```

## 📧 Contact

Competition Host: [Eden Canlilar](https://www.kaggle.com/edenecanlilar)  
Helios AI: [www.helios.sc](https://www.helios.sc)

## 📜 License

See [LICENSE](LICENSE) file for details.

---

**Good luck, and may the correlations be with you!** 🌽📈
