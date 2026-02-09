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
