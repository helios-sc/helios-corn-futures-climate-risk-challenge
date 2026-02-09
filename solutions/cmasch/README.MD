# Helios AI: Feature Engineering Challenge

A comprehensive feature engineering pipeline for the Helios Corn Climate Challenge that generates climate risk features to predict corn futures prices based on environmental stressors affecting corn-growing regions worldwide.

## Overview

This notebook implements advanced feature engineering techniques to analyze the relationship between climate risks and agricultural commodity futures prices. The pipeline supports two operational modes:

- **Standard Mode**: Uses only the Helios dataset
- **Enhanced Mode**: Integrates external data from FRED (Federal Reserve Economic Data) and Open-Meteo weather data

## Key Features

- **Climate Risk Analysis**: Processes heat stress, drought, excess precipitation, and unseasonably cold weather data
- **Temporal Features**: Creates lag-based features (7, 14, 30, 60, 90, 270, 365, 400 days) and rolling statistics
- **External Data Integration**: Enriches analysis with macroeconomic indicators and high-resolution meteorological data
- **Risk Scoring**: Implements weighted risk calculations with configurable thresholds
- **CFCS Metric**: Climate-Futures Correlation Score for evaluating feature quality
- **Geospatial Analysis**: Regional risk assessment and market share weighting

## Requirements

### Python Version
- Python 3.8 or higher recommended

### Dependencies

Install all required packages using:

```bash
pip install -r requirements.txt
```

The main dependencies are:
- `numpy` - Numerical computing
- `pandas` - Data manipulation and analysis
- `matplotlib` - Data visualization
- `seaborn` - Statistical data visualization

## Getting Started

### 1. Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

### 2. Data Requirements

The notebook expects the following input files:

**Required (Helios Data):**
- `corn_climate_risk_futures_daily_master.csv`
- `corn_regional_market_share.csv`

**Recommended (External Data):**
- `fred_economic_data.csv` (FRED economic indicators)
- `open_meteo_weather_data.csv` (Weather data)

> **Note**: External data can be obtained from [Kaggle](https://www.kaggle.com/datasets/cmasch/helios-climate-external-data)

### 3. Configuration

Update the file paths in the notebook configuration cell:

```python
INPUT_MASTER_FILE = "path/to/corn_climate_risk_futures_daily_master.csv"
INPUT_MARKET_SHARE_FILE = "path/to/corn_regional_market_share.csv"
INPUT_FRED_EXTERNAL_DATA = "path/to/fred_economic_data.csv"  # if using external data
INPUT_METEO_EXTERNAL_DATA = "path/to/open_meteo_weather_data.csv"  # if using external data
```

### 4. Running the Notebook

The notebook is designed to run sequentially from top to bottom. Key configuration parameters:

#### Core Parameters

```python
USE_EXTERNAL_DATA = True  # Set to False for standard mode
MIN_SIG_COUNT = 78        # Feature selection threshold (78 with external data, 48 without)
DO_OPTIMIZATION = False   # Enable optimization (may take 0.5-4 hours)
HIGH_RISK_MULTIPLIER = 2  # Multiplier for high-risk climate events
```

#### Stress Thresholds

```python
STRESS_THRESHOLDS = {
    'heat_stress': 0.75,
    'drought': 0.63,
    'excess_precip': 0.5,
    'unseasonably_cold': 0.5
}
```

#### Temporal Parameters

```python
LAGS = [7, 14, 30, 60, 90, 270, 365, 400]  # Lag periods in days
```

## Configuration Modes

### Mode 1: Without External Data

```python
MIN_SIG_COUNT = 48
USE_EXTERNAL_DATA = False
```

This mode uses only the Helios dataset and is faster to run.

### Mode 2: With External Data

```python
MIN_SIG_COUNT = 78
USE_EXTERNAL_DATA = True
```

This mode enriches the analysis with FRED economic data and Open-Meteo weather data for more comprehensive features.


## Key Concepts

### Climate Risk Categories

The notebook processes four main climate risk types:
- **Heat Stress**: High temperature events affecting crop growth
- **Drought**: Prolonged periods of insufficient precipitation
- **Excess Precipitation**: Flooding and oversaturation events
- **Unseasonably Cold**: Frost and cold snaps

### Feature Engineering Techniques

1. **Lag Features**: Historical values at various time intervals
2. **Rolling Statistics**: Moving averages, sums, and volatility measures
3. **Weighted Risk Scores**: Market-share weighted regional aggregations
4. **Composite Indicators**: Combined climate stress metrics
5. **External Enrichment**: Macroeconomic and meteorological features


## 📚 External Data Sources

### FRED (Federal Reserve Economic Data)

The notebook can integrate the following economic indicators:
- Dollar Index (DTWEXBGS)
- Oil Price (DCOILWTICO)
- VIX Volatility Index (VIXCLS)
- CPI, GDP, Interest Rates, and more

**Obtaining FRED Data:**
1. Register for a free API key at [FRED API](https://fred.stlouisfed.org/docs/api/api_key.html)
2. Use the provided script in the notebook or download preprocessed data from Kaggle

### Open-Meteo Weather Data

High-resolution meteorological variables including:
- Temperature (2m)
- Precipitation
- Wind speed
- Soil moisture
- And more

**Obtaining Weather Data:**
- The Open-Meteo API requires no API key
- Use the provided script in the notebook or download preprocessed data from Kaggle

## Additional Resources

- [Helios Corn Climate Challenge](https://www.kaggle.com/competitions/forecasting-the-future-the-helios-corn-climate-challenge)
- [External Data on Kaggle](https://www.kaggle.com/datasets/cmasch/helios-climate-challenge-ext-data-fred-and-meteo)
- [FRED API Documentation](https://fred.stlouisfed.org/docs/api/)
- [Open-Meteo API Documentation](https://open-meteo.com/en/docs)


## Important Notes

1. **Data Leakage Prevention**: The notebook includes a 1-day lag for FRED data to prevent data leakage
2. **NaN Handling**: Missing values are filled using forward-fill → backfill → median strategy
3. **Kaggle Environment**: File paths are configured for Kaggle notebooks and may need adjustment for local execution

## Contact
For questions related to this implementation, please contact me Christopher.Masch@gmail.com.

---

**Last Updated**: 31st January 2025  
**Competition**: Helios Corn Climate Challenge
