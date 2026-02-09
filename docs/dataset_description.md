# Corn Climate Risk and Futures Dataset

## Overview

This dataset combines daily climate risk assessments for corn production with futures market data, created by merging Helios's proprietary climate risk model outputs with commodity futures pricing information. The dataset contains **320,661 records** spanning multiple years and covers corn-growing regions globally.

## Data Sources

### Climate Risk Data (Helios Proprietary Model)
The weather and climate risk data is sourced from Helios's proprietary climate risk model, which evaluates daily conditions for corn production across various global locations. The model determines risk levels (Low, Medium, High) based on crop-specific parameters including temperature thresholds and precipitation requirements.

### Futures Market Data (Barchart API)
Commodity futures pricing data is obtained from Barchart's API, including:
- **ZC*1**: Corn front-month futures
- **ZC*2**: Corn second-month futures  
- **ZW*1**: Wheat front-month futures (contextual)
- **ZS*1**: Soybean front-month futures (contextual)

### Regional Market Share Data
Production share data quantifying each region's contribution to national corn production, providing economic weighting for regional climate risk assessments.

## Risk Classification Methodology

The climate risk model evaluates four key weather factors for corn production:

### Risk Levels
- **High Risk**: Conditions are both anomalous (statistically unusual) AND outside the plant's comfort bounds
- **Medium Risk**: Conditions are outside the plant's comfort bounds AND above normal historical patterns (similar to 1.5-2.5 standard deviations) but not yet anomalous
- **Low Risk**: Normal conditions within acceptable parameters for corn growth

### Risk Categories
1. **Heat Stress Risk** (`daily_too_hot_risk`): Maximum temperature exceeding corn tolerance
2. **Cold Stress Risk** (`daily_too_cold_risk`): Minimum temperature below corn requirements  
3. **Excess Precipitation Risk** (`daily_too_wet_risk`): Rainfall exceeding optimal levels
4. **Drought Risk** (`daily_too_dry_risk`): Insufficient precipitation for corn needs

## Dataset Structure

### Geographic Coverage
- **Crop**: Corn (Commodity Tracked) only
- **Countries**: Multiple countries including Argentina, Brazil, United States, and others
- **Regions**: Sub-national administrative regions within each country
- **Locations**: Individual points of interest (POIs) within each region

### Temporal Coverage
- **Date Range**: 2015-2025 (historical and forecasted data)
- **Frequency**: Daily observations
- **Seasonality**: Includes harvest periods and growing season classifications

## Column Descriptions

### Geographic Identifiers
- `crop_name`: Always "Corn: Commodity Tracked"
- `country_name`: Country where the region is located
- `country_code`: ISO country code (e.g., "AR", "US", "BR")
- `region_name`: Administrative region name
- `region_id`: Unique identifier for the region

### Temporal Information
- `date_on`: Observation date (YYYY-MM-DD)
- `harvest_period`: Growing season phase (e.g., "Harvest", "Planting")
- `growing_season_year`: Agricultural year for the growing season
- `date_on_year`: Calendar year extracted from date
- `date_on_month`: Calendar month extracted from date  
- `date_on_year_month`: Year-month combination (YYYY_MM format)

### Climate Risk Metrics
Each risk category has three columns counting locations by risk level:

**Heat Stress Risk:**
- `climate_risk_cnt_locations_heat_stress_risk_low`
- `climate_risk_cnt_locations_heat_stress_risk_medium` 
- `climate_risk_cnt_locations_heat_stress_risk_high`

**Cold Stress Risk:**
- `climate_risk_cnt_locations_unseasonably_cold_risk_low`
- `climate_risk_cnt_locations_unseasonably_cold_risk_medium`
- `climate_risk_cnt_locations_unseasonably_cold_risk_high`

**Excess Precipitation Risk:**
- `climate_risk_cnt_locations_excess_precip_risk_low`
- `climate_risk_cnt_locations_excess_precip_risk_medium`
- `climate_risk_cnt_locations_excess_precip_risk_high`

**Drought Risk:**
- `climate_risk_cnt_locations_drought_risk_low`
- `climate_risk_cnt_locations_drought_risk_medium`
- `climate_risk_cnt_locations_drought_risk_high`

### Futures Market Data

**Price Data:**
- `futures_close_ZC_1`: Corn front-month futures closing price
- `futures_close_ZC_2`: Corn second-month futures closing price
- `futures_close_ZW_1`: Wheat front-month futures closing price
- `futures_close_ZS_1`: Soybean front-month futures closing price

**Technical Indicators:**
- `futures_zc1_ret_pct`: Daily percentage return for corn front-month
- `futures_zc1_ret_log`: Daily log return for corn front-month
- `futures_zc_term_spread`: Price difference between ZC*2 and ZC*1
- `futures_zc_term_ratio`: Price ratio of ZC*2 to ZC*1

**Moving Averages:**
- `futures_zc1_ma_20`: 20-day moving average of corn prices
- `futures_zc1_ma_60`: 60-day moving average of corn prices  
- `futures_zc1_ma_120`: 120-day moving average of corn prices

**Volatility Measures:**
- `futures_zc1_vol_20`: 20-day rolling volatility of corn returns
- `futures_zc1_vol_60`: 60-day rolling volatility of corn returns

**Cross-Commodity Relationships:**
- `futures_zw_zc_spread`: Wheat-corn price spread
- `futures_zc_zw_ratio`: Corn-to-wheat price ratio
- `futures_zs_zc_spread`: Soybean-corn price spread  
- `futures_zc_zs_ratio`: Corn-to-soybean price ratio

## Data Quality Notes

- **Missing Values**: Futures data may have gaps on non-trading days (weekends, holidays)
- **Outlier Handling**: Daily returns are capped at ±25% to remove extreme outliers
- **Alignment**: Climate risk data is available daily, while futures data follows market trading schedules


## Regional Market Share Data

### Overview
The `corn_regional_market_share.csv` file provides production share information for corn-growing regions across major producing countries. This data enables economic weighting of climate risk assessments based on each region's contribution to national corn production.

### Structure
- **Records**: 86 regional entries across 12 countries
- **Coverage**: Major corn-producing nations including United States, Brazil, Argentina, China, Russia, and others
- **Granularity**: Sub-national administrative regions within each country

### Columns
- `country_name`: Country name (e.g., "United States", "Brazil", "Argentina")
- `country_code`: ISO country code (e.g., "US", "BR", "AR") 
- `region_name`: Administrative region name (e.g., "Iowa", "Mato Grosso", "Córdoba")
- `region_id`: Unique identifier matching the main climate risk dataset
- `percent_country_production`: Percentage of national corn production from this region

### Key Production Regions
**United States**: Iowa (16.19%), Illinois (15.03%), Nebraska (12.62%)  
**Brazil**: Mato Grosso (37%), Rio Grande do Sul (22%), Paraná (11%)  
**Argentina**: Córdoba (35%), Buenos Aires (27%), Entre Ríos (12%)  
**China**: Heilongjiang (15%), Jilin (12%), Inner Mongolia (11%)  
**Russia**: Krasnodar Krai (20%), Voronezh (10%), Belgorod (7%)

### Applications
- **Risk Weighting**: Weight regional climate risks by economic importance
- **Impact Assessment**: Quantify potential production effects of weather events
- **Market Analysis**: Understand geographic concentration of corn production
- **Portfolio Management**: Assess exposure to specific production regions

## Data Quality Notes

- **Missing Values**: Futures data may have gaps on non-trading days (weekends, holidays)
- **Outlier Handling**: Daily returns are capped at ±25% to remove extreme outliers
- **Alignment**: Climate risk data is available daily, while futures data follows market trading schedules
- **Forecasted Data**: Recent dates may include forecasted climate risk assessments
- **Production Shares**: Some regions may have missing or zero production percentages

## File Information

### Main Dataset
- **Filename**: `corn_climate_risk_futures_daily_master.csv`
- **Size**: ~95 MB
- **Records**: 320,661 daily observations
- **Columns**: 40 variables
- **Format**: CSV with comma delimiters

### Regional Market Share
- **Filename**: `corn_regional_market_share.csv`
- **Size**: ~4 KB
- **Records**: 86 regional entries
- **Columns**: 5 variables
- **Format**: CSV with comma delimiters
#
# Feature Engineering Guidelines

### **Column Naming Requirements**

**For Correlation Computation:**

- **Climate Features**: All engineered climate risk features must start with `climate_risk_`
  - Original columns: `climate_risk_cnt_locations_heat_stress_risk_low`, etc.
  - Your features: `climate_risk_temperature_stress`, `climate_risk_drought_weighted`, etc.

- **Futures Features**: All futures market variables start with `futures_`
  - These are provided in the evaluation system - do not modify or include in submissions
  - Examples: `futures_close_ZC_1`, `futures_zc1_ret_pct`, etc.

- **Required Metadata**: Include these columns in your submission:
  - `date_on`: Date in YYYY-MM-DD format
  - `country_name`: Country name (must match original data)
  - `region_name`: Region name (recommended for regional analysis)

**The evaluation metric uses these prefixes to automatically identify feature types. Incorrect naming will result in features being ignored during scoring.**