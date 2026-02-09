#!/usr/bin/env python
# coding: utf-8

# ## Imports

# In[1]:


import pandas as pd
import numpy as np
from pathlib import Path

pd.set_option('display.max_columns', 50)
SEED = 42


# ## Data I/O

# Please enter your exact data path

# In[2]:


DATA_DIR = Path("data")

df = pd.read_csv(f"{DATA_DIR}/corn_climate_risk_futures_daily_master.csv", parse_dates=["date_on"])
shares = pd.read_csv(f"{DATA_DIR}/corn_regional_market_share.csv")

df.shape, df["date_on"].min(), df["date_on"].max()


# ## Configuration

# In[15]:


RISKS = [
    "heat_stress",
    "unseasonably_cold",
    "excess_precip",
    "drought",
]

BASE_COLS = [
    "ID",
    "date_on",
    "crop_name",
    "country_name",
    "region_name",
]

# we are not using futures columns during feature engineering. This definition is just for submission format.
FUTURES_COLS = [
    c for c in df.columns
    if c.startswith("futures_")
]

ROLL_WINDOWS = [400, 410, 420, 430] # long term patterns


# ## Feature Engineering

# In[4]:


def sort_and_reset(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    return df.sort_values(by, kind='mergesort').reset_index(drop=True)

def add_production_share(df: pd.DataFrame, shares: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.merge(
        shares[["region_id", "percent_country_production"]],
        on="region_id",
        how="left",
    )
    df["percent_country_production"] = df["percent_country_production"].fillna(1.0)
    return df

def compute_risk_score(df: pd.DataFrame, risk: str) -> pd.Series:
    low = f"climate_risk_cnt_locations_{risk}_risk_low"
    med = f"climate_risk_cnt_locations_{risk}_risk_medium"
    high = f"climate_risk_cnt_locations_{risk}_risk_high"
    total = df[[low, med, high]].sum(axis=1)
    return (df[med] + 2 * df[high]) / (total + 1e-6)

def add_risk_features(df: pd.DataFrame, risks: list[str]) -> pd.DataFrame:
    df = df.copy()
    for r in risks:
        score = compute_risk_score(df, r)
        df[f"climate_risk_{r}_score"] = score
        df[f"climate_risk_{r}_weighted"] = (score * df["percent_country_production"] / 100)
    return df

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    dt = df["date_on"]
    df["day_of_year"] = dt.dt.dayofyear
    df["quarter"] = dt.dt.quarter
    return df

def add_composites(df: pd.DataFrame, risks: list[str]) -> pd.DataFrame:
    df = df.copy()
    df["climate_risk_temperature_stress"] = df[
        ["climate_risk_heat_stress_score", "climate_risk_unseasonably_cold_score"]
    ].max(axis=1)
    df["climate_risk_precipitation_stress"] = df[
        ["climate_risk_excess_precip_score", "climate_risk_drought_score"]
    ].max(axis=1)
    cols = [f"climate_risk_{r}_score" for r in risks]
    df["climate_risk_overall_stress"] = df[cols].max(axis=1)
    df["climate_risk_combined_stress"] = df[cols].mean(axis=1)
    return df

def add_rolling_features(df: pd.DataFrame, risks: list[str], windows: list[int]) -> pd.DataFrame:
    df = df.copy()
    df['_original_order'] = range(len(df))
    df = sort_and_reset(df, ["region_id", "date_on"])

    for w in windows:
        for r in risks:
            col = f"climate_risk_{r}_score"
            grp = df.groupby("region_id", sort=False)[col]

            df[f"{col}_ma_{w}d"] = (
                grp.rolling(w, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )

            df[f"{col}_max_{w}d"] = (
                grp.rolling(w, min_periods=1)
                .max()
                .reset_index(level=0, drop=True)
            )

    df = df.sort_values('_original_order').reset_index(drop=True)
    df = df.drop(columns='_original_order')
    return df

def add_momentum_features(df: pd.DataFrame, risks: list[str]) -> pd.DataFrame:
    df = df.copy()
    df['_original_order'] = range(len(df))
    df = sort_and_reset(df, ["region_id", "date_on"])

    for r in risks:
        col = f"climate_risk_{r}_score"
        grp = df.groupby("region_id", sort=False)[col]

        df[f"{col}_change_1d"] = grp.diff(1)
        df[f"{col}_change_7d"] = grp.diff(7)
        df[f"{col}_acceleration"] = (
            df.groupby("region_id", sort=False)[f"{col}_change_1d"].diff()
        )

    df = df.sort_values('_original_order').reset_index(drop=True)
    df = df.drop(columns='_original_order')
    return df

def add_country_aggregates(df: pd.DataFrame, risks: list[str]) -> pd.DataFrame:
    df = df.copy()

    for r in risks:
        score = f"climate_risk_{r}_score"
        weight = f"climate_risk_{r}_weighted"

        agg = (
            df.groupby(["country_name", "date_on"], sort=False)
            .agg(
                **{
                    f"climate_risk_country_{r}_mean": (score, "mean"),
                    f"climate_risk_country_{r}_max": (score, "max"),
                    f"climate_risk_country_{r}_std": (score, "std"),
                    f"climate_risk_country_{r}_weighted_sum": (weight, "sum"),
                }
            )
            .reset_index()
            .round(4)
        )

        df = df.merge(agg, on=["country_name", "date_on"], how="left")

    return df

def add_country_weighted_rollings(df: pd.DataFrame, risks: list[str], windows: list[int]) -> pd.DataFrame:
    df = df.copy()
    df['_original_order'] = range(len(df))
    df = sort_and_reset(df, ["country_name", "date_on"])

    for r in risks:
        col = f"climate_risk_{r}_weighted"
        grp = df.groupby("country_name", sort=False)[col]

        for w in windows:
            df[f"{col}_country_sum_ma_{w}d"] = (
                grp.rolling(w, min_periods=1)
                .sum()
                .reset_index(level=0, drop=True)
            )

            df[f"{col}_country_max_ma_{w}d"] = (
                grp.rolling(w, min_periods=1)
                .max()
                .reset_index(level=0, drop=True)
            )

    df = df.sort_values('_original_order').reset_index(drop=True)
    df = df.drop(columns='_original_order')
    return df

def add_country_weighted_cumsum(df: pd.DataFrame, risks: list[str]) -> pd.DataFrame:
    df = df.copy()
    df = sort_and_reset(df, ["country_name", "date_on"])

    for r in risks:
        w = f"climate_risk_{r}_weighted"
        df[f"{w}_country_cumsum"] = (
            df.groupby("country_name", sort=False)[w].cumsum()
        )

    return df


# In[5]:


df_engineered = (
    df
    .pipe(add_time_features)
    .pipe(add_production_share, shares)
    .pipe(add_risk_features, RISKS)
    .pipe(add_composites, RISKS)
    .pipe(add_rolling_features, RISKS, ROLL_WINDOWS)
    .pipe(add_momentum_features, RISKS)
    .pipe(add_country_aggregates, RISKS)
    .pipe(add_country_weighted_rollings, RISKS, ROLL_WINDOWS)
    .pipe(add_country_weighted_cumsum, ["drought"])
)


# In[6]:


from scipy import stats

def compute_monthly_climate_futures_correlations(
    df: pd.DataFrame,
) -> pd.DataFrame:
    climate_cols = [c for c in df.columns if c.startswith("climate_risk_")]
    futures_cols = [c for c in df.columns if c.startswith("futures_")]

    df = df.copy()
    df["month"] = df["date_on"].dt.month

    out = []

    for (crop, country, month), g in df.groupby(
        ["crop_name", "country_name", "month"]
    ):
        x = g[climate_cols].to_numpy()
        y = g[futures_cols].to_numpy()

        x_std = x.std(axis=0)
        y_std = y.std(axis=0)

        x_mask = x_std > 0
        y_mask = y_std > 0

        if not (x_mask.any() and y_mask.any()):
            continue

        x = x[:, x_mask]
        y = y[:, y_mask]

        x_cols = np.array(climate_cols)[x_mask]
        y_cols = np.array(futures_cols)[y_mask]

        x = x - x.mean(axis=0)
        y = y - y.mean(axis=0)

        denom_x = np.sqrt((x ** 2).sum(axis=0))
        denom_y = np.sqrt((y ** 2).sum(axis=0))

        corr = (x.T @ y) / np.outer(denom_x, denom_y)

        for i, cx in enumerate(x_cols):
            for j, fy in enumerate(y_cols):
                out.append({
                    "crop_name": crop,
                    "country_name": country,
                    "month": month,
                    "climate_variable": cx,
                    "futures_variable": fy,
                    "correlation": round(float(corr[i, j]), 5),
                })

    return pd.DataFrame(out)

def calculate_cfcs_from_arrays(
    abs_corr: np.ndarray,
    sig_thr: float,
) -> dict:
    if abs_corr.size == 0:
        return {
            "cfcs_score": 0.0,
            "avg_significant_correlation": 0.0,
            "max_abs_correlation": 0.0,
            "significant_correlations_pct": 0.0,
            "total_correlations": 0,
            "significant_correlations": 0,
        }

    sig = abs_corr >= sig_thr

    avg_sig = abs_corr[sig].mean() if sig.any() else 0.0
    max_corr = abs_corr.max()
    sig_pct = sig.mean()

    cfcs = (
        0.5 * min(100.0, avg_sig * 100)
        + 0.3 * min(100.0, max_corr * 100)
        + 0.2 * (sig_pct * 100)
    )

    return {
        "cfcs_score": round(cfcs, 2),
        "avg_significant_correlation": round(avg_sig, 4),
        "max_abs_correlation": round(max_corr, 4),
        "significant_correlations_pct": round(sig_pct * 100, 2),
        "total_correlations": int(abs_corr.size),
        "significant_correlations": int(sig.sum()),
    }

def calculate_cfcs_score(
    correlations_df: pd.DataFrame,
    sig_thr: float = 0.5,
) -> dict:
    abs_corr = correlations_df["correlation"].dropna().abs().to_numpy()
    return calculate_cfcs_from_arrays(abs_corr, sig_thr)


# ## Holdout Validation

# In[7]:


# only last year as a holdout validation set
df_2025 = df_engineered[df_engineered["date_on"].dt.year == 2025].dropna()

monthly_corr_2025 = compute_monthly_climate_futures_correlations(df_2025)

score_2025 = calculate_cfcs_score(monthly_corr_2025)

print("=== 2025 CFCS SCORE ===")
for k, v in score_2025.items():
    print(f"{k}: {v}")


# In[8]:


def build_feature_aggregates(
    correlations_df: pd.DataFrame,
    sig_thr: float,
) -> dict[str, dict]:
    stats = {}

    for f, g in correlations_df.groupby("climate_variable"):
        abs_corr = g["correlation"].abs().to_numpy()
        sig = abs_corr >= sig_thr

        stats[f] = {
            "n_total": abs_corr.size,
            "n_sig": int(sig.sum()),
            "sum_sig_abs": float(abs_corr[sig].sum()),
            "max_abs": float(abs_corr.max()),
        }

    return stats

def calculate_cfcs_from_stats(
    feature_stats: dict[str, dict],
    features: list[str],
    sig_thr: float,
) -> dict:
    n_total = n_sig = 0
    sum_sig = max_abs = 0.0

    for f in features:
        s = feature_stats.get(f)
        if s is None:
            continue
        n_total += s["n_total"]
        n_sig += s["n_sig"]
        sum_sig += s["sum_sig_abs"]
        max_abs = max(max_abs, s["max_abs"])

    abs_corr = np.empty(0) if n_total == 0 else None

    if n_total == 0:
        return calculate_cfcs_from_arrays(abs_corr, sig_thr)

    avg_sig = sum_sig / n_sig if n_sig > 0 else 0.0
    sig_pct = n_sig / n_total

    cfcs = (
        0.5 * min(100.0, avg_sig * 100)
        + 0.3 * min(100.0, max_abs * 100)
        + 0.2 * (sig_pct * 100)
    )

    return {
        "cfcs_score": round(cfcs, 2),
        "avg_significant_correlation": round(avg_sig, 4),
        "max_abs_correlation": round(max_abs, 4),
        "significant_correlations_pct": round(sig_pct * 100, 2),
        "total_correlations": int(n_total),
        "significant_correlations": int(n_sig),
    }


# ## Feature Selection with Beam Search

# In[9]:


import heapq

def beam_cfcs_selection(
    correlations_df: pd.DataFrame,
    sig_thr: float = 0.5,
    beam_width: int = 20,
    max_features: int = 8,
) -> tuple[list[str], dict]:
    corr_df = correlations_df[
        ["climate_variable", "correlation"]
    ].dropna()

    stats = build_feature_aggregates(corr_df, sig_thr)
    features = list(stats)

    def score(feats: list[str]) -> float:
        return calculate_cfcs_from_stats(
            stats,
            feats,
            sig_thr,
        )["cfcs_score"]

    # (score, feature_list)
    beam: list[tuple[float, list[str]]] = [(0.0, [])]
    best: tuple[float, list[str]] = (0.0, [])

    for _ in range(max_features):
        candidates: list[tuple[float, list[str]]] = []

        for _, feats in beam:
            used = set(feats)
            for f in features:
                if f in used:
                    continue

                new_feats = feats + [f]
                s = score(new_feats)

                candidates.append((s, new_feats))

        # Keep top beam_width candidates
        beam = heapq.nlargest(
            beam_width,
            candidates,
            key=lambda x: x[0],
        )

        if not beam:
            break

        if beam[0][0] > best[0]:
            best = beam[0]

    best_stats = calculate_cfcs_from_stats(
        stats,
        best[1],
        sig_thr,
    )

    return best[1], best_stats


# In[10]:


beam_selected_features, beam_cfcs_stats = beam_cfcs_selection(
    correlations_df=monthly_corr_2025,
    sig_thr=0.5,
    beam_width=50,
    max_features=30,
)
print(beam_cfcs_stats)
print()
print(beam_selected_features)


# ## Holdout Validation with Selected Features

# In[11]:


selected_df_2025 = (
    df_engineered[df_engineered["date_on"].dt.year == 2025]
    .dropna()
    [BASE_COLS + beam_selected_features + FUTURES_COLS] 
)

selected_monthly_corr_2025 = compute_monthly_climate_futures_correlations(
    selected_df_2025
)

selected_score_2025 = calculate_cfcs_score(selected_monthly_corr_2025)

print("=== 2025 CFCS SCORE (Selected Feature/Features) ===")
for k, v in selected_score_2025.items():
    print(f"{k}: {v}")


# ## Let's Submit the Solution

# In[12]:


valid_idx = df_engineered.dropna().index
df_submit = df_engineered.loc[
    valid_idx,
    BASE_COLS + beam_selected_features + FUTURES_COLS,
]

output_path = (
    DATA_DIR
    / "submission.csv"
)
df_submit.to_csv(output_path, index=False)
print(f"Saved to {output_path}")
print(f"Rows: {len(df_submit)}, Columns: {len(df_submit.columns)}")


# In[ ]:




