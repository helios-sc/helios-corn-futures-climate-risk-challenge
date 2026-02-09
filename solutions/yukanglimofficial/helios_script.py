#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

What this script does
---------------------
1) Reconstructs the sample_submission-style engineered dataframe and uses `.dropna()` to
   reproduce the Kaggle evaluation row-set.

2) Builds country-day climate signals using the exact weighting scheme from the sample submission
(production-share normalization by country, with missing shares filled as 1.0).

3) Scores *single-feature gated submissions*:
   - One climate_risk_* feature at a time
   - Non-zero only inside one (country, month) bucket
   - Broadcast across all region rows for that country-date
   This prevents CFCS dilution.

4) Runs a 2-stage sweep:
   Stage 0 (baseline): scan ALL (country, month, signal) quickly to rank promising groups.
   Stage 1 (deep): refine only the top groups with pruned search + local refinements.

Outputs
-------
(out_dir)
- cache/ base_cache_*.npz, cd_cache_*.npz
- baseline_scan_all.csv
- grids/ grid__mMM_.csv.gz
- grid_all_candidates.csv
- recommendations.txt
"""
from __future__ import annotations

# ---- Threading hygiene (avoid oversubscription when using many processes) ----
import os as _os
_os.environ.setdefault("OMP_NUM_THREADS", "1")
_os.environ.setdefault("MKL_NUM_THREADS", "1")
_os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
_os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import gzip
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from joblib import Parallel, delayed
except Exception as e:
    raise RuntimeError("joblib is required. Install via: pip install joblib") from e


# =============================================================================
# Logging
# =============================================================================

def now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def log(msg: str) -> None:
    print(f"[{now_str()}] {msg}", flush=True)


# =============================================================================
# CFCS scoring (matches sample submission logic)
# =============================================================================

def _pearson_corr_nan(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return float("nan")
    xv = x[m]; yv = y[m]
    sx = xv.std(ddof=1); sy = yv.std(ddof=1)
    if sx <= 0 or sy <= 0:
        return float("nan")
    return float(np.corrcoef(xv, yv)[0, 1])

def corr_vector(z: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Correlation between z (n,) and each column of Y (n,k), with NaN handling.
    """
    z = np.asarray(z, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)

    # Fast path if fully finite
    if np.isfinite(z).all() and np.isfinite(Y).all():
        z0 = z - z.mean()
        denom_z = np.sqrt(np.dot(z0, z0))
        if denom_z <= 0:
            return np.full((Y.shape[1],), np.nan, dtype=np.float64)
        Y0 = Y - Y.mean(axis=0, keepdims=True)
        denom_y = np.sqrt((Y0 * Y0).sum(axis=0))
        denom = denom_z * denom_y
        out = (z0[:, None] * Y0).sum(axis=0) / np.where(denom > 0, denom, np.nan)
        return out.astype(np.float64)

    out = np.full((Y.shape[1],), np.nan, dtype=np.float64)
    for j in range(Y.shape[1]):
        out[j] = _pearson_corr_nan(z, Y[:, j])
    return out

def cfcs_from_corrs(corrs_1d: np.ndarray) -> Tuple[float, Dict[str, Any]]:
    s = pd.Series(np.asarray(corrs_1d, dtype=np.float64)).dropna()
    if s.empty:
        meta = dict(
            cfcs_score=0.0,
            avg_significant_correlation=0.0,
            max_abs_correlation=0.0,
            significant_correlations_pct=0.0,
            total_correlations=0,
            significant_correlations=0,
        )
        return 0.0, meta

    # Kaggle rounds correlations before thresholding
    s = s.round(5)
    abs_corrs = s.abs()
    sig = abs_corrs[abs_corrs >= 0.5]
    sig_count = int(sig.shape[0])
    total = int(abs_corrs.shape[0])

    if sig_count > 0:
        avg_sig = float(sig.mean())
        avg_sig_score = min(100.0, avg_sig * 100.0)
    else:
        avg_sig = 0.0
        avg_sig_score = 0.0

    max_abs = float(abs_corrs.max())
    max_score = min(100.0, max_abs * 100.0)

    sig_pct = (sig_count / total) * 100.0 if total else 0.0
    cfcs = (0.5 * avg_sig_score) + (0.3 * max_score) + (0.2 * sig_pct)

    meta = dict(
        cfcs_score=float(round(cfcs, 2)),
        avg_significant_correlation=float(round(avg_sig, 4)),
        max_abs_correlation=float(round(max_abs, 4)),
        significant_correlations_pct=float(round(sig_pct, 2)),
        total_correlations=total,
        significant_correlations=sig_count,
    )
    return float(round(cfcs, 2)), meta

def cfcs_score(z: np.ndarray, Y: np.ndarray) -> Tuple[float, Dict[str, Any]]:
    return cfcs_from_corrs(corr_vector(z, Y))


# =============================================================================
# Rolling + transforms 
# =============================================================================

def shift_array(x: np.ndarray, shift: int) -> np.ndarray:
    """
    pandas-like shift:
      out[t] = x[t - shift]
    shift > 0 => lag (uses past)
    shift < 0 => lead (uses future)
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    out = np.full((n,), np.nan, dtype=np.float64)
    if shift == 0:
        out[:] = x
        return out
    if shift > 0:
        out[shift:] = x[: n - shift]
    else:
        s = -shift
        out[: n - s] = x[s:]
    return out

def ffill_bfill_0(x: np.ndarray) -> np.ndarray:
    """Replicate pandas: s.ffill().bfill().fillna(0) on a 1D float array."""
    x = np.asarray(x, dtype=np.float64).copy()
    n = x.shape[0]
    prev = np.nan
    for i in range(n):
        if not np.isfinite(x[i]):
            x[i] = prev
        else:
            prev = x[i]
    nxt = np.nan
    for i in range(n - 1, -1, -1):
        if not np.isfinite(x[i]):
            x[i] = nxt
        else:
            nxt = x[i]
    x[~np.isfinite(x)] = 0.0
    return x

def rolling_mean_min1(x: np.ndarray, w: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    if w <= 1:
        return x.copy()
    out = np.full((n,), np.nan, dtype=np.float64)
    xx = np.where(np.isfinite(x), x, 0.0)
    cs = np.zeros(n + 1, dtype=np.float64)
    cs[1:] = np.cumsum(xx)
    cnt = np.zeros(n + 1, dtype=np.int32)
    cnt[1:] = np.cumsum(np.isfinite(x).astype(np.int32))
    for i in range(n):
        j0 = max(0, i - w + 1)
        s = cs[i + 1] - cs[j0]
        c = cnt[i + 1] - cnt[j0]
        if c > 0:
            out[i] = s / c
    return out

def rolling_std_min1(x: np.ndarray, w: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    if w <= 1:
        return np.zeros_like(x, dtype=np.float64)
    out = np.full((n,), np.nan, dtype=np.float64)
    xx = np.where(np.isfinite(x), x, 0.0)
    cs = np.zeros(n + 1, dtype=np.float64)
    cs2 = np.zeros(n + 1, dtype=np.float64)
    cs[1:] = np.cumsum(xx)
    cs2[1:] = np.cumsum(xx * xx)
    cnt = np.zeros(n + 1, dtype=np.int32)
    cnt[1:] = np.cumsum(np.isfinite(x).astype(np.int32))
    for i in range(n):
        j0 = max(0, i - w + 1)
        s = cs[i + 1] - cs[j0]
        s2 = cs2[i + 1] - cs2[j0]
        c = cnt[i + 1] - cnt[j0]
        if c > 1:
            mean = s / c
            var = max(0.0, (s2 / c) - (mean * mean))
            out[i] = math.sqrt(var)
        elif c == 1:
            out[i] = 0.0
    return out

def rolling_max_min1(x: np.ndarray, w: int) -> np.ndarray:
    from collections import deque
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    if w <= 1:
        return x.copy()
    out = np.full((n,), np.nan, dtype=np.float64)
    dq: "deque[int]" = deque()
    for i in range(n):
        j0 = i - w + 1
        while dq and dq[0] < j0:
            dq.popleft()
        xi = x[i]
        if np.isfinite(xi):
            while dq and x[dq[-1]] <= xi:
                dq.pop()
            dq.append(i)
        if dq:
            out[i] = x[dq[0]]
    return out

def ewm_mean(x: np.ndarray, span: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    if span <= 1:
        return x.copy()
    out = np.full((n,), np.nan, dtype=np.float64)
    alpha = 2.0 / (span + 1.0)
    prev = np.nan
    for i in range(n):
        xi = x[i]
        if not np.isfinite(xi):
            out[i] = prev
            continue
        if not np.isfinite(prev):
            prev = xi
        else:
            prev = (1.0 - alpha) * prev + alpha * xi
        out[i] = prev
    return out

def streak_fraction(x: np.ndarray, w: int, thr: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    if w <= 1:
        b = (x >= thr).astype(np.float64)
        b[~np.isfinite(x)] = np.nan
        return b
    out = np.full((n,), np.nan, dtype=np.float64)
    b = (x >= thr).astype(np.float64)
    b[~np.isfinite(x)] = np.nan
    b0 = np.where(np.isfinite(b), b, 0.0)
    cs = np.zeros(n + 1, dtype=np.float64)
    cs[1:] = np.cumsum(b0)
    cnt = np.zeros(n + 1, dtype=np.int32)
    cnt[1:] = np.cumsum(np.isfinite(b).astype(np.int32))
    for i in range(n):
        j0 = max(0, i - w + 1)
        s = cs[i + 1] - cs[j0]
        c = cnt[i + 1] - cnt[j0]
        if c > 0:
            out[i] = s / c
    return out

def runlen_current(x: np.ndarray, thr: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    out = np.zeros((n,), dtype=np.float64)
    cur = 0
    for i in range(n):
        xi = x[i]
        if (not np.isfinite(xi)) or (xi < thr):
            cur = 0
        else:
            cur += 1
        out[i] = float(cur)
    return out

def apply_transform(x: np.ndarray, name: str) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    name = (name or "identity").lower()
    if name == "identity":
        return x
    if name == "square":
        # signed square (matches submission.py)
        return np.sign(x) * (x * x)
    if name == "signlog1p":
        return np.sign(x) * np.log1p(np.abs(x))
    if name == "abs":
        return np.abs(x)
    if name == "tanh":
        return np.tanh(x)
    raise ValueError(f"Unknown transform: {name}")

def compute_agg_no_shift(x_full: np.ndarray, agg: str, w: int, *, q_for_streakq: float = 0.85) -> np.ndarray:
    """
    Compute aggregator on unshifted timeline; shift output later (translation invariance).
    """
    agg_l = (agg or "ma").lower().strip()
    w = int(max(1, w))
    if agg_l in ("identity", "raw"):
        return np.asarray(x_full, dtype=np.float64).copy()
    if agg_l == "ma":
        return rolling_mean_min1(x_full, w)
    if agg_l == "std":
        return rolling_std_min1(x_full, w)
    if agg_l == "max":
        return rolling_max_min1(x_full, w)
    if agg_l == "ewm":
        return ewm_mean(x_full, span=max(2, w))
    if agg_l.startswith("streakq"):
        q = q_for_streakq
        tail = agg_l.replace("streakq", "")
        if tail:
            try:
                q = float(tail) / 100.0
            except Exception:
                q = q_for_streakq
        q = float(np.clip(q, 0.5, 0.99))
        xf = np.asarray(x_full, dtype=np.float64)
        m = np.isfinite(xf)
        thr = float(np.nanquantile(xf[m], q)) if m.any() else 0.0
        if not np.isfinite(thr):
            thr = 0.0
        return streak_fraction(x_full, w, thr)
    if agg_l.startswith("streakthr"):
        tail = agg_l.replace("streakthr", "")
        thr = float(tail) if tail else 0.5
        return streak_fraction(x_full, w, thr)
    if agg_l.startswith("runlenth"):
        tail = agg_l.replace("runlenth", "")
        thr = float(tail) if tail else 0.5
        return runlen_current(x_full, thr)
    raise ValueError(f"Unknown agg: {agg}")

def compute_feature_series(
    x_full: np.ndarray,
    *,
    agg: str,
    w: int,
    shift: int,
    transform: str,
    q_for_streakq: float,
) -> np.ndarray:
    base = compute_agg_no_shift(x_full, agg=agg, w=w, q_for_streakq=q_for_streakq)
    z = apply_transform(shift_array(base, int(shift)), transform)
    return ffill_bfill_0(z)


# =============================================================================
# Row-set alignment
# =============================================================================

RISK_CATEGORIES = ["heat_stress", "unseasonably_cold", "excess_precip", "drought"]

def build_kaggle_rowset_base(
    main_df: pd.DataFrame,
    share_df: pd.DataFrame,
    *,
    rolling_windows: Sequence[int] = (7, 14, 30),
) -> pd.DataFrame:
    df = main_df.copy()
    df["date_on"] = pd.to_datetime(df["date_on"], errors="coerce")

    merged_daily_df = df.copy()
    merged_daily_df["day_of_year"] = merged_daily_df["date_on"].dt.dayofyear
    merged_daily_df["quarter"] = merged_daily_df["date_on"].dt.quarter

    share_cols = share_df[["region_id", "percent_country_production"]].copy()
    merged_daily_df = merged_daily_df.merge(share_cols, on="region_id", how="left")
    merged_daily_df["percent_country_production"] = merged_daily_df["percent_country_production"].fillna(1.0)

    for risk_type in RISK_CATEGORIES:
        low_col  = f"climate_risk_cnt_locations_{risk_type}_risk_low"
        med_col  = f"climate_risk_cnt_locations_{risk_type}_risk_medium"
        high_col = f"climate_risk_cnt_locations_{risk_type}_risk_high"
        total_locations = merged_daily_df[low_col] + merged_daily_df[med_col] + merged_daily_df[high_col]
        risk_score = (merged_daily_df[med_col] + 2.0 * merged_daily_df[high_col]) / (total_locations + 1e-6)
        weighted_risk = risk_score * (merged_daily_df["percent_country_production"] / 100.0)
        merged_daily_df[f"climate_risk_{risk_type}_score"] = risk_score
        merged_daily_df[f"climate_risk_{risk_type}_weighted"] = weighted_risk

    temperature_risks = ["heat_stress", "unseasonably_cold"]
    precipitation_risks = ["excess_precip", "drought"]
    temp_scores = [f"climate_risk_{r}_score" for r in temperature_risks]
    precip_scores = [f"climate_risk_{r}_score" for r in precipitation_risks]
    all_scores = [f"climate_risk_{r}_score" for r in RISK_CATEGORIES]

    merged_daily_df["climate_risk_temperature_stress"] = merged_daily_df[temp_scores].max(axis=1)
    merged_daily_df["climate_risk_precipitation_stress"] = merged_daily_df[precip_scores].max(axis=1)
    merged_daily_df["climate_risk_overall_stress"] = merged_daily_df[all_scores].max(axis=1)
    merged_daily_df["climate_risk_combined_stress"] = merged_daily_df[all_scores].mean(axis=1)

    merged_daily_df = merged_daily_df.sort_values(["region_id", "date_on"])

    for window in rolling_windows:
        for risk_type in RISK_CATEGORIES:
            score_col = f"climate_risk_{risk_type}_score"
            merged_daily_df[f"climate_risk_{risk_type}_ma_{window}d"] = (
                merged_daily_df.groupby("region_id")[score_col]
                .rolling(window=window, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )
            merged_daily_df[f"climate_risk_{risk_type}_max_{window}d"] = (
                merged_daily_df.groupby("region_id")[score_col]
                .rolling(window=window, min_periods=1)
                .max()
                .reset_index(level=0, drop=True)
            )

    for risk_type in RISK_CATEGORIES:
        score_col = f"climate_risk_{risk_type}_score"
        merged_daily_df[f"climate_risk_{risk_type}_change_1d"] = merged_daily_df.groupby("region_id")[score_col].diff(1)
        merged_daily_df[f"climate_risk_{risk_type}_change_7d"] = merged_daily_df.groupby("region_id")[score_col].diff(7)
        merged_daily_df[f"climate_risk_{risk_type}_acceleration"] = (
            merged_daily_df.groupby("region_id")[f"climate_risk_{risk_type}_change_1d"].diff(1)
        )

    for risk_type in RISK_CATEGORIES:
        score_col = f"climate_risk_{risk_type}_score"
        weighted_col = f"climate_risk_{risk_type}_weighted"
        country_agg = (
            merged_daily_df.groupby(["country_name", "date_on"]).agg({
                score_col: ["mean", "max", "std"],
                weighted_col: "sum",
                "percent_country_production": "sum",
            }).round(4)
        )
        country_agg.columns = [f"country_{risk_type}_{'_'.join(col).strip()}" for col in country_agg.columns]
        country_agg = country_agg.reset_index()
        merged_daily_df = merged_daily_df.merge(country_agg, on=["country_name", "date_on"], how="left")

    return merged_daily_df.dropna().copy()


# =============================================================================
# Country-day signals 
# =============================================================================

RISK_MAP = {
    "heat": "heat_stress",
    "cold": "unseasonably_cold",
    "wet":  "excess_precip",
    "dry":  "drought",
}

def build_country_day_signals(
    main_df: pd.DataFrame,
    share_df: pd.DataFrame,
    *,
    country_whitelist: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    df = main_df.copy()
    df["date_on"] = pd.to_datetime(df["date_on"], errors="coerce")

    # region weights (submission.py style)
    regions = df[["region_id", "country_name"]].drop_duplicates().copy()
    regions = regions.merge(share_df[["region_id", "percent_country_production"]], on="region_id", how="left")
    w_raw = regions["percent_country_production"].astype(np.float64).fillna(1.0).to_numpy()
    w_raw = np.where(w_raw <= 0, 1.0, w_raw)
    regions["w_raw"] = w_raw
    regions["prod_w"] = regions["w_raw"] / regions.groupby("country_name")["w_raw"].transform("sum")
    if country_whitelist is not None:
        regions = regions[regions["country_name"].isin(list(country_whitelist))].copy()
    w_map = regions.set_index("region_id")["prod_w"]

    df["prod_w"] = df["region_id"].map(w_map).fillna(0.0).astype(np.float64)

    eps = 1e-6
    tmp = df[["country_name", "date_on", "prod_w"]].copy()

    signal_cols: List[str] = []
    for short, kind in RISK_MAP.items():
        low = f"climate_risk_cnt_locations_{kind}_risk_low"
        med = f"climate_risk_cnt_locations_{kind}_risk_medium"
        high = f"climate_risk_cnt_locations_{kind}_risk_high"
        tot = (df[low] + df[med] + df[high]).astype(np.float64)

        sev = (df[med].astype(np.float64) + 2.0 * df[high].astype(np.float64)) / (tot + eps)
        hi  = df[high].astype(np.float64) / (tot + eps)
        wapr = (df[med].astype(np.float64) + df[high].astype(np.float64)) / (tot + eps)

        tmp[f"w_{short}_sev"] = tmp["prod_w"] * sev
        tmp[f"w_{short}_high"] = tmp["prod_w"] * hi
        tmp[f"w_{short}_wapr"] = tmp["prod_w"] * wapr

    sum_cols = ["prod_w"] + [c for c in tmp.columns if c.startswith("w_")]
    cd = tmp.groupby(["country_name", "date_on"], sort=False)[sum_cols].sum().reset_index()
    den = cd["prod_w"].to_numpy(np.float64) + 1e-12

    out = pd.DataFrame({
        "country_name": cd["country_name"].values,
        "date_on": cd["date_on"].values,
    })

    for short in RISK_MAP.keys():
        out[f"{short}_sev_wmean"]  = (cd[f"w_{short}_sev"].to_numpy(np.float64)  / den).astype(np.float32)
        out[f"{short}_high_wmean"] = (cd[f"w_{short}_high"].to_numpy(np.float64) / den).astype(np.float32)
        out[f"{short}_wapr_wmean"] = (cd[f"w_{short}_wapr"].to_numpy(np.float64) / den).astype(np.float32)
        signal_cols.extend([f"{short}_sev_wmean", f"{short}_high_wmean", f"{short}_wapr_wmean"])

    # Interactions (include what your earlier sweeps found useful)
    out["heat_dry_prod"] = (out["heat_sev_wmean"] * out["dry_sev_wmean"]).astype(np.float32)
    out["wet_dry_diff"]  = (out["wet_sev_wmean"]  - out["dry_sev_wmean"]).astype(np.float32)
    out["temp_stress_max"] = out[["heat_sev_wmean", "cold_sev_wmean"]].max(axis=1).astype(np.float32)
    out["precip_stress_max"] = out[["wet_sev_wmean", "dry_sev_wmean"]].max(axis=1).astype(np.float32)
    out["overall_stress_max"] = out[["heat_sev_wmean","cold_sev_wmean","wet_sev_wmean","dry_sev_wmean"]].max(axis=1).astype(np.float32)
    out["overall_stress_mean"] = out[["heat_sev_wmean","cold_sev_wmean","wet_sev_wmean","dry_sev_wmean"]].mean(axis=1).astype(np.float32)
    signal_cols.extend(["heat_dry_prod","wet_dry_diff","temp_stress_max","precip_stress_max","overall_stress_max","overall_stress_mean"])

    out["date_on"] = pd.to_datetime(out["date_on"])
    out["year"] = out["date_on"].dt.year.astype(np.int16)
    out["month"] = out["date_on"].dt.month.astype(np.int8)
    out = out.sort_values(["country_name", "date_on"]).reset_index(drop=True)
    return out, signal_cols


# =============================================================================
# Helper: window grid (coarse)
# =============================================================================

def make_window_grid(w_min: int, w_max: int) -> List[int]:
    w_min = int(max(1, w_min))
    w_max = int(max(w_min, w_max))
    ws: List[int] = []

    def add_range(a: int, b: int, step: int) -> None:
        for v in range(a, b + 1, step):
            ws.append(v)

    add_range(w_min, min(w_max, 120), 1)
    if w_max > 120:
        add_range(121, min(w_max, 400), 2)
    if w_max > 400:
        add_range(405, min(w_max, 800), 5)
    if w_max > 800:
        add_range(810, min(w_max, 1500), 10)
    if w_max > 1500:
        add_range(1525, min(w_max, 2500), 25)
    if w_max > 2500:
        add_range(2600, w_max, 50)

    specials = [7, 14, 21, 30, 45, 60, 90, 98, 102, 112, 120, 140, 168, 180, 200, 224, 240, 252,
                280, 300, 330, 365, 400, 450, 500, 540, 600, 730, 900, 1095, 1200, 1461, 1500, 1800, 2000, 2500]
    for v in specials:
        if w_min <= v <= w_max:
            ws.append(int(v))

    return sorted(set(ws))


# =============================================================================
# Time split helpers
# =============================================================================

def last_n_years_subset(years: np.ndarray, n: int) -> Optional[np.ndarray]:
    years = np.asarray(years, dtype=np.int16)
    uniq = np.array(sorted(set(int(y) for y in years if np.isfinite(y))), dtype=np.int16)
    if uniq.size < n:
        return None
    last = set(uniq[-n:].tolist())
    return np.where(np.isin(years, list(last)))[0]


# =============================================================================
# Cache format (npz): to keep worker args small
# =============================================================================

def date_to_int_days(dts: np.ndarray) -> np.ndarray:
    dd = dts.astype("datetime64[D]")
    return dd.astype(np.int32)

def load_or_build_caches(
    *,
    main_csv: str,
    share_csv: str,
    out_dir: str,
    rebuild: bool,
    kaggle_rowset: bool,
) -> Tuple[Path, Path, Dict[str, Any]]:
    out = Path(out_dir)
    cache_dir = out / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    base_npz = cache_dir / ("base_cache_kaggle.npz" if kaggle_rowset else "base_cache_raw.npz")
    cd_npz = cache_dir / ("cd_cache_kaggle.npz" if kaggle_rowset else "cd_cache_raw.npz")
    meta_json = cache_dir / ("cache_meta_kaggle.json" if kaggle_rowset else "cache_meta_raw.json")

    if (not rebuild) and base_npz.exists() and cd_npz.exists() and meta_json.exists():
        meta = json.loads(meta_json.read_text(encoding="utf-8"))
        return base_npz, cd_npz, meta

    log("Loading CSVs to build cache...")
    df_main = pd.read_csv(main_csv)
    df_share = pd.read_csv(share_csv)

    futures_cols = [c for c in df_main.columns if c.startswith("futures_")]
    if not futures_cols:
        raise RuntimeError("No futures_* columns found in main_csv.")

    if kaggle_rowset:
        log("Building Kaggle-aligned base row-set (sample_submission-style + dropna)...")
        base = build_kaggle_rowset_base(df_main, df_share, rolling_windows=(7,14,30))
        log(f"Base rows after dropna: {len(base):,}")
    else:
        log("Using RAW row-set (no sample_submission dropna).")
        base = df_main.copy()
        base["date_on"] = pd.to_datetime(base["date_on"], errors="coerce")
        base = base.dropna(subset=["date_on", "country_name"]).copy()

    # Keep only columns needed for scoring
    keep_cols = ["date_on", "country_name"] + futures_cols
    base = base[keep_cols].copy()
    base["date_on"] = pd.to_datetime(base["date_on"], errors="coerce")
    base = base.dropna(subset=["date_on"]).copy()

    countries = sorted(base["country_name"].astype(str).unique().tolist())
    cat = pd.Categorical(base["country_name"].astype(str), categories=countries)
    base_country_code = cat.codes.astype(np.int16)

    base_date_int = date_to_int_days(base["date_on"].to_numpy())
    dt_series = pd.to_datetime(base["date_on"])
    base_month = dt_series.dt.month.astype(np.int8).to_numpy()
    base_year = dt_series.dt.year.astype(np.int16).to_numpy()

    Y = base[futures_cols].to_numpy(dtype=np.float32, copy=False)

    log("Building country-day signals (submission.py weighting)...")
    cd_df, signal_cols = build_country_day_signals(df_main, df_share, country_whitelist=countries)

    cd_cat = pd.Categorical(cd_df["country_name"].astype(str), categories=countries)
    cd_country_code = cd_cat.codes.astype(np.int16)
    cd_date_int = date_to_int_days(cd_df["date_on"].to_numpy())
    cd_year = cd_df["year"].to_numpy(dtype=np.int16, copy=False)
    cd_month = cd_df["month"].to_numpy(dtype=np.int8, copy=False)
    Xsig = cd_df[signal_cols].to_numpy(dtype=np.float32, copy=False)

    log(f"Saving base cache -> {base_npz}")
    np.savez(
        base_npz,
        base_date_int=base_date_int.astype(np.int32),
        base_country_code=base_country_code.astype(np.int16),
        base_month=base_month.astype(np.int8),
        base_year=base_year.astype(np.int16),
        futures_cols=np.array(futures_cols, dtype=object),
        Y=Y.astype(np.float32),
        countries=np.array(countries, dtype=object),
    )

    log(f"Saving country-day cache -> {cd_npz}")
    np.savez(
        cd_npz,
        cd_country_code=cd_country_code.astype(np.int16),
        cd_date_int=cd_date_int.astype(np.int32),
        cd_year=cd_year.astype(np.int16),
        cd_month=cd_month.astype(np.int8),
        signal_cols=np.array(signal_cols, dtype=object),
        Xsig=Xsig.astype(np.float32),
    )

    meta = dict(
        built_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        kaggle_rowset=bool(kaggle_rowset),
        n_base_rows=int(base.shape[0]),
        n_countries=int(len(countries)),
        n_futures_cols=int(len(futures_cols)),
        n_cd_rows=int(cd_df.shape[0]),
        n_signal_cols=int(len(signal_cols)),
    )
    meta_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return base_npz, cd_npz, meta


def _load_npz(path: Path) -> Dict[str, Any]:
    with np.load(path, allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


# =============================================================================
# Fast date mapping helpers (precompute pos+ok per bucket)
# =============================================================================

def make_date_mapper(cd_dates: np.ndarray, bucket_dates: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    cd_dates = np.asarray(cd_dates, dtype=np.int32)
    bucket_dates = np.asarray(bucket_dates, dtype=np.int32)
    pos = np.searchsorted(cd_dates, bucket_dates)
    ok = (pos >= 0) & (pos < cd_dates.size)
    pos_clip = pos.clip(0, cd_dates.size - 1)
    ok = ok & (cd_dates[pos_clip] == bucket_dates)
    return pos.astype(np.int32), ok

def map_with_mapper(z_full: np.ndarray, pos: np.ndarray, ok: np.ndarray) -> np.ndarray:
    out = np.zeros(pos.shape[0], dtype=np.float64)
    if ok.any():
        out[ok] = z_full[pos[ok]]
    return out


# =============================================================================
# Stage 0 baseline scan
# =============================================================================

@dataclass
class BaselineHit:
    country: str
    month: int
    signal: str
    agg: str
    transform: str
    w: int
    shift: int
    proxy_cfcs: float
    robust_min_cfcs: float
    per_vy: Dict[int, float]
    n_rows: int
    n_years: int

def score_on_bucket(
    *,
    z_full: np.ndarray,
    mapper_pos: np.ndarray,
    mapper_ok: np.ndarray,
    bucket_years: np.ndarray,
    Y_bucket: np.ndarray,
    val_years_list: Sequence[int],
) -> Tuple[float, float, Dict[int, float]]:
    z = map_with_mapper(z_full, mapper_pos, mapper_ok)
    proxy, _ = cfcs_score(z, Y_bucket)
    per: Dict[int, float] = {}
    vals = []
    for vy in val_years_list:
        idx = last_n_years_subset(bucket_years, int(vy))
        if idx is None or idx.size < 30:
            per[int(vy)] = float("nan")
            continue
        s, _ = cfcs_score(z[idx], Y_bucket[idx])
        per[int(vy)] = float(s)
        if math.isfinite(s):
            vals.append(float(s))
    robust = float(min(vals)) if vals else 0.0
    return float(proxy), float(robust), per

def stage0_scan_country(
    *,
    base_npz: Path,
    cd_npz: Path,
    country_code: int,
    stage0_windows: Sequence[int],
    stage0_aggs: Sequence[str],
    stage0_transforms: Sequence[str],
    val_years_list: Sequence[int],
    min_rows: int,
    q_for_streakq: float,
    time_budget_end: float,
) -> List[BaselineHit]:
    """
    For one country:
      - precompute month buckets and their date mappers
      - for each signal, precompute agg series for each (agg,w) ONCE
      - score transforms quickly
      - return best config per (month, signal)
    """
    if time.time() > time_budget_end:
        return []

    base = _load_npz(base_npz)
    cd = _load_npz(cd_npz)

    countries = base["countries"].tolist()
    signal_cols = cd["signal_cols"].tolist()

    bc = base["base_country_code"].astype(np.int16)
    mask_country = (bc == int(country_code))
    if not mask_country.any():
        return []

    base_dates_all = base["base_date_int"].astype(np.int32)
    base_month_all = base["base_month"].astype(np.int8)
    base_year_all = base["base_year"].astype(np.int16)
    Y_all = base["Y"].astype(np.float32)

    idx_country = np.where(mask_country)[0]
    months_present = sorted(set(int(m) for m in base_month_all[idx_country].tolist()))

    # Country-day slice
    cdc = cd["cd_country_code"].astype(np.int16)
    idx_cd_country = np.where(cdc == int(country_code))[0]
    if idx_cd_country.size == 0:
        return []
    cd_dates = cd["cd_date_int"].astype(np.int32)[idx_cd_country]
    order_cd = np.argsort(cd_dates, kind="mergesort")
    idx_cd_country = idx_cd_country[order_cd]
    cd_dates = cd_dates[order_cd]
    Xsig = cd["Xsig"].astype(np.float32)[idx_cd_country]  # [n_days, n_signals]

    # Precompute bucket data per month: indices, mapper, years, Y
    buckets: Dict[int, Dict[str, Any]] = {}
    for month in months_present:
        idx_m = idx_country[base_month_all[idx_country] == int(month)]
        if idx_m.size < int(min_rows):
            continue
        bucket_dates = base_dates_all[idx_m]
        bucket_years = base_year_all[idx_m]
        Y_bucket = Y_all[idx_m].astype(np.float64)
        pos, ok = make_date_mapper(cd_dates, bucket_dates)
        buckets[int(month)] = dict(
            bucket_years=bucket_years,
            Y_bucket=Y_bucket,
            pos=pos,
            ok=ok,
            n_rows=int(idx_m.size),
            n_years=len(set(int(y) for y in bucket_years.tolist())),
        )
    if not buckets:
        return []

    country_name = str(countries[int(country_code)])

    hits: List[BaselineHit] = []
    # For each signal and month: find best
    for si, sig in enumerate(signal_cols):
        if time.time() > time_budget_end:
            break
        x_full = Xsig[:, si].astype(np.float64, copy=False)

        # cache agg series for this signal
        agg_cache: Dict[Tuple[str, int], np.ndarray] = {}
        def get_agg(agg: str, w: int) -> np.ndarray:
            key = (agg, int(w))
            if key in agg_cache:
                return agg_cache[key]
            s = compute_agg_no_shift(x_full, agg=agg, w=int(w), q_for_streakq=q_for_streakq)
            agg_cache[key] = s.astype(np.float64, copy=False)
            return agg_cache[key]

        for month, b in buckets.items():
            if time.time() > time_budget_end:
                break

            best: Optional[BaselineHit] = None
            for agg in stage0_aggs:
                # precompute agg series per window once
                for w in stage0_windows:
                    base_series = get_agg(agg, int(w))
                    # shift is fixed 0 in stage0
                    for transform in stage0_transforms:
                        z_full = apply_transform(base_series, transform)
                        z_full = ffill_bfill_0(z_full)
                        proxy, robust, per = score_on_bucket(
                            z_full=z_full,
                            mapper_pos=b["pos"],
                            mapper_ok=b["ok"],
                            bucket_years=b["bucket_years"],
                            Y_bucket=b["Y_bucket"],
                            val_years_list=val_years_list,
                        )
                        cand = BaselineHit(
                            country=country_name,
                            month=int(month),
                            signal=str(sig),
                            agg=str(agg),
                            transform=str(transform),
                            w=int(w),
                            shift=0,
                            proxy_cfcs=float(proxy),
                            robust_min_cfcs=float(robust),
                            per_vy=per,
                            n_rows=int(b["n_rows"]),
                            n_years=int(b["n_years"]),
                        )
                        if (best is None) or (cand.proxy_cfcs > best.proxy_cfcs):
                            best = cand
            if best is not None:
                hits.append(best)

    return hits


# =============================================================================
# Deep sweep for one (country, month, signal)
# =============================================================================

def deep_sweep_group(
    *,
    base_npz: Path,
    cd_npz: Path,
    country_name: str,
    month: int,
    signal: str,
    out_dir: Path,
    val_years_list: Sequence[int],
    min_rows: int,
    w_min: int,
    w_max: int,
    shift_min: int,
    shift_max: int,
    coarse_shifts: Sequence[int],
    aggs: Sequence[str],
    transforms: Sequence[str],
    top_windows_keep: int,
    top_pairs_keep: int,
    refine_w_radius: int,
    refine_shift_radius: int,
    q_for_streakq: float,
    save_top: int,
    save_top_public: int,
    public_vy: int,
    time_budget_end: float,
    resume: bool,
) -> Optional[str]:
    if time.time() > time_budget_end:
        return None

    safe_country = country_name.replace(" ", "_")
    out_csv = out_dir / f"grid_{safe_country}_m{int(month):02d}_{signal}.csv.gz"
    if resume and out_csv.exists():
        return str(out_csv)

    base = _load_npz(base_npz)
    cd = _load_npz(cd_npz)

    countries = base["countries"].tolist()
    try:
        country_code = int(countries.index(country_name))
    except ValueError:
        return None

    signal_cols = cd["signal_cols"].tolist()
    if signal not in signal_cols:
        return None
    sig_idx = int(signal_cols.index(signal))

    bc = base["base_country_code"].astype(np.int16)
    bm = base["base_month"].astype(np.int8)
    idx = np.where((bc == country_code) & (bm == int(month)))[0]
    if idx.size < int(min_rows):
        return None

    bucket_dates = base["base_date_int"].astype(np.int32)[idx]
    bucket_years = base["base_year"].astype(np.int16)[idx]
    Y_bucket = base["Y"].astype(np.float32)[idx].astype(np.float64)

    cdc = cd["cd_country_code"].astype(np.int16)
    idx_cd = np.where(cdc == country_code)[0]
    if idx_cd.size == 0:
        return None
    cd_dates = cd["cd_date_int"].astype(np.int32)[idx_cd]
    order_cd = np.argsort(cd_dates, kind="mergesort")
    idx_cd = idx_cd[order_cd]
    cd_dates = cd_dates[order_cd]
    x_full = cd["Xsig"].astype(np.float32)[idx_cd, sig_idx].astype(np.float64, copy=False)

    mapper_pos, mapper_ok = make_date_mapper(cd_dates, bucket_dates)

    w_min = int(max(1, w_min))
    w_max = int(max(w_min, w_max))
    w_grid = [w for w in make_window_grid(w_min, w_max) if w_min <= w <= w_max]

    coarse = sorted(set(int(s) for s in coarse_shifts if int(shift_min) <= int(s) <= int(shift_max)))
    if 0 not in coarse and int(shift_min) <= 0 <= int(shift_max):
        coarse = [0] + coarse

    rows: List[Dict[str, Any]] = []

    agg_cache: Dict[Tuple[str, int], np.ndarray] = {}
    def get_agg_series(agg: str, w: int) -> np.ndarray:
        key = (agg, int(w))
        if key in agg_cache:
            return agg_cache[key]
        s = compute_agg_no_shift(x_full, agg=agg, w=int(w), q_for_streakq=q_for_streakq)
        agg_cache[key] = s.astype(np.float64, copy=False)
        return agg_cache[key]

    def score_one(agg: str, transform: str, w: int, shift: int) -> Tuple[float, float, Dict[int, float]]:
        base_series = get_agg_series(agg, w)
        z_full = apply_transform(shift_array(base_series, int(shift)), transform)
        z_full = ffill_bfill_0(z_full)
        proxy, robust, per = score_on_bucket(
            z_full=z_full,
            mapper_pos=mapper_pos,
            mapper_ok=mapper_ok,
            bucket_years=bucket_years,
            Y_bucket=Y_bucket,
            val_years_list=val_years_list,
        )
        return proxy, robust, per

    # Stage A: shift=0, rank windows per (agg,transform)
    seeds: List[Tuple[float, str, str, int, int]] = []
    for agg in aggs:
        for transform in transforms:
            if time.time() > time_budget_end:
                break
            scored_w: List[Tuple[float, int]] = []
            for w in w_grid:
                if time.time() > time_budget_end:
                    break
                proxy, _, _ = score_one(agg, transform, int(w), 0)
                scored_w.append((float(proxy), int(w)))
            if not scored_w:
                continue
            scored_w.sort(reverse=True, key=lambda t: t[0])
            keep_ws = [w for _, w in scored_w[: min(int(top_windows_keep), len(scored_w))]]

            # Stage B: for each kept window, score coarse shifts; keep top 3 shifts
            for w in keep_ws:
                scored_s: List[Tuple[float, int]] = []
                for s in coarse:
                    if time.time() > time_budget_end:
                        break
                    proxy, _, _ = score_one(agg, transform, int(w), int(s))
                    scored_s.append((float(proxy), int(s)))
                scored_s.sort(reverse=True, key=lambda t: t[0])
                for proxy, s in scored_s[:3]:
                    seeds.append((float(proxy), agg, transform, int(w), int(s)))

    if not seeds:
        return None
    seeds.sort(reverse=True, key=lambda t: t[0])
    seeds = seeds[: min(int(top_pairs_keep), len(seeds))]

    # Refinement per seed: 1D shift refine, 1D window refine, small shift refine
    for _, agg, transform, w0, s0 in seeds:
        if time.time() > time_budget_end:
            break

        # shift refine (w fixed)
        s_lo = max(int(shift_min), int(s0) - int(refine_shift_radius))
        s_hi = min(int(shift_max), int(s0) + int(refine_shift_radius))
        best_s = int(s0)
        best_proxy = -1e9
        for s in range(s_lo, s_hi + 1):
            proxy, _, _ = score_one(agg, transform, int(w0), int(s))
            if proxy > best_proxy:
                best_proxy = float(proxy)
                best_s = int(s)

        # window refine (shift fixed)
        w_lo = max(int(w_min), int(w0) - int(refine_w_radius))
        w_hi = min(int(w_max), int(w0) + int(refine_w_radius))
        best_w = int(w0)
        best_proxy2 = -1e9
        for w in range(w_lo, w_hi + 1):
            proxy, _, _ = score_one(agg, transform, int(w), int(best_s))
            if proxy > best_proxy2:
                best_proxy2 = float(proxy)
                best_w = int(w)

        # small shift refine (w fixed)
        sr = max(5, int(refine_shift_radius // 3))
        s_lo2 = max(int(shift_min), int(best_s) - sr)
        s_hi2 = min(int(shift_max), int(best_s) + sr)
        best_s2 = int(best_s)
        best_proxy3 = -1e9
        for s in range(s_lo2, s_hi2 + 1):
            proxy, _, _ = score_one(agg, transform, int(best_w), int(s))
            if proxy > best_proxy3:
                best_proxy3 = float(proxy)
                best_s2 = int(s)

        proxy, robust, per = score_one(agg, transform, int(best_w), int(best_s2))
        row: Dict[str, Any] = dict(
            country=country_name,
            month=int(month),
            signal=str(signal),
            agg=str(agg),
            transform=str(transform),
            w=int(best_w),
            shift=int(best_s2),
            proxy_cfcs=float(proxy),
            robust_min_cfcs=float(robust),
            n_rows=int(idx.size),
        )
        for vy, sc in per.items():
            row[f"vy{int(vy)}_cfcs"] = float(sc) if math.isfinite(sc) else float("nan")
        rows.append(row)

    if not rows:
        return None

    keycols = ["country","month","signal","agg","transform","w","shift"]
    df_all = pd.DataFrame(rows).drop_duplicates(subset=keycols)

    # Always keep robust-first rows
    df_rob = df_all.sort_values(["robust_min_cfcs", "proxy_cfcs"], ascending=False).head(int(save_top))

    # Also keep public-proxy rows (so they don't get truncated away)
    pub_col = f"vy{int(public_vy)}_cfcs"
    if int(save_top_public) > 0 and pub_col in df_all.columns:
        df_pub = df_all.sort_values([pub_col, "proxy_cfcs"], ascending=False).head(int(save_top_public))
        df_out = pd.concat([df_rob, df_pub], ignore_index=True).drop_duplicates(subset=keycols)
    else:
        df_out = df_rob

    # Final ordering for the saved grid file (robust-first)
    df_out = df_out.sort_values(["robust_min_cfcs", "proxy_cfcs"], ascending=False)

    out_dir.mkdir(parents=True, exist_ok=True)
    with gzip.open(out_csv, "wt", encoding="utf-8") as f:
        df_out.to_csv(f, index=False)

    return str(out_csv)


# =============================================================================
# Main orchestration
# =============================================================================

def parse_int_list(s: str) -> List[int]:
    s = (s or "").strip()
    if not s:
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip()]

def parse_str_list(s: str) -> List[str]:
    s = (s or "").strip()
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--main_csv", required=True)
    ap.add_argument("--share_csv", required=True)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--n_jobs", type=int, default=24)
    ap.add_argument("--time_budget_hours", type=float, default=12.0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--rebuild_cache", action="store_true")

    ap.add_argument("--no_kaggle_rowset", action="store_true",
                    help="Disable Kaggle row-set reconstruction + dropna (debug only).")

    ap.add_argument("--val_years_list", type=str, default="2,3,4,5")
    ap.add_argument("--min_rows", type=int, default=60)

    # Stage 0 controls
    ap.add_argument("--stage0_windows", type=str, default="7,14,21,30,45,60,90,98,102,112,120,140,168,180,200,224,240,252,280,300,330,365,400,450,540,730,1095")
    ap.add_argument("--stage0_aggs", type=str, default="ma")
    ap.add_argument("--stage0_transforms", type=str, default="identity,square,signlog1p")
    ap.add_argument("--stage0_keep_per_country_month", type=int, default=3)

    # Deep selection
    ap.add_argument("--deep_top_groups", type=int, default=400,
                    help="Deep sweep top-N (country,month,signal) groups by baseline proxy. 0 => deep all.")
    ap.add_argument("--deep_min_proxy", type=float, default=70.0)

    # Deep sweep search space
    ap.add_argument("--w_min", type=int, default=2)
    ap.add_argument("--w_max", type=int, default=2500)
    ap.add_argument("--shift_min", type=int, default=-365)
    ap.add_argument("--shift_max", type=int, default=365)
    ap.add_argument("--coarse_shifts", type=str, default="-365,-240,-180,-150,-120,-90,-60,-30,0,30,60,90,120,150,180,240,365")
    ap.add_argument("--aggs", type=str, default="ma,max,ewm,std,streakq85,streakthr0.5,runlenth0.5")
    ap.add_argument("--transforms", type=str, default="identity,square,signlog1p")
    ap.add_argument("--top_windows_keep", type=int, default=60)
    ap.add_argument("--top_pairs_keep", type=int, default=80)
    ap.add_argument("--refine_w_radius", type=int, default=80)
    ap.add_argument("--refine_shift_radius", type=int, default=45)
    ap.add_argument("--save_top", type=int, default=500)
    ap.add_argument("--streak_q", type=float, default=0.85)
    ap.add_argument("--public_vy", type=int, default=2,
                help="Which vyN_cfcs column to treat as the public-LB proxy (e.g. 2 => vy2_cfcs).")
    ap.add_argument("--save_top_public", type=int, default=200,
                help="Per-group: also keep this many rows ranked by vy{public_vy}_cfcs so public-strong rows don't get truncated.")


    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    time_budget_end = time.time() + float(args.time_budget_hours) * 3600.0
    val_years_list = parse_int_list(args.val_years_list)
    stage0_windows = parse_int_list(args.stage0_windows)
    stage0_aggs = parse_str_list(args.stage0_aggs)
    stage0_transforms = parse_str_list(args.stage0_transforms)
    coarse_shifts = parse_int_list(args.coarse_shifts)
    aggs = parse_str_list(args.aggs)
    transforms = parse_str_list(args.transforms)

    kaggle_rowset = not bool(args.no_kaggle_rowset)

    log("Preparing caches...")
    base_npz, cd_npz, meta = load_or_build_caches(
        main_csv=args.main_csv,
        share_csv=args.share_csv,
        out_dir=str(out_dir),
        rebuild=bool(args.rebuild_cache),
        kaggle_rowset=bool(kaggle_rowset),
    )
    log(f"Cache meta: {meta}")

    # ---------------- Stage 0 ----------------
    baseline_path = out_dir / "baseline_scan_all.csv"
    if args.resume and baseline_path.exists():
        log(f"Resume: loading existing baseline scan: {baseline_path}")
        baseline_all = pd.read_csv(baseline_path)
    else:
        log("Stage 0: baseline scan across ALL countries...")
        base = _load_npz(base_npz)
        n_countries = int(len(base["countries"].tolist()))
        country_codes = list(range(n_countries))

        hits_nested: List[List[BaselineHit]] = Parallel(n_jobs=int(args.n_jobs), backend="loky", batch_size=1)(
            delayed(stage0_scan_country)(
                base_npz=base_npz,
                cd_npz=cd_npz,
                country_code=cc,
                stage0_windows=stage0_windows,
                stage0_aggs=stage0_aggs,
                stage0_transforms=stage0_transforms,
                val_years_list=val_years_list,
                min_rows=int(args.min_rows),
                q_for_streakq=float(args.streak_q),
                time_budget_end=time_budget_end,
            )
            for cc in country_codes
        )

        hits = [h for sub in hits_nested for h in sub]
        if not hits:
            raise SystemExit("Stage 0 produced no hits (check min_rows / cache / data).")

        rows = []
        for h in hits:
            r = dict(
                country=h.country,
                month=int(h.month),
                signal=h.signal,
                agg=h.agg,
                transform=h.transform,
                w=int(h.w),
                shift=int(h.shift),
                proxy_cfcs=float(h.proxy_cfcs),
                robust_min_cfcs=float(h.robust_min_cfcs),
                n_rows=int(h.n_rows),
                n_years=int(h.n_years),
            )
            for vy, sc in h.per_vy.items():
                r[f"vy{int(vy)}_cfcs"] = float(sc) if math.isfinite(sc) else float("nan")
            rows.append(r)

        baseline_all = pd.DataFrame(rows)
        baseline_all = baseline_all.sort_values(["proxy_cfcs","robust_min_cfcs"], ascending=False).reset_index(drop=True)
        baseline_all.to_csv(baseline_path, index=False)
        log(f"Wrote: {baseline_path}")

    # Keep only top signals per (country, month)
    baseline_all = baseline_all.sort_values(["country","month","proxy_cfcs"], ascending=[True, True, False]).copy()
    baseline_top = (
        baseline_all.groupby(["country","month"], as_index=False, sort=True)
        .head(int(args.stage0_keep_per_country_month))
        .copy()
    )

    # Select deep groups
    baseline_top = baseline_top.sort_values(["proxy_cfcs","robust_min_cfcs"], ascending=False).reset_index(drop=True)
    baseline_top = baseline_top[baseline_top["proxy_cfcs"] >= float(args.deep_min_proxy)].copy()

    if int(args.deep_top_groups) > 0:
        deep_groups = baseline_top.head(int(args.deep_top_groups)).copy()
    else:
        deep_groups = baseline_top.copy()

    if deep_groups.empty:
        raise SystemExit("No deep groups after filtering. Lower --deep_min_proxy or increase --stage0_keep_per_country_month.")

    deep_list = list(deep_groups[["country","month","signal"]].itertuples(index=False, name=None))
    log(f"Deep sweep groups: {len(deep_list)} (deep_top_groups={args.deep_top_groups}, deep_min_proxy={args.deep_min_proxy})")

    coarse_windows_count = len(make_window_grid(int(args.w_min), int(args.w_max)))
    log(f"Deep sweep search summary: aggs={len(aggs)} transforms={len(transforms)} coarse_windows≈{coarse_windows_count} coarse_shifts={len(coarse_shifts)}")
    if int(args.deep_top_groups) == 0:
        log("WARNING: deep_top_groups=0 means 'deep all'. This can take a VERY long time.")

    # ---------------- Stage 1 ----------------
    grids_dir = out_dir / "grids"
    grids_dir.mkdir(parents=True, exist_ok=True)

    def _deep_one(tup: Tuple[str, int, str]) -> Optional[str]:
        c, m, s = tup
        return deep_sweep_group(
            base_npz=base_npz,
            cd_npz=cd_npz,
            country_name=str(c),
            month=int(m),
            signal=str(s),
            out_dir=grids_dir,
            val_years_list=val_years_list,
            min_rows=int(args.min_rows),
            w_min=int(args.w_min),
            w_max=int(args.w_max),
            shift_min=int(args.shift_min),
            shift_max=int(args.shift_max),
            coarse_shifts=coarse_shifts,
            aggs=aggs,
            transforms=transforms,
            top_windows_keep=int(args.top_windows_keep),
            top_pairs_keep=int(args.top_pairs_keep),
            refine_w_radius=int(args.refine_w_radius),
            refine_shift_radius=int(args.refine_shift_radius),
            q_for_streakq=float(args.streak_q),
            save_top=int(args.save_top),
            time_budget_end=time_budget_end,
            resume=bool(args.resume),
            public_vy=int(args.public_vy),
            save_top_public=int(args.save_top_public),
        )

    log("Stage 1: starting deep sweep...")
    deep_paths = Parallel(n_jobs=int(args.n_jobs), backend="loky", batch_size=1)(
        delayed(_deep_one)(t) for t in deep_list
    )
    deep_paths = [p for p in deep_paths if p is not None]
    log(f"Deep sweep produced {len(deep_paths)} grid files.")

    # Consolidate
    log("Consolidating grid files...")
    frames = []
    for p in sorted(grids_dir.glob("grid_*.csv.gz")):
        try:
            df = pd.read_csv(p)
            df["grid_file"] = str(p.name)
            frames.append(df)
        except Exception:
            continue
    if not frames:
        raise SystemExit("No grid files found to consolidate.")

    grid_all = pd.concat(frames, ignore_index=True)
    grid_all_path = out_dir / "grid_all_candidates.csv"
    grid_all.to_csv(grid_all_path, index=False)
    log(f"Wrote: {grid_all_path} | rows={len(grid_all):,}")

    # Recommendations
    grid_all2 = grid_all.copy()
    grid_all2["abs_shift"] = grid_all2["shift"].abs()

    robust_pick = grid_all2.sort_values(["robust_min_cfcs","proxy_cfcs","abs_shift"], ascending=[False, False, True]).head(25)

    best_rob = float(grid_all2["robust_min_cfcs"].max())
    aggr_pool = grid_all2[grid_all2["robust_min_cfcs"] >= (best_rob - 1.0)].copy()
    aggr_pick = aggr_pool.sort_values(["proxy_cfcs","robust_min_cfcs","abs_shift"], ascending=[False, False, True]).head(25)

    rec_path = out_dir / "recommendations.txt"
    lines = []
    lines.append("TOP-25 ROBUST (max robust_min_cfcs, tie proxy, prefer smaller |shift|)\n")
    lines.append(robust_pick[["country","month","signal","agg","transform","w","shift","robust_min_cfcs","proxy_cfcs"]].to_string(index=False))
    lines.append("\n\nTOP-25 AGGRESSIVE (max proxy_cfcs among robust>=best-1)\n")
    lines.append(aggr_pick[["country","month","signal","agg","transform","w","shift","robust_min_cfcs","proxy_cfcs"]].to_string(index=False))
    lines.append("\n\nNOTES\n")
    lines.append(" - This script scores ONE gated feature per row (single-feature submissions).\n")
    lines.append(" - If you want Kaggle-faithful CFCS, keep only one climate_risk_* column in your submission.\n")
    lines.append(" - shift < 0 = lead (uses future climate relative to futures date); shift > 0 = lag.\n")
    rec_path.write_text("\n".join(lines), encoding="utf-8")
    log(f"Wrote: {rec_path}")

    pub_col = f"vy{int(args.public_vy)}_cfcs"
    if pub_col in grid_all.columns:
        g = grid_all.copy()
        g["abs_shift"] = g["shift"].abs()

        # pick public-proxy: prioritize vyN, then proxy, then robustness, then prefer shift closer to 0
        g = g.sort_values([pub_col, "proxy_cfcs", "robust_min_cfcs", "abs_shift"],
                        ascending=[False, False, False, True])

        pub_pick = g.drop_duplicates(subset=["country","month","signal","agg","transform","w","shift"]).head(25)

        outp = out_dir / "recommendations_public.txt"
        lines = []
        lines.append(f"Top PUBLIC-PROXY picks (ranked by {pub_col})")
        lines.append("")
        for _, r in pub_pick.iterrows():
            lines.append(
                f"{r['country']}\tm{int(r['month'])}\t{r['signal']}\t{r['agg']}\t{r['transform']}\t"
                f"w={int(r['w'])}\tshift={int(r['shift'])}\t"
                f"{pub_col}={float(r[pub_col]):.2f}\trobust_min={float(r['robust_min_cfcs']):.2f}\tproxy={float(r['proxy_cfcs']):.2f}"
            )
        outp.write_text("\n".join(lines), encoding="utf-8")
        log(f"Wrote: {outp}")

    log("Done.")

if __name__ == "__main__":
    main()