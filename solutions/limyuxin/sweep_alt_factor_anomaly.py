#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""sweep_alt_factor_anomaly.py

This script uses a *factor-screen + seasonal de-trending* workflow:

1) **Seasonal anomaly variants**
   Markets often care about *surprises* vs the seasonal norm. We create
   day-of-year z-scores (zDOY) for every country-date climate signal.

2) **Futures factor screening**
   For each (country, month) bucket, we compute a weighted PC1 factor of the
   futures columns and screen candidate climate features by correlation to this
   factor. This quickly finds features that should correlate with *many* futures
   columns (helpful for SigCount/AvgSig CFCS components).

3) **Coordinate-descent refinement**
   Instead of scanning huge grids, we refine (window, shift, transform) around
   the screened candidates using a few focused sweeps.
"""

from __future__ import annotations

# ---- Threading hygiene (avoid oversubscription when using many threads) ----
import os as _os
_os.environ.setdefault("OMP_NUM_THREADS", "1")
_os.environ.setdefault("MKL_NUM_THREADS", "1")
_os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
_os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import heapq
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from joblib import Parallel, delayed
except Exception as e:
    raise RuntimeError("joblib is required. Install via: pip install joblib") from e


# =============================================================================
# Logging
# =============================================================================

def _now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def log(msg: str) -> None:
    print(f"[{_now_str()}] {msg}", flush=True)


# =============================================================================
# Argument parsing helpers
# =============================================================================

def parse_csv_list(s: str) -> List[str]:
    s = (s or "").strip()
    if not s:
        return []
    return [p.strip() for p in s.split(",") if p.strip()]


def parse_ints(s: str) -> List[int]:
    out: List[int] = []
    for tok in parse_csv_list(s):
        out.append(int(tok))
    return out


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Metric: CFCS (Kaggle-style rounding + significance threshold)
# =============================================================================

def weighted_corr(x: np.ndarray, y: np.ndarray, w: np.ndarray, eps: float = 1e-12) -> float:
    """Weighted Pearson correlation.

    If you duplicated each observation i exactly w[i] times, the ordinary Pearson
    correlation on that expanded dataset matches this weighted correlation.

    We treat non-finite values as missing.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)

    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    if m.sum() < 3:
        return float("nan")

    xv = x[m]
    yv = y[m]
    wv = w[m]
    sw = float(wv.sum())
    if sw <= eps:
        return float("nan")

    mx = float((wv * xv).sum() / sw)
    my = float((wv * yv).sum() / sw)

    dx = xv - mx
    dy = yv - my

    num = float((wv * dx * dy).sum())
    denx = float((wv * dx * dx).sum())
    deny = float((wv * dy * dy).sum())
    if denx <= eps or deny <= eps:
        return float("nan")

    return float(num / math.sqrt(denx * deny))


def corr_vector_weighted(z: np.ndarray, Y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Correlation between z (n,) and each Y[:,j] (n,k) under weights w (n,)."""
    z = np.asarray(z, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)

    out = np.full((Y.shape[1],), np.nan, dtype=np.float64)
    for j in range(Y.shape[1]):
        out[j] = weighted_corr(z, Y[:, j], w)
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


def cfcs_score_weighted(z: np.ndarray, Y: np.ndarray, w: np.ndarray) -> Tuple[float, Dict[str, Any]]:
    return cfcs_from_corrs(corr_vector_weighted(z, Y, w))


# =============================================================================
# Rolling/shift/transform utilities (submission-like semantics)
# =============================================================================

def shift_array(x: np.ndarray, shift: int) -> np.ndarray:
    """pandas-like shift: out[t] = x[t - shift]."""
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


def ffill_bfill_fill0(x: np.ndarray) -> np.ndarray:
    """Forward-fill, then backward-fill, then fill remaining NaNs with 0."""
    x = np.asarray(x, dtype=np.float64).copy()
    x[~np.isfinite(x)] = np.nan

    # forward fill
    last = np.nan
    for i in range(x.shape[0]):
        if np.isfinite(x[i]):
            last = x[i]
        else:
            x[i] = last

    # backward fill
    last = np.nan
    for i in range(x.shape[0] - 1, -1, -1):
        if np.isfinite(x[i]):
            last = x[i]
        else:
            x[i] = last

    x[~np.isfinite(x)] = 0.0
    return x


def rolling_mean_min1(x: np.ndarray, w: int) -> np.ndarray:
    """Rolling mean with min_periods=1, ignoring NaNs, vectorized."""
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    if w <= 1:
        return x.copy()

    xx = np.where(np.isfinite(x), x, 0.0)
    m = np.isfinite(x).astype(np.int32)

    cs = np.concatenate([[0.0], np.cumsum(xx)])
    cnt = np.concatenate([[0], np.cumsum(m)])

    idx = np.arange(n, dtype=np.int32)
    j0 = np.maximum(0, idx - w + 1)

    s = cs[idx + 1] - cs[j0]
    c = cnt[idx + 1] - cnt[j0]

    out = np.full((n,), np.nan, dtype=np.float64)
    good = c > 0
    out[good] = s[good] / c[good]
    return out


def rolling_std_min1(x: np.ndarray, w: int) -> np.ndarray:
    """Rolling std with min_periods=1, ignoring NaNs, vectorized."""
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    if w <= 1:
        return np.zeros_like(x, dtype=np.float64)

    xx = np.where(np.isfinite(x), x, 0.0)
    m = np.isfinite(x).astype(np.int32)

    cs = np.concatenate([[0.0], np.cumsum(xx)])
    cs2 = np.concatenate([[0.0], np.cumsum(xx * xx)])
    cnt = np.concatenate([[0], np.cumsum(m)])

    idx = np.arange(n, dtype=np.int32)
    j0 = np.maximum(0, idx - w + 1)

    s = cs[idx + 1] - cs[j0]
    s2 = cs2[idx + 1] - cs2[j0]
    c = cnt[idx + 1] - cnt[j0]

    out = np.full((n,), np.nan, dtype=np.float64)
    good = c > 0
    mean = np.zeros_like(out)
    mean[good] = s[good] / c[good]
    var = np.zeros_like(out)
    var[good] = np.maximum(0.0, (s2[good] / c[good]) - (mean[good] * mean[good]))
    out[good] = np.sqrt(var[good])
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
    with np.errstate(invalid="ignore"):
        b = (x >= thr).astype(np.float64)
    b[~np.isfinite(x)] = np.nan
    return rolling_mean_min1(b, w)


def apply_agg(x_shift: np.ndarray, agg: str, w: int) -> np.ndarray:
    agg = (agg or "").strip().lower()
    if agg == "ma":
        return rolling_mean_min1(x_shift, int(w))
    if agg == "std":
        return rolling_std_min1(x_shift, int(w))
    if agg == "ewm":
        return ewm_mean(x_shift, int(w))
    if agg.startswith("streakq"):
        q = float(agg.replace("streakq", "")) / 100.0
        if np.all(~np.isfinite(x_shift)):
            thr = 0.0
        else:
            thr = float(np.nanquantile(x_shift, q))
            if not np.isfinite(thr):
                thr = 0.0
        return streak_fraction(x_shift, int(w), thr)
    if agg.startswith("streakthr"):
        thr = float(agg.replace("streakthr", ""))
        return streak_fraction(x_shift, int(w), thr)
    raise ValueError(f"Unsupported agg: {agg}")


def apply_transform(x: np.ndarray, name: str) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    name = (name or "").strip().lower()
    if name == "identity":
        return x
    if name == "square":
        return np.sign(x) * (x * x)
    if name == "signlog1p":
        return np.sign(x) * np.log1p(np.abs(x))
    raise ValueError(f"Unknown transform: {name}")


def compute_feature_series(
    x_base: np.ndarray,
    *,
    agg: str,
    w: int,
    shift: int,
    transform: str,
) -> np.ndarray:
    x_shift = shift_array(x_base, int(shift))
    x_agg = apply_agg(x_shift, agg, int(w))
    x_tr = apply_transform(x_agg, transform)
    x_fill = ffill_bfill_fill0(x_tr)
    return x_fill


# =============================================================================
# Kaggle-aligned row-set (sample-submission style)
# =============================================================================

RISK_MAP = {
    "heat": "heat_stress",
    "cold": "unseasonably_cold",
    "wet": "excess_precip",
    "dry": "drought",
}


def build_kaggle_rowset_base(
    df: pd.DataFrame,
    share_df: pd.DataFrame,
    *,
    rolling_windows: Sequence[int] = (7, 14, 30),
) -> pd.DataFrame:
    """Recreate the sample-submission engineered dataframe and dropna().

    This matches the Kaggle evaluator's effective row-setOW-set (what your submission.py
    must map onto), so we get leaderboard-aligned scoring.

    We keep this function intentionally close to sample_submission behavior:
    - compute region-level risk scores
    - rolling mean/max per region
    - momentum features (diff/accel) per region
    - country-level aggregates (mean/max/std/sums)
    - dropna
    """

    df = df.copy()
    df["date_on"] = pd.to_datetime(df["date_on"], errors="coerce")

    # merge production shares (sample submission fills missing with 1.0)
    share = share_df[["region_id", "percent_country_production"]].copy()
    df = df.merge(share, on="region_id", how="left")
    df["percent_country_production"] = df["percent_country_production"].astype(np.float64).fillna(1.0)

    eps = 1e-6

    # compute scores per risk type
    score_cols: Dict[str, str] = {}
    weighted_cols: Dict[str, str] = {}
    for short, kind in RISK_MAP.items():
        low = f"climate_risk_cnt_locations_{kind}_risk_low"
        med = f"climate_risk_cnt_locations_{kind}_risk_medium"
        high = f"climate_risk_cnt_locations_{kind}_risk_high"

        tot = (df[low] + df[med] + df[high]).astype(np.float64)
        score = (df[med].astype(np.float64) + 2.0 * df[high].astype(np.float64)) / (tot + eps)

        score_col = f"climate_risk_{short}_score"
        df[score_col] = score
        score_cols[short] = score_col

        wcol = f"climate_risk_{short}_weighted"
        # sample submission uses (percent/100) weights in a *sum* aggregation
        df[wcol] = score * (df["percent_country_production"].astype(np.float64) / 100.0)
        weighted_cols[short] = wcol

    # composite indices
    df["climate_risk_composite_max"] = df[list(score_cols.values())].max(axis=1)
    df["climate_risk_composite_mean"] = df[list(score_cols.values())].mean(axis=1)

    # region-level rolling features + momentum
    df = df.sort_values(["region_id", "date_on"]).reset_index(drop=True)
    for short, sc in score_cols.items():
        # rolling
        for w in rolling_windows:
            df[f"{sc}_ma_{w}"] = (
                df.groupby("region_id", sort=False)[sc]
                  .transform(lambda s: s.rolling(window=w, min_periods=1).mean())
            )
            df[f"{sc}_max_{w}"] = (
                df.groupby("region_id", sort=False)[sc]
                  .transform(lambda s: s.rolling(window=w, min_periods=1).max())
            )

        # momentum
        df[f"{sc}_change_1d"] = df.groupby("region_id", sort=False)[sc].diff(1)
        df[f"{sc}_change_7d"] = df.groupby("region_id", sort=False)[sc].diff(7)
        df[f"{sc}_acceleration"] = df.groupby("region_id", sort=False)[f"{sc}_change_1d"].diff(1)

    # country-level aggregates
    for short, sc in score_cols.items():
        wcol = weighted_cols[short]
        agg = (
            df.groupby(["country_name", "date_on"], sort=False)
              .agg(
                  **{
                      f"country_{short}_score_mean": (sc, "mean"),
                      f"country_{short}_score_max": (sc, "max"),
                      f"country_{short}_score_std": (sc, "std"),
                      f"country_{short}_weighted_sum": (wcol, "sum"),
                      f"country_{short}_production_sum": ("percent_country_production", "sum"),
                  }
              )
              .reset_index()
        )
        df = df.merge(agg, on=["country_name", "date_on"], how="left")

    # final row-set
    df = df.dropna().reset_index(drop=True)
    return df


# =============================================================================
# Climate signals (different weighting modes + seasonal variants)
# =============================================================================

WEIGHT_MODES = ("share_norm_fill1", "share_only_norm", "share_plus_locations")


def build_country_day_signals(
    df: pd.DataFrame,
    share_df: pd.DataFrame,
    *,
    weight_mode: str,
    country_whitelist: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """Build country-date signals (production-weighted means).

    Differences vs earlier scripts:
    - Supports multiple weighting modes:
      * share_norm_fill1: missing share -> 1.0 then normalize (old behavior)
      * share_only_norm: missing share -> 0.0 then normalize among known regions
      * share_plus_locations: missing share -> mean #locations proxy then normalize

    Returns
    -------
    cd_df: DataFrame with columns [country_name, date_on, year, month, ]
    signal_cols: list of signal column names
    """

    weight_mode = (weight_mode or "").strip().lower()
    if weight_mode not in WEIGHT_MODES:
        raise ValueError(f"Unknown weight_mode={weight_mode}. Choose from {WEIGHT_MODES}")

    df = df[[
        "country_name",
        "region_id",
        "date_on",
        *[f"climate_risk_cnt_locations_{kind}_risk_{lvl}" for kind in RISK_MAP.values() for lvl in ("low", "medium", "high")],
    ]].copy()

    df["date_on"] = pd.to_datetime(df["date_on"], errors="coerce")

    # region weights
    regions = df[["region_id", "country_name"]].drop_duplicates().copy()
    regions = regions.merge(share_df[["region_id", "percent_country_production"]], on="region_id", how="left")

    if weight_mode == "share_plus_locations":
        # proxy by average total locations in heat-stress counts
        heat_low = "climate_risk_cnt_locations_heat_stress_risk_low"
        heat_med = "climate_risk_cnt_locations_heat_stress_risk_medium"
        heat_high = "climate_risk_cnt_locations_heat_stress_risk_high"
        tot_loc = (df[heat_low] + df[heat_med] + df[heat_high]).astype(np.float64)
        loc_proxy = df[["region_id"]].copy()
        loc_proxy["loc_proxy"] = tot_loc
        loc_proxy = loc_proxy.groupby("region_id", sort=False)["loc_proxy"].mean().reset_index()
        regions = regions.merge(loc_proxy, on="region_id", how="left")
    else:
        regions["loc_proxy"] = np.nan

    w_raw = regions["percent_country_production"].astype(np.float64)

    if weight_mode == "share_norm_fill1":
        w_raw = w_raw.fillna(1.0)
        w_raw = w_raw.where(w_raw > 0, 1.0)
    elif weight_mode == "share_only_norm":
        w_raw = w_raw.fillna(0.0)
        w_raw = w_raw.where(w_raw > 0, 0.0)
        # fallback: if a country has all zeros, use 1.0 for all its regions
        # (avoids producing all-zero signals)
        sums = w_raw.groupby(regions["country_name"]).transform("sum")
        w_raw = np.where(sums.to_numpy() <= 0, 1.0, w_raw.to_numpy())
        w_raw = pd.Series(w_raw, index=regions.index)
    elif weight_mode == "share_plus_locations":
        # if share present and >0 use it; else fallback to loc_proxy; else 1.0
        w_loc = regions["loc_proxy"].astype(np.float64)
        w_raw = w_raw.where(w_raw.notna() & (w_raw > 0), np.nan)
        w_raw = w_raw.fillna(w_loc)
        w_raw = w_raw.fillna(1.0)
        w_raw = w_raw.where(w_raw > 0, 1.0)

    regions["w_raw"] = w_raw.astype(np.float64)

    if country_whitelist is not None:
        regions = regions[regions["country_name"].isin(list(country_whitelist))].copy()

    regions["prod_w"] = regions["w_raw"] / regions.groupby("country_name", sort=False)["w_raw"].transform("sum")
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

        sev2 = (df[med].astype(np.float64) + 2.0 * df[high].astype(np.float64)) / (tot + eps)
        sev1 = (df[med].astype(np.float64) + 1.0 * df[high].astype(np.float64)) / (tot + eps)
        hi = df[high].astype(np.float64) / (tot + eps)

        tmp[f"w_{short}_sev"] = tmp["prod_w"] * sev2
        tmp[f"w_{short}_wapr"] = tmp["prod_w"] * sev1
        tmp[f"w_{short}_high"] = tmp["prod_w"] * hi

    sum_cols = ["prod_w"] + [c for c in tmp.columns if c.startswith("w_")]
    cd = tmp.groupby(["country_name", "date_on"], sort=False)[sum_cols].sum().reset_index()

    den = cd["prod_w"].to_numpy(np.float64) + 1e-12

    out = pd.DataFrame({
        "country_name": cd["country_name"].astype(str).values,
        "date_on": pd.to_datetime(cd["date_on"]).values,
    })

    for short in RISK_MAP.keys():
        out[f"{short}_sev_wmean"] = (cd[f"w_{short}_sev"].to_numpy(np.float64) / den).astype(np.float32)
        out[f"{short}_wapr_wmean"] = (cd[f"w_{short}_wapr"].to_numpy(np.float64) / den).astype(np.float32)
        out[f"{short}_high_wmean"] = (cd[f"w_{short}_high"].to_numpy(np.float64) / den).astype(np.float32)
        signal_cols.extend([f"{short}_sev_wmean", f"{short}_wapr_wmean", f"{short}_high_wmean"])

    # composites
    out["wet_dry_diff"] = (out["wet_sev_wmean"] - out["dry_sev_wmean"]).astype(np.float32)
    out["wet_dry_wapr_diff"] = (out["wet_wapr_wmean"] - out["dry_wapr_wmean"]).astype(np.float32)
    out["temp_stress_max"] = out[["heat_sev_wmean", "cold_sev_wmean"]].max(axis=1).astype(np.float32)
    out["precip_stress_max"] = out[["wet_sev_wmean", "dry_sev_wmean"]].max(axis=1).astype(np.float32)
    out["overall_stress_max"] = out[["heat_sev_wmean", "cold_sev_wmean", "wet_sev_wmean", "dry_sev_wmean"]].max(axis=1).astype(np.float32)
    out["overall_stress_mean"] = out[["heat_sev_wmean", "cold_sev_wmean", "wet_sev_wmean", "dry_sev_wmean"]].mean(axis=1).astype(np.float32)
    signal_cols.extend(["wet_dry_diff", "wet_dry_wapr_diff", "temp_stress_max", "precip_stress_max", "overall_stress_max", "overall_stress_mean"])

    out["date_on"] = pd.to_datetime(out["date_on"])
    out["year"] = out["date_on"].dt.year.astype(np.int16)
    out["month"] = out["date_on"].dt.month.astype(np.int8)

    out = out.sort_values(["country_name", "date_on"]).reset_index(drop=True)
    return out, signal_cols


# =============================================================================
# Futures factor (PC1) per (country, month)
# =============================================================================

def futures_pc1_factor(Y: np.ndarray, w: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Compute a weighted PC1 score series for futures matrix Y (n,k)."""
    Y = np.asarray(Y, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)

    m = np.isfinite(Y).all(axis=1) & np.isfinite(w) & (w > 0)
    if m.sum() < 5:
        return np.full((Y.shape[0],), np.nan, dtype=np.float64)

    Ym = Y[m]
    wm = w[m]
    sw = float(wm.sum())

    mu = (wm[:, None] * Ym).sum(axis=0) / sw
    Z = Ym - mu
    sig = np.sqrt((wm[:, None] * (Z * Z)).sum(axis=0) / sw)
    sig = np.where(sig > eps, sig, 1.0)
    Z = Z / sig

    # weighted covariance (k x k)
    C = (Z.T * wm) @ Z / sw

    # eigenvector of largest eigenvalue
    vals, vecs = np.linalg.eigh(C)
    v = vecs[:, int(np.argmax(vals))]

    scores = Z @ v

    out = np.full((Y.shape[0],), np.nan, dtype=np.float64)
    out[m] = scores
    # fill gaps to avoid NaNs in correlation
    out = ffill_bfill_fill0(out)
    return out


# =============================================================================
# Country data container
# =============================================================================

@dataclass
class CountryData:
    country: str
    dates: np.ndarray          # (n,) datetime64
    year: np.ndarray           # (n,) int16
    month: np.ndarray          # (n,) int8
    weight: np.ndarray         # (n,) float64
    Y: np.ndarray              # (n,k) float32
    X_raw: np.ndarray          # (n,s) float32
    X_zdoy: np.ndarray         # (n,s) float32


# =============================================================================
# Candidate containers
# =============================================================================

@dataclass
class ScreenHit:
    country: str
    month: int
    variant: str
    signal: str
    agg: str
    w: int
    shift: int
    corr_abs: float
    corr_signed: float


@dataclass
class RefinedCandidate:
    country: str
    month: int
    variant: str
    signal: str
    agg: str
    w: int
    shift: int
    transform: str

    proxy_corr_abs: float

    # CFCS scores
    cfcs_all: float
    cfcs_vy2: float
    cfcs_vy3: float
    cfcs_vy4: float
    cfcs_vy5: float
    robust_min: float

    # optional robustness checks
    detrended_vy2: float


# =============================================================================
# Time subset helpers
# =============================================================================

def last_n_years_mask(year_arr: np.ndarray, idx: np.ndarray, n: int) -> Optional[np.ndarray]:
    """Return boolean mask over idx selecting last n unique years within idx."""
    years = year_arr[idx]
    uniq = np.array(sorted(set(int(y) for y in years if np.isfinite(y))), dtype=np.int16)
    if uniq.size < n:
        return None
    last = set(uniq[-n:].tolist())
    return np.isin(year_arr, np.array(list(last), dtype=np.int16)) & idx


def detrend_by_year(x: np.ndarray, y: np.ndarray, years: np.ndarray, w: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Year-demean x and each column of y within each year (weighted)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    years = np.asarray(years, dtype=np.int16)
    w = np.asarray(w, dtype=np.float64)

    xd = x.copy()
    yd = y.copy()

    for yr in np.unique(years):
        m = (years == yr) & np.isfinite(w) & (w > 0)
        if m.sum() < 3:
            continue
        sw = w[m].sum()
        if sw <= 0:
            continue
        mx = (w[m] * xd[m]).sum() / sw
        xd[m] = xd[m] - mx

        my = (w[m][:, None] * yd[m]).sum(axis=0) / sw
        yd[m] = yd[m] - my

    return xd, yd, w


# =============================================================================
# Scoring candidate for a bucket
# =============================================================================

def score_bucket_cfcs(
    *,
    z: np.ndarray,
    Y: np.ndarray,
    year: np.ndarray,
    weight: np.ndarray,
    idx_month: np.ndarray,
    val_years_list: Sequence[int],
    min_rows: int = 40,
) -> Dict[str, float]:
    """Compute CFCS on full data and on last-n-years slices."""

    out: Dict[str, float] = {}

    # full
    if idx_month.sum() >= min_rows:
        out["all"] = cfcs_score_weighted(z[idx_month], Y[idx_month], weight[idx_month])[0]
    else:
        out["all"] = float("nan")

    for vy in val_years_list:
        m = last_n_years_mask(year, idx_month, int(vy))
        if m is None or m.sum() < min_rows:
            out[f"vy{vy}"] = float("nan")
        else:
            out[f"vy{vy}"] = cfcs_score_weighted(z[m], Y[m], weight[m])[0]

    return out


def score_bucket_detrended_vy2(
    *,
    z: np.ndarray,
    Y: np.ndarray,
    year: np.ndarray,
    weight: np.ndarray,
    idx_month: np.ndarray,
    min_rows: int = 40,
) -> float:
    """Extra robustness check: CFCS on last-2-years after year-demeaning."""
    m = last_n_years_mask(year, idx_month, 2)
    if m is None or m.sum() < min_rows:
        return float("nan")

    zd, Yd, wd = detrend_by_year(z[m], Y[m], year[m], weight[m])
    return cfcs_score_weighted(zd, Yd, wd)[0]


# =============================================================================
# Screening: stream through coarse candidates, keep top K per month
# =============================================================================

def screen_one_country(
    cdata: CountryData,
    *,
    signal_names: Sequence[str],
    variants: Sequence[str],
    aggs: Sequence[str],
    windows: Sequence[int],
    shifts: Sequence[int],
    top_k_per_month: int,
    min_abs_corr: float,
    min_rows_per_month: int,
) -> List[ScreenHit]:
    country = cdata.country

    # month indices and futures factors
    month_idx: Dict[int, np.ndarray] = {}
    month_factor: Dict[int, np.ndarray] = {}
    for m in range(1, 13):
        idx = (cdata.month == m)
        if idx.sum() < max(5, min_rows_per_month):
            continue
        month_idx[m] = idx
        month_factor[m] = futures_pc1_factor(cdata.Y[idx], cdata.weight[idx])

    if not month_idx:
        return []

    # Heaps per month
    # NOTE: heapq breaks ties by comparing the next tuple element. Since ScreenHit
    # is not orderable, include a deterministic tiebreaker integer.
    heaps: Dict[int, List[Tuple[float, int, ScreenHit]]] = {m: [] for m in month_idx.keys()}
    tie_counter = 0

    def push(m: int, hit: ScreenHit) -> None:
        nonlocal tie_counter
        h = heaps[m]
        key = hit.corr_abs
        tie_counter += 1
        item = (key, tie_counter, hit)
        if len(h) < top_k_per_month:
            heapq.heappush(h, item)
        else:
            # min-heap by corr_abs
            if key > h[0][0]:
                heapq.heapreplace(h, item)


    # variant -> matrix
    X_by_variant = {
        "raw": cdata.X_raw,
        "zdoy": cdata.X_zdoy,
    }

    for variant in variants:
        if variant not in X_by_variant:
            continue
        X = X_by_variant[variant]

        for j, sig_name in enumerate(signal_names):
            x_base = X[:, j].astype(np.float64, copy=False)

            for shift in shifts:
                x_shift = shift_array(x_base, int(shift))

                for agg in aggs:
                    for w in windows:
                        try:
                            x_feat = compute_feature_series(
                                x_shift,  # already shifted; compute_feature_series will shift again if we pass shift
                                agg=agg,
                                w=int(w),
                                shift=0,
                                transform="identity",
                            )
                        except Exception:
                            continue

                        # Evaluate in each month bucket
                        for m, idx in month_idx.items():
                            fac = month_factor[m]
                            corr = weighted_corr(x_feat[idx], fac, cdata.weight[idx])
                            if not np.isfinite(corr):
                                continue
                            ac = abs(float(corr))
                            if ac < min_abs_corr:
                                continue

                            push(
                                m,
                                ScreenHit(
                                    country=country,
                                    month=int(m),
                                    variant=variant,
                                    signal=sig_name,
                                    agg=agg,
                                    w=int(w),
                                    shift=int(shift),
                                    corr_abs=ac,
                                    corr_signed=float(corr),
                                ),
                            )

    hits: List[ScreenHit] = []
    for m, h in heaps.items():
        # sort descending by corr_abs
        hits.extend([t[2] for t in sorted(h, key=lambda z: z[0], reverse=True)])
    return hits


# =============================================================================
# Refinement: coordinate descent around a screened hit
# =============================================================================

def refine_one_hit(
    cdata: CountryData,
    *,
    signal_names: Sequence[str],
    hit: ScreenHit,
    transforms: Sequence[str],
    val_years_list: Sequence[int],
    w_min: int,
    w_max: int,
    shift_min: int,
    shift_max: int,
    refine_w_radius: int,
    refine_shift_radius: int,
    min_rows: int,
) -> Optional[RefinedCandidate]:

    # locate signal column
    try:
        j = list(signal_names).index(hit.signal)
    except ValueError:
        return None

    X = cdata.X_raw if hit.variant == "raw" else cdata.X_zdoy
    x_base = X[:, j].astype(np.float64, copy=False)

    month = int(hit.month)
    idx_month = (cdata.month == month)
    if idx_month.sum() < min_rows:
        return None

    # objective helper
    def eval_params(w: int, shift: int, transform: str) -> Tuple[float, Dict[str, float], float, np.ndarray]:
        z = compute_feature_series(x_base, agg=hit.agg, w=w, shift=shift, transform=transform)
        scores = score_bucket_cfcs(
            z=z,
            Y=cdata.Y,
            year=cdata.year,
            weight=cdata.weight,
            idx_month=idx_month,
            val_years_list=val_years_list,
            min_rows=min_rows,
        )
        # robust objective: min over vy scores (ignore NaNs)
        vy_scores = [scores.get(f"vy{vy}") for vy in val_years_list]
        vy_scores = [s for s in vy_scores if np.isfinite(s)]
        robust = float(min(vy_scores)) if vy_scores else float("nan")

        # proxy = vy2 if present else all
        proxy = float(scores.get("vy2", scores.get("all", float("nan"))))

        # combined objective (robust-first)
        obj = robust + 0.02 * proxy if np.isfinite(robust) and np.isfinite(proxy) else float("nan")
        return obj, scores, robust, z

    # coordinate descent parameters
    w0 = int(hit.w)
    s0 = int(hit.shift)

    w0 = max(w_min, min(w_max, w0))
    s0 = max(shift_min, min(shift_max, s0))

    best_overall: Optional[RefinedCandidate] = None
    best_obj = -1e18

    for transform in transforms:
        # start
        w_cur, s_cur = w0, s0

        # coarse w sweep
        w_lo = max(w_min, w_cur - refine_w_radius)
        w_hi = min(w_max, w_cur + refine_w_radius)
        w_grid = list(range(w_lo, w_hi + 1, 10))
        if w_cur not in w_grid:
            w_grid.append(w_cur)
        w_grid = sorted(set(w_grid))

        best_local_obj = -1e18
        best_local_scores: Dict[str, float] = {}
        best_local_robust = float("nan")
        best_local_z = None

        for w in w_grid:
            obj, scores, robust, z = eval_params(w, s_cur, transform)
            if np.isfinite(obj) and obj > best_local_obj:
                best_local_obj = obj
                w_cur = w
                best_local_scores = scores
                best_local_robust = robust
                best_local_z = z

        # coarse shift sweep
        s_lo = max(shift_min, s_cur - refine_shift_radius)
        s_hi = min(shift_max, s_cur + refine_shift_radius)
        s_grid = list(range(s_lo, s_hi + 1, 3))
        if s_cur not in s_grid:
            s_grid.append(s_cur)
        s_grid = sorted(set(s_grid))

        for s in s_grid:
            obj, scores, robust, z = eval_params(w_cur, s, transform)
            if np.isfinite(obj) and obj > best_local_obj:
                best_local_obj = obj
                s_cur = s
                best_local_scores = scores
                best_local_robust = robust
                best_local_z = z

        # fine w sweep around current +/- 10
        w_lo2 = max(w_min, w_cur - 10)
        w_hi2 = min(w_max, w_cur + 10)
        for w in range(w_lo2, w_hi2 + 1):
            obj, scores, robust, z = eval_params(w, s_cur, transform)
            if np.isfinite(obj) and obj > best_local_obj:
                best_local_obj = obj
                w_cur = w
                best_local_scores = scores
                best_local_robust = robust
                best_local_z = z

        # fine shift sweep around current +/- 5
        s_lo2 = max(shift_min, s_cur - 5)
        s_hi2 = min(shift_max, s_cur + 5)
        for s in range(s_lo2, s_hi2 + 1):
            obj, scores, robust, z = eval_params(w_cur, s, transform)
            if np.isfinite(obj) and obj > best_local_obj:
                best_local_obj = obj
                s_cur = s
                best_local_scores = scores
                best_local_robust = robust
                best_local_z = z

        if best_local_z is None:
            continue

        # compute detrended vy2 robustness check
        detr2 = score_bucket_detrended_vy2(
            z=best_local_z,
            Y=cdata.Y,
            year=cdata.year,
            weight=cdata.weight,
            idx_month=idx_month,
            min_rows=min_rows,
        )

        # proxy corr (factor) at final params (for debugging)
        fac = futures_pc1_factor(cdata.Y[idx_month], cdata.weight[idx_month])
        corr = weighted_corr(best_local_z[idx_month], fac, cdata.weight[idx_month])
        corr_abs = abs(float(corr)) if np.isfinite(corr) else float("nan")

        cand = RefinedCandidate(
            country=cdata.country,
            month=month,
            variant=hit.variant,
            signal=hit.signal,
            agg=hit.agg,
            w=int(w_cur),
            shift=int(s_cur),
            transform=transform,
            proxy_corr_abs=float(corr_abs),
            cfcs_all=float(best_local_scores.get("all", float("nan"))),
            cfcs_vy2=float(best_local_scores.get("vy2", float("nan"))),
            cfcs_vy3=float(best_local_scores.get("vy3", float("nan"))),
            cfcs_vy4=float(best_local_scores.get("vy4", float("nan"))),
            cfcs_vy5=float(best_local_scores.get("vy5", float("nan"))),
            robust_min=float(best_local_robust),
            detrended_vy2=float(detr2),
        )

        if np.isfinite(best_local_obj) and best_local_obj > best_obj:
            best_obj = best_local_obj
            best_overall = cand

    return best_overall


# =============================================================================
# Caches
# =============================================================================

def load_or_build_cache(
    *,
    main_csv: str,
    share_csv: str,
    out_dir: Path,
    resume: bool,
    rebuild_cache: bool,
    weight_modes: Sequence[str],
) -> Tuple[pd.DataFrame, List[str], pd.DataFrame, List[str]]:
    """Return (base_cd, futures_cols, signals_df, signal_cols)."""

    cache_dir = out_dir / "cache"
    ensure_dir(cache_dir)

    base_path = cache_dir / "base_country_date.csv.gz"
    sig_path = cache_dir / f"cd_signals_{'_'.join([m.replace(',', '') for m in weight_modes])}.csv.gz"
    meta_path = cache_dir / "cache_meta.json"

    if resume and (not rebuild_cache) and base_path.exists() and sig_path.exists() and meta_path.exists():
        log("Loading caches...")
        base_cd = pd.read_csv(base_path, compression="gzip", parse_dates=["date_on"])
        signals = pd.read_csv(sig_path, compression="gzip", parse_dates=["date_on"])
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        futures_cols = meta["futures_cols"]
        signal_cols = meta["signal_cols"]
        return base_cd, futures_cols, signals, signal_cols

    log("Reading CSVs...")
    df_main = pd.read_csv(main_csv)
    df_share = pd.read_csv(share_csv)

    futures_cols = [c for c in df_main.columns if c.startswith("futures_")]
    if not futures_cols:
        raise RuntimeError("No futures_* columns found in main_csv")

    log("Building Kaggle-aligned base row-set (sample-submission-style + dropna)...")
    base = build_kaggle_rowset_base(df_main, df_share, rolling_windows=(7, 14, 30))
    log(f"Base rows after dropna: {len(base):,}")

    keep = ["country_name", "date_on", "region_id"] + futures_cols
    base = base[keep].copy()

    log("Collapsing to country-date (with region-row weights)...")
    agg_dict = {c: "first" for c in futures_cols}
    agg_dict["region_id"] = "size"  # weight
    base_cd = (
        base.groupby(["country_name", "date_on"], sort=False)
            .agg(agg_dict)
            .reset_index()
            .rename(columns={"region_id": "weight"})
    )
    base_cd["date_on"] = pd.to_datetime(base_cd["date_on"], errors="coerce")
    base_cd["year"] = base_cd["date_on"].dt.year.astype(np.int16)
    base_cd["month"] = base_cd["date_on"].dt.month.astype(np.int8)

    countries = sorted(base_cd["country_name"].astype(str).unique().tolist())

    log(f"Building climate signals for weight_modes={list(weight_modes)}...")
    all_sig = None
    signal_cols: List[str] = []

    for mode in weight_modes:
        cd_df, cols = build_country_day_signals(df_main, df_share, weight_mode=mode, country_whitelist=countries)
        # prefix signal names so they are unique across modes
        rename = {c: f"{mode}__{c}" for c in cols}
        cd_df = cd_df[["country_name", "date_on"] + cols].copy()
        cd_df = cd_df.rename(columns=rename)
        cols2 = [rename[c] for c in cols]

        if all_sig is None:
            all_sig = cd_df
        else:
            all_sig = all_sig.merge(cd_df, on=["country_name", "date_on"], how="outer")

        signal_cols.extend(cols2)

    assert all_sig is not None
    all_sig["date_on"] = pd.to_datetime(all_sig["date_on"], errors="coerce")
    all_sig = all_sig.sort_values(["country_name", "date_on"]).reset_index(drop=True)

    meta = dict(
        built_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        futures_cols=futures_cols,
        signal_cols=signal_cols,
        n_base_country_dates=int(base_cd.shape[0]),
        n_signal_rows=int(all_sig.shape[0]),
        n_signal_cols=int(len(signal_cols)),
        weight_modes=list(weight_modes),
    )
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    log(f"Writing base cache -> {base_path}")
    base_cd.to_csv(base_path, index=False, compression="gzip")

    log(f"Writing signals cache -> {sig_path}")
    all_sig.to_csv(sig_path, index=False, compression="gzip")

    return base_cd, futures_cols, all_sig, signal_cols


# =============================================================================
# Build per-country arrays and zDOY variants
# =============================================================================

def build_country_data(
    base_cd: pd.DataFrame,
    futures_cols: Sequence[str],
    signals: pd.DataFrame,
    signal_cols: Sequence[str],
) -> Tuple[List[CountryData], List[str]]:

    futures_cols = list(futures_cols)
    signal_cols = list(signal_cols)

    countries = sorted(base_cd["country_name"].astype(str).unique().tolist())
    out: List[CountryData] = []

    for country in countries:
        b = base_cd.loc[base_cd["country_name"] == country].sort_values("date_on").reset_index(drop=True)
        if b.empty:
            continue

        s = signals.loc[signals["country_name"] == country].sort_values("date_on").reset_index(drop=True)
        m = b[["date_on"]].merge(s, on="date_on", how="left")

        # fill missing signal values
        m = m.sort_values("date_on").reset_index(drop=True)
        for c in signal_cols:
            if c not in m.columns:
                m[c] = np.nan
        m[signal_cols] = m[signal_cols].ffill().bfill().fillna(0.0)

        X_raw = m[signal_cols].to_numpy(dtype=np.float32, copy=False)

        # zDOY
        dt = pd.to_datetime(m["date_on"], errors="coerce")
        doy = dt.dt.dayofyear.to_numpy(dtype=np.int16)

        X_z = np.zeros_like(X_raw, dtype=np.float32)
        for j in range(X_raw.shape[1]):
            x = X_raw[:, j].astype(np.float64)
            # compute per-DOY mean/std
            df_tmp = pd.DataFrame({"doy": doy, "x": x})
            g = df_tmp.groupby("doy", sort=False)["x"].agg(["mean", "std"]).reset_index()
            mu_map = dict(zip(g["doy"].astype(int).tolist(), g["mean"].astype(float).tolist()))
            sd_map = dict(zip(g["doy"].astype(int).tolist(), g["std"].astype(float).fillna(0.0).tolist()))

            mu = np.array([mu_map.get(int(d), 0.0) for d in doy], dtype=np.float64)
            sd = np.array([sd_map.get(int(d), 0.0) for d in doy], dtype=np.float64)
            sd = np.where(sd > 1e-6, sd, 1.0)
            z = (x - mu) / sd
            z = np.clip(z, -10.0, 10.0)
            X_z[:, j] = z.astype(np.float32)

        cdata = CountryData(
            country=country,
            dates=b["date_on"].to_numpy(),
            year=b["year"].to_numpy(dtype=np.int16, copy=False),
            month=b["month"].to_numpy(dtype=np.int8, copy=False),
            weight=b["weight"].to_numpy(dtype=np.float64, copy=False),
            Y=b[list(futures_cols)].to_numpy(dtype=np.float32, copy=False),
            X_raw=X_raw,
            X_zdoy=X_z,
        )
        out.append(cdata)

    return out, list(signal_cols)


# =============================================================================
# Recommendations writer
# =============================================================================

def write_recommendations(path: Path, title: str, rows: pd.DataFrame, top_n: int = 25) -> None:
    lines: List[str] = []
    lines.append(title)
    lines.append("=" * len(title))
    lines.append("")

    if rows.empty:
        lines.append("(no rows)")
        path.write_text("\n".join(lines), encoding="utf-8")
        return

    show = rows.head(top_n).copy()
    for i, r in show.iterrows():
        lines.append(
            f"#{i+1:03d}  {r['country']}  m{int(r['month']):02d}  {r['variant']}  {r['signal']}  "
            f"agg={r['agg']} w={int(r['w'])} shift={int(r['shift'])} transform={r['transform']}"
        )
        lines.append(
            f"      robust_min={r['robust_min']:.2f}  vy2={r['cfcs_vy2']:.2f}  all={r['cfcs_all']:.2f}  detrended_vy2={r['detrended_vy2']:.2f}"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--main_csv", required=True)
    ap.add_argument("--share_csv", required=True)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--n_jobs", type=int, default=24)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--rebuild_cache", action="store_true")

    ap.add_argument("--weight_modes", type=str, default="share_only_norm,share_norm_fill1")

    ap.add_argument("--variants", type=str, default="raw,zdoy")
    ap.add_argument("--aggs", type=str, default="ma")
    ap.add_argument("--transforms", type=str, default="identity,square,signlog1p")

    ap.add_argument("--coarse_windows", type=str, default="2,3,5,7,10,14,21,30,45,60,90,120,180,240,300,365,450,600,730,900,1095,1500,2000,2500")
    ap.add_argument("--coarse_shifts", type=str, default="-60,-30,-15,0,15,30,60")

    ap.add_argument("--screen_top_k_per_month", type=int, default=8)
    ap.add_argument("--screen_keep_per_bucket", type=int, default=3)
    ap.add_argument("--screen_min_abs_corr", type=float, default=0.25)
    ap.add_argument("--min_rows_per_month", type=int, default=40)

    ap.add_argument("--val_years_list", type=str, default="2,3,4,5")

    ap.add_argument("--w_min", type=int, default=2)
    ap.add_argument("--w_max", type=int, default=2500)
    ap.add_argument("--shift_min", type=int, default=-60)
    ap.add_argument("--shift_max", type=int, default=60)

    ap.add_argument("--refine_w_radius", type=int, default=80)
    ap.add_argument("--refine_shift_radius", type=int, default=45)

    ap.add_argument("--save_top", type=int, default=5000)

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    weight_modes = [m.strip() for m in parse_csv_list(args.weight_modes) if m.strip()]
    for m in weight_modes:
        if m not in WEIGHT_MODES:
            raise ValueError(f"Unknown weight_mode {m}. Choose from {WEIGHT_MODES}")

    variants = [v.strip().lower() for v in parse_csv_list(args.variants) if v.strip()]
    aggs = [a.strip().lower() for a in parse_csv_list(args.aggs) if a.strip()]
    transforms = [t.strip().lower() for t in parse_csv_list(args.transforms) if t.strip()]

    windows = [int(w) for w in parse_ints(args.coarse_windows) if int(w) >= 1]
    shifts = [int(s) for s in parse_ints(args.coarse_shifts)]

    val_years_list = [int(v) for v in parse_ints(args.val_years_list)]

    base_cd, futures_cols, sig_df, signal_cols = load_or_build_cache(
        main_csv=args.main_csv,
        share_csv=args.share_csv,
        out_dir=out_dir,
        resume=bool(args.resume),
        rebuild_cache=bool(args.rebuild_cache),
        weight_modes=weight_modes,
    )

    log("Preparing per-country arrays (raw + zDOY)...")
    countries_data, signal_cols_used = build_country_data(base_cd, futures_cols, sig_df, signal_cols)
    log(f"Prepared {len(countries_data)} countries | signals={len(signal_cols_used)}")

    # ---------------------------
    # Stage 0: screening
    # ---------------------------
    baseline_path = out_dir / "baseline_screen.csv"
    if args.resume and baseline_path.exists():
        log(f"Loading existing baseline_screen.csv -> {baseline_path}")
        baseline_df = pd.read_csv(baseline_path)
    else:
        log("Stage 0: factor-screening coarse candidates...")

        hits_nested = Parallel(n_jobs=int(args.n_jobs), backend="threading")(
            delayed(screen_one_country)(
                c,
                signal_names=signal_cols_used,
                variants=variants,
                aggs=aggs,
                windows=windows,
                shifts=shifts,
                top_k_per_month=int(args.screen_top_k_per_month),
                min_abs_corr=float(args.screen_min_abs_corr),
                min_rows_per_month=int(args.min_rows_per_month),
            )
            for c in countries_data
        )

        hits: List[ScreenHit] = [h for sub in hits_nested for h in sub]
        baseline_df = pd.DataFrame([h.__dict__ for h in hits])
        baseline_df = baseline_df.sort_values(["corr_abs"], ascending=False).reset_index(drop=True)
        baseline_df.to_csv(baseline_path, index=False)
        log(f"Wrote: {baseline_path} | rows={len(baseline_df):,}")

    if baseline_df.empty:
        log("No candidates passed screening; try lowering --screen_min_abs_corr")
        return

    # choose candidates to refine: top K per (country, month)
    baseline_df["bucket"] = baseline_df["country"].astype(str) + "|" + baseline_df["month"].astype(str)
    baseline_df["rank_in_bucket"] = baseline_df.groupby("bucket")["corr_abs"].rank(method="first", ascending=False)
    refine_df = baseline_df.loc[baseline_df["rank_in_bucket"] <= int(args.screen_keep_per_bucket)].copy()
    refine_df = refine_df.sort_values(["corr_abs"], ascending=False).reset_index(drop=True)

    # ---------------------------
    # Stage 1: refinement
    # ---------------------------
    grid_path = out_dir / "grid_all_candidates.csv"
    existing_keys: set = set()
    if args.resume and grid_path.exists():
        try:
            existing = pd.read_csv(grid_path)
            for _, r in existing.iterrows():
                k = (r.get("country"), int(r.get("month")), r.get("variant"), r.get("signal"), r.get("agg"))
                existing_keys.add(k)
        except Exception:
            existing_keys = set()

    tasks: List[Tuple[CountryData, ScreenHit]] = []
    country_map = {c.country: c for c in countries_data}
    for _, r in refine_df.iterrows():
        k = (r["country"], int(r["month"]), r["variant"], r["signal"], r["agg"])
        if k in existing_keys:
            continue
        cdata = country_map.get(r["country"])
        if cdata is None:
            continue
        tasks.append(
            (
                cdata,
                ScreenHit(
                    country=r["country"],
                    month=int(r["month"]),
                    variant=r["variant"],
                    signal=r["signal"],
                    agg=r["agg"],
                    w=int(r["w"]),
                    shift=int(r["shift"]),
                    corr_abs=float(r["corr_abs"]),
                    corr_signed=float(r["corr_signed"]),
                ),
            )
        )

    log(f"Stage 1: refining {len(tasks)} screened candidates...")

    refined_list = Parallel(n_jobs=int(args.n_jobs), backend="threading")(
        delayed(refine_one_hit)(
            cdata,
            signal_names=signal_cols_used,
            hit=hit,
            transforms=transforms,
            val_years_list=val_years_list,
            w_min=int(args.w_min),
            w_max=int(args.w_max),
            shift_min=int(args.shift_min),
            shift_max=int(args.shift_max),
            refine_w_radius=int(args.refine_w_radius),
            refine_shift_radius=int(args.refine_shift_radius),
            min_rows=int(args.min_rows_per_month),
        )
        for (cdata, hit) in tasks
    )

    refined = [r for r in refined_list if r is not None]
    refined_df = pd.DataFrame([r.__dict__ for r in refined])

    if refined_df.empty:
        log("No refined candidates. Try increasing --screen_keep_per_bucket or lowering screening threshold.")
        return

    # If resuming, append
    if args.resume and grid_path.exists():
        try:
            old = pd.read_csv(grid_path)
            refined_df = pd.concat([old, refined_df], ignore_index=True)
        except Exception:
            pass

    # keep top N overall by robust_min (and proxy as tie-break)
    refined_df = refined_df.sort_values(["robust_min", "cfcs_vy2"], ascending=False).reset_index(drop=True)
    refined_df = refined_df.head(int(args.save_top)).copy()

    refined_df.to_csv(grid_path, index=False)
    log(f"Wrote: {grid_path} | rows={len(refined_df):,}")

    # recommendations
    rec_public = refined_df.sort_values(["cfcs_vy2", "robust_min"], ascending=False).reset_index(drop=True)
    rec_robust = refined_df.sort_values(["robust_min", "cfcs_vy2"], ascending=False).reset_index(drop=True)

    write_recommendations(out_dir / "recommendations_public.txt", "Recommendations (public-chasing: sort by vy2)", rec_public)
    write_recommendations(out_dir / "recommendations_robust.txt", "Recommendations (robust: sort by robust_min)", rec_robust)

    log(f"Wrote: {out_dir / 'recommendations_public.txt'}")
    log(f"Wrote: {out_dir / 'recommendations_robust.txt'}")

    log("Done.")


if __name__ == "__main__":
    main()