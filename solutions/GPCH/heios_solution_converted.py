#!/usr/bin/env python
# coding: utf-8

# Ref: 
# 
# https://www.kaggle.com/code/erguntiryaki/starter-notebook-with-baseline

# In[1]:


# get_ipython().system('cp -r /kaggle/input/git-clone-utils-script/script/src ./src')


# In[2]:


class CFG:
    seed = 42

    max_iters_optim_seasonal_weights = 0    # 10000
    topk_feats_to_keep = 1
    weighted_feats_objective = 'avg_corr'    # cfcs , avg_corr    # for better performance : cfcs  # for faster : avg_corr

    # Fixed
    drop_real_clim_risk_feats = False
    shift_window = 0
    add_futures_to_clim_risk_feats = False


# In[3]:


import pickle
import random
import os
import numpy as np
# import torch

def save_object(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

def load_object(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    # torch.manual_seed(seed) # un-comment if torch is used


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from pathlib import Path
from scipy import stats
import warnings
from scipy.optimize import minimize
from scipy.stats import pearsonr
import time

# warnings.filterwarnings('ignore')
plt.rcParams['figure.figsize'] = (20, 4)

pd.options.display.max_columns = 100 


# In[4]:


def normalize(arr, upper_bound=1, lower_bound=0, eps=1e-12):
    multi = upper_bound - lower_bound
    denom = arr.max() - arr.min()
    denom = denom + eps if np.all(denom == 0) else denom
    ret = lower_bound + multi * (arr - arr.min()) / denom
    return ret

def apply_sin(arr, apply_norm=True):
    arr = normalize(arr) if apply_norm else arr
    return np.sin(2 * np.pi * arr)


# In[5]:


def get_colnames_by_prefix(df, prefix):
    return [c for c in df.columns if c.startswith(prefix)]

def get_colnames_by_suffix(df, suffix):
    return [c for c in df.columns if c.endswith(suffix)]

def drop_added_date_on_cols(df, original_date_on_colnames, colname_prefix="date_on"):
    assert all([c.startswith(colname_prefix) for c in original_date_on_colnames]), \
        f"Some original_date_on_colnames does not have a colname_prefix \'{colname_prefix}\',"\
        f"got : {original_date_on_colnames}"
    df = df.copy()
    candidate_colanmes = get_colnames_by_prefix(df, colname_prefix)
    drop_colnames = set(candidate_colanmes) - set(original_date_on_colnames)
    return df.drop(columns=drop_colnames)

def remove_futures_columns(df):
    keep_cols = [c for c in df.columns if not(c.startswith('futures_'))]
    return df[keep_cols].copy()

class SubmissionWarning(UserWarning):
    pass

class SubmissionCleaner():
    def __init__(self, original_date_on_colnames, dropna=True, required_rows = 219161, verbose=True):
        self.original_date_on_colnames = original_date_on_colnames
        self.dropna = dropna
        self.required_rows = required_rows
        self.verbose = verbose

    def _check_num_rows(self, df):
        if len(df) != self.required_rows:
            warnings.warn("Failed in _check_num_rows.", SubmissionWarning)

    def _check_id_column(self, df):
        if not('ID' in df.columns):
            warnings.warn("Failed in _check_id_column.", SubmissionWarning)

    def _check_null(self, df):
        if df.isnull().sum().sum() != 0:
            warnings.warn("Failed in _check_null.", SubmissionWarning)

    def process(self, df, msg_starts_with=""):
        verbose = self.verbose
        if verbose: 
            print(
                msg_starts_with + 
                (": " if len(msg_starts_with) > 0 else " ") + 
                "Cleaning & checking df..."
            )
        df = df.copy()
        original_date_on_colnames = self.original_date_on_colnames

        # if verbose: print("\t- remove_futures_columns...")
        # df = remove_futures_columns(df)

        if verbose: print("\t- drop_added_date_on_cols...")
        df = drop_added_date_on_cols(
            df=df, 
            original_date_on_colnames=original_date_on_colnames
        )
        if self.dropna:
            if verbose: print("\t- dropna...")
            df = df.dropna()

        # final checks
        if verbose: print("\t- final checks...")
        self._check_num_rows(df)
        self._check_id_column(df)
        self._check_null(df)
        return df


# In[6]:


DATA_PATH   = './'
OUTPUT_PATH = './'

df = pd.read_csv(f'{DATA_PATH}corn_climate_risk_futures_daily_master.csv')
df['date_on'] = pd.to_datetime(df['date_on'])
market_share_df = pd.read_csv(f'{DATA_PATH}corn_regional_market_share.csv')

print(f"📊 Dataset: {df.shape[0]:,} rows")


# In[7]:


ORIGINAL_DATE_ON_COLNAMES = get_colnames_by_prefix(df, "date_on")
submission_cleaner = SubmissionCleaner(ORIGINAL_DATE_ON_COLNAMES)    # to be used just before saving as csv

ORIGINAL_DATE_ON_COLNAMES


# In[8]:


df['date_on_dayofyear'] = df['date_on'].dt.dayofyear
df['date_on_day'] = df['date_on'].dt.day
df['date_on_month'] = df['date_on'].dt.month
df['date_on_quarter'] = df['date_on'].dt.quarter

DATE_ON_COLS = [c for c in df.columns if c.startswith('date_on_') and len(c.split("_"))==3]
DATE_ON_NORM_COLS = [c + "_norm" for c in DATE_ON_COLS]
df[DATE_ON_NORM_COLS] = apply_sin(df[DATE_ON_COLS])    # normalize(df[DATE_ON_COLS])

DATE_ON_COLS, DATE_ON_NORM_COLS


# In[9]:


df[DATE_ON_NORM_COLS]


# In[10]:


for c in df.columns:
    if c.startswith("climate_risk_"):
        plt.figure(figsize=(20, 2))
        plt.plot(df[c])
        plt.title(c)
        plt.show()


# # Analyze Seasonl Changes in "future_" Columns

# In[11]:


def fft_series(x, dt=1.0):
    """
    Compute FFT of a 1D real-valued time series.

    Parameters
    ----------
    x : array-like, shape (n,)
        Input time series.
    dt : float, optional
        Sampling interval (time step) between observations.

    Returns
    -------
    freqs : ndarray
        One-sided frequency array (same units as 1/dt).
    amps : ndarray
        Complex FFT values corresponding to freqs.
    """
    x = np.asarray(x, dtype=float)
    n = x.size

    # Real FFT and corresponding frequencies
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, d=dt)

    return freqs, X

def ifft_keep_topk(x, dt=1., topk=None, ignore_amplitude=False):
    """
    Reconstruct a time series from an rFFT spectrum, optionally keeping only
    the top-k frequency components by magnitude and optionally ignoring amplitude.
    """
    freqs, X = fft_series(x, dt=dt)
    X = np.asarray(X, dtype=complex)
    m = X.size

    # Select which coefficients to keep
    if topk is None or topk >= m:
        mask = np.ones(m, dtype=bool)
    else:
        mags = np.abs(X)
        idx_topk = np.argpartition(mags, -topk)[-topk:]
        mask = np.zeros(m, dtype=bool)
        mask[idx_topk] = True

    X_used = np.zeros_like(X)

    if ignore_amplitude:
        # Use only phase: X = exp(i * phase) for kept indices
        phase = np.angle(X[mask])
        X_used[mask] = np.exp(1j * phase)
    else:
        # Keep original complex values on kept indices
        X_used[mask] = X[mask]

    x_recon = np.fft.irfft(X_used, n=x.size)
    return x_recon, X_used, freqs, X


# In[12]:


futures_cols = [c for c in df.columns if c.startswith('futures_')]
# fft_series(df.fillna(-9999)[futures_cols[0]])
x = df.dropna()[futures_cols[0]].values
x_k, X_k, freqs, X = ifft_keep_topk(x, topk=2, ignore_amplitude=False)    # topk >= 2 for reasonable performance


# In[13]:


plt.figure(figsize=(20,5))
plt.plot(normalize(x_k), alpha=0.5, label='x_k')
plt.plot(normalize(x), alpha=0.5, label='x')
plt.legend()
plt.show()


# In[14]:


pd.DataFrame([x_k, x]).transpose().corr()


# In[ ]:





# ---
# ## Phase 1: Build Sample Baseline (Exact Match)  
# Replicate the [sample baseline provided by organizer](https://www.kaggle.com/code/edenecanlilar/sample-submission-notebook?scriptVersionId=287073253)

# In[15]:


def compute_competition_metric(avg_sig_score, max_score, sig_pct):
    return (0.5 * avg_sig_score) + (0.3 * max_score) + (0.2 * sig_pct)


# In[16]:


# # WIP compute_cfcs ; v2
# def compute_cfcs(df, climate_colname_prefix="climate_risk_", futures_colname_prefix="futures_"):
#     assert 'country_name' in df.columns and 'date_on_month' in df.columns

#     climate_cols = [c for c in df.columns if c.startswith(climate_colname_prefix)]
#     futures_cols = [c for c in df.columns if c.startswith(futures_colname_prefix)]

#     # Group by country+month, compute all pairwise correlations vectorized
#     grouped = df.groupby(['country_name', 'date_on_month'])

#     correlations = []
#     for name, group in grouped:
#         # Skip groups too small for correlation
#         if len(group) < 2:
#             continue

#         # Vectorized corr across all climate x futures pairs
#         for clim in climate_cols:
#             for fut in futures_cols:
#                 if group[clim].std() > 0 and group[fut].std() > 0:
#                     corr = group[[clim, fut]].corr().iloc[0, 1]
#                     correlations.append(corr)

#     # Rest unchanged (already efficient)
#     correlations = pd.Series(correlations).dropna()
#     abs_corrs = correlations.abs()
#     sig_corrs = abs_corrs[abs_corrs >= 0.5]

#     avg_sig = sig_corrs.mean() if len(sig_corrs) > 0 else 0
#     max_corr = abs_corrs.max()
#     sig_pct = len(sig_corrs) / len(correlations) * 100 if len(correlations) > 0 else 0

#     avg_sig_score = min(100, avg_sig * 100)
#     max_score = min(100, max_corr * 100)

#     cfcs = compute_competition_metric(avg_sig_score=avg_sig_score, max_score=max_score, sig_pct=sig_pct)

#     return {
#         'cfcs': cfcs, 'avg_sig_corr': avg_sig, 'max_corr': max_corr,
#         'sig_count': len(sig_corrs), 'total': len(correlations), 'sig_pct': sig_pct
#     }


# In[17]:


def compute_cfcs(df, climate_colname_prefix="climate_risk_", futures_colname_prefix="futures_"):
    """Compute CFCS score for a dataframe."""
    all_colnames = df.columns
    for n in ['country_name', 'date_on_month']:
        assert n in all_colnames, f"Missing {n} in df"
    climate_cols = [c for c in df.columns if c.startswith(climate_colname_prefix)]
    futures_cols = [c for c in df.columns if c.startswith(futures_colname_prefix)]

    correlations = []

    for country in df['country_name'].unique():
        df_country = df[df['country_name'] == country]

        for month in df_country['date_on_month'].unique():
            df_month = df_country[df_country['date_on_month'] == month]

            for clim in climate_cols:
                for fut in futures_cols:
                    if df_month[clim].std() > 0 and df_month[fut].std() > 0:
                        corr = df_month[[clim, fut]].corr().iloc[0, 1]
                        correlations.append(corr)

    correlations = pd.Series(correlations).dropna()
    abs_corrs = correlations.abs()
    sig_corrs = abs_corrs[abs_corrs >= 0.5]

    avg_sig = sig_corrs.mean() if len(sig_corrs) > 0 else 0
    max_corr = abs_corrs.max()
    sig_pct = len(sig_corrs) / len(correlations) * 100 if len(correlations) > 0 else 0

    avg_sig_score = min(100, avg_sig * 100)
    max_score = min(100, max_corr * 100)

    cfcs = compute_competition_metric(
        avg_sig_score = avg_sig_score, 
        max_score = max_score,
        sig_pct = sig_pct
    )
    # cfcs = (0.5 * avg_sig_score) + (0.3 * max_score) + (0.2 * sig_pct)

    return {
        'cfcs': cfcs,
        'avg_sig_corr': avg_sig, 
        'max_corr': max_corr, 
        'sig_count': len(sig_corrs),
        'total': len(correlations),
        'sig_pct': sig_pct, 
    }


# In[18]:


RISK_CATEGORIES = ['heat_stress', 'unseasonably_cold', 'excess_precip', 'drought']

merged_daily_df = df.copy()
merged_daily_df['day_of_year'] = merged_daily_df['date_on'].dt.dayofyear
merged_daily_df['quarter'] = merged_daily_df['date_on'].dt.quarter

merged_daily_df = merged_daily_df.merge(
    market_share_df[['region_id', 'percent_country_production']], 
    on='region_id', how='left'
)
merged_daily_df['percent_country_production'] = merged_daily_df['percent_country_production'].fillna(1.0)

# Track which features we create
CREATED_FEATURES = []

print("✅ Base setup")


# In[19]:


# Risk scores
for risk_type in RISK_CATEGORIES:
    low_col = f'climate_risk_cnt_locations_{risk_type}_risk_low'
    med_col = f'climate_risk_cnt_locations_{risk_type}_risk_medium' 
    high_col = f'climate_risk_cnt_locations_{risk_type}_risk_high'

    total = merged_daily_df[low_col] + merged_daily_df[med_col] + merged_daily_df[high_col]
    risk_score = (merged_daily_df[med_col] + 2 * merged_daily_df[high_col]) / (total + 1e-6)
    weighted = risk_score * (merged_daily_df['percent_country_production'] / 100)

    merged_daily_df[f'climate_risk_{risk_type}_score'] = risk_score
    merged_daily_df[f'climate_risk_{risk_type}_weighted'] = weighted
    CREATED_FEATURES.extend([f'climate_risk_{risk_type}_score', f'climate_risk_{risk_type}_weighted'])

print(f"✅ Risk scores: {len(CREATED_FEATURES)} features")


# In[20]:


# Composite indices
score_cols = [f'climate_risk_{r}_score' for r in RISK_CATEGORIES]

merged_daily_df['climate_risk_temperature_stress'] = merged_daily_df[[f'climate_risk_{r}_score' for r in ['heat_stress', 'unseasonably_cold']]].max(axis=1)
merged_daily_df['climate_risk_precipitation_stress'] = merged_daily_df[[f'climate_risk_{r}_score' for r in ['excess_precip', 'drought']]].max(axis=1)
merged_daily_df['climate_risk_overall_stress'] = merged_daily_df[score_cols].max(axis=1)
merged_daily_df['climate_risk_combined_stress'] = merged_daily_df[score_cols].mean(axis=1)

CREATED_FEATURES.extend(['climate_risk_temperature_stress', 'climate_risk_precipitation_stress',
                         'climate_risk_overall_stress', 'climate_risk_combined_stress'])

print(f"✅ Composites: {len(CREATED_FEATURES)} total features")


# In[21]:


# Rolling features
merged_daily_df = merged_daily_df.sort_values(['region_id', 'date_on'])

for window in [7, 14, 30]:
    for risk_type in RISK_CATEGORIES:
        score_col = f'climate_risk_{risk_type}_score'

        ma_col = f'climate_risk_{risk_type}_ma_{window}d'
        max_col = f'climate_risk_{risk_type}_max_{window}d'

        merged_daily_df[ma_col] = (
            merged_daily_df.groupby('region_id')[score_col]
            .rolling(window=window, min_periods=1).mean()
            .reset_index(level=0, drop=True)
        )
        merged_daily_df[max_col] = (
            merged_daily_df.groupby('region_id')[score_col]
            .rolling(window=window, min_periods=1).max()
            .reset_index(level=0, drop=True)
        )
        CREATED_FEATURES.extend([ma_col, max_col])

print(f"✅ Rolling: {len(CREATED_FEATURES)} total features")


# In[22]:


# Momentum features (create NaN - determines valid rows)
for risk_type in RISK_CATEGORIES:
    score_col = f'climate_risk_{risk_type}_score'

    c1 = f'climate_risk_{risk_type}_change_1d'
    c7 = f'climate_risk_{risk_type}_change_7d'
    acc = f'climate_risk_{risk_type}_acceleration'

    merged_daily_df[c1] = merged_daily_df.groupby('region_id')[score_col].diff(1)
    merged_daily_df[c7] = merged_daily_df.groupby('region_id')[score_col].diff(7)
    merged_daily_df[acc] = merged_daily_df.groupby('region_id')[c1].diff(1)

    CREATED_FEATURES.extend([c1, c7, acc])

print(f"✅ Momentum: {len(CREATED_FEATURES)} total features")


# In[23]:


# Country aggregations
for risk_type in RISK_CATEGORIES:
    score_col = f'climate_risk_{risk_type}_score'
    weighted_col = f'climate_risk_{risk_type}_weighted'

    country_agg = merged_daily_df.groupby(['country_name', 'date_on']).agg({
        score_col: ['mean', 'max', 'std'],
        weighted_col: 'sum',
        'percent_country_production': 'sum'
    }).round(4)

    country_agg.columns = [f'country_{risk_type}_{"_".join(col).strip()}' for col in country_agg.columns]
    country_agg = country_agg.reset_index()

    new_cols = [c for c in country_agg.columns if c not in ['country_name', 'date_on']]
    CREATED_FEATURES.extend(new_cols)

    merged_daily_df = merged_daily_df.merge(country_agg, on=['country_name', 'date_on'], how='left')

print(f"✅ Country aggs: {len(CREATED_FEATURES)} total features")


# In[24]:


# Get valid rows
print(f"\n📊 Before dropna: {len(merged_daily_df):,}")
baseline_df = merged_daily_df.dropna()
print(f"📊 After dropna: {len(baseline_df):,} (expected: 219,161)")


# In[25]:


if CFG.drop_real_clim_risk_feats:
    climate_risk_colnames = [c for c in baseline_df.columns if c.startswith("climate_risk_")]
    baseline_df = baseline_df.drop(columns=climate_risk_colnames)
    print(f"Dropped these columns : \n{climate_risk_colnames}")


# In[26]:


baseline_df.head(1)


# In[27]:


if CFG.add_futures_to_clim_risk_feats:
    future_colnames = [c for c in baseline_df.columns if c.startswith("future")]
    # inputs
    new_colnames = [f"climate_risk_{i}_{CFG.shift_window}" for i in future_colnames]
    baseline_df[new_colnames] = baseline_df[future_colnames]\
        .shift(CFG.shift_window)\
        .fillna(method="ffill")\
        .fillna(method="bfill")
    print(f"Added columns : \n{new_colnames}")


# ## Combine and Optimize

# In[28]:


def fit_corr_weights_by_cfcs(df, feat_colnames, w0=None, options:dict={'maxiter': 10000}):
    def cfcs_objective(w, df):
        ret = compute_cfcs(df)
        return ret['cfcs']

    def print_progress_with_count(w):
        current_objective_value = avg_corr_objective(w, X, Y)
        progress['iterations'] += 1
        progress['current_time'] = time.time()
        time_elpased = progress['current_time'] - progress['last_time']
        print(f"Iter {progress['iterations']} | Objective Value: {-current_objective_value:.4f} | Time Elpased : {time_elpased:.4f} sec")

    progress = {'iterations': 0, 'last_time': time.time(), 'current_time': time.time()}

    n_feats = len(feat_colnames)
    if w0 is None:
        w0 = np.ones(n_feats) / n_feats

    res = minimize(
        cfcs_objective, w0,
        args=(df),
        method="L-BFGS-B",  # or another gradient-based method
        options = options,
        callback = print_progress_with_count,
    )
    return res.x, res

# def fit_corr_weights_by_avg_corr(X_feats, Ys, w0=None, options:dict={'maxiter': 10000}):
#     def avg_corr_objective(w, X, Y):
#         y_pred = X @ w          # (n_samples,)
#         corrs = []
#         for j in range(Y.shape[1]):
#             r, _ = pearsonr(y_pred, Y[:, j])
#             corrs.append(r)
#         # negative because we will *minimize*
#         return -np.mean(np.abs(corrs))  # or np.mean(corrs) if sign matters

#     def print_progress_with_count(w):
#         current_objective_value = avg_corr_objective(w, X, Y)
#         progress['iterations'] += 1
#         progress['current_time'] = time.time()
#         time_elpased = progress['current_time'] - progress['last_time']
#         print(f"Iter {progress['iterations']} | Objective Value: {-current_objective_value:.4f} | Time Elpased : {time_elpased:.2f} sec")
#         progress['last_time'] = progress['current_time']

#     progress = {'iterations': 0, 'last_time': time.time(), 'current_time': time.time()}

#     X = np.asarray(X_feats, dtype=float)
#     Y = np.asarray(Ys, dtype=float)

#     n_feats = X.shape[1]
#     if w0 is None:
#         w0 = np.ones(n_feats) / n_feats

#     res = minimize(
#         avg_corr_objective, w0,
#         args=(X, Y),
#         method="L-BFGS-B",  # or another gradient-based method
#         options = options, 
#         callback = print_progress_with_count
#     )
#     return res.x, res

def fit_corr_weights_by_avg_corr(
    X_feats,
    Ys,
    w0=None,
    X_feats_multi_arr=None,
    w0_feats_multi=None,
    options: dict = {"maxiter": 10000},
    penalty_type='L1', 
    lambda_penalty=0.01,
):
    """
    Fit weights to maximize average absolute correlation.

    Parameters
    ----------
    X_feats : array-like, shape (n_samples, n_features)
        Base feature matrix.

    Ys : array-like, shape (n_samples, n_targets)
        Target matrix.

    w0 : array-like, shape (n_features,), optional
        Initial weights for base features.

    X_feats_multi_arr : array-like, shape (n_samples, n_multi), optional
        Additional multiplicative features. Each row is combined with X_feats
        and a learnable per-row weight matrix w_feats_multi.

    w0_feats_multi : array-like, shape (n_samples, n_features * n_multi), optional
        Initial values for w_feats_multi, flattened.

    options : dict
        Options for scipy.optimize.minimize.

    Returns
    -------
    w_opt : ndarray, shape (n_features,)
        Learned base feature weights.

    w_multi_opt : ndarray or None
        Learned multiplicative weights, shape (n_samples, n_features * n_multi),
        or None if X_feats_multi_arr is not provided.

    res : OptimizeResult
        Full result object from scipy.optimize.minimize.
    """

    X = np.asarray(X_feats, dtype=float)
    Y = np.asarray(Ys, dtype=float)
    n_samples, n_feats = X.shape

    use_multi = X_feats_multi_arr is not None
    if use_multi:
        X_multi = np.asarray(X_feats_multi_arr, dtype=float)
        if X_multi.shape[0] != n_samples:
            raise ValueError(
                "X_feats_multi_arr must have the same number of rows as X_feats"
            )
        n_multi = X_multi.shape[1]
        # total parameters = n_feats (w) + n_samples * (n_feats * n_multi) (w_feats_multi)
        n_w = n_feats
        n_w_multi = n_samples * n_feats * n_multi
    else:
        n_multi = 0
        n_w = n_feats
        n_w_multi = 0

    def unpack_params(theta):
        w = theta[:n_w]
        if use_multi:
            w_multi_flat = theta[n_w:]
            w_multi = w_multi_flat.reshape(n_samples, n_feats * n_multi)
        else:
            w_multi = None
        return w, w_multi

    def compute_objective_components(theta, X, Y, penalty_type, lambda_penalty):
        """Extract corr_loss and penalty separately"""
        penalty_type = penalty_type.upper()
        w, w_multi = unpack_params(theta)

        y_pred = X @ w
        if use_multi:  # assuming use_multi defined in outer scope
            contrib = np.empty(n_samples)
            for i in range(n_samples):
                outer = np.outer(X[i, :], X_multi[i, :])
                contrib[i] = np.dot(outer.ravel(), w_multi[i, :])
            y_pred = y_pred + contrib

        corrs = [pearsonr(y_pred, Y[:, j])[0] for j in range(Y.shape[1])]
        corr_loss = -np.mean(np.abs(corrs))

        if penalty_type == 'L1':
            penalty = lambda_penalty * (np.sum(np.abs(w)) + (np.sum(np.abs(w_multi)) if use_multi else 0))
        elif penalty_type == 'L2':
            penalty = lambda_penalty * (np.sum(w**2) + (np.sum(w_multi**2) if use_multi else 0))
        else:
            penalty = 0.0

        return corr_loss, penalty

    def avg_corr_objective(theta, X, Y, penalty_type, lambda_penalty):
        corr_loss, penalty = compute_objective_components(theta, X, Y, penalty_type, lambda_penalty)
        return corr_loss + penalty

    def print_progress_with_count(theta):
        corr_loss, penalty = compute_objective_components(theta, X, Y, penalty_type, lambda_penalty)
        current_objective_value =  corr_loss + penalty
        progress["iterations"] += 1
        progress["current_time"] = time.time()
        time_elapsed = progress["current_time"] - progress["last_time"]
        print(
            f"Iter {progress['iterations']} | "\
            f"Objective Value : {current_objective_value:.4f} | "
            f"corr_loss : {corr_loss:.4f} | "\
            f"penalty : {penalty:.4f} | "\
            f"Time Elapsed : {time_elapsed:.2f} sec"
        )
        progress["last_time"] = progress["current_time"]

    progress = {
        "iterations": 0,
        "last_time": time.time(),
        "current_time": time.time(),
    }

    # Build initial theta
    if w0 is None:
        w0 = np.ones(n_feats) / n_feats
    w0 = np.asarray(w0, dtype=float)
    if w0.shape[0] != n_feats:
        raise ValueError("w0 must have length equal to number of features in X_feats")

    if use_multi:
        if w0_feats_multi is None:
            # initialize multiplicative weights to zero
            w0_feats_multi = np.zeros(n_samples * n_feats * n_multi)
        else:
            w0_feats_multi = np.asarray(w0_feats_multi, dtype=float)
            if w0_feats_multi.size != n_samples * n_feats * n_multi:
                raise ValueError(
                    "w0_feats_multi must have size n_samples * n_feats * n_multi"
                )
        theta0 = np.concatenate([w0, w0_feats_multi])
    else:
        theta0 = w0

    res = minimize(
        avg_corr_objective,
        theta0,
        args=(X, Y, penalty_type, lambda_penalty),
        method="L-BFGS-B",
        options=options,
        callback=print_progress_with_count,
    )

    w_opt, w_multi_opt = unpack_params(res.x)
    return w_opt, w_multi_opt, res


# In[29]:


class WeightedFeaturesOptimizer:
    """Unified class to optimize feature weights by different objectives."""

    def __init__(self, df, feat_colnames, w0_feats_multi=None, target_colnames=None, objective='avg_corr'):
        self.df = df
        self.feat_colnames = feat_colnames
        self.target_colnames = target_colnames or []
        self.w0_feats_multi = w0_feats_multi
        self.objective = objective
        self.w_opt = None
        self.w_opt_multi = None
        self.res = None
        self.y_weighted = None

    def optimize(self, w0=None, bounds=None, constraints=None, **kwargs):
        """Fit optimal weights based on objective."""
        if w0 is None:
            w0 = np.random.uniform(low=-1, high=1, size=len(self.feat_colnames))

        print(f"kwargs : {kwargs}")
        if self.objective in ['cfcs', 'cfc']:
            self.w_opt, self.res = fit_corr_weights_by_cfcs(
                df=self.df, feat_colnames=self.feat_colnames, w0=w0, w0_feats_multi=self.w0_feats_multi, **kwargs
            )
        elif self.objective in ['avg_corrs', 'avg_corr', 'mean_corr', 'mean_corrs']:
            X_feats = self.df[self.feat_colnames]
            Ys = self.df[self.target_colnames] if self.target_colnames else None
            self.w_opt, self.w_opt_multi, self.res = fit_corr_weights_by_avg_corr(
                X_feats=X_feats, Ys=Ys, w0=w0,**kwargs
            )
        else:
            raise ValueError(f"Unknown objective: {self.objective}")

        # Compute weighted features
        X_matrix = self.df[self.feat_colnames].values
        self.y_weighted = X_matrix @ self.w_opt

        return self.w_opt, self.w_opt_multi, self.res

    def add_to_df(self, df=None, new_colname='climate_risk_weighted_feats_1'):
        """Add weighted features to dataframe."""
        target_df = self.df if df is None else df
        target_df[new_colname] = self.y_weighted
        return target_df

    def get_weighted_features(self):
        """Return weighted features as Series."""
        return pd.Series(self.y_weighted, index=self.df.index, 
                        name='climate_risk_weighted_feats_1')


# In[30]:


seed_everything(CFG.seed)


# In[31]:


get_ipython().run_cell_magic('time', '', '\n# Get target weighted futures first; DO NOT add to climate_risk_*\n\n# inputs\nfuture_colnames = [c for c in baseline_df.columns if c.startswith(\'futures_\')]\nfeat_colnames = future_colnames\noptimizer = WeightedFeaturesOptimizer(\n    df=baseline_df,\n    feat_colnames=future_colnames,\n    target_colnames=future_colnames,  # optional for avg_corr\n    # w0_feats_multi=w0_feats_multi,\n    objective=CFG.weighted_feats_objective,\n)\n\nw_opt, w_multi_opt, res = optimizer.optimize(\n    w0=np.random.uniform(low=-1, high=1, size=len(feat_colnames)),\n    # options = {\'maxiter\': 3},\n    penalty_type=\'L1\',\n    lambda_penalty=1,    # 1: 0.6378 ; 0.1: 0.6266 (iter 9X) ; 2: 0.6264 (iter 88)\n)\nbaseline_df = optimizer.add_to_df(baseline_df, \'_weighted_feats_1\')\n\n# optimizer.optimize(w0=previous_w_opt)  # warm start\nprint(f"Best objective: {-res.fun}")    # 0.6264 (iter 21)\nprint(f"w_opt : {w_opt}")\nw_opt_futures = w_opt\n')


# In[32]:


save_object(w_opt_futures, "w_opt_futures.pkl")


# In[33]:


# use merged_daily_df for FFT because it hasnt dropped nan fows
merged_daily_df['_weighted_feats_1'] = baseline_df['_weighted_feats_1']
merged_daily_df['_weighted_feats_1']  # .dropna()


# In[34]:


plt.bar(x=feat_colnames, height=w_opt_futures)
plt.xticks(rotation=90)
plt.show()


# In[35]:


# rfft(baseline_df['_weighted_feats_1'].dropna())
for topk in range(1, 10+1):
    x_recon, X_used, freqs, X = ifft_keep_topk(
        merged_daily_df['_weighted_feats_1'].fillna(method='ffill').fillna(method='bfill'), 
        topk=topk, 
        ignore_amplitude=False
    )
    save_object(
        {"topk": topk, "x_recon" : x_recon, "X_used" : X_used, "freqs" : freqs, "X": X},
        f"fft_recon_weighted_feats_topk_{topk}.pkl"
    )


# In[36]:


# x_recon


# In[37]:


plt.figure(figsize=(20,5))
i = 0
j = i + 20000
plt.plot(merged_daily_df.loc[i:j, "_weighted_feats_1"])
# plt.plot(merged_daily_df.loc[i:j, "_weighted_feats_1"].fillna(method='ffill').fillna(method='bfill'))
plt.plot(x_recon[i:j])


# In[38]:


baseline_df[['_weighted_feats_1']+future_colnames].corr()


# In[39]:


merged_daily_df.columns


# In[40]:


get_ipython().run_cell_magic('time', '', '\nmerged_daily_df = merged_daily_df.drop(columns = [c for c in merged_daily_df.columns if c.endswith("_fft")])\n\n# inputs\nfeat_colnames = [c for c in merged_daily_df.columns if c.startswith("date_on_")]   # c.startswith(\'climate_risk_\') or\n# apply FFT\nmerged_daily_df[[c+"_fft" for c in feat_colnames]] = merged_daily_df.fillna(method="ffill").fillna(method="bfill")[feat_colnames].apply(\n    lambda x: ifft_keep_topk(x, topk=2, ignore_amplitude=False)[0]\n)\n')


# In[41]:


i = 0
j = 50000
for c in [c+"_fft" for c in feat_colnames]:
    try:
        temp = merged_daily_df.loc[i:j, [c, "_weighted_feats_1"]]
        plt.figure(figsize=(20, 4))
        plt.plot(temp.apply(normalize))
        plt.title(c)
        plt.show()
        display(temp.corr())
    except Exception as e:
        print(f"Exception for column {c} : {e}")


# In[42]:


def shift_series_continuous(z, lag):
    z = np.asarray(z, dtype=float)
    n = z.size
    idx = np.arange(n)
    shifted_idx = idx - lag
    z_shift = np.interp(shifted_idx, xp=idx, fp=z, left=np.nan, right=np.nan)
    return z_shift

def neg_corr_objective(lag_array, x, Y):
    # minimize passes x as an array, even for scalar problems
    lag = lag_array[0]
    x_shift = shift_series_continuous(x, lag)
    mask = ~np.isnan(x_shift) & ~np.isnan(Y)
    if mask.sum() < 2:
        return 1.0  # poor correlation
    xs = x_shift[mask]
    ys = Y[mask]
    c = np.corrcoef(xs, ys)[0, 1]
    return -c  # minimize negative correlation

def best_shift_with_minimize(x, Y, dt=1.0, topk=None, ignore_amplitude=False,
                             lag_bounds=None):
    # 1) Reconstruct from FFT
    x_recon, X_used, freqs, X_full = ifft_keep_topk(
        x, dt=dt, topk=topk, ignore_amplitude=ignore_amplitude
    )

    x_recon = np.asarray(x_recon, dtype=float)
    Y = np.asarray(Y, dtype=float)
    assert x_recon.shape == Y.shape, \
        f"x_recon.shape : {x_recon.shape} \nY.shape : {Y.shape}"

    # 2) Bounds (in samples)
    n = len(x_recon)
    if lag_bounds is None:
        lag_bounds = (-n / 4, n / 4)

    # 3) Optimize lag with scipy.optimize.minimize
    x0 = np.array([0.0])  # initial guess
    bounds = [lag_bounds]  # list of (low, high) tuples for each variable[web:63][web:70]

    res = minimize(
        neg_corr_objective,
        x0=x0,
        args=(x_recon, Y),
        bounds=bounds,
        method="L-BFGS-B",   # supports simple box bounds[web:63][web:64]
        options={"ftol": 1e-6}
    )

    best_lag = float(res.x[0])   # scalar
    best_val = float(res.fun)
    best_corr = -best_val

    x_recon_shifted = shift_series_continuous(x_recon, best_lag)
    return best_lag, best_corr, x_recon, x_recon_shifted, res


# In[43]:


# obtain the best shift
feat_colnames = [c for c in merged_daily_df.columns if c.startswith("date_on_")]   # c.startswith('climate_risk_') or

for c in feat_colnames:
    best_lag, best_corr, x_recon, x_recon_shifted, res = best_shift_with_minimize(
        merged_daily_df[c].fillna(method='ffill').fillna(method='bfill').values,
        merged_daily_df['_weighted_feats_1'].fillna(method='ffill').fillna(method='bfill').values,
        dt=1,
        topk=2,
        ignore_amplitude=False,
        lag_bounds=(-90, 90)
    )
    merged_daily_df[c+f"_fft"] = x_recon
    merged_daily_df[c+f"_fft_bestshift"] = x_recon_shifted


# In[44]:


# fillna, and select rows to match other candidate submissions
merged_daily_dropna_df = merged_daily_df.fillna(method="ffill").fillna(method="bfill").loc[baseline_df.index]    # .dropna()
merged_daily_dropna_df.shape


# # Climate Risks Weighted by Seasonal Features

# In[45]:


climate_risk_colnames = [c for c in merged_daily_dropna_df.columns if c.startswith("climate_risk_")]
future_colnames = [c for c in merged_daily_dropna_df.columns if c.startswith("futures_")]
climate_risk_colnames, future_colnames


# In[46]:


selected_date_on_colnames = [
    c for c in merged_daily_dropna_df.columns if c.startswith("date_on_") and \
    merged_daily_dropna_df[c].dtype in [np.dtype('float64'), np.dtype('float32'), np.dtype('int64'), np.dtype('int32')]
]
selected_date_on_colnames


# In[47]:


# get topk correlation
corr = merged_daily_dropna_df[[*climate_risk_colnames, *selected_date_on_colnames, '_weighted_feats_1']].corr()


# In[48]:


corr.loc[[*climate_risk_colnames, *selected_date_on_colnames], '_weighted_feats_1'].abs().nlargest(10)  # sort_values(ascending=False)


# ## Some Experiments for Post Processing the "date_on_year_norm"
# 
# **Caution: Keep all the commented-out blocks**

# In[49]:


# # abs correlation: 0.73621425
# temp = merged_daily_dropna_df["date_on_year_norm"].copy()
# temp[temp>0] = temp[temp>0].max()
# temp.loc[1:1e4].plot()

# np.corrcoef(temp, merged_daily_dropna_df._weighted_feats_1)


# In[50]:


# # abs correlation: 0.75047258
# i = 0
# j = i+1e4

# temp = merged_daily_dropna_df["date_on_year_norm"].copy()
# plt.plot(temp.loc[i:j])

# temp[temp>0] = temp[temp>0].max()
# temp[temp<=-0.75] = temp[temp<=-0.75].min()
# temp.loc[i:j].plot()

# np.corrcoef(temp, merged_daily_dropna_df._weighted_feats_1)


# In[51]:


# # abs correlation: 0.73937056
# i = 0
# j = i+1e4

# plt.figure(figsize=(20,4))

# temp = merged_daily_dropna_df["date_on_year_norm"].copy()
# plt.plot(temp.loc[i:j], label="original")

# temp[temp>0] = temp[temp>0].max()
# temp[temp<=-0.25] = temp[temp<=-0.25].min()

# plt.plot(temp.loc[i:j], label="post proc")
# plt.plot(
#     merged_daily_dropna_df.loc[i:j, ['_weighted_feats_1']].apply(normalize, args=(temp.max(), temp.min())),
#     label = 'weighted_feats_1'
# )

# plt.legend()

# np.corrcoef(temp, merged_daily_dropna_df._weighted_feats_1)


# In[52]:


# # abs correlation: 0.78354333
# i = 0
# j = i+1e4

# plt.figure(figsize=(20,4))

# temp = merged_daily_dropna_df["date_on_year_norm"].copy()
# plt.plot(
#     temp.loc[i:j], 
#     label="original (corr : {:.4f})".format(
#         np.corrcoef(temp, merged_daily_dropna_df._weighted_feats_1)[0, 1]
#     )
# )

# temp[temp>=0] = temp[temp>=0].min()
# temp[temp<=-0.75] = temp[temp<=-0.75].min()

# plt.plot(
#     temp.loc[i:j], 
#     label="post-proc (corr : {:.4f})".format(
#         np.corrcoef(temp, merged_daily_dropna_df._weighted_feats_1)[0, 1]
#     )     
# )
# plt.plot(
#     merged_daily_dropna_df.loc[i:j, ['_weighted_feats_1']].apply(normalize, args=(temp.max(), temp.min())),
#     label = 'weighted_feats_1'
# )
# plt.legend()

# np.corrcoef(temp, merged_daily_dropna_df._weighted_feats_1)


# In[53]:


# # abs correlation: 0.78626189
# i = 0
# j = i+1e4

# plt.figure(figsize=(20,4))

# temp = merged_daily_dropna_df["date_on_year_norm"].copy()
# plt.plot(
#     temp.loc[i:j], 
#     label="original (corr : {:.4f})".format(
#         np.corrcoef(temp, merged_daily_dropna_df._weighted_feats_1)[0, 1]
#     )
# )

# temp[temp>=-0.05] = temp[temp>=-0.05].max()
# temp[temp<=-0.75] = temp[temp<=-0.75].max()

# plt.plot(
#     temp.loc[i:j], 
#     label="post-proc (corr : {:.4f})".format(
#         np.corrcoef(temp, merged_daily_dropna_df._weighted_feats_1)[0, 1]
#     )     
# )
# plt.plot(
#     merged_daily_dropna_df.loc[i:j, ['_weighted_feats_1']].apply(normalize, args=(temp.max(), temp.min())),
#     label = 'weighted_feats_1'
# )
# plt.legend()

# np.corrcoef(temp, merged_daily_dropna_df._weighted_feats_1)


# In[54]:


# # abs correlation: 0.79719795
# i = 0
# j = i+1e4

# plt.figure(figsize=(20,4))

# temp = merged_daily_dropna_df["date_on_year_norm"].copy()
# plt.plot(temp.loc[i:j], label="original")

# temp[temp>=0] = temp[temp>=0].max()
# temp[temp<=-0.75] = temp[temp<=-0.75].min()

# plt.plot(temp.loc[i:j], label="post proc")
# plt.plot(
#     merged_daily_dropna_df.loc[i:j, ['_weighted_feats_1']].apply(normalize, args=(temp.max(), temp.min())),
#     label = 'weighted_feats_1'
# )

# plt.legend()

# np.corrcoef(temp, merged_daily_dropna_df._weighted_feats_1)


# In[55]:


def clip_data(df, col_name, upper_bound, lower_bound, 
              target_col="_weighted_feats_1", 
              i=0, j=10000, show_plot=False):
    """
    Plots processed data based on bounds and returns the modified array and correlation.
    """   
    # Use 'arr' as requested
    arr = df[col_name].copy()
    target = df[target_col]

    # Post-processing logic using args
    arr[arr >= upper_bound] = arr[arr >= upper_bound].max()
    arr[arr <= lower_bound] = arr[arr <= lower_bound].min()

    if show_plot:
        plt.figure(figsize=(20, 4))
        # Plot original
        plt.plot(
            df.loc[i:j, col_name], 
            label="original (corr : {:.4f})".format(np.corrcoef(df[col_name], target)[0, 1])
        )
        plt.plot(
            arr.loc[i:j], 
            label="post-proc (corr : {:.4f})".format(np.corrcoef(arr, target)[0, 1])     
        )
        display_label = target_col if not target_col.startswith("_") else f" {target_col}"
        plt.plot(
            df.loc[i:j, [target_col]].apply(normalize, args=(arr.max(), arr.min())),
            label=display_label
        )

        plt.legend(loc='center')
        plt.show()

    # Function output
    return arr, np.corrcoef(arr, target)


# In[56]:


# # SUPERSEDED because of the sensitivity of w0
# def clip_data_objective(params, df, col_name):
#     upper_bound, lower_bound = params
#     # Negate because we want to MAXIMIZE result
#     _, result = clip_data(
#         df=df,
#         col_name=col_name,
#         upper_bound=upper_bound,
#         lower_bound=lower_bound,
#         show_plot=False
#     )
#     result = -np.abs(result[0, 1])
#     if np.all(np.isnan(result)):
#         # penalize
#         result = np.inf
#     return result

# # x0 = np.sort(
# #     np.random.uniform(low=-1, high=1, size=(2))
# # )[::-1]

# res = minimize(
#     fun=clip_data_objective,
#     # x0=np.array([0, 0]),
#     # x0 = np.sort(
#     #     np.random.uniform(low=-1, high=1, size=(2))
#     # )[::-1], 
#     args=(merged_daily_dropna_df, "date_on_year_norm"),
#     method="Powell", 
#     options={"disp": True, "xtol": 5e-1, "xatol": 5e-1, }, 
# )
# res


# In[57]:


merged_daily_dropna_df['date_on_year_norm_clipped'], _ = clip_data(df=merged_daily_dropna_df, col_name="date_on_year_norm", upper_bound=-0.05, lower_bound=-0.75, show_plot=True)


# In[58]:


i = 0
j = i + 10000
plt.figure(figsize=(20,4))
plt.plot(normalize(temp).loc[i:j])
_plt_colnames = [
    'date_on_year_norm_clipped', 
    'date_on_year_norm', 
    'date_on_year_norm_fft', 
    '_weighted_feats_1',
] 
plt.plot(
    merged_daily_dropna_df.loc[i:j, _plt_colnames].apply(normalize),
    label = _plt_colnames
)
plt.legend()


# In[59]:


# pd.DataFrame(
#     [temp, temp]
# ).transpose().corr()


# ## Combination of seasonal features and climate risk

# In[60]:


selected_date_on_colnames = [
    c for c in merged_daily_dropna_df.columns if c.startswith("date_on_") and \
    merged_daily_dropna_df[c].dtype in [np.dtype('float64'), np.dtype('float32'), np.dtype('int64'), np.dtype('int32')]
]
len(selected_date_on_colnames)


# In[61]:


# input; use len(selected_date_on_colnames) for using all
topk_date_on_colnames = merged_daily_dropna_df[[*selected_date_on_colnames, '_weighted_feats_1']]\
    .corr()\
    .loc[selected_date_on_colnames, '_weighted_feats_1']\
    .abs()\
    .nlargest(100)\
    .index\
    .tolist()


# In[62]:


topk_date_on_colnames


# In[63]:


future_colnames


# In[64]:


get_ipython().run_cell_magic('time', '', '# inputs \nfeat_colnames = topk_date_on_colnames + climate_risk_colnames    # Order sensitive; need to match with w0\n\n# w0 = np.array([\n#     *[1 for _ in topk_date_on_colnames], \n#     *[0 for _ in climate_risk_colnames], \n#     # *np.random.uniform(low=-1, high=1, size=len(climate_risk_colnames))\n# ])\n\n_ones_for_first_n = 1\nw0 = np.array([\n    *[0.1 for _ in range(_ones_for_first_n)], \n    *[0 for _ in feat_colnames[_ones_for_first_n:]], \n    # *np.random.uniform(low=-1, high=1, size=len(climate_risk_colnames))\n])\n\n\nprint(f"feat_colnames : {feat_colnames}")\nprint(f"w0 : {w0}")\n\noptimizer = WeightedFeaturesOptimizer(\n    df=merged_daily_dropna_df.copy(),\n    feat_colnames=feat_colnames,\n    target_colnames = future_colnames,  # optional for avg_corr ; future_colnames; ["_weighted_feats_1"]\n    # w0_feats_multi=w0_feats_multi,\n    objective=CFG.weighted_feats_objective,\n)\n\nw_opt, w_multi_opt, res = optimizer.optimize(\n    w0 = w0,\n    penalty_type=\'L1\', \n    lambda_penalty=0.,    # 0.01: 0.520X ; 0: 0.521X\n    options = {"maxiter": CFG.max_iters_optim_seasonal_weights},\n)\nmerged_daily_dropna_df = optimizer.add_to_df(merged_daily_dropna_df, \'climate_risk_season_weighted_feats_1\')\n\nprint(f"Best objective: {-res.fun}")    # to weighted_feats_1: 0.8484035725841691\nprint(f"w_opt : {w_opt}")\n')


# In[65]:


w_opt_climate_risk_season_weighted_feats_1 = w_opt


# In[66]:


save_object(w_opt_climate_risk_season_weighted_feats_1, "w_opt_climate_risk_season_weighted_feats_1.pkl")


# In[67]:


plt.bar(
    x = feat_colnames,
    height=w_opt_climate_risk_season_weighted_feats_1
)
plt.title("w_opt_climate_risk_season_weighted_feats_1")
plt.xticks(rotation=90)
plt.show()


# In[68]:


# # add date_on_ to climate_risk_
# for c in merged_daily_dropna_df.columns:
#     if c.startswith("date_on_"):
#         _dtype = merged_daily_dropna_df[c].dtype
#         if _dtype in [np.dtype('float64'), np.dtype('float32'), np.dtype('int64'), np.dtype('int32')]:
#             merged_daily_dropna_df[f'climate_risk_{c}'] = merged_daily_dropna_df[c]
#         else:
#             print(f"Unsupported dtype for column {c}, got {_dtype}.")
# merged_daily_dropna_df


# ---
# ## Phase 2: Analyze Feature Contributions
# 
# Find which features have significant correlations and which are dead weight.

# In[69]:


def analyze_feature_contribution(df):
    """
    For each climate feature, count how many significant correlations it produces.
    """
    climate_cols = [c for c in df.columns if c.startswith('climate_risk_')]
    futures_cols = [c for c in df.columns if c.startswith('futures_')]
    feature_stats = {col: {'sig_count': 0, 'total': 0, 'max_corr': 0, 'sig_corrs': []} 
                     for col in climate_cols}

    for country in df['country_name'].unique():
        df_country = df[df['country_name'] == country]

        for month in df_country['date_on_month'].unique():
            df_month = df_country[df_country['date_on_month'] == month]

            for clim in climate_cols:
                for fut in futures_cols:
                    if df_month[clim].std() > 0 and df_month[fut].std() > 0:
                        corr = df_month[[clim, fut]].corr().iloc[0, 1]

                        feature_stats[clim]['total'] += 1

                        if abs(corr) >= 0.5:
                            feature_stats[clim]['sig_count'] += 1
                            feature_stats[clim]['sig_corrs'].append(abs(corr))

                        if abs(corr) > feature_stats[clim]['max_corr']:
                            feature_stats[clim]['max_corr'] = abs(corr)

    # Convert to DataFrame
    results = []
    for col, stats in feature_stats.items():
        sig_pct = stats['sig_count'] / stats['total'] * 100 if stats['total'] > 0 else 0
        max_corr = 100*stats['max_corr']

        avg_sig = 0
        std_corr = 0
        min_corr = 0
        if len(stats['sig_corrs'])>0:
            avg_sig = 100*np.mean(stats['sig_corrs']) if stats['sig_corrs'] else 0
            std_corr = np.std(stats['sig_corrs'])
            min_corr = np.min(stats['sig_corrs'])

        results.append({
            'feature': col,
            'sig_count': stats['sig_count'],
            'total': stats['total'],
            'sig_pct': sig_pct, 
            'max_corr': max_corr,
            'min_corr': min_corr,
            'avg_sig_corr': avg_sig,
            'std_corr': std_corr, 
            'cfcs': compute_competition_metric(
                avg_sig_score = avg_sig, 
                max_score = max_corr,
                sig_pct = sig_pct
            ),
        })
    return pd.DataFrame(results).sort_values('cfcs', ascending=False)


# In[70]:


get_ipython().run_cell_magic('time', '', 'print("📊 Analyzing feature contributions for merged_daily_dropna_df...\\n")\nclimate_cols = [c for c in merged_daily_dropna_df.columns if c.startswith(\'climate_risk_\')]\nfutures_cols = [c for c in merged_daily_dropna_df.columns if c.startswith(\'futures_\')]\nfeature_analysis = analyze_feature_contribution(merged_daily_dropna_df)\n')


# In[71]:


print("\n🔝 TOP 20 Features by Significant Correlation Count:")
print("="*80)
print(feature_analysis.head(20).to_string(index=False))


# In[72]:


# print("\n❌ BOTTOM 20 Features (candidates for removal):")
# print("="*80)
# print(feature_analysis.tail(20).to_string(index=False))


# In[73]:


# # Identify features with ZERO significant correlations
# sig_count_col = feature_analysis['sig_count']
# quantile = .99
# zero_sig_features = feature_analysis[feature_analysis['sig_count'] == 0]['feature'].tolist()
# low_sig_features = feature_analysis[sig_count_col<=sig_count_col.quantile(quantile)]['feature'].tolist()

# cfcs_col = feature_analysis['cfcs']
# quantile = .99
# zero_sig_features = feature_analysis[cfcs_col == 0]['feature'].tolist()
# low_sig_features = feature_analysis[cfcs_col<=cfcs_col.quantile(quantile)]['feature'].tolist()

cfcs_col = feature_analysis['cfcs']
zero_sig_features = feature_analysis[cfcs_col == 0]['feature'].tolist()
low_sig_features = feature_analysis[cfcs_col < cfcs_col.nlargest(CFG.topk_feats_to_keep).min()]['feature'].tolist()


print(f"\n📊 Feature Analysis Summary:")
print(f"   Total climate features: {len(climate_cols)}")
print(f"   Features with 0 significant correlations: {len(zero_sig_features)}")
print(f"   Features with ≤1 significant correlations: {len(low_sig_features)}")
print(f"   Remaining features : \n{set(climate_cols) - set(low_sig_features)}")

print(f"\n   Total significant correlations: {feature_analysis['sig_count'].sum()}")


# ---
# ## Phase 3: Optimize - Remove Weak Features
# 
# Remove features that contribute 0 or very few significant correlations.

# In[74]:


# Decide removal threshold
# We need to be careful: removing features reduces total correlations (denominator)
# but also removes their (few) significant correlations (numerator)

FEATURES_TO_REMOVE = (zero_sig_features + low_sig_features).copy()

# But KEEP the original 12 climate_risk_cnt_* columns (required by competition)
original_cols = [c for c in FEATURES_TO_REMOVE if 'cnt_locations' in c]
# original_cols = []
FEATURES_TO_REMOVE = [c for c in FEATURES_TO_REMOVE if c not in original_cols]

print(f"🗑️ Features to remove: {len(FEATURES_TO_REMOVE)}")
if len(FEATURES_TO_REMOVE) > 0:
    for f in FEATURES_TO_REMOVE[:10]:
        print(f"   - {f}")
    if len(FEATURES_TO_REMOVE) > 10:
        print(f"   ... and {len(FEATURES_TO_REMOVE) - 10} more")


# In[75]:


# Create optimized dataset
optimized_df = merged_daily_dropna_df.copy() # baseline_df.copy()

# Remove weak features
cols_before = len([c for c in optimized_df.columns if c.startswith('climate_risk_')])
optimized_df = optimized_df.drop(columns=FEATURES_TO_REMOVE, errors='ignore')
cols_after = len([c for c in optimized_df.columns if c.startswith('climate_risk_')])

print(f"\n📊 Climate features: {cols_before} → {cols_after} (removed {cols_before - cols_after})")


# ---
# ## Phase 4: Calculate Optimized CFCS Score

# In[76]:


print("📊 Computing CFCS scores...\n")

print("merged_daily_dropna_df (sample features):")
merged_daily_dropna_score = compute_cfcs(merged_daily_dropna_df)
print(f"   CFCS: {merged_daily_dropna_score['cfcs']}")
print(f"   Sig: {merged_daily_dropna_score['sig_count']}/{merged_daily_dropna_score['total']} ({merged_daily_dropna_score['sig_pct']:.4f}%)")

print("Baseline (sample features):")
baseline_score = compute_cfcs(baseline_df)
print(f"   CFCS: {baseline_score['cfcs']}")
print(f"   Sig: {baseline_score['sig_count']}/{baseline_score['total']} ({baseline_score['sig_pct']:.4f}%)")

print("\nOptimized (weak features removed):")
optimized_score = compute_cfcs(optimized_df)
print(f"   CFCS: {optimized_score['cfcs']}")
print(f"   Sig: {optimized_score['sig_count']}/{optimized_score['total']} ({optimized_score['sig_pct']:.4f}%)")

print(f"\n{'📈 IMPROVEMENT!' if optimized_score['cfcs'] > baseline_score['cfcs'] else '📉 No improvement'}")
print(f"   Delta: {optimized_score['cfcs'] - baseline_score['cfcs']:+.2f}")


# ---
# ## Phase 5: Try Alternative - Add High-Value Features Only
# 
# Instead of just removing, let's try adding ONLY features likely to have high correlations.

# In[77]:


# Start fresh from baseline
enhanced_df = baseline_df.copy()

# Based on feature analysis, the TOP features are typically:
# - Rolling MA features (especially 30d)
# - excess_precip related features
# - unseasonably_cold related features

# Add ONLY high-potential features:

# 1. Extended MA for top-performing risk types
enhanced_df = enhanced_df.sort_values(['region_id', 'date_on'])

try:
    for risk_type in ['excess_precip', 'unseasonably_cold']:  # Focus on top performers
        score_col = f'climate_risk_{risk_type}_score'

        # 60-day MA (longer trend)
        enhanced_df[f'climate_risk_{risk_type}_ma_60d'] = (
            enhanced_df.groupby('region_id')[score_col]
            .transform(lambda x: x.rolling(60, min_periods=1).mean())
        )

    print("✅ Added 60-day MA for top risk types")

    # 2. Cumulative stress (30-day sum) - often shows strong correlations
    for risk_type in ['excess_precip', 'drought']:
        score_col = f'climate_risk_{risk_type}_score'
        enhanced_df[f'climate_risk_{risk_type}_cumsum_30d'] = (
            enhanced_df.groupby('region_id')[score_col]
            .transform(lambda x: x.rolling(30, min_periods=1).sum())
        )

    print("✅ Added cumulative 30-day stress")

    # 3. Cross-risk ratio (relative stress)
    enhanced_df['climate_risk_precip_drought_ratio'] = (
        enhanced_df['climate_risk_excess_precip_score'] / 
        (enhanced_df['climate_risk_drought_score'] + 0.01)
    )

    enhanced_df['climate_risk_temp_ratio'] = (
        enhanced_df['climate_risk_heat_stress_score'] / 
        (enhanced_df['climate_risk_unseasonably_cold_score'] + 0.01)
    )

    print("✅ Added risk ratios")

    # Fill any NaN
    for col in enhanced_df.columns:
        if enhanced_df[col].isnull().any():
            enhanced_df[col] = enhanced_df[col].fillna(0)

    print(f"\n📊 Enhanced features: {len([c for c in enhanced_df.columns if c.startswith('climate_risk_')])}")
except KeyError:
    pass


# In[78]:


print("\nEnhanced (selective additions):")
enhanced_score = compute_cfcs(enhanced_df)
print(f"   CFCS: {enhanced_score['cfcs']}")
print(f"   Sig: {enhanced_score['sig_count']}/{enhanced_score['total']} ({enhanced_score['sig_pct']:.4f}%)")


# ---
# ## Phase 6: Select Best Version & Submit

# In[79]:


submission_cleaner.process(merged_daily_dropna_df, "merged_daily_dropna_df").to_csv("merged_daily_dropna_df.csv", index=False)
submission_cleaner.process(baseline_df, "baseline_df").to_csv("baseline_df.csv", index=False)
submission_cleaner.process(optimized_df, "optimized_df").to_csv("optimized_df.csv", index=False)
submission_cleaner.process(enhanced_df, "enhanced_df").to_csv("enhanced_df.csv", index=False)


# In[80]:


# Compare all versions
print("="*60)
print("📊 SCORE COMPARISON")
print("="*60)
print(f"Baseline:  {baseline_score['cfcs']} (sample notebook equivalent)")
print(f"Optimized: {optimized_score['cfcs']} (weak features removed)")
print(f"Enhanced:  {enhanced_score['cfcs']} (selective additions)")
print("="*60)

# Pick the best
scores = {
    'merged_daily_dropna': (merged_daily_dropna_df, merged_daily_dropna_score['cfcs']),
    'baseline': (baseline_df, baseline_score['cfcs']),
    'optimized': (optimized_df, optimized_score['cfcs']),
    'enhanced': (enhanced_df, enhanced_score['cfcs']),
}

best_name = max(scores.keys(), key=lambda k: scores[k][1])
best_df, best_score = scores[best_name]

print(f"\n🏆 Best version: {best_name} (CFCS: {best_score})")


# In[81]:


# Use the best version for submission
submission = best_df.copy()

# Validation
REQUIRED_ROWS = 219161

print("\n" + "="*60)
print("✅ SUBMISSION VALIDATION")
print("="*60)

checks = [
    ('Row count', len(submission) == REQUIRED_ROWS, f"{len(submission):,}/{REQUIRED_ROWS:,}"),
    ('ID column', 'ID' in submission.columns, str('ID' in submission.columns)),
    ('No nulls', submission.isnull().sum().sum() == 0, f"{submission.isnull().sum().sum()} nulls"),
]

all_passed = True
for name, passed, detail in checks:
    print(f"{'✅' if passed else '❌'} {name}: {detail}")
    all_passed = all_passed and passed

print("="*60)


# In[82]:


# Save
output_file = f'{OUTPUT_PATH}submission.csv'
submission.to_csv(output_file, index=False)


print(f"\n📁 Saved: {output_file}")
print(f"   Version: {best_name}")
print(f"   CFCS: {best_score}")
print(f"   Rows: {len(submission):,}")
print(f"   Climate features: {len([c for c in submission.columns if c.startswith('climate_risk_')])}")


# ---
# ## 📊 Summary
# 
# ### Key Insights
# 
# 1. **Adding features can HURT your score** if they don't contribute significant correlations
# 2. The `Sig_Count%` component penalizes feature bloat
# 3. **Quality > Quantity**: Focus on features with high correlation potential
# 
# ### Optimization Strategies Tested
# 
# | Strategy | Approach |
# |----------|----------|
# | **Baseline** | Sample notebook features |
# | **Optimized** | Remove features with 0 significant correlations |
# | **Enhanced** | Add only high-potential features |
# 
# ### Recommendations for Further Improvement
# 
# 1. Analyze which country-month combinations have strongest correlations
# 2. Create features targeted at those specific contexts
# 3. Consider lagged correlations (weather affects prices days/weeks later)
# 4. Test interaction features between top-performing risk types
# 
# **Good luck! 🌽📈**

# In[ ]:




