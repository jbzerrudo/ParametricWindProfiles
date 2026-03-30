"""
compare_profiles.py
Main comparison pipeline for parametric TC wind profiles.

For each valid snapshot in the IBTrACS catalog:
  1. Reconstruct radial wind profile with each of the 6 models
  2. Extract predicted wind radii (R34, R50, R64) from each profile
  3. Compare against observed JTWC wind radii
  4. Compute error metrics by intensity category, latitude band, size class

Outputs:
  - metrics_by_snapshot.csv  (per-snapshot, per-model errors)
  - metrics_summary.csv      (aggregated statistics)
  - Diagnostic figures
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wind_profiles import PROFILES, coriolis
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

# ═══════════════════════════════════════════════════════════════
# Helper: extract wind radius from a profile
# ═══════════════════════════════════════════════════════════════
def extract_wind_radius(r, v, threshold):
    """
    Find the outermost radius where V >= threshold.
    Returns NaN if threshold is never reached.
    """
    mask = v >= threshold
    if not mask.any():
        return np.nan
    return r[mask][-1]  # outermost crossing


def extract_all_wind_radii(r, v):
    """Extract R34, R50, R64 from a profile."""
    return {
        'R34': extract_wind_radius(r, v, 34.0),
        'R50': extract_wind_radius(r, v, 50.0),
        'R64': extract_wind_radius(r, v, 64.0),
    }


# ═══════════════════════════════════════════════════════════════
# Helper: observed mean wind radii from JTWC quadrants
# ═══════════════════════════════════════════════════════════════
def obs_mean_radius(row, prefix):
    """Mean of available quadrants for a given wind radius (R34/R50/R64)."""
    cols = [f'{prefix}_NE', f'{prefix}_SE', f'{prefix}_SW', f'{prefix}_NW']
    vals = [row[c] for c in cols if pd.notna(row.get(c))]
    return np.mean(vals) if vals else np.nan


# ═══════════════════════════════════════════════════════════════
# Intensity / stratification helpers
# ═══════════════════════════════════════════════════════════════
def intensity_cat(v):
    if v < 34:   return 'TD'
    elif v < 64:  return 'TS'
    elif v < 96:  return 'C1-2'
    else:         return 'C3-5'

def lat_band(lat):
    if lat < 15:   return '00-15N'
    elif lat < 25:  return '15-25N'
    else:           return '25-35N'


# ═══════════════════════════════════════════════════════════════
# Load data
# ═══════════════════════════════════════════════════════════════
print("Loading snapshot catalog...")
df = pd.read_csv(r'D:\2026\ParametricWindModel\Models\snapshot_catalog.csv')

# Coerce numerics
num_cols = ['LAT', 'LON', 'USA_WIND', 'USA_PRES', 'USA_RMW',
            'USA_R34_NE', 'USA_R34_SE', 'USA_R34_SW', 'USA_R34_NW',
            'USA_R50_NE', 'USA_R50_SE', 'USA_R50_SW', 'USA_R50_NW',
            'USA_R64_NE', 'USA_R64_SE', 'USA_R64_SW', 'USA_R64_NW']
for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors='coerce')

# Compute observed mean radii
df['OBS_R34'] = df.apply(lambda row: obs_mean_radius(row, 'USA_R34'), axis=1)
df['OBS_R50'] = df.apply(lambda row: obs_mean_radius(row, 'USA_R50'), axis=1)
df['OBS_R64'] = df.apply(lambda row: obs_mean_radius(row, 'USA_R64'), axis=1)

# Stratification
df['INTENSITY_CAT'] = df['USA_WIND'].apply(intensity_cat)
df['LAT_BAND'] = df['LAT'].apply(lat_band)

# Size class from R34 (terciles)
q33, q66 = df['OBS_R34'].quantile([0.33, 0.66])
def size_class(r34):
    if pd.isna(r34): return 'unknown'
    if r34 <= q33:   return 'compact'
    elif r34 <= q66:  return 'average'
    else:             return 'large'
df['SIZE_CLASS'] = df['OBS_R34'].apply(size_class)

print(f"Total snapshots: {len(df)}")
print(f"With R34: {df['OBS_R34'].notna().sum()}")
print(f"With R50: {df['OBS_R50'].notna().sum()}")
print(f"With R64: {df['OBS_R64'].notna().sum()}")
print(f"With USA_PRES: {df['USA_PRES'].notna().sum()}")

# ═══════════════════════════════════════════════════════════════
# Run all profiles on all snapshots
# ═══════════════════════════════════════════════════════════════
print("\nReconstructing profiles...")

# Radial grid: 0 to 500 nm, 1 nm resolution
r_grid = np.arange(0, 501, 1.0)

results = []
n_total = len(df)

for idx, (i, row) in enumerate(df.iterrows()):
    if idx % 2000 == 0:
        print(f"  {idx}/{n_total} ({100*idx/n_total:.0f}%)")
    
    vmax = row['USA_WIND']
    rmax = row['USA_RMW']
    lat = row['LAT']
    pc = row['USA_PRES']  # may be NaN
    r34_mean = row['OBS_R34']
    
    # Common params
    params = dict(
        r=r_grid,
        vmax=vmax,
        rmax=rmax,
        lat=lat,
        pc=pc if pd.notna(pc) else np.nan,
        penv=1013.0,
        r34_mean=r34_mean if pd.notna(r34_mean) else None,
    )
    
    snap_result = {
        'idx': i,
        'SID': row['SID'],
        'ISO_TIME': row['ISO_TIME'],
        'VMAX': vmax,
        'RMAX': rmax,
        'LAT': lat,
        'PC': pc,
        'OBS_R34': row['OBS_R34'],
        'OBS_R50': row['OBS_R50'],
        'OBS_R64': row['OBS_R64'],
        'INTENSITY_CAT': row['INTENSITY_CAT'],
        'LAT_BAND': row['LAT_BAND'],
        'SIZE_CLASS': row['SIZE_CLASS'],
    }
    
    for name, func in PROFILES.items():
        # Skip Holland models if no pressure
        #if name in ('Holland1980', 'Holland2010') and pd.isna(pc):
        if name == 'Holland1980' and pd.isna(pc):
            for rad in ['R34', 'R50', 'R64']:
                snap_result[f'{name}_{rad}'] = np.nan
            continue
        
        try:
            v = func(**params)
            radii = extract_all_wind_radii(r_grid, v)
            for rad in ['R34', 'R50', 'R64']:
                snap_result[f'{name}_{rad}'] = radii[rad]
        except Exception as e:
            for rad in ['R34', 'R50', 'R64']:
                snap_result[f'{name}_{rad}'] = np.nan
    
    results.append(snap_result)

print(f"  {n_total}/{n_total} (100%)")

# ═══════════════════════════════════════════════════════════════
# Assemble results
# ═══════════════════════════════════════════════════════════════
res = pd.DataFrame(results)

# Compute errors (predicted - observed) for each model and radius
model_names = list(PROFILES.keys())
for name in model_names:
    for rad in ['R34', 'R50', 'R64']:
        pred_col = f'{name}_{rad}'
        obs_col = f'OBS_{rad}'
        err_col = f'{name}_{rad}_ERR'
        res[err_col] = res[pred_col] - res[obs_col]

res.to_csv('metrics_by_snapshot.csv', index=False)
print(f"\nSaved metrics_by_snapshot.csv ({len(res)} rows)")

# ═══════════════════════════════════════════════════════════════
# Summary statistics
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("OVERALL WIND RADII ERRORS (predicted - observed, nm)")
print("="*70)

summary_rows = []

for name in model_names:
    for rad in ['R34', 'R50', 'R64']:
        err_col = f'{name}_{rad}_ERR'
        errs = res[err_col].dropna()
        if len(errs) == 0:
            continue
        row = {
            'Model': name,
            'Radius': rad,
            'N': len(errs),
            'Bias': errs.mean(),
            'MAE': errs.abs().mean(),
            'RMSE': np.sqrt((errs**2).mean()),
            'Median_Err': errs.median(),
            'P10': errs.quantile(0.10),
            'P90': errs.quantile(0.90),
        }
        summary_rows.append(row)

summary = pd.DataFrame(summary_rows)
summary.to_csv(r'D:\2026\ParametricWindModel\metrics_summary_v2.csv', index=False)

# Print nicely
for rad in ['R34', 'R50', 'R64']:
    sub = summary[summary['Radius'] == rad]
    if len(sub) == 0:
        continue
    print(f"\n── {rad} ──")
    print(f"{'Model':<18} {'N':>6} {'Bias':>8} {'MAE':>8} {'RMSE':>8} {'Median':>8}")
    for _, r in sub.iterrows():
        print(f"{r['Model']:<18} {r['N']:>6.0f} {r['Bias']:>8.1f} {r['MAE']:>8.1f} "
              f"{r['RMSE']:>8.1f} {r['Median_Err']:>8.1f}")

# ═══════════════════════════════════════════════════════════════
# Stratified summary: by intensity
# ═══════════════════════════════════════════════════════════════
print("\n\n" + "="*70)
print("R34 BIAS BY INTENSITY CATEGORY (nm)")
print("="*70)
print(f"{'Model':<18}", end="")
for cat in ['TD', 'TS', 'C1-2', 'C3-5']:
    print(f" {cat:>10}", end="")
print()

for name in model_names:
    err_col = f'{name}_R34_ERR'
    print(f"{name:<18}", end="")
    for cat in ['TD', 'TS', 'C1-2', 'C3-5']:
        sub = res.loc[res['INTENSITY_CAT'] == cat, err_col].dropna()
        if len(sub) > 5:
            print(f" {sub.mean():>10.1f}", end="")
        else:
            print(f" {'---':>10}", end="")
    print()

# ═══════════════════════════════════════════════════════════════
# Stratified: by latitude band
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("R34 BIAS BY LATITUDE BAND (nm)")
print("="*70)
print(f"{'Model':<18}", end="")
for band in ['00-15N', '15-25N', '25-35N']:
    print(f" {band:>10}", end="")
print()

for name in model_names:
    err_col = f'{name}_R34_ERR'
    print(f"{name:<18}", end="")
    for band in ['00-15N', '15-25N', '25-35N']:
        sub = res.loc[res['LAT_BAND'] == band, err_col].dropna()
        if len(sub) > 5:
            print(f" {sub.mean():>10.1f}", end="")
        else:
            print(f" {'---':>10}", end="")
    print()

# ═══════════════════════════════════════════════════════════════
# Stratified: by size class
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("R34 BIAS BY SIZE CLASS (nm)")
print("="*70)
print(f"{'Model':<18}", end="")
for sc in ['compact', 'average', 'large']:
    print(f" {sc:>10}", end="")
print()

for name in model_names:
    err_col = f'{name}_R34_ERR'
    print(f"{name:<18}", end="")
    for sc in ['compact', 'average', 'large']:
        sub = res.loc[res['SIZE_CLASS'] == sc, err_col].dropna()
        if len(sub) > 5:
            print(f" {sub.mean():>10.1f}", end="")
        else:
            print(f" {'---':>10}", end="")
    print()

print("\nDone.")
