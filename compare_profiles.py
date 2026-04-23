"""
compare_profiles.py
Main comparison pipeline for parametric TC wind profiles.

For each valid snapshot in the IBTrACS catalog:
  1. Reconstruct radial wind profile with each of the 6 models
     (Chavas2015 is now the proper CLE15 implementation; r0 is
     solved internally, so the old 'Chavas_clim' configuration
     is no longer needed.)
  2. Extract predicted wind radii (R34, R50, R64) from each profile
  3. Compare against observed JTWC wind radii
  4. Compute error metrics by intensity category, latitude band, size class
  5. Bootstrap 95% CIs for RMSE (1000 iterations, storm-level resampling)

Outputs:
  - metrics_by_snapshot.csv  (per-snapshot, per-model errors)
  - metrics_summary_v4.csv   (aggregated statistics with bootstrap CIs & r)
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from wind_profiles import PROFILES
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

# ═══════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════
INPUT_CSV  = r'D:\2026\ParametricWindModel\CorrectedScripts\snapshot_catalog.csv'
OUTPUT_DIR = r'D:\2026\ParametricWindModel\NEWOUTS'
N_BOOT     = 1000   # bootstrap iterations for 95% CI

# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════
def extract_wind_radius(r, v, threshold):
    """Outermost radius where V >= threshold. NaN if never reached."""
    mask = v >= threshold
    if not mask.any():
        return np.nan
    return r[mask][-1]


def intensity_cat(v):
    if v < 34:   return 'TD'
    elif v < 64: return 'TS'
    elif v < 96: return 'C1-2'
    else:        return 'C3-5'


def lat_band(lat):
    if lat < 15:   return '00-15N'
    elif lat < 25: return '15-25N'
    else:          return '25-35N'


def bootstrap_rmse_ci(errors_by_storm, n_boot=1000, ci=0.95):
    """
    Bootstrap 95% CI for RMSE, resampling at the storm level
    to account for temporal autocorrelation within TCs.

    errors_by_storm: dict  {SID: array of errors}
    Returns: (rmse_lo, rmse_hi)
    """
    sids = list(errors_by_storm.keys())
    n_storms = len(sids)
    if n_storms < 5:
        return (np.nan, np.nan)

    rng = np.random.default_rng(42)
    rmse_samples = np.empty(n_boot)

    for b in range(n_boot):
        idx = rng.integers(0, n_storms, size=n_storms)
        boot_errs = np.concatenate([errors_by_storm[sids[i]] for i in idx])
        rmse_samples[b] = np.sqrt(np.mean(boot_errs**2))

    alpha = (1.0 - ci) / 2.0
    return (np.quantile(rmse_samples, alpha),
            np.quantile(rmse_samples, 1.0 - alpha))


# ═══════════════════════════════════════════════════════════════
# Load data
# ═══════════════════════════════════════════════════════════════
print("Loading snapshot catalog...")
df = pd.read_csv(INPUT_CSV)

num_cols = ['LAT', 'LON', 'USA_WIND', 'USA_PRES', 'USA_RMW',
            'USA_R34_NE', 'USA_R34_SE', 'USA_R34_SW', 'USA_R34_NW',
            'USA_R50_NE', 'USA_R50_SE', 'USA_R50_SW', 'USA_R50_NW',
            'USA_R64_NE', 'USA_R64_SE', 'USA_R64_SW', 'USA_R64_NW']
for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors='coerce')

# Vectorised observed mean radii
for prefix, obs_col in [('USA_R34', 'OBS_R34'), ('USA_R50', 'OBS_R50'),
                         ('USA_R64', 'OBS_R64')]:
    quad_cols = [f'{prefix}_{q}' for q in ('NE', 'SE', 'SW', 'NW')]
    df[obs_col] = df[quad_cols].mean(axis=1, skipna=True)

# Stratification
df['INTENSITY_CAT'] = df['USA_WIND'].apply(intensity_cat)
df['LAT_BAND']      = df['LAT'].apply(lat_band)

q33, q66 = df['OBS_R34'].quantile([0.33, 0.66])
def size_class(r34):
    if pd.isna(r34): return 'unknown'
    if r34 <= q33:   return 'compact'
    elif r34 <= q66: return 'average'
    else:            return 'large'
df['SIZE_CLASS'] = df['OBS_R34'].apply(size_class)

print(f"Total snapshots: {len(df)}")
print(f"  With R34: {df['OBS_R34'].notna().sum()}")
print(f"  With R50: {df['OBS_R50'].notna().sum()}")
print(f"  With R64: {df['OBS_R64'].notna().sum()}")

# ═══════════════════════════════════════════════════════════════
# Model registry: 6 profiles
# ═══════════════════════════════════════════════════════════════
model_names = list(PROFILES.keys())

# ═══════════════════════════════════════════════════════════════
# Run all profiles on all snapshots
# ═══════════════════════════════════════════════════════════════
print("\nReconstructing profiles...")

r_grid = np.arange(0, 501, 1.0)
n_total = len(df)
results = []

for idx, (i, row) in enumerate(df.iterrows()):
    if idx % 2000 == 0:
        print(f"  {idx}/{n_total} ({100*idx/n_total:.0f}%)")

    vmax     = row['USA_WIND']
    rmax     = row['USA_RMW']
    lat      = row['LAT']
    pc       = row['USA_PRES']

    params = dict(
        r=r_grid, vmax=vmax, rmax=rmax, lat=lat,
        pc=pc if pd.notna(pc) else np.nan,
        penv=1013.0,
    )

    snap = {
        'idx': i, 'SID': row['SID'], 'ISO_TIME': row['ISO_TIME'],
        'VMAX': vmax, 'RMAX': rmax, 'LAT': lat, 'PC': pc,
        'OBS_R34': row['OBS_R34'], 'OBS_R50': row['OBS_R50'],
        'OBS_R64': row['OBS_R64'],
        'INTENSITY_CAT': row['INTENSITY_CAT'],
        'LAT_BAND': row['LAT_BAND'],
        'SIZE_CLASS': row['SIZE_CLASS'],
    }

    for name in model_names:
        # Holland 1980 and Holland 2010 both need pc
        if name in ('Holland1980', 'Holland2010') and pd.isna(pc):
            for rad in ('R34', 'R50', 'R64'):
                snap[f'{name}_{rad}'] = np.nan
            continue

        try:
            v = PROFILES[name](**params)

            for rad, thresh in [('R34', 34.0), ('R50', 50.0), ('R64', 64.0)]:
                snap[f'{name}_{rad}'] = extract_wind_radius(r_grid, v, thresh)
        except Exception:
            for rad in ('R34', 'R50', 'R64'):
                snap[f'{name}_{rad}'] = np.nan

    results.append(snap)

print(f"  {n_total}/{n_total} (100%)")

# ═══════════════════════════════════════════════════════════════
# Assemble & compute errors
# ═══════════════════════════════════════════════════════════════
res = pd.DataFrame(results)

for name in model_names:
    for rad in ('R34', 'R50', 'R64'):
        res[f'{name}_{rad}_ERR'] = res[f'{name}_{rad}'] - res[f'OBS_{rad}']

out_snap = f'{OUTPUT_DIR}\\Models\\metrics_by_snapshot.csv'
res.to_csv(out_snap, index=False)
print(f"\nSaved {out_snap} ({len(res)} rows)")

# ═══════════════════════════════════════════════════════════════
# Summary statistics with bootstrap CIs and Pearson r
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("OVERALL WIND RADII ERRORS (predicted - observed, nm)")
print("=" * 80)

summary_rows = []

for name in model_names:
    for rad in ('R34', 'R50', 'R64'):
        err_col  = f'{name}_{rad}_ERR'
        pred_col = f'{name}_{rad}'
        obs_col  = f'OBS_{rad}'

        mask = res[err_col].notna()
        errs = res.loc[mask, err_col]
        if len(errs) == 0:
            continue

        # Pearson r
        valid = mask & res[pred_col].notna() & res[obs_col].notna()
        if valid.sum() > 10:
            corr, _ = pearsonr(res.loc[valid, pred_col], res.loc[valid, obs_col])
        else:
            corr = np.nan

        # Storm-level bootstrap for RMSE 95% CI
        sub = res.loc[mask, ['SID', err_col]]
        errors_by_storm = {sid: grp[err_col].values
                           for sid, grp in sub.groupby('SID')}
        ci_lo, ci_hi = bootstrap_rmse_ci(errors_by_storm, N_BOOT)

        row = {
            'Model': name, 'Radius': rad,
            'N': len(errs),
            'Bias': errs.mean(),
            'MAE': errs.abs().mean(),
            'r': corr,
            'RMSE': np.sqrt((errs**2).mean()),
            'CI_lo': ci_lo, 'CI_hi': ci_hi,
        }
        summary_rows.append(row)

summary = pd.DataFrame(summary_rows)
out_summary = f'{OUTPUT_DIR}\\metrics_summary_v4.csv'
summary.to_csv(out_summary, index=False)
print(f"Saved {out_summary}")

# Pretty-print
for rad in ('R34', 'R50', 'R64'):
    sub = summary[summary['Radius'] == rad]
    if len(sub) == 0:
        continue
    print(f"\n-- {rad} --")
    print(f"{'Model':<18} {'N':>6} {'Bias':>8} {'MAE':>8} {'r':>6} "
          f"{'RMSE':>8} {'[95% CI]':>20}")
    for _, r in sub.iterrows():
        ci_str = f"[{r['CI_lo']:.1f}, {r['CI_hi']:.1f}]"
        print(f"{r['Model']:<18} {r['N']:>6.0f} {r['Bias']:>+8.1f} "
              f"{r['MAE']:>8.1f} {r['r']:>6.2f} {r['RMSE']:>8.1f} {ci_str:>20}")

# ═══════════════════════════════════════════════════════════════
# Stratified: R34 bias by intensity
# ═══════════════════════════════════════════════════════════════
print("\n\n" + "=" * 80)
print("R34 BIAS BY INTENSITY CATEGORY (nm)")
print("=" * 80)
cats = ['TS', 'C1-2', 'C3-5']
print(f"{'Model':<18}", end="")
for cat in cats:
    print(f" {cat:>10}", end="")
print()

for name in model_names:
    err_col = f'{name}_R34_ERR'
    print(f"{name:<18}", end="")
    for cat in cats:
        sub = res.loc[res['INTENSITY_CAT'] == cat, err_col].dropna()
        if len(sub) > 5:
            print(f" {sub.mean():>+10.1f}", end="")
        else:
            print(f" {'---':>10}", end="")
    print()

# ═══════════════════════════════════════════════════════════════
# Stratified: R34 bias by latitude band
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("R34 BIAS BY LATITUDE BAND (nm)")
print("=" * 80)
bands = ['00-15N', '15-25N', '25-35N']
print(f"{'Model':<18}", end="")
for band in bands:
    print(f" {band:>10}", end="")
print()

for name in model_names:
    err_col = f'{name}_R34_ERR'
    print(f"{name:<18}", end="")
    for band in bands:
        sub = res.loc[res['LAT_BAND'] == band, err_col].dropna()
        if len(sub) > 5:
            print(f" {sub.mean():>+10.1f}", end="")
        else:
            print(f" {'---':>10}", end="")
    print()

# ═══════════════════════════════════════════════════════════════
# Stratified: R34 bias by size class
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("R34 BIAS BY SIZE CLASS (nm)")
print("=" * 80)
sizes = ['compact', 'average', 'large']
print(f"{'Model':<18}", end="")
for sc in sizes:
    print(f" {sc:>10}", end="")
print()

for name in model_names:
    err_col = f'{name}_R34_ERR'
    print(f"{name:<18}", end="")
    for sc in sizes:
        sub = res.loc[res['SIZE_CLASS'] == sc, err_col].dropna()
        if len(sub) > 5:
            print(f" {sub.mean():>+10.1f}", end="")
        else:
            print(f" {'---':>10}", end="")
    print()

print("\nDone.")
