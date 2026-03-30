"""
select_storms.py
Filter IBTrACS WP to storms usable for parametric wind profile comparison.
Criteria:
  - 0-35°N domain
  - Observed USA_RMW present
  - Observed USA_R34 wind radii present (at least one quadrant)
  - USA_WIND (JTWC intensity) present
Stratify by:
  - Intensity category (TD/TS/C1-2/C3-5)
  - Latitude band (0-15, 15-25, 25-35)
  - Size (compact/average/large based on mean R34)
Output: storm_catalog.csv (one row per storm, summary stats)
        snapshot_catalog.csv (all valid 6-hourly snapshots)
"""
import pandas as pd
import numpy as np

# ── Load ──
df = pd.read_csv('filtered_ibtracs_v2.csv')

# ── Coerce numerics ──
num_cols = ['LAT', 'LON', 'USA_WIND', 'USA_PRES', 'USA_RMW', 'WMO_WIND', 'WMO_PRES',
            'USA_R34_NE', 'USA_R34_SE', 'USA_R34_SW', 'USA_R34_NW',
            'USA_R50_NE', 'USA_R50_SE', 'USA_R50_SW', 'USA_R50_NW',
            'USA_R64_NE', 'USA_R64_SE', 'USA_R64_SW', 'USA_R64_NW']
for c in num_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

# ── Filter: domain, RMW, wind, R34 ──
mask = (
    (df['LAT'] >= 0) & (df['LAT'] <= 35) &
    df['USA_RMW'].notna() &
    df['USA_WIND'].notna() &
    (df['USA_WIND'] >= 20) &  # at least tropical depression
    df[['USA_R34_NE', 'USA_R34_SE', 'USA_R34_SW', 'USA_R34_NW']].notna().any(axis=1)
)
snaps = df[mask].copy()

# ── Derived fields ──
# Mean R34 (average of available quadrants)
r34_cols = ['USA_R34_NE', 'USA_R34_SE', 'USA_R34_SW', 'USA_R34_NW']
snaps['R34_MEAN'] = snaps[r34_cols].mean(axis=1, skipna=True)

# Intensity category
def intensity_cat(v):
    if v < 34:   return 'TD'
    elif v < 64:  return 'TS'
    elif v < 96:  return 'C1-2'
    else:         return 'C3-5'

snaps['INTENSITY_CAT'] = snaps['USA_WIND'].apply(intensity_cat)

# Latitude band
def lat_band(lat):
    if lat < 15:   return '00-15N'
    elif lat < 25:  return '15-25N'
    else:           return '25-35N'

snaps['LAT_BAND'] = snaps['LAT'].apply(lat_band)

# ── Storm-level summary ──
storm_summary = snaps.groupby('SID').agg(
    NAME=('NAME', 'first'),
    SEASON=('SEASON', 'first'),
    N_SNAPS=('ISO_TIME', 'count'),
    VMAX=('USA_WIND', 'max'),
    MIN_PRES=('USA_PRES', 'min'),
    MEAN_RMW=('USA_RMW', 'mean'),
    MEAN_R34=('R34_MEAN', 'mean'),
    MEAN_LAT=('LAT', 'mean'),
    MAX_CAT=('USA_WIND', lambda x: intensity_cat(x.max())),
    LAT_BAND_MODE=('LAT_BAND', lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'unknown')
).reset_index()

# Size classification (terciles of mean R34)
q33, q66 = storm_summary['MEAN_R34'].quantile([0.33, 0.66])
def size_class(r34):
    if r34 <= q33:   return 'compact'
    elif r34 <= q66:  return 'average'
    else:             return 'large'

storm_summary['SIZE_CLASS'] = storm_summary['MEAN_R34'].apply(size_class)

# ── Print summary ──
print(f"Valid snapshots: {len(snaps)}")
print(f"Unique storms:  {storm_summary.shape[0]}")
print(f"Season range:   {storm_summary['SEASON'].min()}-{storm_summary['SEASON'].max()}")
print(f"\nBy intensity category:")
print(storm_summary['MAX_CAT'].value_counts().sort_index())
print(f"\nBy latitude band:")
print(storm_summary['LAT_BAND_MODE'].value_counts().sort_index())
print(f"\nBy size class:")
print(storm_summary['SIZE_CLASS'].value_counts())
print(f"\nR34 tercile thresholds: compact <= {q33:.0f} nm, large > {q66:.0f} nm")

# ── R50/R64 availability ──
r50_any = snaps[['USA_R50_NE','USA_R50_SE','USA_R50_SW','USA_R50_NW']].notna().any(axis=1)
r64_any = snaps[['USA_R64_NE','USA_R64_SE','USA_R64_SW','USA_R64_NW']].notna().any(axis=1)
print(f"\nSnapshots with R50: {r50_any.sum()} ({100*r50_any.mean():.1f}%)")
print(f"Snapshots with R64: {r64_any.sum()} ({100*r64_any.mean():.1f}%)")

# ── Save ──
storm_summary.to_csv('storm_catalog.csv', index=False)
snaps.to_csv('snapshot_catalog.csv', index=False)
print("\nSaved storm_catalog.csv and snapshot_catalog.csv")
