"""
diagnose_chavas_failures.py
Identify the 173 snapshots where CLE15 fails, and characterize them.

This script re-runs chavas2015() on each snapshot in the catalog,
flags the failures, and summarizes their distribution across VMAX,
LAT, RMAX, and PC.

Run time: ~10-20 minutes (it's re-running CLE15 on all 18,772 snapshots).
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

from wind_profiles import chavas2015

# ── CONFIG ──
INPUT_CSV = r'D:\2026\ParametricWindModel\CorrectedScripts\snapshot_catalog.csv'
OUTPUT_CSV = r'D:\2026\ParametricWindModel\NEWOUTS\chavas_failures.csv'
# ─────────────

print("Loading snapshot catalog...")
df = pd.read_csv(INPUT_CSV)

num_cols = ['LAT', 'LON', 'USA_WIND', 'USA_PRES', 'USA_RMW']
for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors='coerce')

print(f"Total snapshots: {len(df)}")

# Evaluate chavas2015 on a dummy radial grid — we only care whether it succeeds
r_grid = np.array([50.0])  # single radius, just to trigger the solver
failures = []
successes = 0

print("\nRe-running CLE15 on all snapshots (this takes 10-20 minutes)...")
for i, row in df.iterrows():
    if i % 2000 == 0:
        print(f"  {i}/{len(df)} ({100*i/len(df):.0f}%)  [failures so far: {len(failures)}]")

    vmax = row['USA_WIND']
    rmax = row['USA_RMW']
    lat  = row['LAT']

    if pd.isna(vmax) or pd.isna(rmax) or pd.isna(lat):
        continue
    if vmax < 20:  # below our sample threshold
        continue

    try:
        v = chavas2015(r=r_grid, vmax=vmax, rmax=rmax, lat=lat)
        if np.all(np.isnan(v)):
            failures.append({
                'SID':  row.get('SID', ''),
                'ISO_TIME': row.get('ISO_TIME', ''),
                'VMAX': vmax,
                'RMAX': rmax,
                'LAT':  lat,
                'PC':   row.get('USA_PRES', np.nan),
            })
        else:
            successes += 1
    except Exception as e:
        failures.append({
            'SID':  row.get('SID', ''),
            'ISO_TIME': row.get('ISO_TIME', ''),
            'VMAX': vmax,
            'RMAX': rmax,
            'LAT':  lat,
            'PC':   row.get('USA_PRES', np.nan),
            'ERROR': str(e)[:80],
        })

print(f"\nTotal: {len(df)} | Successes: {successes} | Failures: {len(failures)}")

if not failures:
    print("No failures found.")
    raise SystemExit(0)

fail_df = pd.DataFrame(failures)
fail_df.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved {OUTPUT_CSV}")

# ════════════════════════════════════════════════════════════════
# SUMMARIZE FAILURE PATTERNS
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("FAILURE SUMMARY")
print("=" * 60)

print(f"\nN failures: {len(fail_df)}")
print(f"\n-- VMAX (kt) --")
print(f"  min/median/max: {fail_df.VMAX.min():.1f} / {fail_df.VMAX.median():.1f} / {fail_df.VMAX.max():.1f}")
print(f"  < 35 kt (weak):    {(fail_df.VMAX < 35).sum()} ({100*(fail_df.VMAX < 35).mean():.0f}%)")
print(f"  35-64 kt (TS):     {((fail_df.VMAX >= 35) & (fail_df.VMAX < 64)).sum()}")
print(f"  >= 64 kt (TY+):    {(fail_df.VMAX >= 64).sum()}")

print(f"\n-- LAT (deg N) --")
print(f"  min/median/max: {fail_df.LAT.min():.1f} / {fail_df.LAT.median():.1f} / {fail_df.LAT.max():.1f}")
print(f"  < 10N (deep tropics):  {(fail_df.LAT < 10).sum()} ({100*(fail_df.LAT < 10).mean():.0f}%)")
print(f"  10-20N:                {((fail_df.LAT >= 10) & (fail_df.LAT < 20)).sum()}")
print(f"  >= 20N:                {(fail_df.LAT >= 20).sum()}")

print(f"\n-- RMAX (nm) --")
print(f"  min/median/max: {fail_df.RMAX.min():.1f} / {fail_df.RMAX.median():.1f} / {fail_df.RMAX.max():.1f}")
print(f"  very large (>100 nm): {(fail_df.RMAX > 100).sum()}")
print(f"  small (<=15 nm):      {(fail_df.RMAX <= 15).sum()}")

# Cross-tab: are failures concentrated in weak-AND-equatorial storms?
print(f"\n-- CROSS-TAB --")
weak_trop = ((fail_df.VMAX < 35) & (fail_df.LAT < 10)).sum()
weak_only = ((fail_df.VMAX < 35) & (fail_df.LAT >= 10)).sum()
trop_only = ((fail_df.VMAX >= 35) & (fail_df.LAT < 10)).sum()
neither   = ((fail_df.VMAX >= 35) & (fail_df.LAT >= 10)).sum()
print(f"  Weak AND equatorial:    {weak_trop}")
print(f"  Weak but not equatorial: {weak_only}")
print(f"  Equatorial but not weak: {trop_only}")
print(f"  Neither:                 {neither}")