"""
01_diagnose_basin_contamination.py

Purpose
-------
Quantify how many fixes in the original sample are NOT western North Pacific
(IBTrACS BASIN != 'WP'), and show that restricting to BASIN == 'WP' leaves the
model ranking unchanged. This is the diagnostic that motivated the corrected
sample used in the revision.

Inputs (place next to this script, or edit the paths below)
-----------------------------------------------------------
  snapshot_catalog.csv     : per-fix catalog (must contain BASIN, LON, SID, ISO_TIME)
  metrics_by_snapshot.csv  : per-snapshot observed radii + 6-model predictions/errors
                             (columns OBS_R34/R50/R64 and <Model>_<R>_ERR)

Output
------
Prints basin composition, the non-WP fix/storm counts, and overall RMSE by
basin filter for R34/R50/R64.
"""
import numpy as np
import pandas as pd

CATALOG = "snapshot_catalog.csv"
METRICS = "metrics_by_snapshot.csv"
MODELS = ["Rankine", "Holland1980", "Holland2010", "Willoughby2006", "Emanuel2004", "Chavas2015"]


def rmse_bias(sub, model, radius):
    err = sub[f"{model}_{radius}_ERR"]
    obs = sub[f"OBS_{radius}"]
    mask = err.notna() & obs.notna()
    e = err[mask].values
    return len(e), e.mean(), np.sqrt((e ** 2).mean())


def main():
    cat = pd.read_csv(CATALOG)
    met = pd.read_csv(METRICS)

    print("BASIN composition:")
    print(cat["BASIN"].value_counts().to_string())
    non_wp = cat[cat["BASIN"] != "WP"]
    print(f"\nnon-WP fixes: {len(non_wp)} across {non_wp['SID'].nunique()} storms")
    print(f"LON range in catalog: {cat['LON'].min()} .. {cat['LON'].max()} degE")

    d = met.merge(cat[["SID", "ISO_TIME", "BASIN"]], on=["SID", "ISO_TIME"], how="left")
    for radius in ["R34", "R50", "R64"]:
        print(f"\n== {radius} ==")
        for label, sub in [("ALL", d), ("WP-only", d[d["BASIN"] == "WP"])]:
            rows = [(m,) + rmse_bias(sub, m, radius) for m in MODELS]
            rows.sort(key=lambda x: x[3])  # by RMSE
            best = rows[0]
            wil = next(r for r in rows if r[0] == "Willoughby2006")
            print(f"  {label:8s} n={len(sub):6d}  best={best[0]} (RMSE {best[3]:.1f})  "
                  f"Willoughby RMSE={wil[3]:.1f} bias={wil[2]:+.1f}")


if __name__ == "__main__":
    main()
