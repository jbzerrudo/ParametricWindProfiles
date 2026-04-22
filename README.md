# Parametric Wind Profiles for the Western North Pacific

Python code and processed datasets for Zerrudo & Bala (2026),
"A Comparison of Parametric Tropical Cyclone Wind Profiles for the
Western North Pacific," submitted to *Tropical Cyclone Research and Review*.

## Contents

- `wind_profiles.py` — Six parametric profile implementations
  (modified Rankine, Holland 1980, Holland 2010, Willoughby 2006,
  Emanuel 2004, simplified Chavas-type).
- `compare_profiles.py` — Main comparison pipeline. Reads an IBTrACS
  snapshot catalog, reconstructs profiles, and computes error metrics
  with storm-level bootstrap 95% CIs.
- `update_all_figs_chavas_clim.py` — Regenerates Figures 1–6 from
  `metrics_by_snapshot.csv`.
- `metrics_summary_v3.csv` — Summary statistics (Table 1 of the paper).

## Requirements

Python 3.9+, numpy, pandas, scipy, matplotlib.

## Data source

IBTrACS v04r01 (Western Pacific basin), 2001–2024. Available from
NOAA NCEI. A `snapshot_catalog.csv` must be built from IBTrACS before
running `compare_profiles.py`.

## Citation

Zerrudo, J. B., & Bala, M. S. (2026). A Comparison of Parametric
Tropical Cyclone Wind Profiles for the Western North Pacific.
*Tropical Cyclone Research and Review* (in review).

## Contact

Jeferson B. Zerrudo — jbzerrudo@pagasa.dost.gov.ph