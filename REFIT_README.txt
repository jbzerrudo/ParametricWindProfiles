================================================================================
Out-of-sample basin-specific re-fit  (adds Table 4 to the revised manuscript)
================================================================================

These three scripts extend the analysis in the main bundle. They read the same
inputs already shipped in the repository:
    metrics_by_snapshot.csv   (per-snapshot inputs + as-published predictions)
    snapshot_catalog.csv      (BASIN, ISO_TIME for the WNP filter and the split)
    wind_profiles.py          (the six model implementations)
Place these three scripts next to those files and run with Python 3.9+
(numpy, pandas, scipy).

--------------------------------------------------------------------------------
WHAT THEY DO
--------------------------------------------------------------------------------
refit_all.py
    Produces Table 4. For each model with an adjustable coefficient, fits the
    WNP-specific parameter on 2016-2020 and evaluates R34 RMSE on the held-out
    2021-2024 period:
        Rankine    alpha(Vmax,|lat|)      [grid-based R34, caps at 500 nm]
        Holland80  shape B  (WNP factor)
        Holland10  peakedness b_s (WNP factor)
        Willoughby far-field length X1 (WNP scale)
        CLE15      Ck/Cd (WNP scale; calls wind_profiles.chavas2015 directly)
        Emanuel    -- no adjustable coefficient
    Prints the as-published vs WNP-refit held-out RMSE per model.

refit_verify.py
    Equation check. Runs each re-implementation with the AS-PUBLISHED
    coefficients and compares, snapshot by snapshot, to the paper's precomputed
    R34 columns (which Tables 1-3 are built from). Holland80, Holland10,
    Willoughby, and CLE15 reproduce them exactly (0.000 nm).

refit_rankine_grid.py
    Confirms the Rankine row using the same grid-based R34 extraction as the
    other models (as-published 58.5 nm matches the precomputed column; held-out
    re-fit 52.3 nm).

--------------------------------------------------------------------------------
NOTES
--------------------------------------------------------------------------------
* All held-out RMSEs are computed on the full 2021-2024 validation set.
* CLE15 is the slow model (per-snapshot root-finding). refit_all.py fits its
  Ck/Cd on the full 2016-2020 training set by default; set CLE_FIT_N near the
  top to a smaller integer for a fast approximate fit (same result).
* Randomness is seeded (numpy seed 20260715) for reproducibility.
