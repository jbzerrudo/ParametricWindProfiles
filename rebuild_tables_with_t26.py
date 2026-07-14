#!/usr/bin/env python3
"""
rebuild_tables_with_t26.py
--------------------------------------------------------------------------------
Regenerates Tables 1, 2 and 3 for the WNP parametric-profile comparison with the
Tao (2026) [T26] model INCLUDED and FLAGGED, and verifies every T26 value against
the numbers printed in the manuscript.

Why this script exists:
  02_build_wnp_tables.py builds Table 1 for the six core models only, so the T26
  row of Table 1 (its N/bias/MAE/r/RMSE/CI) was never emitted by code. This script
  closes that gap: it rebuilds all three tables for all seven models from the
  shipped data, writes CSVs, and prints a PASS/FAIL check.

Run from inside the revision-bundle folder. Needs:
  snapshot_catalog.csv, metrics_by_snapshot.csv, t26_predictions.csv
Outputs:
  table1_wnp_full.csv, table2_wnp_full.csv, table3_era_full.csv

Notes:
  * Point estimates (N, bias, MAE, r, RMSE, stratified biases, era RMSE) are exact.
  * Bootstrap 95% CIs are deterministic given the seed but may differ ~0.1 nm from
    the manuscript's printed T26 CI, which was computed off-script.
  * The six-model Table 1 rows reproduce table1_wnp.csv exactly (same seed/order).
"""
import numpy as np, pandas as pd

CAT = "snapshot_catalog.csv"; MET = "metrics_by_snapshot.csv"; T26 = "t26_predictions.csv"
MODELS = ["Rankine", "Holland1980", "Holland2010", "Willoughby2006", "Emanuel2004", "Chavas2015"]
ALL = MODELS + ["Tao2026"]
LABEL = {"Rankine": "Rankine", "Holland1980": "Holland (1980)", "Holland2010": "Holland (2010)",
         "Willoughby2006": "Willoughby (2006)", "Emanuel2004": "Emanuel (2004), hyp.",
         "Chavas2015": "Chavas (2015)", "Tao2026": "Tao (2026) [R34 inferred]"}


def boot_ci(err, sid, nboot=1000):
    """Storm-level bootstrap 95% CI for RMSE (identical to 02_build_wnp_tables.py)."""
    g = pd.DataFrame({"sse": err ** 2, "sid": sid}).groupby("sid")["sse"].agg(["sum", "count"])
    sse, nn = g["sum"].values, g["count"].values; S = len(sse)
    idx = np.random.randint(0, S, size=(nboot, S))
    rb = np.sqrt(sse[idx].sum(1) / nn[idx].sum(1))
    return np.percentile(rb, 2.5), np.percentile(rb, 97.5)


def stat_row(wp, model, radius):
    err = wp[f"{model}_{radius}_ERR"]; pred = wp[f"{model}_{radius}"]; obs = wp[f"OBS_{radius}"]
    m = err.notna() & obs.notna(); e = err[m].values
    if len(e) < 5:
        return None
    lo, hi = boot_ci(e, wp["SID"].values[m])
    return dict(Model=LABEL[model], Radius=radius, N=len(e), Bias=round(e.mean(), 1),
                MAE=round(np.abs(e).mean(), 1), r=round(np.corrcoef(pred[m], obs[m])[0, 1], 2),
                RMSE=round(np.sqrt((e ** 2).mean()), 1), CI_lo=round(lo, 1), CI_hi=round(hi, 1))


def check(label, got, want, tol=0.15):
    ok = (want is None) or (got == want) or (isinstance(got, float) and abs(got - want) <= tol)
    print(f"    {label:28s} computed={got!s:>8}  manuscript={want!s:>8}  {'OK' if ok else 'MISMATCH'}")
    return ok


def main():
    met = pd.read_csv(MET); cat = pd.read_csv(CAT)[["SID", "ISO_TIME", "BASIN"]]; t26 = pd.read_csv(T26)
    wp = met.merge(cat, on=["SID", "ISO_TIME"], how="left")
    wp = wp[wp["BASIN"] == "WP"].copy()
    wp = wp.merge(t26[["idx", "Tao2026_R34", "Tao2026_R50", "Tao2026_R64",
                       "Tao2026_R34_ERR", "Tao2026_R50_ERR", "Tao2026_R64_ERR"]], on="idx", how="left")
    wp["YEAR"] = pd.to_datetime(wp["ISO_TIME"], errors="coerce").dt.year
    print(f"WNP sample: {len(wp)} fixes, {wp['SID'].nunique()} storms  (expect 18147 / 512)\n")

    # ---------- TABLE 1 : seven models, full columns ----------
    np.random.seed(42)                                   # six-model rows == table1_wnp.csv
    rows = [r for R in ["R34", "R50", "R64"] for r in [stat_row(wp, m, R) for m in MODELS] if r]
    np.random.seed(42)                                   # deterministic T26 rows
    rows += [stat_row(wp, "Tao2026", R) for R in ["R34", "R50", "R64"]]
    t1 = pd.DataFrame(rows); t1.to_csv("table1_wnp_full.csv", index=False)
    print("=== TABLE 1 (all seven models) -> table1_wnp_full.csv ===")
    print(t1.to_string(index=False))

    # ---------- TABLE 2 : R34 bias by strata, incl. Tao ----------
    q33, q66 = wp["OBS_R34"].quantile([0.33, 0.66])
    wp["SIZE_CLASS"] = wp["OBS_R34"].apply(
        lambda x: np.nan if pd.isna(x) else ("compact" if x <= q33 else ("average" if x <= q66 else "large")))
    strata = [("INTENSITY_CAT", ["TS", "C1-2", "C3-5"]),
              ("LAT_BAND", ["00-15N", "15-25N", "25-35N"]),
              ("SIZE_CLASS", ["compact", "average", "large"])]
    t2rows = []
    for m in ALL:
        row = {"Model": LABEL[m]}
        for col, cats in strata:
            for c in cats:
                e = wp.loc[wp[col] == c, f"{m}_R34_ERR"].dropna()
                row[c] = round(e.mean(), 1) if len(e) > 5 else np.nan
        t2rows.append(row)
    t2 = pd.DataFrame(t2rows); t2.to_csv("table2_wnp_full.csv", index=False)
    print("\n=== TABLE 2 (R34 bias by strata) -> table2_wnp_full.csv ===")
    print(t2.to_string(index=False))

    # ---------- TABLE 3 : R34 RMSE by era, incl. Tao ----------
    eras = [("2001-2024", 2001), ("2016-2024", 2016), ("2020-2024", 2020)]
    t3rows = []
    for m in ALL:
        row = {"Model": LABEL[m]}
        for lab, y0 in eras:
            e = wp.loc[wp.YEAR >= y0, f"{m}_R34_ERR"].dropna()
            row[lab] = round(np.sqrt((e ** 2).mean()), 1) if len(e) > 5 else np.nan
        t3rows.append(row)
    t3 = pd.DataFrame(t3rows); t3.to_csv("table3_era_full.csv", index=False)
    print("\n=== TABLE 3 (R34 RMSE by era) -> table3_era_full.csv ===")
    print(t3.to_string(index=False))

    # ---------- VERIFICATION against the manuscript ----------
    print("\n================ VERIFICATION vs manuscript ================")
    g = {(r["Radius"]): r for r in rows if r["Model"].startswith("Tao")}
    print("  Table 1  Tao (2026):")
    ok = True
    for R, (N, B, MA, RR, RM) in {"R34": (17015, -27.6, 39.9, 0.46, 53.3),
                                  "R50": (9205, -17.8, 23.2, 0.44, 31.9),
                                  "R64": (6400, -7.4, 12.6, 0.40, 17.4)}.items():
        ok &= check(f"{R} N",   g[R]["N"], N, 0)
        ok &= check(f"{R} bias", g[R]["Bias"], B)
        ok &= check(f"{R} MAE",  g[R]["MAE"], MA)
        ok &= check(f"{R} r",    g[R]["r"], RR, 0.01)
        ok &= check(f"{R} RMSE", g[R]["RMSE"], RM)
    tao2 = next(r for r in t2rows if r["Model"].startswith("Tao"))
    print("  Table 2  Tao (2026) R34 bias:")
    for c, w in {"TS": -18.3, "C1-2": -41.5, "C3-5": -27.1, "00-15N": -15.5, "15-25N": -27.3,
                 "25-35N": -39.0, "compact": -0.4, "average": -26.7, "large": -61.4}.items():
        ok &= check(c, tao2[c], w)
    tao3 = next(r for r in t3rows if r["Model"].startswith("Tao"))
    print("  Table 3  Tao (2026) R34 RMSE:")
    for c, w in {"2001-2024": 53.3, "2016-2024": 59.6, "2020-2024": 55.1}.items():
        ok &= check(c, tao3[c], w)
    rank3 = next(r for r in t3rows if r["Model"] == "Rankine")
    print("  (bonus) fixed-alpha Rankine R34 RMSE, 2016-2024 -- settles the 59.3-vs-59.4 point:")
    check("Rankine 2016-2024", rank3["2016-2024"], 59.4)

    print("\n  RESULT:", "ALL T26 POINT ESTIMATES REPRODUCE." if ok else "SOME VALUES DID NOT MATCH -- investigate.")


if __name__ == "__main__":
    main()
