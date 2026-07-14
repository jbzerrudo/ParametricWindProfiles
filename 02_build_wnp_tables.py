"""
02_build_wnp_tables.py
Table 1 (overall errors + storm-level bootstrap CIs, six core models) and Table 2
(R34 bias by intensity/latitude/size). If t26_predictions.csv is present (run 03
first), Tao (2026) is added to Table 2, flagged: for T26, R34 is an inferred size
input (via Chavas & Knaff 2022), not an independent prediction.
Inputs : snapshot_catalog.csv, metrics_by_snapshot.csv, [t26_predictions.csv]
Outputs: table1_wnp.csv, table2_wnp.csv
"""
import os, numpy as np, pandas as pd
np.random.seed(42)
CATALOG="snapshot_catalog.csv"; METRICS="metrics_by_snapshot.csv"
MODELS=["Rankine","Holland1980","Holland2010","Willoughby2006","Emanuel2004","Chavas2015"]
LABEL={"Rankine":"Rankine","Holland1980":"Holland (1980)","Holland2010":"Holland (2010)",
       "Willoughby2006":"Willoughby (2006)","Emanuel2004":"Emanuel (2004), hyp.","Chavas2015":"Chavas (2015)",
       "Tao2026":"Tao (2026) [R34 inferred]"}

def boot_ci(err,sid,nboot=1000):
    g=pd.DataFrame({"sse":err**2,"sid":sid}).groupby("sid")["sse"].agg(["sum","count"])
    sse,nn=g["sum"].values,g["count"].values; S=len(sse)
    idx=np.random.randint(0,S,size=(nboot,S)); rb=np.sqrt(sse[idx].sum(1)/nn[idx].sum(1))
    return np.percentile(rb,2.5),np.percentile(rb,97.5)

def stat_row(sub,model,radius):
    err=sub[f"{model}_{radius}_ERR"]; pred=sub[f"{model}_{radius}"]; obs=sub[f"OBS_{radius}"]
    mask=err.notna()&obs.notna(); e=err[mask].values
    if len(e)<5: return None
    lo,hi=boot_ci(e,sub["SID"].values[mask])
    return dict(Model=LABEL[model],Radius=radius,N=len(e),Bias=round(e.mean(),1),MAE=round(np.abs(e).mean(),1),
                r=round(np.corrcoef(pred[mask],obs[mask])[0,1],2),RMSE=round(np.sqrt((e**2).mean()),1),CI_lo=round(lo,1),CI_hi=round(hi,1))

def main():
    met=pd.read_csv(METRICS); cat=pd.read_csv(CATALOG)[["SID","ISO_TIME","BASIN"]]
    wp=met.merge(cat,on=["SID","ISO_TIME"],how="left"); wp=wp[wp["BASIN"]=="WP"].copy()
    print(f"true-WNP sample: {len(wp)} fixes, {wp['SID'].nunique()} storms")
    q33,q66=wp["OBS_R34"].quantile([0.33,0.66])
    wp["SIZE_CLASS"]=wp["OBS_R34"].apply(lambda x:"unknown" if pd.isna(x) else ("compact" if x<=q33 else ("average" if x<=q66 else "large")))
    # Table 1 (six core models)
    t1=pd.DataFrame([r for R in ["R34","R50","R64"] for r in [stat_row(wp,m,R) for m in MODELS] if r])
    t1.to_csv("table1_wnp.csv",index=False)
    # Table 2 -- add Tao if available
    models_t2=MODELS[:]
    if os.path.exists("t26_predictions.csv"):
        _t=pd.read_csv("t26_predictions.csv"); wp=wp.merge(_t[["idx","Tao2026_R34_ERR"]],on="idx",how="left"); models_t2=MODELS+["Tao2026"]
    strata=[("TS","C1-2","C3-5","INTENSITY_CAT"),("00-15N","15-25N","25-35N","LAT_BAND"),("compact","average","large","SIZE_CLASS")]
    rows=[]
    for m in models_t2:
        row={"Model":LABEL[m]}
        for *cats,col in strata:
            for c in cats:
                e=wp.loc[wp[col]==c,f"{m}_R34_ERR"].dropna(); row[c]=round(e.mean(),1) if len(e)>5 else np.nan
        rows.append(row)
    t2=pd.DataFrame(rows); t2.to_csv("table2_wnp.csv",index=False)
    print("\n=== TABLE 1 (WNP, six models) ==="); [print(t1[t1.Radius==R].sort_values("RMSE").to_string(index=False)) for R in ["R34","R50","R64"]]
    print("\n=== TABLE 2 (WNP): R34 bias by strata"+(" (Tao added, flagged)" if len(models_t2)>6 else "")+" ==="); print(t2.to_string(index=False))

if __name__=="__main__": main()
