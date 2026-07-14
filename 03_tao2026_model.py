"""
03_tao2026_model.py
Add the Tao et al. (2026) analytical model (T26) on the WNP sample, driven from
Rmax on the same footing as the other six models by inverting the Chavas & Knaff
(2022) Rmax model to infer R34. Reports the fair (Rmax-driven) skill, a native
(observed-R34) upper bound, and a common-sample R64 comparison.

Requires: pip install tcwindprofile
Inputs  : metrics_by_snapshot.csv, snapshot_catalog.csv (BASIN)
Output  : t26_predictions.csv (+ prints)
"""
import numpy as np, pandas as pd, io, contextlib, warnings
from scipy.optimize import brentq
from tcwindprofile import generate_wind_profile
from tcwindprofile.tc_rmax_estimatefromR34kt import predict_Rmax_from_R34kt
warnings.filterwarnings("ignore")
KT_MS=0.5144444; NM_KM=1.852; MS_KT=1/KT_MS
METRICS="metrics_by_snapshot.csv"; CATALOG="snapshot_catalog.csv"

def _silent(fn,*a,**k):
    with contextlib.redirect_stdout(io.StringIO()): return fn(*a,**k)

def invert_ck22(vmax_ms,rmax_km,lat):
    def h(r34km): return _silent(predict_Rmax_from_R34kt,vmax_ms,r34km,lat)-rmax_km
    try:
        if np.sign(h(8.0))==np.sign(h(900.0)): return np.nan
        return brentq(h,8.0,900.0,xtol=1e-3,maxiter=80)
    except Exception: return np.nan

def cross(rr_km,vv_ms,thr_kt):
    with np.errstate(all="ignore"):
        r_nm=rr_km/NM_KM; v_kt=vv_ms*MS_KT; a=v_kt>=thr_kt
    return float(r_nm[np.where(a)[0][-1]]) if a.any() else np.nan

def t26_from_rmax(vmax_kt,rmax_nm,lat):
    if not(np.isfinite(vmax_kt) and rmax_nm>0 and vmax_kt>=34 and np.isfinite(lat)): return (np.nan,np.nan,np.nan)
    vms=vmax_kt*KT_MS; rkm=rmax_nm*NM_KM; r34=invert_ck22(vms,rkm,lat)
    if not np.isfinite(r34): return (np.nan,np.nan,np.nan)
    try:
        with np.errstate(all="ignore"): rr,vv,R0=_silent(generate_wind_profile,vms,rkm,r34,lat)
    except Exception: return (np.nan,np.nan,np.nan)
    return cross(rr,vv,34),cross(rr,vv,50),cross(rr,vv,64)

def t26_from_obsR34(vmax_kt,rmax_nm,r34_nm,lat):
    if not(np.isfinite(vmax_kt) and rmax_nm>0 and r34_nm>0 and vmax_kt>=34 and np.isfinite(lat)): return (np.nan,np.nan)
    try:
        with np.errstate(all="ignore"): rr,vv,R0=_silent(generate_wind_profile,vmax_kt*KT_MS,rmax_nm*NM_KM,r34_nm*NM_KM,lat)
    except Exception: return (np.nan,np.nan)
    return cross(rr,vv,50),cross(rr,vv,64)

def rmse_bias(pred,obs):
    m=pred.notna()&obs.notna(); e=pred[m]-obs[m]; return int(m.sum()),e.mean(),np.sqrt((e**2).mean())

def main():
    df=pd.read_csv(METRICS)
    cat=pd.read_csv(CATALOG)[["SID","ISO_TIME","BASIN"]]
    df=df.merge(cat,on=["SID","ISO_TIME"],how="left"); df=df[df.BASIN=="WP"].copy()
    res=np.array([t26_from_rmax(v,r,l) for v,r,l in zip(df.VMAX,df.RMAX,df.LAT)])
    for i,R in enumerate(["R34","R50","R64"]):
        df[f"Tao2026_{R}"]=res[:,i]; df[f"Tao2026_{R}_ERR"]=df[f"Tao2026_{R}"]-df[f"OBS_{R}"]
    keep=["idx","SID","ISO_TIME","VMAX","RMAX","LAT","OBS_R34","OBS_R50","OBS_R64",
          "Tao2026_R34","Tao2026_R50","Tao2026_R64","Tao2026_R34_ERR","Tao2026_R50_ERR","Tao2026_R64_ERR"]
    df[keep].to_csv("t26_predictions.csv",index=False)
    print("T26 (Rmax-driven, fair) on WNP:")
    for R in ["R34","R50","R64"]:
        n,b,rm=rmse_bias(df[f"Tao2026_{R}"],df[f"OBS_{R}"])
        note=" [R34 = inferred size input, not independent]" if R=="R34" else ""
        print(f"  {R}: N={n} bias={b:+.1f} RMSE={rm:.1f}{note}")
    nat=np.array([t26_from_obsR34(v,r,o,l) for v,r,o,l in zip(df.VMAX,df.RMAX,df.OBS_R34,df.LAT)])
    nats=pd.DataFrame(nat,columns=["R50","R64"],index=df.index)
    print("native mode (given observed R34), upper bound:")
    for R in ["R50","R64"]:
        n,b,rm=rmse_bias(nats[R],df[f"OBS_{R}"]); print(f"  {R}: N={n} bias={b:+.1f} RMSE={rm:.1f}")
    com=df["Willoughby2006_R64"].notna()&df["Chavas2015_R64"].notna()&df["Tao2026_R64"].notna()&df["OBS_R64"].notna()
    print(f"\ncommon-sample R64 (N={com.sum()}):")
    for lab,col in [("Willoughby2006","Willoughby2006_R64"),("Chavas2015","Chavas2015_R64"),("Tao2026","Tao2026_R64")]:
        e=df.loc[com,col]-df.loc[com,"OBS_R64"]; print(f"  {lab:15s} bias={e.mean():+.1f} RMSE={np.sqrt((e**2).mean()):.1f}")

if __name__=="__main__": main()
