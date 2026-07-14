"""
04_recent_era_and_refit.py
(1) Recent-era RMSE by period (2001-2024, 2016-2024, 2020-2024). If t26_predictions.csv
    is present, Tao (2026) is included (its R34 is the inferred size input -- flagged).
(2) Basin-specific re-fit demo on 2016-2024 R34 (Rankine alpha; Willoughby X1).
Inputs: metrics_by_snapshot.csv, snapshot_catalog.csv, wind_profiles.py, [t26_predictions.csv]
"""
import os, numpy as np, pandas as pd, sys
from scipy.optimize import minimize_scalar, minimize
sys.path.insert(0,"."); import wind_profiles as wp
METRICS="metrics_by_snapshot.csv"; CATALOG="snapshot_catalog.csv"
MODELS=["Rankine","Holland1980","Holland2010","Willoughby2006","Emanuel2004","Chavas2015"]

def load_wnp():
    df=pd.read_csv(METRICS); cat=pd.read_csv(CATALOG)[["SID","ISO_TIME","BASIN"]]
    df=df.merge(cat,on=["SID","ISO_TIME"],how="left"); df=df[df.BASIN=="WP"].copy()
    df["YEAR"]=pd.to_datetime(df["ISO_TIME"]).dt.year
    if os.path.exists("t26_predictions.csv"):
        _t=pd.read_csv("t26_predictions.csv")
        df=df.merge(_t[["idx","Tao2026_R34","Tao2026_R50","Tao2026_R64","Tao2026_R34_ERR","Tao2026_R50_ERR","Tao2026_R64_ERR"]],on="idx",how="left")
    return df

def recent_era(df):
    mods=MODELS+(["Tao2026"] if "Tao2026_R34_ERR" in df.columns else [])
    flag=" (Tao R34 = inferred size input, flagged)" if "Tao2026_R34_ERR" in df.columns else ""
    print("=== Recent-era RMSE by period (WNP)"+flag+" ===")
    for R in ["R34","R50","R64"]:
        print(f"-- {R} --")
        for y0,lab in [(2001,"2001-2024"),(2016,"2016-2024"),(2020,"2020-2024")]:
            s=df[df.YEAR>=y0]; rows=[]
            for m in mods:
                e=s[f"{m}_{R}_ERR"]; o=s[f"OBS_{R}"]; k=e.notna()&o.notna(); ee=e[k]
                if len(ee)>=5: rows.append((m,np.sqrt((ee**2).mean())))
            rows.sort(key=lambda x:x[1])
            print(f"  {lab}: best={rows[0][0]} ({rows[0][1]:.1f}) | "+" ".join(f"{m[:8]}={v:.1f}" for m,v in rows))

def refit(df):
    print("\n=== Basin-specific re-fit demo (WNP, 2016-2024, R34) ===")
    d=df[(df.YEAR>=2016)&(df.VMAX>=34)&(df.RMAX>0)&df.LAT.notna()&df.OBS_R34.notna()]
    V=d.VMAX.values; Rm=d.RMAX.values; La=np.abs(d.LAT.values); O=d.OBS_R34.values
    rmse=lambda p: np.sqrt(np.nanmean((p-O)**2)); print(f"N={len(d)}")
    rank=lambda a: Rm*(V/34.0)**(1.0/a)
    print(f"  Rankine alpha=0.50            : RMSE={rmse(rank(0.5)):.1f}")
    r=minimize_scalar(lambda a:rmse(rank(a)),bounds=(0.2,1.2),method="bounded"); print(f"  Rankine best single alpha={r.x:.3f}: RMSE={r.fun:.1f}")
    def rankp(c):
        inv=np.clip(c[0]+c[1]*V+c[2]*La,0.5,6.0); return Rm*(V/34.0)**inv
    rr=minimize(lambda c:rmse(rankp(c)),[2.0,0,0],method="Nelder-Mead",options={"xatol":1e-4,"fatol":1e-3,"maxiter":4000})
    print(f"  Rankine alpha(Vmax,|lat|)     : RMSE={rr.fun:.1f}")
    def willR(v,rm,la,scale,thr=34):
        r=np.arange(0,500.001,1.0); vms=v*wp.KT_TO_MS; laa=abs(la)
        n=np.clip(0.4067+0.0144*vms-0.0038*laa,0.2,2.4); X1=max(317.1-2.026*vms+1.915*laa,50.0)*scale; X2=25.0
        A=np.clip(0.0696+0.0049*vms-0.0064*laa,0,1); rmk=rm*wp.NM_TO_M/1000; rk=r*wp.NM_TO_M/1000
        Xeff=(1-A)*X1+A*X2; R1=wp._find_R1(rmk,Xeff,n,25.0); R2=R1+25
        vin=v*(np.maximum(rk,1e-6)/rmk)**n; dr=rk-rmk; vout=v*((1-A)*np.exp(-dr/X1)+A*np.exp(-dr/X2))
        xi=np.clip((rk-R1)/25.,0,1); w=wp._bellramp(xi); vt=vin*(1-w)+vout*w
        vv=np.where(rk<=R1,vin,np.where(rk<=R2,vt,vout)); vv[r==0]=0; a=vv>=thr
        return float(r[np.where(a)[0][-1]]) if a.any() else np.nan
    base=np.array([willR(v,rm,la,1.0) for v,rm,la in zip(V,Rm,d.LAT.values)])
    print(f"  Willoughby X1 scale=1.00      : RMSE={rmse(base):.1f}")
    best=min(np.round(np.arange(0.7,1.51,0.05),2),key=lambda s:rmse(np.array([willR(v,rm,la,s) for v,rm,la in zip(V,Rm,d.LAT.values)])))
    bb=np.array([willR(v,rm,la,best) for v,rm,la in zip(V,Rm,d.LAT.values)])
    print(f"  Willoughby best X1 scale={best:.2f}  : RMSE={rmse(bb):.1f}")

if __name__=="__main__":
    df=load_wnp(); recent_era(df); refit(df)
