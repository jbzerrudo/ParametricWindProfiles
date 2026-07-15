"""Recompute the Rankine row with the SAME grid-based R34 extraction (cap at 500 nm,
1-nm resolution) used by the paper's pipeline, so as-published and re-fit are on
identical footing. Other models already matched the precomputed values exactly."""
import numpy as np, pandas as pd, sys
from scipy.optimize import minimize
sys.path.insert(0,'TCRR20260037_revision_bundle'); import wind_profiles as wp
BUN='TCRR20260037_revision_bundle'; R=np.arange(0,500.001,1.0)
df=pd.read_csv(f'{BUN}/metrics_by_snapshot.csv')
cat=pd.read_csv(f'{BUN}/snapshot_catalog.csv')[['SID','ISO_TIME','BASIN']]
df=df.merge(cat,on=['SID','ISO_TIME'],how='left'); df=df[df.BASIN=='WP'].copy()
df['YEAR']=pd.to_datetime(df.ISO_TIME).dt.year
base=df[(df.VMAX>=34)&(df.RMAX>0)&df.LAT.notna()&df.OBS_R34.notna()].copy()
tr=base[(base.YEAR>=2016)&(base.YEAR<=2020)]; va=base[(base.YEAR>=2021)&(base.YEAR<=2024)]
def rmse(p,o):
    p=np.asarray(p,float);o=np.asarray(o,float);m=np.isfinite(p)&np.isfinite(o)
    return float(np.sqrt(np.mean((p[m]-o[m])**2)))
def last_cross(vkt):
    ge=vkt>=34.0;anyc=ge.any(axis=1);last=ge.shape[1]-1-np.argmax(ge[:,::-1],axis=1)
    return np.where(anyc,R[last],np.nan)
def rankine_grid(V,Rm,inv):           # inv = 1/alpha (scalar or per-snapshot array)
    inv=np.broadcast_to(inv,V.shape); alpha=1.0/inv
    r=R[None,:]; rm=Rm[:,None]
    vkt=np.where(r<=rm, V[:,None]*r/np.maximum(rm,1e-9),
                        V[:,None]*(rm/np.maximum(r,1e-6))**alpha[:,None])
    return last_cross(vkt)
Vt,Rt,Lt,Ot=tr.VMAX.values,tr.RMAX.values,np.abs(tr.LAT.values),tr.OBS_R34.values
Vv,Rv,Lv,Ov=va.VMAX.values,va.RMAX.values,np.abs(va.LAT.values),va.OBS_R34.values
# as-published alpha=0.5 (inv=2) -- should match precomputed 58.5 on val
pub_va=rankine_grid(Vv,Rv,2.0)
print(f"Rankine as-published, grid: val RMSE={rmse(pub_va,Ov):.1f}  (precomputed col={rmse(va['Rankine_R34'].values,Ov):.1f})")
# re-fit inv=c0+c1*V+c2*|lat|, grid-based, minimise train RMSE
def invc(c,V,La): return np.clip(c[0]+c[1]*V+c[2]*La,0.5,6.0)
r=minimize(lambda c:rmse(rankine_grid(Vt,Rt,invc(c,Vt,Lt)),Ot),[2.,0,0],
           method='Nelder-Mead',options={'xatol':1e-4,'fatol':1e-3,'maxiter':8000})
fit_tr=rmse(rankine_grid(Vt,Rt,invc(r.x,Vt,Lt)),Ot)
fit_va=rmse(rankine_grid(Vv,Rv,invc(r.x,Vv,Lv)),Ov)
print(f"Rankine re-fit alpha(V,|lat|), grid: train RMSE={fit_tr:.1f}, HELD-OUT val RMSE={fit_va:.1f}")
print(f"coeffs (c0,c1*V,c2*|lat|)={np.round(r.x,4)}")
print(f"\n=> Rankine row (held-out 2021-2024): as-published {rmse(pub_va,Ov):.1f} -> WNP re-fit {fit_va:.1f} nm")
