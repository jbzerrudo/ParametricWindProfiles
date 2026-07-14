"""
05_regenerate_figures.py
Regenerate Figures 1-6 on the corrected WNP sample, plus Figure 7 (domain map).
If t26_predictions.csv (from 03_tao2026_model.py) is present, the Tao (2026) model
is added to EVERY figure as a 7th model. Because T26's R34 is an inferred size
input (Chavas & Knaff 2022 Rmax->R34 inversion), not an independent prediction,
its R34 entries are always FLAGGED: hatched bars/boxes in Figs 1-5 and 6, and a
"(R34 inferred)" title in the scatter (Fig 5). This is consistent with Table 1,
where the Tao R34 value is italicised.

Requires: matplotlib; for the map: pip install global-land-mask
Inputs   : metrics_by_snapshot.csv, snapshot_catalog.csv, [t26_predictions.csv]
Outputs  : fig1_overall_boxplots.png ... fig6_rmse_summary.png, fig7_domain_map.png
"""
import os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
matplotlib.rcParams["font.size"]=10
METRICS="metrics_by_snapshot.csv"; CATALOG="snapshot_catalog.csv"; T26="t26_predictions.csv"; OUT="."
MODELS=["Rankine","Holland1980","Holland2010","Willoughby2006","Emanuel2004","Chavas2015"]
MLAB=["Rankine","Holland\n1980","Holland\n2010","Willoughby\n2006","E04\n(hyp.)","Chavas\n2015"]
COLORS=["#2E5C8A","#D97A2B","#4A7C3A","#B23A2C","#E0B020","#6B4423"]
TAOCOL="#7B3FA0"
LEG={"Rankine":"Rankine","Holland1980":"Holland 1980","Holland2010":"Holland 2010",
     "Willoughby2006":"Willoughby 2006","Emanuel2004":"Emanuel 2004 (hyp.)","Chavas2015":"Chavas 2015","Tao2026":"Tao 2026"}
W=0.12

def load_wnp():
    m=pd.read_csv(METRICS); c=pd.read_csv(CATALOG)
    res=m.merge(c[["SID","ISO_TIME","BASIN"]],on=["SID","ISO_TIME"],how="left")
    res=res[res.BASIN=="WP"].copy()
    has_t26=os.path.exists(T26)
    if has_t26:
        t=pd.read_csv(T26)
        res=res.merge(t[["idx","Tao2026_R34","Tao2026_R50","Tao2026_R64",
                          "Tao2026_R34_ERR","Tao2026_R50_ERR","Tao2026_R64_ERR"]],on="idx",how="left")
    res["INTENSITY_CAT"]=res.VMAX.apply(lambda v:"TD" if v<34 else "TS" if v<64 else "C1-2" if v<96 else "C3-5")
    res["LAT_BAND"]=res.LAT.apply(lambda l:"00-15N" if l<15 else "15-25N" if l<25 else "25-35N")
    q33,q66=res.OBS_R34.quantile([0.33,0.66])
    res["SIZE_CLASS"]=res.OBS_R34.apply(lambda r:"unknown" if pd.isna(r) else "compact" if r<=q33 else "average" if r<=q66 else "large")
    return res,c,has_t26

def figures(res,has_t26):
    tao = has_t26 and "Tao2026_R34_ERR" in res.columns
    m7=MODELS+(["Tao2026"] if tao else []); lab7=MLAB+(["Tao\n2026"] if tao else []); col7=COLORS+([TAOCOL] if tao else [])
    # Fig 1: boxplots (7 models when T26 available; Tao R34 box hatched)
    fig,ax=plt.subplots(1,3,figsize=(16,5))
    for a,rad in zip(ax,["R34","R50","R64"]):
        data=[res[f"{n}_{rad}_ERR"].dropna().values for n in m7]
        bp=a.boxplot(data,tick_labels=lab7,showfliers=False,patch_artist=True,widths=0.6,medianprops=dict(color="black",linewidth=1.5))
        for p,c in zip(bp["boxes"],col7): p.set_facecolor(c); p.set_alpha(0.6)
        if tao and rad=="R34": bp["boxes"][-1].set_hatch("///")  # Tao R34 = inferred size input
        a.axhline(0,color="black",lw=0.8); a.set_ylabel("Error (nm)" if a is ax[0] else ""); a.set_title(f"{rad} Wind Radius Error"); a.grid(True,axis="y",alpha=0.3)
    plt.suptitle("Predicted - Observed Wind Radii (nm), All Snapshots",fontsize=13,y=1.02); plt.tight_layout()
    plt.savefig(f"{OUT}/fig1_overall_boxplots.png",dpi=150,bbox_inches="tight"); plt.close()
    # Fig 2-4: R34 bias bars (Tao added, hatched + "(R34 inf.)" label, when available)
    def barfig(col,cats,xlabel,title,fname):
        mods=MODELS+(["Tao2026"] if tao else []); cols=COLORS+([TAOCOL] if tao else [])
        fig,a=plt.subplots(figsize=(12,5)); x=np.arange(len(cats))
        for i,(n,c) in enumerate(zip(mods,cols)):
            b=[res.loc[res[col]==cc,f"{n}_R34_ERR"].dropna().mean() if (res[col]==cc).sum()>5 else np.nan for cc in cats]
            bars=a.bar(x+i*W,b,W,label=LEG[n]+(" (R34 inf.)" if n=="Tao2026" else ""),color=c,alpha=0.85)
            if n=="Tao2026":
                for bb in bars: bb.set_hatch("///")
        a.axhline(0,color="black",lw=0.8); a.set_xticks(x+W*(len(mods)-1)/2); a.set_xticklabels(cats)
        a.set_xlabel(xlabel); a.set_ylabel("R34 Bias (nm)"); a.set_title(title); a.legend(fontsize=9,ncol=4); a.grid(True,axis="y",alpha=0.3)
        plt.tight_layout(); plt.savefig(f"{OUT}/{fname}",dpi=150,bbox_inches="tight"); plt.close()
    barfig("INTENSITY_CAT",["TS","C1-2","C3-5"],"Intensity Category","R34 Bias by Intensity Category","fig2_r34_by_intensity.png")
    barfig("LAT_BAND",["00-15N","15-25N","25-35N"],"Latitude Band","R34 Bias by Latitude Band","fig3_r34_by_latitude.png")
    barfig("SIZE_CLASS",["compact","average","large"],"Size Class","R34 Bias by Size Class","fig4_r34_by_size.png")
    # Fig 5: scatter (7 panels when T26; Tao panel title-flagged "(R34 inferred)")
    sm=MODELS+(["Tao2026"] if tao else []); scol=COLORS+([TAOCOL] if tao else [])
    ncol=4 if len(sm)>6 else 3; nrow=2
    fig,ax=plt.subplots(nrow,ncol,figsize=(4.7*ncol,9)); ax=ax.flatten()
    for idx,(n,c) in enumerate(zip(sm,scol)):
        a=ax[idx]; pred=res[f"{n}_R34"]; obs=res["OBS_R34"]; v=pred.notna()&obs.notna(); p,o=pred[v].values,obs[v].values
        a.scatter(o,p,s=2,alpha=0.15,color=c,rasterized=True); a.plot([0,500],[0,500],"k--",lw=0.8)
        a.set_xlim(0,500); a.set_ylim(0,500); a.set_xlabel("Observed R34 (nm)"); a.set_ylabel("Predicted R34 (nm)")
        a.set_title(LEG[n]+(" (R34 inferred)" if n=="Tao2026" else "")); a.set_aspect("equal")
        a.text(0.05,0.95,f"Bias: {(p-o).mean():+.1f}\nRMSE: {np.sqrt(((p-o)**2).mean()):.1f}\nr: {np.corrcoef(o,p)[0,1]:.2f}",transform=a.transAxes,va="top",fontsize=8,bbox=dict(boxstyle="round",facecolor="white",alpha=0.8))
    for k in range(len(sm),len(ax)): ax[k].axis("off")
    plt.suptitle("Predicted vs Observed R34 (nm)",fontsize=13,y=1.00); plt.tight_layout()
    plt.savefig(f"{OUT}/fig5_scatter_r34.png",dpi=150,bbox_inches="tight"); plt.close()
    # Fig 6: RMSE summary (7 models when T26 available; Tao R34 bar hatched)
    fig,a=plt.subplots(figsize=(13,5)); x=np.arange(len(m7)); wb=0.25
    for j,(rad,rc) in enumerate(zip(["R34","R50","R64"],["#FF0000","#00FF00","#0000FF"])):
        rm=[np.sqrt((res[f"{n}_{rad}_ERR"].dropna()**2).mean()) for n in m7]
        bars=a.bar(x+j*wb,rm,wb,label=rad,color=rc,alpha=0.85,edgecolor="black",linewidth=0.5)
        if tao and rad=="R34": bars[-1].set_hatch("///")  # Tao R34 = inferred size input
    a.set_xticks(x+wb); a.set_xticklabels([m.replace(chr(10)," ") for m in lab7],rotation=15); a.set_ylabel("RMSE (nm)")
    a.set_title("Wind Radius RMSE by Model and Threshold"); a.legend(title="Wind Radius"); a.grid(True,axis="y",alpha=0.3)
    if tao: a.text(0.99,0.97,"Hatched: Tao 2026 R34 (inferred size input)",transform=a.transAxes,ha="right",va="top",fontsize=8,style="italic")
    plt.tight_layout(); plt.savefig(f"{OUT}/fig6_rmse_summary.png",dpi=150,bbox_inches="tight"); plt.close()

def domain_map(cat):
    from global_land_mask import globe
    c2=cat[cat.BASIN=="WP"].sort_values(["SID","ISO_TIME"])
    lons=np.arange(100,180.01,0.1); lats=np.arange(0,45.01,0.1); LON,LAT=np.meshgrid(lons,lats)
    land=globe.is_land(LAT,np.where(LON>180,LON-360,LON)).astype(float)
    fig,ax=plt.subplots(figsize=(9,6.2))
    ax.contourf(LON,LAT,land,levels=[0.5,1.5],colors=["#ece6da"],zorder=0)
    ax.contour(LON,LAT,land,levels=[0.5],colors="#555",linewidths=0.4,zorder=1)
    for sid,g in c2.groupby("SID"):
        ax.plot(g.LON,g.LAT,color=plt.cm.viridis(min(g.USA_WIND.max(),160)/160.),lw=0.5,alpha=0.5,zorder=2)
    sm=plt.cm.ScalarMappable(cmap="viridis",norm=plt.Normalize(0,160)); sm.set_array([])
    cb=plt.colorbar(sm,ax=ax,shrink=0.85,pad=0.02); cb.set_label("Lifetime max $V_{max}$ (kt)")
    ax.axhline(35,color="crimson",ls="--",lw=0.9,alpha=0.85); ax.text(101,35.5,"35 N study limit",color="crimson",fontsize=8)
    ax.set_xlim(100,180); ax.set_ylim(0,45); ax.set_xlabel("Longitude (degE)"); ax.set_ylabel("Latitude (degN)")
    ax.set_title(f"Study domain: {c2.SID.nunique()} WNP TCs, {len(c2):,} 6-hourly fixes (2001-2024)")
    ax.set_aspect("equal",adjustable="box"); plt.tight_layout()
    plt.savefig(f"{OUT}/fig7_domain_map.png",dpi=200,bbox_inches="tight"); plt.close()

if __name__=="__main__":
    res,cat,has_t26=load_wnp(); figures(res,has_t26)
    print("figures 1-6 written; Tao 2026 (flagged) in all figures:", has_t26 and "Tao2026_R34_ERR" in res.columns)
    try: domain_map(cat); print("fig7 domain map written")
    except Exception as e: print("domain map skipped (need global-land-mask):",e)
