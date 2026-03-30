"""
plot_diagnostics.py
Key diagnostic figures for the parametric wind profile comparison paper.

Fig 1: Overall R34/R50/R64 bias boxplots
Fig 2: R34 bias by intensity category 
Fig 3: R34 bias by latitude band
Fig 4: R34 bias by size class
Fig 5: Scatter: predicted vs observed R34 (6 panels)
Fig 6: Example radial profiles for 3 selected storms
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.size'] = 10

res = pd.read_csv('metrics_by_snapshot.csv')
model_names = ['Rankine', 'Holland1980', 'Holland2010', 'Willoughby2006', 'Emanuel2004', 'Chavas2015']
model_labels = ['Rankine', 'Holland\n1980', 'Holland\n2010', 'Willoughby\n2006', 'Emanuel\n2004', 'Chavas\n2015']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

# ═══════════════════════════════════════════════════════════════
# Fig 1: Overall bias boxplots for R34, R50, R64
# ═══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=False)

for ax, rad in zip(axes, ['R34', 'R50', 'R64']):
    data = []
    for name in model_names:
        err_col = f'{name}_{rad}_ERR'
        errs = res[err_col].dropna().values
        data.append(errs)
    
    bp = ax.boxplot(data, labels=model_labels, showfliers=False, patch_artist=True,
                    widths=0.6, medianprops=dict(color='black', linewidth=1.5))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax.axhline(0, color='black', ls='-', lw=0.8)
    ax.set_ylabel('Error (nm)' if ax == axes[0] else '')
    ax.set_title(f'{rad} Wind Radius Error')
    ax.grid(True, axis='y', alpha=0.3)

plt.suptitle('Predicted − Observed Wind Radii (nm), All Snapshots', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig('fig1_overall_boxplots.png', dpi=150, bbox_inches='tight')
print("Saved fig1_overall_boxplots.png")
plt.close()

# ═══════════════════════════════════════════════════════════════
# Fig 2: R34 bias by intensity category (bar chart)
# ═══════════════════════════════════════════════════════════════
cats = ['TS', 'C1-2', 'C3-5']
fig, ax = plt.subplots(figsize=(10, 5))

x = np.arange(len(cats))
width = 0.13
for i, (name, label, color) in enumerate(zip(model_names, model_labels, colors)):
    err_col = f'{name}_R34_ERR'
    biases = []
    for cat in cats:
        sub = res.loc[res['INTENSITY_CAT'] == cat, err_col].dropna()
        biases.append(sub.mean() if len(sub) > 5 else np.nan)
    ax.bar(x + i * width, biases, width, label=name, color=color, alpha=0.8)

ax.axhline(0, color='black', lw=0.8)
ax.set_xticks(x + width * 2.5)
ax.set_xticklabels(cats)
ax.set_xlabel('Intensity Category')
ax.set_ylabel('R34 Bias (nm)')
ax.set_title('R34 Bias by Intensity Category')
ax.legend(fontsize=8, ncol=3)
ax.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('fig2_r34_by_intensity.png', dpi=150, bbox_inches='tight')
print("Saved fig2_r34_by_intensity.png")
plt.close()

# ═══════════════════════════════════════════════════════════════
# Fig 3: R34 bias by latitude band
# ═══════════════════════════════════════════════════════════════
bands = ['00-15N', '15-25N', '25-35N']
fig, ax = plt.subplots(figsize=(10, 5))

x = np.arange(len(bands))
for i, (name, label, color) in enumerate(zip(model_names, model_labels, colors)):
    err_col = f'{name}_R34_ERR'
    biases = []
    for band in bands:
        sub = res.loc[res['LAT_BAND'] == band, err_col].dropna()
        biases.append(sub.mean() if len(sub) > 5 else np.nan)
    ax.bar(x + i * width, biases, width, label=name, color=color, alpha=0.8)

ax.axhline(0, color='black', lw=0.8)
ax.set_xticks(x + width * 2.5)
ax.set_xticklabels(bands)
ax.set_xlabel('Latitude Band')
ax.set_ylabel('R34 Bias (nm)')
ax.set_title('R34 Bias by Latitude Band')
ax.legend(fontsize=8, ncol=3)
ax.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('fig3_r34_by_latitude.png', dpi=150, bbox_inches='tight')
print("Saved fig3_r34_by_latitude.png")
plt.close()

# ═══════════════════════════════════════════════════════════════
# Fig 4: R34 bias by size class
# ═══════════════════════════════════════════════════════════════
sizes = ['compact', 'average', 'large']
fig, ax = plt.subplots(figsize=(10, 5))

x = np.arange(len(sizes))
for i, (name, label, color) in enumerate(zip(model_names, model_labels, colors)):
    err_col = f'{name}_R34_ERR'
    biases = []
    for sc in sizes:
        sub = res.loc[res['SIZE_CLASS'] == sc, err_col].dropna()
        biases.append(sub.mean() if len(sub) > 5 else np.nan)
    ax.bar(x + i * width, biases, width, label=name, color=color, alpha=0.8)

ax.axhline(0, color='black', lw=0.8)
ax.set_xticks(x + width * 2.5)
ax.set_xticklabels(sizes)
ax.set_xlabel('Size Class')
ax.set_ylabel('R34 Bias (nm)')
ax.set_title('R34 Bias by Size Class')
ax.legend(fontsize=8, ncol=3)
ax.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('fig4_r34_by_size.png', dpi=150, bbox_inches='tight')
print("Saved fig4_r34_by_size.png")
plt.close()

# ═══════════════════════════════════════════════════════════════
# Fig 5: Scatter — predicted vs observed R34 (6 panels)
# ═══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(14, 9))
axes = axes.flatten()

for ax, name, color in zip(axes, model_names, colors):
    pred = res[f'{name}_R34'].dropna()
    obs = res.loc[pred.index, 'OBS_R34']
    valid = pred.notna() & obs.notna()
    p, o = pred[valid].values, obs[valid].values
    
    ax.scatter(o, p, s=2, alpha=0.15, color=color, rasterized=True)
    lim = max(o.max(), p.max()) * 1.05
    ax.plot([0, lim], [0, lim], 'k--', lw=0.8)
    ax.set_xlim(0, 400)
    ax.set_ylim(0, 400)
    ax.set_xlabel('Observed R34 (nm)')
    ax.set_ylabel('Predicted R34 (nm)')
    ax.set_title(name)
    ax.set_aspect('equal')
    
    # Stats annotation
    bias = (p - o).mean()
    rmse = np.sqrt(((p - o)**2).mean())
    corr = np.corrcoef(o, p)[0, 1]
    ax.text(0.05, 0.95, f'Bias: {bias:.1f}\nRMSE: {rmse:.1f}\nr: {corr:.2f}',
            transform=ax.transAxes, va='top', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.suptitle('Predicted vs Observed R34 (nm)', fontsize=13)
plt.tight_layout()
plt.savefig('fig5_scatter_r34.png', dpi=150, bbox_inches='tight')
print("Saved fig5_scatter_r34.png")
plt.close()

# ═══════════════════════════════════════════════════════════════
# Fig 6: RMSE summary bar chart (R34, R50, R64 side by side)
# ═══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5))

x = np.arange(len(model_names))
width_bar = 0.25

for j, (rad, hatch) in enumerate(zip(['R34', 'R50', 'R64'], ['', '//', '..'])):
    rmses = []
    for name in model_names:
        err_col = f'{name}_{rad}_ERR'
        errs = res[err_col].dropna()
        rmses.append(np.sqrt((errs**2).mean()) if len(errs) > 0 else np.nan)
    ax.bar(x + j * width_bar, rmses, width_bar, label=rad, 
           color=[colors[i] for i in range(len(model_names))],
           alpha=0.5 + 0.2*j, edgecolor='black', linewidth=0.5)

ax.set_xticks(x + width_bar)
ax.set_xticklabels(model_names, rotation=15)
ax.set_ylabel('RMSE (nm)')
ax.set_title('Wind Radius RMSE by Model and Threshold')
ax.legend(title='Wind Radius')
ax.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('fig6_rmse_summary.png', dpi=150, bbox_inches='tight')
print("Saved fig6_rmse_summary.png")
plt.close()

print("\nAll figures saved.")
