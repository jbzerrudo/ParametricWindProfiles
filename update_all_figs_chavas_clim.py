"""
update_all_figs_chavas_clim.py
Regenerate ALL figures (1-6) with 7 models including Chavas-clim.

Run this AFTER compare_profiles.py has produced metrics_by_snapshot.csv
(which now includes Chavas_clim columns).
"""

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
matplotlib.rcParams['font.size'] = 10

# ── CONFIG: adjust these paths ──
METRICS_CSV = r'D:\2026\ParametricWindModel\NEWOUTS\Models\metrics_by_snapshot.csv'
OUTPUT_DIR  = r'D:\2026\ParametricWindModel\NEWOUTS\UPDATEDFIGS'
# ─────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load data ──
print("Loading metrics...")
res = pd.read_csv(METRICS_CSV)

# ── Stratification columns ──
def intensity_cat(v):
    if v < 34:   return 'TD'
    elif v < 64: return 'TS'
    elif v < 96: return 'C1-2'
    else:        return 'C3-5'

def lat_band(lat):
    if lat < 15:   return '00-15N'
    elif lat < 25: return '15-25N'
    else:          return '25-35N'

res['INTENSITY_CAT'] = res['VMAX'].apply(intensity_cat)
res['LAT_BAND'] = res['LAT'].apply(lat_band)

q33, q66 = res['OBS_R34'].quantile([0.33, 0.66])
def size_class(r34):
    if pd.isna(r34): return 'unknown'
    if r34 <= q33:   return 'compact'
    elif r34 <= q66: return 'average'
    else:            return 'large'
res['SIZE_CLASS'] = res['OBS_R34'].apply(size_class)

print(f"Loaded {len(res)} snapshots.")

# ── Model lists ──
model_names  = ['Rankine', 'Holland1980', 'Holland2010', 'Willoughby2006',
                'Emanuel2004', 'Chavas2015', 'Chavas_clim']
model_labels = ['Rankine', 'Holland\n1980', 'Holland\n2010', 'Willoughby\n2006',
                'Emanuel\n2004', 'Chavas\n(obs)', 'Chavas\n(clim)']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']

# ═══════════════════════════════════════════════════════════════
# Fig 1: Overall bias boxplots (7 models)
# ═══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)

for ax, rad in zip(axes, ['R34', 'R50', 'R64']):
    data = []
    for name in model_names:
        err_col = f'{name}_{rad}_ERR'
        errs = res[err_col].dropna().values
        data.append(errs)

    bp = ax.boxplot(data, tick_labels=model_labels, showfliers=False, patch_artist=True,
                    widths=0.6, medianprops=dict(color='black', linewidth=1.5))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.axhline(0, color='black', ls='-', lw=0.8)
    ax.set_ylabel('Error (nm)' if ax == axes[0] else '')
    ax.set_title(f'{rad} Wind Radius Error')
    ax.grid(True, axis='y', alpha=0.3)

plt.suptitle('Predicted \u2212 Observed Wind Radii (nm), All Snapshots', fontsize=13, y=1.02)
plt.tight_layout()
outpath = os.path.join(OUTPUT_DIR, 'fig1_overall_boxplots.png')
plt.savefig(outpath, dpi=150, bbox_inches='tight')
print(f"Saved {outpath}")
plt.close()

# ═══════════════════════════════════════════════════════════════
# Fig 2: R34 bias by intensity category (7 models)
# ═══════════════════════════════════════════════════════════════
cats = ['TS', 'C1-2', 'C3-5']
fig, ax = plt.subplots(figsize=(12, 5))

x = np.arange(len(cats))
width = 0.11
for i, (name, color) in enumerate(zip(model_names, colors)):
    err_col = f'{name}_R34_ERR'
    biases = []
    for cat in cats:
        sub = res.loc[res['INTENSITY_CAT'] == cat, err_col].dropna()
        biases.append(sub.mean() if len(sub) > 5 else np.nan)
    ax.bar(x + i * width, biases, width, label=name.replace('_', ' '), color=color, alpha=0.8)

ax.axhline(0, color='black', lw=0.8)
ax.set_xticks(x + width * 3)
ax.set_xticklabels(cats)
ax.set_xlabel('Intensity Category')
ax.set_ylabel('R34 Bias (nm)')
ax.set_title('R34 Bias by Intensity Category')
ax.legend(fontsize=8, ncol=4)
ax.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
outpath = os.path.join(OUTPUT_DIR, 'fig2_r34_by_intensity.png')
plt.savefig(outpath, dpi=150, bbox_inches='tight')
print(f"Saved {outpath}")
plt.close()

# ═══════════════════════════════════════════════════════════════
# Fig 3: R34 bias by latitude band (7 models)
# ═══════════════════════════════════════════════════════════════
bands = ['00-15N', '15-25N', '25-35N']
fig, ax = plt.subplots(figsize=(12, 5))

x = np.arange(len(bands))
for i, (name, color) in enumerate(zip(model_names, colors)):
    err_col = f'{name}_R34_ERR'
    biases = []
    for band in bands:
        sub = res.loc[res['LAT_BAND'] == band, err_col].dropna()
        biases.append(sub.mean() if len(sub) > 5 else np.nan)
    ax.bar(x + i * width, biases, width, label=name.replace('_', ' '), color=color, alpha=0.8)

ax.axhline(0, color='black', lw=0.8)
ax.set_xticks(x + width * 3)
ax.set_xticklabels(bands)
ax.set_xlabel('Latitude Band')
ax.set_ylabel('R34 Bias (nm)')
ax.set_title('R34 Bias by Latitude Band')
ax.legend(fontsize=8, ncol=4)
ax.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
outpath = os.path.join(OUTPUT_DIR, 'fig3_r34_by_latitude.png')
plt.savefig(outpath, dpi=150, bbox_inches='tight')
print(f"Saved {outpath}")
plt.close()

# ═══════════════════════════════════════════════════════════════
# Fig 4: R34 bias by size class (7 models)
# ═══════════════════════════════════════════════════════════════
sizes = ['compact', 'average', 'large']
fig, ax = plt.subplots(figsize=(12, 5))

x = np.arange(len(sizes))
for i, (name, color) in enumerate(zip(model_names, colors)):
    err_col = f'{name}_R34_ERR'
    biases = []
    for sc in sizes:
        sub = res.loc[res['SIZE_CLASS'] == sc, err_col].dropna()
        biases.append(sub.mean() if len(sub) > 5 else np.nan)
    ax.bar(x + i * width, biases, width, label=name.replace('_', ' '), color=color, alpha=0.8)

ax.axhline(0, color='black', lw=0.8)
ax.set_xticks(x + width * 3)
ax.set_xticklabels(sizes)
ax.set_xlabel('Size Class')
ax.set_ylabel('R34 Bias (nm)')
ax.set_title('R34 Bias by Size Class')
ax.legend(fontsize=8, ncol=4)
ax.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
outpath = os.path.join(OUTPUT_DIR, 'fig4_r34_by_size.png')
plt.savefig(outpath, dpi=150, bbox_inches='tight')
print(f"Saved {outpath}")
plt.close()

# ═══════════════════════════════════════════════════════════════
# Fig 5: Scatter — predicted vs observed R34 (7 panels + summary)
# ═══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 4, figsize=(18, 9))
axes = axes.flatten()

for idx, (name, color) in enumerate(zip(model_names, colors)):
    ax = axes[idx]
    pred_col = f'{name}_R34'
    pred = res[pred_col].copy()
    obs = res['OBS_R34'].copy()
    valid = pred.notna() & obs.notna()
    p, o = pred[valid].values, obs[valid].values

    ax.scatter(o, p, s=2, alpha=0.15, color=color, rasterized=True)
    ax.plot([0, 400], [0, 400], 'k--', lw=0.8)
    ax.set_xlim(0, 400)
    ax.set_ylim(0, 400)
    ax.set_xlabel('Observed R34 (nm)')
    ax.set_ylabel('Predicted R34 (nm)')

    display_name = (name.replace('_', ' ')
                        .replace('Chavas2015', 'Chavas (obs)')
                        .replace('Chavas clim', 'Chavas (clim)'))
    ax.set_title(display_name)
    ax.set_aspect('equal')

    bias = (p - o).mean()
    rmse = np.sqrt(((p - o)**2).mean())
    corr = np.corrcoef(o, p)[0, 1]
    ax.text(0.05, 0.95, f'Bias: {bias:+.1f}\nRMSE: {rmse:.1f}\nr: {corr:.2f}',
            transform=ax.transAxes, va='top', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Panel 8: summary table
ax = axes[7]
ax.axis('off')
ax.set_title('R34 Summary', fontsize=10)
header = ['Model', 'Bias', 'RMSE', 'r']
rows = []
for name in model_names:
    err_col = f'{name}_R34_ERR'
    pred_col = f'{name}_R34'
    errs = res[err_col].dropna()
    valid = res[pred_col].notna() & res['OBS_R34'].notna()
    corr = np.corrcoef(res.loc[valid, pred_col], res.loc[valid, 'OBS_R34'])[0, 1] if valid.sum() > 10 else np.nan
    short = (name.replace('Holland1980', 'H80').replace('Holland2010', 'H10')
                 .replace('Willoughby2006', 'W06').replace('Emanuel2004', 'E04')
                 .replace('Chavas2015', 'Ch-obs').replace('Chavas_clim', 'Ch-clim'))
    rows.append([short, f'{errs.mean():+.1f}', f'{np.sqrt((errs**2).mean()):.1f}', f'{corr:.2f}'])
table = ax.table(cellText=rows, colLabels=header, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1, 1.3)

plt.suptitle('Predicted vs Observed R34 (nm)', fontsize=13)
plt.tight_layout()
outpath = os.path.join(OUTPUT_DIR, 'fig5_scatter_r34.png')
plt.savefig(outpath, dpi=150, bbox_inches='tight')
print(f"Saved {outpath}")
plt.close()

# ═══════════════════════════════════════════════════════════════
# Fig 6: RMSE summary bar chart (7 models, R34/R50/R64)
# ═══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 5))

x = np.arange(len(model_names))
width_bar = 0.25
rad_colors = ['#4477AA', '#EE6677', '#228833']  # distinct per threshold

for j, (rad, rc) in enumerate(zip(['R34', 'R50', 'R64'], rad_colors)):
    rmses = []
    for name in model_names:
        err_col = f'{name}_{rad}_ERR'
        errs = res[err_col].dropna()
        rmses.append(np.sqrt((errs**2).mean()) if len(errs) > 0 else np.nan)

    ax.bar(x + j * width_bar, rmses, width_bar, label=rad,
           color=rc, alpha=0.8, edgecolor='black', linewidth=0.5)

ax.set_xticks(x + width_bar)
ax.set_xticklabels([m.replace('\n', ' ') for m in model_labels], rotation=15)
ax.set_ylabel('RMSE (nm)')
ax.set_title('Wind Radius RMSE by Model and Threshold')
ax.legend(title='Wind Radius')
ax.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
outpath = os.path.join(OUTPUT_DIR, 'fig6_rmse_summary.png')
plt.savefig(outpath, dpi=150, bbox_inches='tight')
print(f"Saved {outpath}")
plt.close()

print("\nAll 6 figures regenerated. Replace fig*.png files in your LaTeX folder.")
