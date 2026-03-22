import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

os.makedirs('figs', exist_ok=True)

FONT_TITLE, FONT_AXIS, FONT_TICK, FONT_LEGEND = 12, 11, 10, 10
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="white")

color_map = {'UB Load/Store': '#517FA4', 'URMA': '#E6B85C', 'UB-XMem (Ours)': '#C00000'}

nodes = [8, 16, 32, 64, 128]
x_labels = [str(n) for n in nodes]

# 第一子图数据
perf_ls = [1.0, 0.85, 0.6, 0.3, 0.15]
perf_urma = [1.0, 0.95, 0.9, 0.82, 0.75]
perf_xmem = [1.0, 0.98, 0.95, 0.90, 0.85]

# 第二子图数据: P99 Tail Latency
tail_ls = [2.0, 5.5, 15.0, 50.0, 150.0]
tail_urma = [10.0, 12.0, 15.0, 20.0, 25.0]
tail_xmem = [2.0, 3.0, 4.0, 5.5, 7.0]

# 改为 1行2列 图
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

# (a) Efficiency
ax = axes[0]
ax.plot(x_labels, perf_ls, marker='o', label='Load/Store', color=color_map['UB Load/Store'])
ax.plot(x_labels, perf_urma, marker='s', label='URMA', color=color_map['URMA'])
ax.plot(x_labels, perf_xmem, marker='*', markersize=10, label='UB-XMem', color=color_map['UB-XMem (Ours)'])
ax.set_xlabel('Number of Nodes', fontsize=FONT_AXIS)
ax.set_ylabel('Norm. Efficiency', fontsize=FONT_AXIS)
ax.set_title('(a) Cluster Scalability', fontsize=FONT_TITLE, fontweight='bold', loc='left')
ax.set_ylim(0, 1.1)

# (b) Tail Latency
ax = axes[1]
ax.plot(x_labels, tail_ls, marker='o', label='Load/Store', color=color_map['UB Load/Store'])
ax.plot(x_labels, tail_urma, marker='s', label='URMA', color=color_map['URMA'])
ax.plot(x_labels, tail_xmem, marker='*', markersize=10, label='UB-XMem', color=color_map['UB-XMem (Ours)'])
ax.set_yscale('log')
ax.set_xlabel('Number of Nodes', fontsize=FONT_AXIS)
ax.set_ylabel('P99 Tail Latency ($\mu$s)', fontsize=FONT_AXIS)
ax.set_title('(b) Tail Latency Validation', fontsize=FONT_TITLE, fontweight='bold', loc='left')

for ax in axes:
    ax.tick_params(labelsize=FONT_TICK)
    ax.grid(axis='both', linestyle='--', alpha=0.5)
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(1.0)

fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center', ncol=3, fontsize=FONT_LEGEND, frameon=False, bbox_to_anchor=(0.5, 1.12))

plt.tight_layout()
out_path = os.path.join('figs', 'fig4_scalability.png')
plt.savefig(out_path, dpi=300, bbox_inches='tight')
print(f"Saved {out_path}")
plt.show()