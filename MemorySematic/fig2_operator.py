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

operators = ['Sparse\nAll-to-All', 'Dense\nAll-Reduce', 'Scatter/\nGather', 'Broadcast']
x = np.arange(len(operators))
width = 0.25

# 第一子图数据: 耗时
time_ls = [1.1, 2.8, 1.2, 2.0]
time_urma = [2.2, 1.05, 1.8, 1.1]
time_xmem = [1.0, 1.0, 1.0, 1.0]

# 第二子图数据: 网络利用率
util_ls = [40, 85, 30, 45]
util_urma = [80, 95, 75, 85]
util_xmem = [85, 98, 80, 90]

# 改为 1行2列
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

# (a) Execution Time
ax = axes[0]
ax.bar(x - width, time_ls, width, label='Load/Store', color=color_map['UB Load/Store'], edgecolor='black', linewidth=1.0)
ax.bar(x, time_urma, width, label='URMA', color=color_map['URMA'], edgecolor='black', linewidth=1.0)
ax.bar(x + width, time_xmem, width, label='UB-XMem', color=color_map['UB-XMem (Ours)'], edgecolor='black', linewidth=1.0, hatch='//')

for container in ax.containers:
    ax.bar_label(container, fmt='%.1f', padding=2, fontsize=FONT_TICK-2)

ax.set_ylabel('Norm. Execution Time', fontsize=FONT_AXIS)
ax.set_xticks(x)
ax.set_xticklabels(operators, fontsize=FONT_TICK)
ax.set_title('(a) Execution Time', fontsize=FONT_TITLE, fontweight='bold', loc='left')

# (b) Net Utilization
ax = axes[1]
ax.bar(x - width, util_ls, width, label='Load/Store', color=color_map['UB Load/Store'], edgecolor='black', linewidth=1.0)
ax.bar(x, util_urma, width, label='URMA', color=color_map['URMA'], edgecolor='black', linewidth=1.0)
ax.bar(x + width, util_xmem, width, label='UB-XMem', color=color_map['UB-XMem (Ours)'], edgecolor='black', linewidth=1.0, hatch='//')

for container in ax.containers:
    ax.bar_label(container, fmt='%d', padding=2, fontsize=FONT_TICK-2)

ax.set_ylabel('Network Util. (%)', fontsize=FONT_AXIS)
ax.set_xticks(x)
ax.set_xticklabels(operators, fontsize=FONT_TICK)
ax.set_title('(b) Bandwidth Utilization', fontsize=FONT_TITLE, fontweight='bold', loc='left')

for ax in axes:
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(1.0)

fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center', ncol=3, fontsize=FONT_LEGEND, frameon=False, bbox_to_anchor=(0.5, 1.1))

plt.tight_layout()
out_path = os.path.join('figs', 'fig2_operator.png')
plt.savefig(out_path, dpi=300, bbox_inches='tight')
print(f"Saved {out_path}")
plt.show()