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
systems = ['Load/Store', 'URMA', 'UB-XMem']
x = np.arange(len(systems))
width = 0.5

prefill_time = [120, 50, 48]
decode_time = [40, 100, 38]
total_time = [160, 150, 86]

# 改为单栏 3行1列 排版
fig, axes = plt.subplots(3, 1, figsize=(5.5, 8.5), sharex=True)
colors = [color_map['UB Load/Store'], color_map['URMA'], color_map['UB-XMem (Ours)']]

# (a) Prefill 阶段
axes[0].bar(x, prefill_time, width, color=colors, edgecolor='black')
axes[0].set_title('(a) Prefill Comm (BW-Intensive)', fontsize=FONT_TITLE, fontweight='bold', loc='left')
axes[0].set_ylabel('Latency (ms)', fontsize=FONT_AXIS)

# (b) Decode 阶段
axes[1].bar(x, decode_time, width, color=colors, edgecolor='black')
axes[1].set_title('(b) Decode Comm (Lat-Sensitive)', fontsize=FONT_TITLE, fontweight='bold', loc='left')
axes[1].set_ylabel('Latency (ms)', fontsize=FONT_AXIS)

# (c) Total 总耗时
axes[2].bar(x, total_time, width, color=colors, edgecolor='black', hatch=['', '', '//'])
axes[2].set_title('(c) Total Inference Comm', fontsize=FONT_TITLE, fontweight='bold', loc='left')
axes[2].set_ylabel('Latency (ms)', fontsize=FONT_AXIS)

for ax in axes:
    ax.set_xticks(x)
    ax.set_xticklabels(systems, fontsize=FONT_TICK)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    for container in ax.containers:
        ax.bar_label(container, fmt='%.0f', padding=2, fontsize=FONT_TICK-1)
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(1.0)

plt.tight_layout()
plt.subplots_adjust(hspace=0.25)
out_path = os.path.join('figs', 'fig3_e2e_llm.png')
plt.savefig(out_path, dpi=300, bbox_inches='tight')
print(f"Saved {out_path}")
plt.show()