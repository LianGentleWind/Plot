import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

os.makedirs('figs', exist_ok=True)

# 字体设定
FONT_TITLE, FONT_AXIS, FONT_TICK, FONT_LEGEND = 12, 11, 10, 10
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="white")

color_map = {'UB Load/Store': '#517FA4', 'URMA': '#E6B85C', 'UB-XMem (Ours)': '#C00000'}

sizes = ['64B', '256B', '1KB', '4KB', '64KB', '1MB']
lat_ls = [0.1, 0.15, 0.3, 1.2, 8.0, 100.0]
lat_urma = [0.8, 0.85, 0.9, 1.1, 3.0, 15.0]
lat_xmem = [0.1, 0.15, 0.3, 1.1, 3.0, 15.0]

bw_ls = [1.5, 6.0, 20.0, 60.0, 100.0, 120.0]
bw_urma = [0.2, 1.0, 10.0, 80.0, 350.0, 390.0]
bw_xmem = [1.5, 6.0, 20.0, 80.0, 350.0, 390.0]

cpu_ls = [10, 15, 30, 80, 95, 99]
cpu_urma = [50, 50, 50, 50, 40, 20]
cpu_xmem = [10, 15, 30, 45, 35, 15]

# 改为单栏 3行1列，共享 X 轴
fig, axes = plt.subplots(3, 1, figsize=(5.5, 8.5), sharex=True)

# (a) Latency
ax = axes[0]
ax.plot(sizes, lat_ls, marker='o', label='Load/Store', color=color_map['UB Load/Store'])
ax.plot(sizes, lat_urma, marker='s', label='URMA', color=color_map['URMA'])
ax.plot(sizes, lat_xmem, marker='*', markersize=10, label='UB-XMem', color=color_map['UB-XMem (Ours)'])
ax.set_yscale('log')
ax.set_ylabel('Latency ($\mu$s)', fontsize=FONT_AXIS)
ax.set_title('(a) Latency vs. Msg Size', fontsize=FONT_TITLE, fontweight='bold', loc='left')

# (b) Throughput
ax = axes[1]
ax.plot(sizes, bw_ls, marker='o', color=color_map['UB Load/Store'])
ax.plot(sizes, bw_urma, marker='s', color=color_map['URMA'])
ax.plot(sizes, bw_xmem, marker='*', markersize=10, color=color_map['UB-XMem (Ours)'])
ax.set_ylabel('Throughput (GB/s)', fontsize=FONT_AXIS)
ax.set_title('(b) Throughput vs. Msg Size', fontsize=FONT_TITLE, fontweight='bold', loc='left')

# (c) CPU Overhead
ax = axes[2]
ax.plot(sizes, cpu_ls, marker='o', color=color_map['UB Load/Store'])
ax.plot(sizes, cpu_urma, marker='s', color=color_map['URMA'])
ax.plot(sizes, cpu_xmem, marker='*', markersize=10, color=color_map['UB-XMem (Ours)'])
ax.set_xlabel('Message Size', fontsize=FONT_AXIS)
ax.set_ylabel('CPU Util. (%)', fontsize=FONT_AXIS)
ax.set_title('(c) CPU Cost vs. Msg Size', fontsize=FONT_TITLE, fontweight='bold', loc='left')

for ax in axes:
    ax.tick_params(labelsize=FONT_TICK)
    ax.grid(axis='both', linestyle='--', alpha=0.5)
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(1.0)

# Legend 在顶部
fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center', ncol=3, 
           fontsize=FONT_LEGEND, frameon=False, bbox_to_anchor=(0.5, 0.98))

plt.tight_layout()
plt.subplots_adjust(top=0.9, hspace=0.2)
out_path = os.path.join('figs', 'fig1_microbenchmark.png')
plt.savefig(out_path, dpi=300, bbox_inches='tight')
print(f"Saved {out_path}")
plt.show()
