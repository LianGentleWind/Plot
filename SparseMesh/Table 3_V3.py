import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

# 1. 数据准备 (仅保留 128P)
data_128p = [
    ["SMesh-N16\n(SP)",      22.19, 55.05, 2_775_040, 7.75],
    ["SMesh-N16\n(opt)",     22.17, 58.52, 2_775_040, 7.75],
    ["FullMesh-N16",              0, 0, 0, 0],
    ["CLOS-N16",                  23.14, 80.0,  2_941_840, 12.98],
    ["Torus-N16",              22.35, 50.0,  2_790_400, 8.26],
]

cols = ['Topology', 'Latency', 'BW', 'SysCost', 'NetCostPct']

def build_df(data, npu_count):
    df = pd.DataFrame(data, columns=cols)
    npu_cost = npu_count * 20000
    df['NetCost'] = df['SysCost'] - npu_cost
    df['NetCost'] = df['NetCost'].replace(0, np.nan)
    df['Latency'] = df['Latency'].replace(0, np.nan)
    
    df['BWCostRaw'] = df['BW'] / df['NetCost']
    max_bw_cost = df['BWCostRaw'].max()
    df['BWCostNorm'] = df['BWCostRaw'] / max_bw_cost if max_bw_cost != 0 else 0
    
    df['LatCostRaw'] = (1.0 / df['Latency']) / df['NetCost']
    max_lat_cost = df['LatCostRaw'].max()
    df['LatCostNorm'] = df['LatCostRaw'] / max_lat_cost if max_lat_cost != 0 else 0
    
    return df.fillna(0)

df_128 = build_df(data_128p, 128)

# 2. 绘图配置
FONT_TITLE, FONT_AXIS, FONT_TICK = 15, 12, 12
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="white")

color_map = {
    'CLOS':       '#E6B85C',
    'FullMesh':   '#6AB187',
    'SMesh':      '#517FA4',
    'Torus':      '#C07CAD',
}

def get_color(name):
    for k, v in color_map.items():
        if k in name: return v
    return '#B0BEC5'

# 3. 绘图: 1行3列，增加 sharey=True 确保坐标对齐
fig, axes = plt.subplots(1, 3, figsize=(8, 3.5), sharey=True,
                         gridspec_kw={'wspace': 0.05, 'width_ratios': [1.6, 0.8, 0.8]})

ax_bw  = axes[0]
ax_mid = axes[1]
ax_lat = axes[2]

df = df_128.copy()
df['Color'] = df['Topology'].apply(get_color)
y_pos = np.arange(len(df))

# --- 左列: BW Cost Efficiency ---
bars_bw = ax_bw.barh(y_pos, df['BWCostNorm'],
                     color=df['Color'], edgecolor='#000',
                     linewidth=0.5, height=0.6, zorder=3)

ax_bw.set_xlim(0, 1.35)
ax_bw.set_xticks([0, 0.5, 1.0])
ax_bw.invert_xaxis()
# 注意：因为 sharey=True 且子图是水平排列，一次 invert_yaxis 会应用到所有共享轴
ax_bw.invert_yaxis() 

for bar in bars_bw:
    w = bar.get_width()
    if w > 0:
        ax_bw.text(w + 0.04, bar.get_y() + bar.get_height() / 2,
                   f'{w:.2f}', ha='right', va='center', fontsize=FONT_TICK, color='#000')

ax_bw.tick_params(axis='y', which='both', left=False, labelleft=False)
ax_bw.tick_params(axis='x', labelsize=FONT_TICK, colors='#000')

# --- 中间列: Topology 文本 ---
ax_mid.axis('off') # 隐藏中间列的轴线和刻度
ax_mid.set_xlim(0, 1)

for y, topo in zip(y_pos, df['Topology']):
    display_topo = topo.replace('\n', ' ')
    ax_mid.text(0.5, y, display_topo, ha='center', va='center', fontsize=FONT_AXIS, color='#000')

# Scale 标签调整位置（使用 transform 确保不受坐标轴范围影响）
ax_mid.text(0.5, 1.08, 'Scale: 128P', transform=ax_mid.transAxes,
            ha='center', va='center', fontsize=FONT_AXIS, fontweight='bold',
            color='#000', bbox=dict(facecolor='#FFE0E0', edgecolor='none', pad=4.0, alpha=0.9))

# --- 右列: Latency Cost Efficiency ---
bars_lat = ax_lat.barh(y_pos, df['LatCostNorm'],
                       color=df['Color'], edgecolor='#000',
                       linewidth=0.5, height=0.6, zorder=3)
ax_lat.set_xlim(0, 1.75)
ax_lat.set_xticks([0, 0.5, 1.0])

for bar in bars_lat:
    w = bar.get_width()
    if w > 0:
        ax_lat.text(w + 0.04, bar.get_y() + bar.get_height() / 2,
                    f'{w:.2f}', ha='left', va='center', fontsize=FONT_TICK, color='#000')

ax_lat.tick_params(axis='y', which='both', left=False, labelleft=False)
ax_lat.tick_params(axis='x', labelsize=FONT_TICK, colors='#000')

# 统一边框颜色
for ax in [ax_bw, ax_lat]:
    for sp in ax.spines.values():
        sp.set_color('#000')

# 列标题
ax_bw.set_title('BW Cost Efficiency', fontsize=FONT_TITLE, pad=30, fontweight='bold', color='#000')
ax_mid.set_title('Topology', fontsize=FONT_TITLE, pad=30, fontweight='bold', color='#000')
ax_lat.set_title('Latency Cost\nEfficiency', fontsize=FONT_TITLE, pad=30, fontweight='bold', color='#000')

# X 轴说明
ax_bw.set_xlabel('Normalized Performance', fontsize=FONT_AXIS+2, labelpad=10, color='#000')
ax_lat.set_xlabel('Normalized \nPerformance', fontsize=FONT_AXIS+2, labelpad=10, color='#000')

# 子图标号
ax_bw.text(0.02, 0.98, '(a)', transform=ax_bw.transAxes,
            fontsize=FONT_TITLE, fontweight='bold', va='top', ha='left', zorder=10)
ax_lat.text(0.98, 0.98, '(b)', transform=ax_lat.transAxes,
            fontsize=FONT_TITLE, fontweight='bold', va='top', ha='right', zorder=10)

plt.tight_layout()
fig.subplots_adjust(top=0.8) # 给标题留出空间
plt.savefig('SparseMesh/fig/table3_V3.png', dpi=300, bbox_inches='tight')
plt.show()
