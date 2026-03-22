import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import os

# 1. 数据准备
data_64p = [
    ["SMesh-N8\n(SP)", 22.16, 100.0,  1_379_840, 7.24],
    ["SMesh-N8\n(opt)",     22.16, 149.98, 1_379_840, 7.24],
    ["FullMesh-N8",              21.99, 200.03, 1_387_520, 7.75],
    ["CLOS-N8",                  23.10, 171.43, 1_470_920, 12.98],
    ["Torus-N8",                 22.24, 100.0,  1_379_840, 7.24],
]

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
    # 避免除以0
    df['NetCost'] = df['NetCost'].replace(0, np.nan)
    df['Latency'] = df['Latency'].replace(0, np.nan)
    
    df['BWCostRaw'] = df['BW'] / df['NetCost']
    # 归一化时忽略 NaN
    max_bw_cost = df['BWCostRaw'].max()
    df['BWCostNorm'] = df['BWCostRaw'] / max_bw_cost if max_bw_cost != 0 else 0
    
    df['LatCostRaw'] = (1.0 / df['Latency']) / df['NetCost']
    max_lat_cost = df['LatCostRaw'].max()
    df['LatCostNorm'] = df['LatCostRaw'] / max_lat_cost if max_lat_cost != 0 else 0
    
    return df.fillna(0)

df_64 = build_df(data_64p, 64)
df_128 = build_df(data_128p, 128)

# 2. 绘图配置 (统一字体)
FONT_TITLE, FONT_AXIS, FONT_TICK = 15, 12, 12
plt.rcParams['font.sans-serif'] = ['DejaVu Sans'] # 使用系统可用字体
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="white")

color_map = {
    'CLOS':       '#E6B85C',   # 暖黄色
    'FullMesh':   '#6AB187',   # 灰绿色
    'SMesh': '#517FA4',   # 雾霾蓝
    'Torus':      '#C07CAD',   # 淡紫色
}

def get_color(name):
    for k, v in color_map.items():
        if k in name: return v
    return '#B0BEC5'

# 3. 绘图: 两行三列，实现“背靠背”
datasets = [('64P', df_64), ('128P', df_128)]

# 调整 width_ratios：增加左侧列宽度，减少右侧列宽度
# 原为 [1, 0.5, 1]，现调整为 [1.6, 0.6, 0.8]
fig, axes = plt.subplots(2, 3, figsize=(7.5, 9), sharey='row',
                         gridspec_kw={'wspace': 0.05, 'hspace': 0.4, 
                                      'width_ratios': [1.6, 0.6, 0.8]})

for i, (scale, df) in enumerate(datasets):
    ax_bw  = axes[i, 0]
    ax_mid = axes[i, 1]
    ax_lat = axes[i, 2]

    df = df.copy()
    df['Color'] = df['Topology'].apply(get_color)
    y_pos = np.arange(len(df))

    # --- 左列: BW Cost Efficiency (镜像翻转) ---
    bars_bw = ax_bw.barh(y_pos, df['BWCostNorm'],
                         color=df['Color'], edgecolor='#000',
                         linewidth=0.5, height=0.6, zorder=3)
    
    ax_bw.set_xlim(0, 1.35)
    ax_bw.set_xticks([0, 0.5, 1.0])
    ax_bw.invert_yaxis()
    ax_bw.invert_xaxis()  # 将 X 轴左右翻转

    # 标注数值 (左列，文字在柱子左侧)
    for bar in bars_bw:
        w = bar.get_width()
        if w > 0:
            ax_bw.text(w + 0.04, bar.get_y() + bar.get_height() / 2,
                       f'{w:.2f}', ha='right', va='center', fontsize=FONT_TICK, color='#000')

    ax_bw.tick_params(axis='y', which='both', left=False, labelleft=False)
    ax_bw.tick_params(axis='x', labelsize=FONT_TICK, colors='#000')
    for sp in ax_bw.spines.values():
        sp.set_color('#000')

    # --- 中间列: Topology 文本 ---
    for sp in ax_mid.spines.values():
        sp.set_visible(False)
    ax_mid.tick_params(axis='both', which='both', bottom=False, left=False,
                        labelbottom=False, labelleft=False)
    ax_mid.set_xlim(0, 1)

    for y, topo in zip(y_pos, df['Topology']):
        display_topo = topo.replace('\n', ' ')
        ax_mid.text(0.5, y, display_topo, ha='center', va='center', fontsize=FONT_AXIS, color='#000')

    # Scale 标签放在中间列上方
    ax_mid.text(0.5, -0.7, f'Scale: {scale}',
                ha='center', va='center', fontsize=FONT_AXIS, fontweight='bold',
                color='#000',
                bbox=dict(facecolor='#FFE0E0', edgecolor='none', pad=4.0, alpha=0.9))

    # --- 右列: Latency Cost Efficiency (常规方向) ---
    bars_lat = ax_lat.barh(y_pos, df['LatCostNorm'],
                           color=df['Color'], edgecolor='#000',
                           linewidth=0.5, height=0.6, zorder=3)
    ax_lat.set_xlim(0, 1.75)
    ax_lat.set_xticks([0, 0.5, 1.0])

    # 标注数值 (右列，文字在柱子右侧)
    for bar in bars_lat:
        w = bar.get_width()
        if w > 0:
            ax_lat.text(w + 0.04, bar.get_y() + bar.get_height() / 2,
                        f'{w:.2f}', ha='left', va='center', fontsize=FONT_TICK, color='#000')

    ax_lat.tick_params(axis='y', which='both', left=False, labelleft=False)
    ax_lat.tick_params(axis='x', labelsize=FONT_TICK, colors='#000')
    for sp in ax_lat.spines.values():
        sp.set_color('#000')

    # 列标题 (仅首行)
    if i == 0:
        ax_bw.set_title('BW Cost Efficiency', fontsize=FONT_TITLE, pad=20, fontweight='bold', color='#000')
        ax_mid.set_title('Topology', fontsize=FONT_TITLE, pad=20, fontweight='bold', color='#000')
        ax_lat.set_title('Latency Cost\nEfficiency', fontsize=FONT_TITLE, pad=20, fontweight='bold', color='#000')

    # X 轴说明 (仅末行)
    
    ax_bw.set_xlabel('Normalized Performance', fontsize=FONT_AXIS+3, labelpad=10, color='#000')
    ax_lat.set_xlabel('Normalized \nPerformance', fontsize=FONT_AXIS+3, labelpad=10, color='#000')

# 整体布局微调
fig.subplots_adjust(left=0.05, right=0.95, top=0.85, bottom=0.12)

# 子图标号 (a)-(d)
subplot_axes = [axes[0, 0], axes[1, 0], axes[0, 2], axes[1, 2]]
for idx, ax in enumerate(subplot_axes):
    label = f'({chr(ord("a") + idx)})'
    x_pos = 0.05 if ax in [axes[0, 0], axes[1, 0]] else 0.95
    align = 'left' if ax in [axes[0, 0], axes[1, 0]] else 'right'
    ax.text(x_pos, 0.96, label, transform=ax.transAxes,
            fontsize=FONT_TITLE, fontweight='bold', va='top', ha=align, zorder=10)

plt.savefig('SparseMesh/fig/table3_V2.png', dpi=300, facecolor='white', bbox_inches='tight', pad_inches=0.1)
plt.close()
