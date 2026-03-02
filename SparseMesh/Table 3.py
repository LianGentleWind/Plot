# Figure 3: Bandwidth cost efficiency comparison across scales (64P / 128P)
# 数据来源: TODO.md 中的 Table: cost_efficiency_combined
# 绘制BW Cost归一化效率，64P和128P分两张子图

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

# 1. 数据准备
# Sys Cost 需要减去 NPU 成本 (每NPU $20,000)
# 64P: 64 * 20000 = 1,280,000
# 128P: 128 * 20000 = 2,560,000
# BW Cost Eff = BW / Net Cost，再归一化

data_64p = [
    ["SparseMesh-N8\n(default)", 22.16, 100.0,  1_379_840, 7.24],
    ["SparseMesh-N8\n(opt)",     22.16, 149.98, 1_379_840, 7.24],
    ["FullMesh-N8",              21.99, 200.03, 1_387_520, 7.75],
    ["Torus-N8",                 22.24, 100.0,  1_379_840, 7.24],
    ["CLOS-N8",                  23.10, 171.43, 1_470_920, 12.98],
]

data_128p = [
    ["SparseMesh-N16\n(default)", 22.19, 40.0,  2_775_040, 7.75],
    ["SparseMesh-N16\n(SP)",      22.19, 55.05, 2_775_040, 7.75],
    ["SparseMesh-N16\n(opt)",     22.17, 58.52, 2_775_040, 7.75],
    ["CLOS-N16",                  23.14, 80.0,  2_941_840, 12.98],
    ["Torus-2D-N16",              22.35, 50.0,  2_790_400, 8.26],
]

cols = ['Topology', 'Latency', 'BW', 'SysCost', 'NetCostPct']


def build_df(data, npu_count):
    df = pd.DataFrame(data, columns=cols)
    npu_cost = npu_count * 20000
    df['NetCost'] = df['SysCost'] - npu_cost
    df['BWCostRaw'] = df['BW'] / df['NetCost']
    df['BWCostNorm'] = df['BWCostRaw'] / df['BWCostRaw'].max()
    return df


df_64 = build_df(data_64p, 64)
df_128 = build_df(data_128p, 128)

# 2. 绘图配置（统一字体：标题14 坐标轴11）
FONT_TITLE, FONT_AXIS, FONT_TICK = 14, 11, 11
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="white")

color_map = {
    'CLOS':       '#E6B85C',   # 暖黄色
    'FullMesh':   '#6AB187',   # 灰绿色
    'SparseMesh': '#517FA4',   # 雾霾蓝
    'Torus':      '#C07CAD',   # 淡紫色
}


def get_color(name):
    for k, v in color_map.items():
        if k in name:
            return v
    return '#B0BEC5'


# 3. 绘图: 两行, 各行三列 (BW Cost | Topology | Latency placeholder – 但按要求只画BW Cost)
# 按TODO要求: 只画BW Cost Eff 归一化，64P和128P分两张子图
datasets = [('64P', df_64), ('128P', df_128)]

fig, axes = plt.subplots(2, 3, figsize=(7, 5), sharey='row',
                         gridspec_kw={'wspace': 0.05, 'hspace': 0.32,
                                      'width_ratios': [1, 0.45, 1]})

for i, (scale, df) in enumerate(datasets):
    ax_bw    = axes[i, 0]
    ax_mid   = axes[i, 1]
    ax_lat   = axes[i, 2]

    df = df.copy()
    df['Color'] = df['Topology'].apply(get_color)
    y_pos = np.arange(len(df))

    # --- 左列: BW Cost Efficiency ---
    bars_bw = ax_bw.barh(y_pos, df['BWCostNorm'],
                         color=df['Color'], edgecolor='#000',
                         linewidth=1.0, height=0.6, zorder=3)
    ax_bw.set_xlim(0, 1.35)
    ax_bw.set_xticks([0, 0.5, 1])
    ax_bw.invert_yaxis()

    for bar in bars_bw:
        w = bar.get_width()
        ax_bw.text(w + 0.03, bar.get_y() + bar.get_height() / 2,
                   f'{w:.2f}', ha='left', va='center', fontsize=FONT_TICK, color='#000')

    ax_bw.tick_params(axis='y', which='both', left=False, labelleft=False)
    ax_bw.tick_params(axis='x', labelsize=FONT_TICK, colors='#000')
    for sp in ax_bw.spines.values():
        sp.set_color('#000')

    # --- 中间列: Topology + Scale label ---
    for sp in ax_mid.spines.values():
        sp.set_visible(False)
    ax_mid.tick_params(axis='both', which='both', bottom=False, left=False,
                       labelbottom=False, labelleft=False)
    ax_mid.set_xlim(0, 1)

    for y, topo in zip(y_pos, df['Topology']):
        ax_mid.text(0.5, y, topo, ha='center', va='center', fontsize=FONT_AXIS, color='#000')

    ax_mid.text(0.5, len(df) + 0.3, f'Scale: {scale}',
                ha='center', va='center', fontsize=FONT_AXIS, fontweight='bold',
                color='#000',
                bbox=dict(facecolor='#FFE0E0', edgecolor='none', pad=4.0, alpha=0.9))

    # --- 右列: Latency Cost Efficiency ---
    # 计算 Latency Cost Eff: (1/Latency) / NetCost, 再归一化
    df['LatCostRaw'] = (1.0 / df['Latency']) / df['NetCost']
    df['LatCostNorm'] = df['LatCostRaw'] / df['LatCostRaw'].max()

    bars_lat = ax_lat.barh(y_pos, df['LatCostNorm'],
                           color=df['Color'], edgecolor='#000',
                           linewidth=1.0, height=0.6, zorder=3)
    ax_lat.set_xlim(0, 1.35)
    ax_lat.set_xticks([0, 0.5, 1])

    for bar in bars_lat:
        w = bar.get_width()
        ax_lat.text(w + 0.03, bar.get_y() + bar.get_height() / 2,
                    f'{w:.2f}', ha='left', va='center', fontsize=FONT_TICK, color='#000')

    ax_lat.tick_params(axis='y', which='both', left=False, labelleft=False)
    ax_lat.tick_params(axis='x', labelsize=FONT_TICK, colors='#000')
    for sp in ax_lat.spines.values():
        sp.set_color('#000')

    # 列标题 (仅首行)
    if i == 0:
        ax_bw.set_title('BW Cost Efficiency', fontsize=FONT_TITLE, pad=12, fontweight='bold', color='#000')
        ax_mid.set_title('Topology', fontsize=FONT_TITLE, pad=12, fontweight='bold', color='#000')
        ax_lat.set_title('Latency Cost Efficiency', fontsize=FONT_TITLE, pad=12, fontweight='bold', color='#000')

    # X 轴说明(仅尾行)
    if i == len(datasets) - 1:
        ax_bw.set_xlabel('Normalized Score', fontsize=FONT_AXIS, labelpad=10, color='#000')
        ax_lat.set_xlabel('Normalized Score', fontsize=FONT_AXIS, labelpad=10, color='#000')

fig.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.10)

# 子图标号 (a)-(d)，仅条形图子图，跳过中间 Topology 列
# 左列 BW Cost、右列 Latency Cost 各有2个，共4个子图
subplot_axes = [axes[i, 0] for i in range(2)] + [axes[i, 2] for i in range(2)]
for idx, ax in enumerate(subplot_axes):
    ax.text(0.98, 0.98, f'({chr(ord("a") + idx)})', transform=ax.transAxes,
            fontsize=FONT_TITLE, fontweight='bold', va='top', ha='right', zorder=10)

plt.savefig('fig/table3.png', dpi=300, facecolor='white', bbox_inches='tight', pad_inches=0)
plt.close()
