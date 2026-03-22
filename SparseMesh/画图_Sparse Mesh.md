File: SparseMesh\Table 1.py
```py
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.gridspec as gridspec

# 1. 准备数据
# 做的修改：N8K6的max load是1.33->1.43; N16K6其实应该是N16K7
data = {
    'Topology': ['N8_K6', 'N8_K6', 'N8_K6', 'N13_K6', 'N13_K6', 'N13_K6', 'N16_K6', 'N16_K6', 'N16_K6'],
    'Routing Strategy': [
        'Default', 'Shortest Multi-Path', 'Optimized (SA)',
        'Default', 'Shortest Multi-Path', 'Optimized (SA)',
        'Default', 'Shortest Multi-Path', 'Optimized (SA)'
    ],
    'Latency (ms)': [11.22, 7.47, 7.47, 13.69, 11.76, 10.26, 28.02, 20.18, 19.04],
    'Norm. Perf.': [1.00, 1.50, 1.50, 1.00, 1.16, 1.33, 1.00, 1.39, 1.47],
    'Max Load': [1.85, 1.43, 1.33, 4.20, 3.65, 3.00, 5.10, 3.95, 3.40]
}
ideal_loads = {'N8_K6': 1.33, 'N13_K6': 3.0, 'N16_K6': 3.28}
df = pd.DataFrame(data)

# 2. 全局配置 (字号 16)
GLOBAL_FONT = 'DejaVu Sans' 
GLOBAL_FONT_SIZE = 16  
LATENCY_YLIM = (0, 35)
PERF_YLIM = (0.7, 1.75)
LOAD_YLIM = (0, 6.5)

# 颜色顺序: 浅黄, 浅绿, 浅蓝
MY_PALETTE = ["#F9E79F", "#A8D5BA", "#A2C4E4"] 

sns.set_theme(style="white")
plt.rcParams['font.sans-serif'] = [GLOBAL_FONT]
plt.rcParams['font.size'] = GLOBAL_FONT_SIZE
plt.rcParams['axes.unicode_minus'] = False

# 尺寸固定为 8x7
fig = plt.figure(figsize=(8, 7))
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.3)

ax1 = fig.add_subplot(gs[0, 0]) 
ax2 = fig.add_subplot(gs[0, 1]) 
ax3 = fig.add_subplot(gs[1, :]) 

def apply_paper_style(ax, title, ylabel, ylim, label, show_legend=False):
    ax.set_title(title, fontsize=GLOBAL_FONT_SIZE, pad=15, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=GLOBAL_FONT_SIZE - 2)
    ax.set_xlabel('Topology', fontsize=GLOBAL_FONT_SIZE - 2)
    ax.set_ylim(ylim)
    
    # 将标号放置在方框内部左上角
    # x=0.02, y=0.96 是相对于坐标轴内部的比例位置
    ax.text(0.02, 0.96, label, transform=ax.transAxes, 
            fontsize=GLOBAL_FONT_SIZE, fontweight='bold', va='top', ha='left', zorder=10)
    
    # 去掉刻度线
    ax.tick_params(axis='both', which='both', length=0, labelsize=GLOBAL_FONT_SIZE - 4)
    
    # 黑色加粗边框
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1.5)
        spine.set_visible(True)
    
    if show_legend:
        # 特殊处理图 (c) 的图例位置，避开左上角的标号
        # loc='upper left' 配合 bbox_to_anchor 稍微向右下方移动
        ax.legend(loc='upper left', bbox_to_anchor=(0.0, 0.85), frameon=True, edgecolor='black', 
                  fontsize=GLOBAL_FONT_SIZE - 6, title_fontsize=GLOBAL_FONT_SIZE - 6)
    else:
        # 显式移除 Seaborn 产生的图例
        leg = ax.get_legend()
        if leg: leg.remove()

def plot_paper_bar(ax, y_col, title, ylabel, ylim, label, show_legend=False):
    sns.barplot(
        data=df, x='Topology', y=y_col, hue='Routing Strategy', 
        ax=ax, palette=MY_PALETTE, edgecolor='black', linewidth=1.2
    )
    apply_paper_style(ax, title, ylabel, ylim, label, show_legend)
    # 柱状图标注
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', padding=2, fontsize=GLOBAL_FONT_SIZE - 8)

# 执行绘图
plot_paper_bar(ax1, 'Latency (ms)', 'Routing Latency', 'Latency (ms)', LATENCY_YLIM, '(a)')
plot_paper_bar(ax2, 'Norm. Perf.', 'Norm. Performance', 'Ratio', PERF_YLIM, '(b)')
# 仅在 ax3 开启 legend
plot_paper_bar(ax3, 'Max Load', 'Max Load vs Theoretical Ideal', 'Load Value', LOAD_YLIM, '(c)', show_legend=True)

# --- 第三幅图 Ideal 参考线标注 ---
for i, topology in enumerate(df['Topology'].unique()):
    ideal_val = ideal_loads[topology]
    ax3.hlines(y=ideal_val, xmin=i-0.4, xmax=i+0.4, color='#E74C3C', 
               linestyle='--', linewidth=2.5, zorder=5)
    
    # 标注 Ideal 数值
    ax3.text(i + 0.02, ideal_val - 0.35, f'Ideal: {ideal_val}', color='#E74C3C', 
             ha='right', va='top', fontsize=GLOBAL_FONT_SIZE - 6, fontweight='bold')

# 保存为无白边高分辨 PNG
plt.savefig('SparseMesh/fig/table1.png', dpi=300, bbox_inches='tight', pad_inches=0.02)
plt.show()
```

File: SparseMesh\Table 3_V3.py
```py
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
```

File: SparseMesh\Table 6_single.py
```py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# 1. 原始数据更新
scales = ['256', '512', '1024']
topologies = ['SMesh 128P rack', 'SMesh 64P rack', 'FullMesh', 'CLOS']

# 吞吐数据 (Throughput)
data_tp = {
    'EP16': [[1.72368, 1.7199, 1.69911], [1.64997, 1.64619, 1.61217], [1.674, 1.6668, 1.62], [1, 0.997, 0.97]],
    'EP32': [[1.752079, 1.748298, 1.736958], [1.701047, 1.697267, 1.685927], [1.682805, 1.679205, 1.666607], [1, 0.998, 0.991]],
    'EP64': [[1.86874, 1.86247, 1.85768], [1.81445, 1.81256, 1.80689], [1.758396, 1.756596, 1.751197], [1, 0.998, 0.995]]
}

# DP暴露数据 (DP Exposure)
data_dp = {
    'EP16': [[2.114, 3.431, 5.689], [2.544, 4.117, 6.761], [2.253, 3.813, 6.788], [1.639, 2.778, 4.978]],
    'EP32': [[3.9, 5.75, 8.897], [3.436, 5.077, 7.905], [4.407, 5.962, 9.206], [3.294, 4.519, 6.76]],
    'EP64': [[0.324, 0.361, 0.421], [0.299, 0.329, 0.389], [0.284, 0.313, 0.37], [0.276, 0.304, 0.36]]
}

# 2. 绘图配置
FONT_TITLE, FONT_AXIS, FONT_TICK = 14, 11, 10
FONT_LEGEND = 12  # 增大图例字号
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="white")

# 莫兰迪配色方案
color_map = {
    'SMesh 128P rack': '#517FA4',  
    'SMesh 64P rack': '#92B4D2',   
    'FullMesh': '#6AB187',        
    'CLOS': '#E6B85C'             
}

# 3. 创建画布 (2行3列)
fig, axes = plt.subplots(2, 3, figsize=(7.5, 5))
ep_list = ['EP16', 'EP32', 'EP64']
x = np.arange(len(scales))
width = 0.2 

# 4. 绘制上排：吞吐性能柱状图 (Shared Y)
for i, ep in enumerate(ep_list):
    ax = axes[0, i]
    for j, topo in enumerate(topologies):
        ax.bar(x + (j - 1.5) * width, data_tp[ep][j], width, 
               label=topo, color=color_map[topo], edgecolor='black', linewidth=0.8)
    
    ax.set_title(f'{ep} ', fontsize=FONT_TITLE, fontweight='bold', pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(scales, fontsize=FONT_TICK)
    ax.set_ylim(0, 2.4) 
    if i == 0:
        ax.set_ylabel('Normalized \nPer-port Throughput', fontsize=FONT_AXIS)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

# 5. 绘制下排：DP暴露折线图
for i, ep in enumerate(ep_list):
    ax = axes[1, i]
    for j, topo in enumerate(topologies):
        ax.plot(scales, data_dp[ep][j], marker='o', markersize=7,  # 增大打点标记
                label=topo, color=color_map[topo], linewidth=2.5, markeredgecolor='black')
    
    ax.set_title(f'{ep}', fontsize=FONT_TITLE, fontweight='bold', pad=10)
    ax.tick_params(labelsize=FONT_TICK)
    if i == 0:
        ax.set_ylabel('DP comm propoertion (%)', fontsize=FONT_AXIS)
    ax.set_xlabel('Experts', fontsize=FONT_AXIS)
    ax.grid(axis='both', linestyle='--', alpha=0.5)

# 6. 标号 (a)-(f)
for idx, ax in enumerate(axes.flat):
    ax.text(0.15, 0.95, f'({chr(ord("a") + idx)})', transform=ax.transAxes,
            fontsize=FONT_TITLE, fontweight='bold', va='top', ha='right')

# 7. 全局图例 (移至顶部)
handles, labels = axes[0, 0].get_legend_handles_labels()
# 调整 bbox_to_anchor 使图例位于 Title 之上
fig.legend(handles, labels, loc='upper center', ncol=4, fontsize=FONT_LEGEND, 
           frameon=False, bbox_to_anchor=(0.5, 1.02), handlelength=2.5, handleheight=1.5)

# 8. 调整布局
# 预留顶部空间给 Legend (top=0.92)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.subplots_adjust(hspace=0.45, wspace=0.25)

# 保存文件
plt.savefig('SparseMesh/fig/table6_Single.png', dpi=300, bbox_inches='tight')
plt.show()
```

File: SparseMesh\Table 6.py
```py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# 1. 原始数据更新
scales = ['256', '512', '1024']
topologies = ['SMesh 128P rack', 'SMesh 64P rack', 'FullMesh', 'CLOS']

# 吞吐数据 (Throughput)
data_tp = {
    'EP16': [[1.72368, 1.7199, 1.69911], [1.64997, 1.64619, 1.61217], [1.674, 1.6668, 1.62], [1, 0.997, 0.97]],
    'EP32': [[1.752079, 1.748298, 1.736958], [1.701047, 1.697267, 1.685927], [1.682805, 1.679205, 1.666607], [1, 0.998, 0.991]],
    'EP64': [[1.86874, 1.86247, 1.85768], [1.81445, 1.81256, 1.80689], [1.758396, 1.756596, 1.751197], [1, 0.998, 0.995]]
}

# DP暴露数据 (DP Exposure)
data_dp = {
    'EP16': [[2.114, 3.431, 5.689], [2.544, 4.117, 6.761], [2.253, 3.813, 6.788], [1.639, 2.778, 4.978]],
    'EP32': [[3.9, 5.75, 8.897], [3.436, 5.077, 7.905], [4.407, 5.962, 9.206], [3.294, 4.519, 6.76]],
    'EP64': [[0.324, 0.361, 0.421], [0.299, 0.329, 0.389], [0.284, 0.313, 0.37], [0.276, 0.304, 0.36]]
}

# 2. 绘图配置
FONT_TITLE, FONT_AXIS, FONT_TICK = 18, 15, 14
FONT_LEGEND = 18  # 增大图例字号
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="white")

# 莫兰迪配色方案
color_map = {
    'SMesh 128P rack': '#517FA4',  
    'SMesh 64P rack': '#92B4D2',   
    'FullMesh': '#6AB187',        
    'CLOS': '#E6B85C'             
}

# 3. 创建画布 (2行3列)
fig, axes = plt.subplots(2, 3, figsize=(15, 7.5))
ep_list = ['EP16', 'EP32', 'EP64']
x = np.arange(len(scales))
width = 0.2 

# 4. 绘制上排：吞吐性能柱状图 (Shared Y)
for i, ep in enumerate(ep_list):
    ax = axes[0, i]
    for j, topo in enumerate(topologies):
        ax.bar(x + (j - 1.5) * width, data_tp[ep][j], width, 
               label=topo, color=color_map[topo], edgecolor='black', linewidth=0.8)
    
    ax.set_title(f'{ep}', fontsize=FONT_TITLE, fontweight='bold', pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(scales, fontsize=FONT_TICK)
    ax.set_ylim(0, 2.4) 
    ax.set_xlabel('Experts', fontsize=FONT_AXIS)
    ax.set_ylabel('Normalized Per-port Throughput', fontsize=FONT_AXIS)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

# 5. 绘制下排：DP暴露折线图
for i, ep in enumerate(ep_list):
    ax = axes[1, i]
    for j, topo in enumerate(topologies):
        ax.plot(scales, data_dp[ep][j], marker='o', markersize=7,  # 增大打点标记
                label=topo, color=color_map[topo], linewidth=2.5, markeredgecolor='black')
    
    ax.set_title(f'{ep}', fontsize=FONT_TITLE, fontweight='bold', pad=10)
    ax.tick_params(labelsize=FONT_TICK)
    ax.set_ylabel('DP comm propoertion (%)', fontsize=FONT_AXIS)
    ax.set_xlabel('Experts', fontsize=FONT_AXIS)
    ax.grid(axis='both', linestyle='--', alpha=0.5)

# 6. 标号 (a)-(f)
for idx, ax in enumerate(axes.flat):
    ax.text(0.1, 0.95, f'({chr(ord("a") + idx)})', transform=ax.transAxes,
            fontsize=FONT_TITLE, fontweight='bold', va='top', ha='right')

# 7. 全局图例 (移至顶部)
handles, labels = axes[0, 0].get_legend_handles_labels()
# 调整 bbox_to_anchor 使图例位于 Title 之上
fig.legend(handles, labels, loc='upper center', ncol=4, fontsize=FONT_LEGEND, 
           frameon=False, bbox_to_anchor=(0.5, 1.02), handlelength=2.5, handleheight=1.5)

# 8. 调整布局
# 预留顶部空间给 Legend (top=0.92)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.subplots_adjust(hspace=0.4, wspace=0.25)

# 保存文件
plt.savefig('SparseMesh/fig/table6_Double.png', dpi=300, bbox_inches='tight')
plt.show()
```

File: SparseMesh\Table5 V3.py
```py
"""
图5: Inference performance comparison (3行3列)
- 列：模型（DSV3 / Qwen2-35B / Kimi25）
- 行：Scale（64P / 128P / 256P）
- 归一化：每列（模型）使用其全局 SM_Thro 最大值进行归一化
"""

import matplotlib.pyplot as plt
import matplotlib.transforms as mtrans
import numpy as np
import seaborn as sns

# === 数据 ===
data_DSV3 = [
    (64, 4096, 20, 64, 64),
    (64, 4096, 50, 204, 202),
    (64, 4096, 100, 417, 393), 
    (64, 8192, 20, 37, 37),
    (64, 8192, 50, 130, 129),
    (64, 8192, 100, 270, 269),
    (128, 4096, 20, 67, 66),
    (128, 4096, 50, 202, 199),
    (128, 4096, 100, 415, 407),
    (128, 8192, 20, 42, 41),
    (128, 8192, 50, 129, 128),
    (128, 8192, 100, 269, 266),
    (256, 4096, 20, 66, 56),
    (256, 4096, 50, 202, 173),
    (256, 4096, 100, 414, 355),
    (256, 8192, 20, 42, 38),
    (256, 8192, 50, 129, 116),
    (256, 8192, 100, 269, 243),
]

data_qwen235B = [
# 64P
(64, 4096, 20, 153, 154),
(64, 4096, 50, 2329, 2306),
(64, 4096, 100, 3133, 3102),
(64, 8192, 20, 100, 101),
(64, 8192, 50, 1381, 1369),
(64, 8192, 100, 1885, 1873),
# 128P
(128, 4096, 20, 1315, 1256),
(128, 4096, 50, 2929, 2845),
(128, 4096, 100, 3133, 3040),
(128, 8192, 20, 765, 754),
(128, 8192, 50, 1746, 1707),
(128, 8192, 100, 1995, 1961),
# 256P
(256, 4096, 20, 2112, 1702),
(256, 4096, 50, 2929, 2465),
(256, 4096, 100, 3133, 2623),
(256, 8192, 20, 1256, 1072),
(256, 8192, 50, 1874, 1671),
(256, 8192, 100, 1995, 1779),
]

data_kimi25 = [
# 64P
(64, 4096, 20, 153, 154),
(64, 4096, 50, 2329, 2306),
(64, 4096, 100, 3133, 3102),
(64, 8192, 20, 100, 101),
(64, 8192, 50, 1381, 1369),
(64, 8192, 100, 1885, 1873),
# 128P
(128, 4096, 20, 1315, 1256),
(128, 4096, 50, 2929, 2845),
(128, 4096, 100, 3133, 3040),
(128, 8192, 20, 765, 754),
(128, 8192, 50, 1746, 1707),
(128, 8192, 100, 1995, 1961),
# 256P
(256, 4096, 20, 2112, 1702),
(256, 4096, 50, 2929, 2465),
(256, 4096, 100, 3133, 2623),
(256, 8192, 20, 1256, 1072),
(256, 8192, 50, 1874, 1671),
(256, 8192, 100, 1995, 1779),
]

# === 计算每模型的全局 SM_Thro 最大值 ===
max_sm_DSV3 = max(d[4] for d in data_DSV3)
max_sm_Qwen = max(d[4] for d in data_qwen235B)
max_sm_Kimi = max(d[4] for d in data_kimi25)

datasets = [
    (data_DSV3, "Deepeek-V3", max_sm_DSV3),
    (data_qwen235B, "Qwen3-VL", max_sm_Qwen),
    (data_kimi25, "Kimi2.5", max_sm_Kimi),
]

scales = [64, 128, 256]
bar_height = 0.3

# 字体设置
FONT_TITLE, FONT_AXIS, FONT_TICK = 15, 12, 12
FONT_GROUP = 13
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="white")

# 创建 3 行 3 列子图
fig, axes = plt.subplots(3, 3, figsize=(21, 6), sharex=True, sharey=False)

for row_idx, scale in enumerate(scales):
    for col_idx, (data, model_name, max_sm_global) in enumerate(datasets):
        ax = axes[row_idx, col_idx]
        subset = [d for d in data if d[0] == scale]
        
        if len(subset) != 6:
            print(f"警告: {model_name} {scale}P 数据数量异常 ({len(subset)} 条)")
            continue

        # 使用该模型的全局 SM 最大值进行归一化
        clos_vals = [d[3] / (max_sm_global * 1.89) for d in subset]
        sm_vals   = [d[4] / max_sm_global for d in subset]

        y = np.arange(6)
        bars1 = ax.barh(y - bar_height/2, clos_vals, bar_height,
                       label='Clos', color='#E6B85C', edgecolor='#000', linewidth=0.5)
        bars2 = ax.barh(y + bar_height/2, sm_vals, bar_height,
                       label='SMesh', color='#517FA4', edgecolor='#000', linewidth=0.5)

        # X轴
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xlim(0, 1.1)
        ax.tick_params(axis='x', labelsize=FONT_TICK)
        ax.xaxis.grid(True, linestyle='--', alpha=0.6, zorder=0)

        # Y轴标签（仅第一列）
        labels = [f'TPOT {d[2]}' for d in subset]
        if col_idx == 0:
            ax.set_yticks(y)
            ax.set_yticklabels(labels, fontsize=FONT_TICK)
        else:
            ax.set_yticks(y)
            ax.set_yticklabels([])

        # 序列长度分隔线
        ax.axhline(2.5, color='#000', linewidth=1.0, linestyle='--', alpha=0.8, zorder=1)

        # 标题：模型名 + Scale
        ax.set_title(f'{model_name} {scale}P', fontsize=FONT_TITLE, fontweight='bold', pad=6)

        # X轴标签（仅最后一行）
        if row_idx == 2:
            ax.set_xlabel('Normalized Per-port Throughput', fontsize=FONT_AXIS + 2)

        ax.invert_yaxis()
        for spine in ax.spines.values():
            spine.set_color('#000')

        # 序列长度标记（仅第一列）
        if col_idx == 0:
            trans = mtrans.blended_transform_factory(ax.transAxes, ax.transData)
            ax.text(-0.18, 1, '4096', transform=trans, fontsize=FONT_GROUP+2, fontweight='bold',
                    va='center', ha='right', color='#C00000', zorder=10, clip_on=False)
            ax.text(-0.18, 4, '8192', transform=trans, fontsize=FONT_GROUP+2, fontweight='bold',
                    va='center', ha='right', color='#C00000', zorder=10, clip_on=False)

        # 子图标号 (a)-(i)
        label_char = chr(ord('a') + row_idx * 3 + col_idx)
        ax.text(0.975, 0.97, f'({label_char})', transform=ax.transAxes,
                fontsize=FONT_TITLE, fontweight='bold', va='top', ha='right',
                bbox=dict(facecolor='white', edgecolor='none', pad=1), zorder=15)
        if col_idx == 0 and row_idx == 0:
            ax.text(-0.145, 1.13, f'Sequence', transform=ax.transAxes,
                fontsize=FONT_TITLE, fontweight='bold', va='top', ha='right',color = '#C00000',
                bbox=dict(facecolor='white', edgecolor='none', pad=1), zorder=15)

# 全局图例（顶部）
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=2, fontsize=FONT_TICK + 4,
           frameon=False, bbox_to_anchor=(0.53, 1.05))

# 布局调整
fig.tight_layout()
fig.subplots_adjust(left=0.13, right=0.98, top=0.93, bottom=0.08, hspace=0.32, wspace=0.06)

# 保存
plt.savefig('SparseMesh/fig/table5_V3.png', dpi=300, facecolor='white',
            bbox_inches='tight', pad_inches=0.1)
plt.show()
```

File: SparseMesh\Table5 V2.py
```py
"""
图5: Inference performance comparison (2行3列 + 局部归一化)
- 上排：DSV3；下排：Qwen2-35B
- 每幅子图独立归一化：SM_Thro 最大值 → 1.0，Clos_Thro 除以 (SM_max * 1.89)
"""

import matplotlib.pyplot as plt
import matplotlib.transforms as mtrans
import numpy as np
import seaborn as sns

# 数据
data_DSV3 = [
    (64, 4096, 20, 64, 64),
    (64, 4096, 50, 204, 202),
    (64, 4096, 100, 417, 393), 
    (64, 8192, 20, 37, 37),
    (64, 8192, 50, 130, 129),
    (64, 8192, 100, 270, 269),
    (128, 4096, 20, 67, 66),
    (128, 4096, 50, 202, 199),
    (128, 4096, 100, 415, 407),
    (128, 8192, 20, 42, 41),
    (128, 8192, 50, 129, 128),
    (128, 8192, 100, 269, 266),
    (256, 4096, 20, 66, 56),
    (256, 4096, 50, 202, 173),
    (256, 4096, 100, 414, 355),
    (256, 8192, 20, 42, 38),
    (256, 8192, 50, 129, 116),
    (256, 8192, 100, 269, 243),
]

data_qwen235B = [
    (64, 4096, 20, 1972, 1952),
    (64, 4096, 50, 2241, 2192),
    (64, 4096, 100, 2298, 2265),
    (64, 8192, 20, 1343, 1318),
    (64, 8192, 50, 1303, 1289),
    (64, 8192, 100, 1122, 1121),
    (128, 4096, 20, 6446, 5966),
    (128, 4096, 50, 6476, 6118),
    (128, 4096, 100, 6476, 6118),
    (128, 8192, 20, 4052, 3808),
    (128, 8192, 50, 4140, 3948),
    (128, 8192, 100, 4140, 3948),
    (256, 4096, 20, 6446, 5224),
    (256, 4096, 50, 6476, 6118),
    (256, 4096, 100, 6476, 6118),
    (256, 8192, 20, 4052, 3182),
    (256, 8192, 50, 4140, 3607),
    (256, 8192, 100, 4140, 3607),
]

data_kimi25 = [
    # 64P
    (64, 4096, 20, 6794, 6669),
    (64, 4096, 50, 7055, 6741),
    (64, 4096, 100, 7067, 6759),
    (64, 8192, 20, 4621, 4557),
    (64, 8192, 50, 5011, 4854),
    (64, 8192, 100, 5011, 4861),
    # 128P
    (128, 4096, 20, 6794, 6334),
    (128, 4096, 50, 7055, 6463),
    (128, 4096, 100, 7067, 6501),
    (128, 8192, 20, 4756, 4536),
    (128, 8192, 50, 5011, 4701),
    (128, 8192, 100, 5011, 4719),
    # 256P
    (256, 4096, 20, 6794, 4726),
    (256, 4096, 50, 7055, 4887),
    (256, 4096, 100, 7067, 4918),
    (256, 8192, 20, 4756, 3642),
    (256, 8192, 50, 5011, 3808),
    (256, 8192, 100, 5011, 3830),
]

datasets = [
    (data_DSV3, "DSV3"),
    (data_qwen235B, "Qwen2-35B")
]

scales = [64, 128, 256]
bar_height = 0.35

# 字体设置
FONT_TITLE, FONT_AXIS, FONT_TICK = 16, 13, 13
FONT_GROUP = 14
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="white")

# 创建 2 行 3 列子图
fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True, sharey=False)

for row_idx, (data, model_name) in enumerate(datasets):
    for col_idx, scale in enumerate(scales):
        ax = axes[row_idx, col_idx]
        subset = [d for d in data if d[0] == scale]
        
        if len(subset) != 6:
            print(f"警告: {model_name} {scale}P 数据数量异常 ({len(subset)} 条)")
            continue

        # === 关键修改：局部最大值归一化 ===
        sm_vals_raw = [d[4] for d in subset]      # 提取 SM_Thro
        max_sm_local = max(sm_vals_raw)           # 本子图内 SM 最大值
        
        clos_vals = [d[3] / (max_sm_local * 1.89) for d in subset]
        sm_vals   = [d[4] / max_sm_local for d in subset]

        y = np.arange(6)
        bars1 = ax.barh(y - bar_height/2, clos_vals, bar_height,
                       label='Clos', color='#E6B85C', edgecolor='#000', linewidth=0.5)
        bars2 = ax.barh(y + bar_height/2, sm_vals, bar_height,
                       label='SMesh', color='#517FA4', edgecolor='#000', linewidth=0.5)

        # X轴
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xlim(0, 1.1)
        ax.tick_params(axis='x', labelsize=FONT_TICK)
        ax.xaxis.grid(True, linestyle='--', alpha=0.6, zorder=0)

        # Y轴标签（仅第一列）
        labels = [f'TPOT {d[2]}' for d in subset]
        if col_idx == 0:
            ax.set_yticks(y)
            ax.set_yticklabels(labels, fontsize=FONT_TICK)
        else:
            ax.set_yticks(y)
            ax.set_yticklabels([])

        # 序列长度分隔线
        ax.axhline(2.5, color='#000', linewidth=1.0, linestyle='--', alpha=0.8, zorder=1)

        # 标题（仅显示 Scale）
        if row_idx == 0:
            ax.set_title(f'Deepeek-V3 {scale}P', fontsize=FONT_TITLE, fontweight='bold', pad=6)
        else:
            ax.set_title(f'Qwen3-VL {scale}P', fontsize=FONT_TITLE, fontweight='bold', pad=6)

        # X轴标签（仅底行）
        if row_idx == 1:
            if col_idx == 1:
                ax.set_xlabel('Normalized Per-port Throughput', fontsize=FONT_AXIS + 6)

        ax.invert_yaxis()
        for spine in ax.spines.values():
            spine.set_color('#000')

        # 序列长度标记
        if col_idx == 0:
            trans = mtrans.blended_transform_factory(ax.transAxes, ax.transData)
            ax.text(-0.35, 1, '4096', transform=trans, fontsize=FONT_GROUP+3, fontweight='bold',
                    va='center', ha='right', color='#C00000', zorder=10, clip_on=False)
            ax.text(-0.35, 4, '8192', transform=trans, fontsize=FONT_GROUP+3, fontweight='bold',
                    va='center', ha='right', color='#C00000', zorder=10, clip_on=False)

        # 子图标号 (a)-(f)
        label_char = chr(ord('a') + row_idx * 3 + col_idx)
        ax.text(0.975, 0.97, f'({label_char})', transform=ax.transAxes,
                fontsize=FONT_TITLE + 1, fontweight='bold', va='top', ha='right',
                bbox=dict(facecolor='white', edgecolor='none', pad=1), zorder=15)

# 全局图例
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=2, fontsize=FONT_TICK + 6,
           frameon=False, bbox_to_anchor=(0.55, 1.03))

# 布局调整
fig.tight_layout()
fig.subplots_adjust(left=0.12, right=0.98, top=0.92, bottom=0.12, hspace=0.2, wspace=0.08)

# 保存
plt.savefig('SparseMesh/fig/table5_V2.png', dpi=300, facecolor='white',
            bbox_inches='tight', pad_inches=0.1)
plt.show()
```

File: SparseMesh\TODO.md
```md
图3。64P和128P分两张图。数据需要修改，成本数据需要减去64*20000或者128*20000，再算成本/Sys Cost得的BW Cost，再归一化。不画其他数据了。
64P的SMesh default改名为SP，128P的SMesh default删除。64P新增FM-N16的配置，数值填成0即可。

\begin{table*}[!h]
\centering
\caption{Bandwidth and latency cost efficiency comparison across scales (64P / 128P)}
\label{tab:cost_efficiency_combined}
\begin{tabular}{l l c c c c c c}
\toprule
Scale & Topology & Latency ($\mu$s) & BW (Gb/s) & Sys Cost (\$) 
& BW Cost Eff. & Lat Cost Eff. & Net Cost \% \\
\midrule

\multirow{5}{*}{64P} 
& SparseMesh-N8 (default) & 22.16 & 100.0  & 1,379,840 & 0.67 & 1.00 & 7.24\% \\
& SparseMesh-N8 (opt)     & 22.16 & 149.98 & 1,379,840 & 1.00 & 1.00 & 7.24\% \\
& FullMesh-N8             & 21.99 & 200.03 & 1,387,520 & 1.33 & 1.00 & 7.75\% \\
& Torus-N8                & 22.24 & 100.0  & 1,379,840 & 0.67 & 1.00 & 7.24\% \\
& CLOS-N8                 & 23.10 & 171.43 & 1,470,920 & 1.07 & 0.90 & 12.98\% \\
\midrule

\multirow{5}{*}{128P}
& SparseMesh-N16 (default) & 22.19 & 40.0  & 2,775,040 & 0.73 & 1.00 & 7.75\% \\
& SparseMesh-N16 (SP)      & 22.19 & 55.05 & 2,775,040 & 1.00 & 1.00 & 7.75\% \\
& SparseMesh-N16 (opt)     & 22.17 & 58.52 & 2,775,040 & 1.06 & 1.00 & 7.75\% \\
& CLOS-N16                 & 23.14 & 80.0  & 2,941,840 & 1.37 & 0.91 & 12.98\% \\
& Torus-2D-N16             & 22.35 & 50.0  & 2,790,400 & 0.90 & 0.99 & 8.26\% \\
\bottomrule
\end{tabular}
\end{table*}  


图4。类似上一张图，画64P，128P，256P下的归一化Cost Efficiency。但是这里开始已经选取了Clos和SM各自的最优策略，所以没有那么多选型了，不再需要分子图，一张图里即可。使用竖向的柱状图。

\begin{table}[h]
\centering
\caption{Training throughput and cost efficiency comparison of 2048-NPU cluster}
\label{Training throughput and cost efficiency}
\begin{tabular}{lcccccc}
\toprule
Topology & \makecell{Norm.\\Throughput} & \makecell{Total\\Cost (\$)} & 
\makecell{Cost\\Efficiency} & \makecell{Net\\Cost \%} \\
\midrule
64P Clos & 1.000 & 47.8M & 1.000 & 14.31\% \\
64P SM & 0.933 & 43.4M & 1.027 & 5.65\% \\
\midrule
128P Clos & 1.009 & 47.8M & 1.009 & 14.31\% \\
128P SM & 0.932 & 43.4M & 1.027 & 5.65\% \\
\midrule
256P Clos & 1.014 & 47.8M & 1.014 & 14.31\% \\
256P SM & 0.933 & 43.4M & 1.028 & 5.65\% \\
\bottomrule
\end{tabular}
\end{table} 


图5。3种rack卡数，2种序列长度，3种TPOT一共18个自变量。绘制的还是Clos和SM各自的Thro吞吐值。
\begin{table}[h]
\centering
\caption{Inference performance comparison}
\label{tab:inference_detailed}
\begin{tabular}{ccccc c c}
\toprule
Scale & Seq Len & \makecell{TPOT\\(ms)} & \makecell{Clos\\Thro.} & \makecell{SM\\Thro.} & \makecell{Perf.\\Change} \\
\midrule

\multirow{6}{*}{64P}
& \multirow{3}{*}{4096}
& 20  & 64  & 64  & -0.16\% \\
&     & 50  & 204 & 202 & -0.78\% \\
&     & 100 & 417 & 413 & -0.90\% \\
\cmidrule(lr){2-6}
& \multirow{3}{*}{8192}
& 20  & 37  & 37  & +0.16\% \\
&     & 50  & 130 & 129 & -0.47\% \\
&     & 100 & 270 & 269 & -0.54\% \\

\midrule

\multirow{6}{*}{128P}
& \multirow{3}{*}{4096}
& 20  & 67  & 66  & -1.49\% \\
&     & 50  & 202 & 199 & -1.85\% \\
&     & 100 & 415 & 407 & -1.99\% \\
\cmidrule(lr){2-6}
& \multirow{3}{*}{8192}
& 20  & 42  & 41  & -1.21\% \\
&     & 50  & 129 & 128 & -1.15\% \\
&     & 100 & 269 & 266 & -1.26\% \\

\midrule

\multirow{6}{*}{256P}
& \multirow{3}{*}{4096}
& 20  & 66  & 56  & -14.87\% \\
&     & 50  & 202 & 173 & -14.16\% \\
&     & 100 & 414 & 355 & -14.23\% \\
\cmidrule(lr){2-6}
& \multirow{3}{*}{8192}
& 20  & 42  & 38  & -9.56\% \\
&     & 50  & 129 & 116 & -10.09\% \\
&     & 100 & 269 & 243 & -9.70\% \\

\bottomrule
\end{tabular}
\end{table}

图6
EP16
||吞吐||||DP暴露|||
|规模|256|512|1024||256|512|1024|
|SMesh 128P rack|1.72368|1.7199|1.69911||2.114|3.431|5.689|
|SMesh 64P rack|1.64997|1.64619|1.61217||2.544|4.117|6.761|
|Fullmesh|1.674|1.6668|1.62||2.253|3.813|6.788|
|CLOS|1|0.997|0.97||1.639|2.778|4.978|

EP32
||吞吐||||DP暴露|||
|规模|256|512|1024||256|512|1024|
|SMesh 128P rack|1.752079|1.748298|1.736958||3.9|5.75|8.897|
|SMesh 64P rack|1.701047|1.697267|1.685927||3.436|5.077|7.905|
|Fullmesh|1.682805|1.679205|1.666607||4.407|5.962|9.206|
|CLOS|1|0.998|0.991||3.294|4.519|6.76|

EP64
||吞吐||||DP暴露|||
|规模|256|512|1024||256|512|1024|
|SMesh 128P rack|1.81445|1.81256|1.80689||0.299|0.329|0.389| % 此行数据需脑补，下一行是脑补结果：
|SMesh 128P rack|1.86874|1.86247|1.85768||0.324|0.361|0.421|
|SMesh 64P rack|1.81445|1.81256|1.80689||0.299|0.329|0.389|
|Fullmesh|1.758396|1.756596|1.751197||0.284|0.313|0.37|
|CLOS|1|0.998|0.995||0.276|0.304|0.36|

吞吐数据：绘制柱状图。双栏图的上半部分，横向分为三个共享Y轴的小子图，分别对应3个EP分组。内部绘制不同规模256,512,1024下的吞吐性能，用不同颜色柱子表示几种拓扑。相同规模下的几种柱子应该紧密贴在一起。

DP暴露数据：绘制折线图，双栏图的下半部分。分为三个独立的小子图，分别绘制不同颜色表示的拓扑折线在规模变化时，DP暴露比例数值的趋势。

```

File: SparseMesh\Table 2.py
```py
# \begin{table}[htbp]
# \centering
# \caption{All-to-All performance under different sparsity levels, port budgets, and traffic volumes}
# \label{tab:smesh_sweetspot_unified}
# \begin{tabular}{l c c c c c}
# \toprule
# Traffic & Topology & Time & Norm. & Nodes & Ports \\
# \midrule
# \multirow{8}{*}{\makecell{7680\\MB}}
# & Clos-N8 & 20.02 ms & 1.00 & 64 & 6 \\
# & FullMesh-N7 & 20.02 ms & 1.00 & 56 & 6 \\
# & SMesh-N8 & 22.80 ms & 0.88 & 64 & 6 \\
# & SMesh-N13 & 33.07 ms & 0.61 & 104 & 6 \\\cmidrule(lr){2-6}
# & Clos-N16 & 20.03 ms & 1.00 & 128 & 12 \\
# & SMesh-N16 & 25.49 ms & 0.79 & 128 & 12 \\
# & SMesh-N31 & 33.95 ms & 0.59 & 248 & 12 \\
# & SMesh-N33 & 84.96 ms & 0.24 & 256 & 12 \\
# \midrule
# \multirow{8}{*}{\makecell{60\\MB}}
# & Clos-N8 & 100.51 $\mu$s & 1.00 & 64 & 6 \\
# & FullMesh-N7 & 100.01 $\mu$s & 1.00 & 56 & 6 \\
# & SMesh-N8 & 111.29 $\mu$s & 0.90 & 64 & 6 \\
# & SMesh-N13 & 139.13 $\mu$s & 0.72 & 104 & 6 \\\cmidrule(lr){2-6}
# & Clos-N16 & 101.20 $\mu$s & 0.99 & 128 & 12 \\
# & SMesh-N16 & 123.12 $\mu$s & 0.81 & 128 & 12 \\
# & SMesh-N31 & 155.76 $\mu$s & 0.64 & 248 & 12 \\
# & SMesh-N33 & 344.66 $\mu$s & 0.29 & 256 & 12 \\
# \midrule
# \multirow{8}{*}{\makecell{300\\KB}}
# & Clos-N8 & 23.94 $\mu$s & 0.95 & 64 & 6 \\
# & FullMesh-N7 & 22.76 $\mu$s & 1.00 & 56 & 6 \\
# & SMesh-N8 & 23.06 $\mu$s & 0.99 & 64 & 6 \\
# & SMesh-N13 & 23.51 $\mu$s & 0.97 & 104 & 6 \\\cmidrule(lr){2-6}
# & Clos-N16 & 24.32 $\mu$s & 0.94 & 128 & 12 \\
# & SMesh-N16 & 23.46 $\mu$s & 0.97 & 128 & 12 \\
# & SMesh-N31 & 24.52 $\mu$s & 0.93 & 248 & 12 \\
# & SMesh-N33 & 26.77 $\mu$s & 0.85 & 256 & 12 \\
# \bottomrule
# \end{tabular}
# \end{table}

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import rcParams

# 1. 数据准备
data_list = [
    ["7680MB", "Clos-N8",     1.00, 6,  20.02, "ms"],
    ["7680MB", "FullMesh-N8", 1.00, 6,  20.02, "ms"],
    ["7680MB", "SMesh-N8",    0.88, 6,  22.80, "ms"],
    ["7680MB", "SMesh-N13",   0.61, 6,  33.07, "ms"],
    ["7680MB", "Clos-N16",    1.00, 12, 20.03, "ms"],
    ["7680MB", "SMesh-N16",   0.79, 12, 25.49, "ms"],
    ["7680MB", "SMesh-N31",   0.59, 12, 33.95, "ms"],

    ["60MB", "Clos-N8",     1.00, 6,  100.51, "µs"],
    ["60MB", "FullMesh-N8", 1.00, 6,  100.01, "µs"],
    ["60MB", "SMesh-N8",    0.90, 6,  111.29, "µs"],
    ["60MB", "SMesh-N13",   0.72, 6,  139.13, "µs"],
    ["60MB", "Clos-N16",    0.99, 12, 101.20, "µs"],
    ["60MB", "SMesh-N16",   0.81, 12, 123.12, "µs"],
    ["60MB", "SMesh-N31",   0.64, 12, 155.76, "µs"],

    ["300KB", "Clos-N8",     0.95, 6,  23.94, "µs"],
    ["300KB", "FullMesh-N8", 1.00, 6,  22.76, "µs"],
    ["300KB", "SMesh-N8",    0.99, 6,  23.06, "µs"],
    ["300KB", "SMesh-N13",   0.97, 6,  23.51, "µs"],
    ["300KB", "Clos-N16",    0.94, 12, 24.32, "µs"],
    ["300KB", "SMesh-N16",   0.97, 12, 23.46, "µs"],
    ["300KB", "SMesh-N31",   0.93, 12, 24.52, "µs"],
]

df = pd.DataFrame(data_list, columns=['Traffic', 'Topology', 'NormPerf', 'Ports', 'Time', 'TimeUnit'])

# 2. 端口效率计算
def calc_eff(row):
    actual_p = row['Ports'] * 2 if 'Clos' in row['Topology'] else row['Ports']
    return row['NormPerf'] / actual_p

df['RawEff'] = df.apply(calc_eff, axis=1)
df['FinalScore'] = df.groupby('Traffic')['RawEff'].transform(lambda x: x / x.max())

# 3. 绘图配置与美化 (统一字体)
FONT_TITLE, FONT_AXIS, FONT_TICK = 14, 12, 12
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="white")

# 莫兰迪色系搭配
color_map = {
    'Clos': '#E6B85C',      # 暖黄色
    'FullMesh': '#6AB187',  # 灰绿色
    'SMesh': '#517FA4'       # 雾霾蓝
}

def get_color(name):
    for k, v in color_map.items():
        if k in name: return v
    return '#B0BEC5'

df['Color'] = df['Topology'].apply(get_color)

# 4. 执行绘图: 三列三行, 优化布局
traffic_types = ["7680MB", "60MB", "300KB"]
# 调整 width_ratios, 左右两列 bar chart 宽度一致
fig, axes = plt.subplots(3, 3, figsize=(7.5, 8.2), sharey='row',
                         gridspec_kw={'wspace': 0.05, 'hspace': 0.3, 'width_ratios': [1, 0.4, 1]})

for i, traffic in enumerate(traffic_types):
    ax_time = axes[i, 0]   
    ax_mid  = axes[i, 1]   
    ax_eff  = axes[i, 2]   
    
    subset = df[df['Traffic'] == traffic].copy()
    y_pos = np.arange(len(subset))

    # --- 左列: Time (镜像翻转) ---
    bars_t = ax_time.barh(y_pos, subset['Time'],
                          color=subset['Color'], edgecolor='#000',
                          linewidth=0.5, height=0.6, zorder=3)
    time_unit = subset['TimeUnit'].iloc[0]
    
    # 设定 X 轴范围并镜像翻转
    if i == 0:   time_lim = 48
    elif i == 1: time_lim = 220
    else:        time_lim = 40
    
    ax_time.set_xlim(0, time_lim)
    ax_time.invert_yaxis() # 保持拓扑顺序一致
    ax_time.invert_xaxis() # 实现“背靠背”效果的核心：将 X 轴左右翻转

    ax_time.axhline(2.5, color='#000', linewidth=1.0, linestyle='--', alpha=0.8, zorder=1)

    # 标注数值 (调整对齐方式和位置)
    max_time_in_subset = subset['Time'].max()
    for bar in bars_t:
        width = bar.get_width()
        # 计算偏离柱子末端的距离 (基于当前数据刻度)
        offset_val = time_lim * 0.03 
        # 在镜像坐标系中，柱子从右向左延伸，末端在视觉左侧。
        # 标注位置需要计算在 width 加上一个偏移量 (数据值变大，视觉更偏左)。
        # 并使用 right 对齐，让文字主体位于计算点的左侧，从而位于柱子外侧。
        ax_time.text(width + offset_val, bar.get_y() + bar.get_height()/2, 
                     f'{width:.2f} {time_unit}', ha='right', va='center', fontsize=FONT_TICK, color='#000')
    
    ax_time.tick_params(axis='y', which='both', left=False, labelleft=False)
    ax_time.tick_params(axis='x', labelsize=FONT_TICK, colors='#000')
    for spine in ax_time.spines.values():
        spine.set_color('#000')
        
    # --- 中间列: Topology ---
    for spine in ax_mid.spines.values():
        spine.set_visible(False)
    ax_mid.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    ax_mid.set_xlim(0, 1)
    
    # 绘制 Topology 文本
    for y, topo in zip(y_pos, subset['Topology']):
        ax_mid.text(0.5, y, topo, ha='center', va='center', fontsize=FONT_AXIS, color='#000')
        
    # 绘制 Traffic 标签
    ax_mid.text(0.5, len(subset) + 0.3, f'Traffic: {traffic}', 
                 ha='center', va='center', fontsize=FONT_AXIS, fontweight='bold', 
                 color='#000', bbox=dict(facecolor='#FFE0E0', edgecolor='none', pad=4.0, alpha=0.9))

    # --- 右列: Port-Normalized Score ---
    bars_e = ax_eff.barh(y_pos, subset['FinalScore'],
                          color=subset['Color'], edgecolor='#000',
                          linewidth=0.5, height=0.6, zorder=3)
    ax_eff.set_xlim(0, 1.25)
    ax_eff.set_xticks([0, 0.5, 1])
    ax_eff.axhline(2.5, color='#000', linewidth=1.0, linestyle='--', alpha=0.8, zorder=1)

    # 标注数值
    for bar in bars_e:
        width = bar.get_width()
        ax_eff.text(width + 0.03, bar.get_y() + bar.get_height()/2, 
                    f'{width:.2f}', ha='left', va='center', fontsize=FONT_TICK, color='#000')
    
    ax_eff.tick_params(axis='y', which='both', left=False, labelleft=False)
    ax_eff.tick_params(axis='x', labelsize=FONT_TICK, colors='#000')
    for spine in ax_eff.spines.values():
        spine.set_color('#000')

    # 处理列标题 (仅首行设置)
    if i == 0:
        ax_time.set_title('All-to-All Time', fontsize=FONT_TITLE, pad=12, fontweight='bold', color='#000')
        ax_mid.set_title('Topology', fontsize=FONT_TITLE, pad=12, fontweight='bold', color='#000')
        ax_eff.set_title('Port-Normalized\nEfficiency Score', fontsize=FONT_TITLE, pad=12, fontweight='bold', color='#000')
    
    # X 轴说明仅尾行设置
    if i == len(traffic_types) - 1:
        ax_time.set_xlabel('Time', fontsize=FONT_AXIS, labelpad=10, color='#000')
        ax_eff.set_xlabel('Efficiency Score', fontsize=FONT_AXIS, labelpad=10, color='#000')

# 整体微调
fig.subplots_adjust(left=0.03, right=0.97, top=0.88, bottom=0.08)

# 子图标号 (a)-(f), 按行排序, (a,c,e)在左, (b,d,f)在右
label_data = []
char_code = ord('a')
for row_idx in range(3):
    # Time 列 (左)
    label_data.append((axes[row_idx, 0], chr(char_code), 'left', 0.02))
    char_code += 1
    # Efficiency 列 (右)
    label_data.append((axes[row_idx, 2], chr(char_code), 'right', 0.98))
    char_code += 1

for ax, label, align, x_pos in label_data:
     ax.text(x_pos, 0.98, f'({label})', transform=ax.transAxes,
            fontsize=FONT_TITLE, fontweight='bold', va='top', ha=align, zorder=10)

plt.savefig('SparseMesh/fig/table2.png', dpi=300, facecolor='white', bbox_inches='tight', pad_inches=0.1)
plt.close()
```

File: SparseMesh\SMesh.py
```py
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as transforms
import numpy as np
import networkx as nx
import os

# 样式常量配置
COLOR_RECT_FILL = '#8FAADC'
COLOR_RECT_EDGE = '#000000'
COLOR_LINE = '#C00000'
LINE_WIDTH = 1.5

def generate_dual_outputs(N, K, R=12, rect_w=4.0, rect_h=1.2):
    # 1. 拓扑合法性检查
    if K >= N: K = N - 1
    if (N * K) % 2 != 0:
        print(f"Warning: N={N}, K={K} 无法构成正则图。已调整 K 为 {K-1}")
        K -= 1

    # 2. 均匀分布拓扑构建
    G = nx.Graph()
    G.add_nodes_from(range(N))
    max_s = N // 2
    num_pairs = K // 2
    
    if num_pairs > 0:
        potential_steps = list(range(1, max_s + (1 if K % 2 == 0 else 0)))
        potential_steps.reverse()
        indices = np.linspace(0, len(potential_steps) - 1, num_pairs, dtype=int)
        steps = [potential_steps[i] for i in set(indices)]
        for i in range(N):
            for s in steps:
                G.add_edge(i, (i + s) % N)
            if K % 2 != 0:
                G.add_edge(i, (i + N // 2) % N)

    # 3. 预计算连接点坐标
    # 矩形长边面向圆心，连接点在内侧长边中点
    conn_points = []
    node_centers = []
    for i in range(N):
        theta = 2 * np.pi * i / N
        cx, cy = R * np.cos(theta), R * np.sin(theta)
        r_inner = R - (rect_h / 2)
        px, py = r_inner * np.cos(theta), r_inner * np.sin(theta)
        conn_points.append((px, py))
        node_centers.append((cx, cy, np.degrees(theta)))

    # --- 输出 1: 标准拓扑图 (带矩形) ---
    render_and_save(N, K, G, conn_points, node_centers, rect_w, rect_h, 
                    show_rects=True, filename='topology_full.png', R=R)

    # --- 输出 2: 纯连线图 (无矩形，无装饰) ---
    render_and_save(N, K, G, conn_points, node_centers, rect_w, rect_h, 
                    show_rects=False, filename='topology_lines_only.png', R=R)

def render_and_save(N, K, G, conn_points, node_centers, rect_w, rect_h, show_rects, filename, R):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect('equal')
    ax.axis('off')

    # 绘制连线
    for u, v in G.edges():
        p1, p2 = conn_points[u], conn_points[v]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 
                color=COLOR_LINE, lw=LINE_WIDTH, alpha=0.9, zorder=1)

    # 绘制矩形 (仅在 show_rects 为 True 时)
    if show_rects:
        for i in range(N):
            cx, cy, deg = node_centers[i]
            angle = deg + 90
            rect = patches.FancyBboxPatch(
                (-rect_w/2, -rect_h/2), rect_w, rect_h,
                boxstyle='round,pad=0.1',
                facecolor=COLOR_RECT_FILL, edgecolor=COLOR_RECT_EDGE,
                linewidth=1.5, zorder=3
            )
            t = transforms.Affine2D().rotate_deg(angle).translate(cx, cy) + ax.transData
            rect.set_transform(t)
            ax.add_patch(rect)
        
        # 计算跳数作为标题
        try:
            if nx.is_connected(G):
                dia = nx.diameter(G)
                plt.title(f"N={N} K={K} | Max Hops: {dia}", fontsize=14, pad=20)
        except:
            pass

    # 统一视图范围
    limit = R + rect_w
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)

    # 保存无白边图片
    plt.savefig(filename, format='png', bbox_inches='tight', pad_inches=0, dpi=300, transparent=not show_rects)
    plt.close(fig)
    print(f"已导出: {filename}")

if __name__ == "__main__":
    # N=数量, K=度数
    generate_dual_outputs(N=16, K=7)
```

File: SparseMesh\Table 3_V2.py
```py
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
```

File: SparseMesh\Table 3.py
```py
# Figure 3: Bandwidth cost efficiency comparison across scales (64P / 128P)
# 数据来源: TODO.md 中的 Table: cost_efficiency_combined
# 绘制BW Cost归一化效率，64P和128P分两张子图

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

# # 原始数据
# data_64p = [
#     ["SMesh-N8\n(default)", 22.16, 100.0,  1_379_840, 7.24],
#     ["SMesh-N8\n(opt)",     22.16, 149.98, 1_379_840, 7.24],
#     ["FullMesh-N8",              21.99, 200.03, 1_387_520, 7.75],
#     ["CLOS-N8",                  23.10, 171.43, 1_470_920, 12.98],
#     ["Torus-N8",                 22.24, 100.0,  1_379_840, 7.24],
# ]

# data_128p = [
#     ["SMesh-N16\n(default)", 22.19, 40.0,  2_775_040, 7.75],
#     ["SMesh-N16\n(SP)",      22.19, 55.05, 2_775_040, 7.75],
#     ["SMesh-N16\n(opt)",     22.17, 58.52, 2_775_040, 7.75],
#     ["CLOS-N16",                  23.14, 80.0,  2_941_840, 12.98],
#     ["Torus-N16",              22.35, 50.0,  2_790_400, 8.26],
# ]

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
    df['BWCostRaw'] = df['BW'] / df['NetCost']
    df['BWCostNorm'] = df['BWCostRaw'] / df['BWCostRaw'].max()
    # 计算 Latency Cost Eff: (1/Latency) / NetCost, 再归一化
    df['LatCostRaw'] = (1.0 / df['Latency']) / df['NetCost']
    df['LatCostNorm'] = df['LatCostRaw'] / df['LatCostRaw'].max()
    return df

df_64 = build_df(data_64p, 64)
df_128 = build_df(data_128p, 128)

# 2. 绘图配置 (统一字体)
FONT_TITLE, FONT_AXIS, FONT_TICK = 14, 11, 11
plt.rcParams['font.sans-serif'] = ['Arial']
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

fig, axes = plt.subplots(2, 3, figsize=(7.5, 9), sharey='row',
                         gridspec_kw={'wspace': 0.05, 'hspace': 0.4, 
                                      'width_ratios': [1, 0.5, 1]})

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
        # 处理换行使其更美观
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
    ax_lat.set_xlim(0, 1.35)
    ax_lat.set_xticks([0, 0.5, 1.0])

    # 标注数值 (右列，文字在柱子右侧)
    for bar in bars_lat:
        w = bar.get_width()
        ax_lat.text(w + 0.04, bar.get_y() + bar.get_height() / 2,
                    f'{w:.2f}', ha='left', va='center', fontsize=FONT_TICK, color='#000')

    ax_lat.tick_params(axis='y', which='both', left=False, labelleft=False)
    ax_lat.tick_params(axis='x', labelsize=FONT_TICK, colors='#000')
    for sp in ax_lat.spines.values():
        sp.set_color('#000')

    # 列标题 (仅首行)
    
    ax_bw.set_title('BW Cost Efficiency', fontsize=FONT_TITLE, pad=20, fontweight='bold', color='#000')
    ax_mid.set_title('Topology', fontsize=FONT_TITLE, pad=20, fontweight='bold', color='#000')
    ax_lat.set_title('Latency Cost Efficiency', fontsize=FONT_TITLE, pad=20, fontweight='bold', color='#000')

    # X 轴说明 (仅末行)
    
    ax_bw.set_xlabel('Normalized Performance', fontsize=FONT_AXIS, labelpad=10, color='#000')
    ax_lat.set_xlabel('Normalized Performance', fontsize=FONT_AXIS, labelpad=10, color='#000')

# 整体布局微调
fig.subplots_adjust(left=0.05, right=0.95, top=0.85, bottom=0.12)

# 子图标号 (a)-(d)
# 顺序：左上(a), 左下(b), 右上(c), 右下(d)
subplot_axes = [axes[0, 0], axes[1, 0], axes[0, 2], axes[1, 2]]
for idx, ax in enumerate(subplot_axes):
    label = f'({chr(ord("a") + idx)})'
    # 对于镜像的左列，标号放在左侧；对于常规的右列，标号放在右侧
    x_pos = 0.05 if ax in [axes[0, 0], axes[1, 0]] else 0.98
    align = 'left' if ax in [axes[0, 0], axes[1, 0]] else 'right'
    ax.text(x_pos, 0.96, label, transform=ax.transAxes,
            fontsize=FONT_TITLE, fontweight='bold', va='top', ha=align, zorder=10)

plt.savefig('SparseMesh/fig/table3.png', dpi=300, facecolor='white', bbox_inches='tight', pad_inches=0.1)
plt.show()
```

File: SparseMesh\Table 4.py
```py
"""
图4: Training throughput and cost efficiency comparison
64P, 128P, 256P 下的归一化 Cost Efficiency。Clos 和 SM 各自最优策略。
使用竖向柱状图，一张图内展示。
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 数据: Scale, Topology, Norm. Throughput, Cost Efficiency, Net Cost %
data = [
    ("64P", "Clos", 1.000, 1.000),
    ("64P", "SM", 0.933, 1.027),
    ("128P", "Clos", 1.009, 1.009),
    ("128P", "SM", 0.932, 1.027),
    ("256P", "Clos", 1.014, 1.014),
    ("256P", "SM", 0.933, 1.028),
]

# 统一字体：标题14 坐标轴11
FONT_TITLE, FONT_AXIS, FONT_TICK = 14, 11, 10
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="white")

scales = ["64P", "128P", "256P"]
x = np.arange(len(scales))
width = 0.35

clos_eff = [1.000, 1.009, 1.014]
sm_eff = [1.027, 1.027, 1.028]

fig, ax = plt.subplots(figsize=(7, 4.5))
bars1 = ax.bar(x - width/2, clos_eff, width, label='Clos', color='#E6B85C', edgecolor='#000', linewidth=1.0)
bars2 = ax.bar(x + width/2, sm_eff, width, label='SM', color='#517FA4', edgecolor='#000', linewidth=1.0)

ax.set_ylabel('Cost Efficiency (Normalized)', fontsize=FONT_AXIS, color='#000')
ax.set_xlabel('Scale', fontsize=FONT_AXIS, color='#000')
ax.set_title('Training Cost Efficiency Comparison (2048-NPU Cluster)', fontsize=FONT_TITLE, fontweight='bold', color='#000')
ax.set_xticks(x)
ax.set_xticklabels(scales, fontsize=FONT_TICK)
ax.tick_params(axis='y', labelsize=FONT_TICK)
ax.legend(loc='upper right', fontsize=FONT_TICK)
ax.set_ylim(0, 1.4)
for spine in ax.spines.values():
    spine.set_color('#000')

# 标注数值
for bar in bars1:
    height = bar.get_height()
    ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 4), textcoords="offset points", ha='center', va='bottom',
                fontsize=FONT_TICK, color='#000', fontweight='bold')
for bar in bars2:
    height = bar.get_height()
    ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 4), textcoords="offset points", ha='center', va='bottom',
                fontsize=FONT_TICK, color='#000', fontweight='bold')

fig.tight_layout()
plt.savefig('SparseMesh/fig/table4.png', dpi=300, facecolor='white', bbox_inches='tight', pad_inches=0)
plt.close()

```

