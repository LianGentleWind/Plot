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

# 1. 数据准备
data_list = [
    ["7680MB", "Clos-N8",     1.00, 6,  20.02, "ms"],
    ["7680MB", "Clos-N16",    1.00, 12, 20.03, "ms"],
    ["7680MB", "FullMesh-N8", 1.00, 6,  20.02, "ms"],
    ["7680MB", "SMesh-N8",    0.88, 6,  22.80, "ms"],
    ["7680MB", "SMesh-N13",   0.61, 6,  33.07, "ms"],
    ["7680MB", "SMesh-N16",   0.79, 12, 25.49, "ms"],
    ["7680MB", "SMesh-N31",   0.59, 12, 33.95, "ms"],

    ["60MB", "Clos-N8",     1.00, 6,  100.51, "µs"],
    ["60MB", "Clos-N16",    0.99, 12, 101.20, "µs"],
    ["60MB", "FullMesh-N8", 1.00, 6,  100.01, "µs"],
    ["60MB", "SMesh-N8",    0.90, 6,  111.29, "µs"],
    ["60MB", "SMesh-N13",   0.72, 6,  139.13, "µs"],
    ["60MB", "SMesh-N16",   0.81, 12, 123.12, "µs"],
    ["60MB", "SMesh-N31",   0.64, 12, 155.76, "µs"],

    ["300KB", "Clos-N8",     0.95, 6,  23.94, "µs"],
    ["300KB", "Clos-N16",    0.94, 12, 24.32, "µs"],
    ["300KB", "FullMesh-N8", 1.00, 6,  22.76, "µs"],
    ["300KB", "SMesh-N8",    0.99, 6,  23.06, "µs"],
    ["300KB", "SMesh-N13",   0.97, 6,  23.51, "µs"],
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

# 3. 绘图配置与美化（统一字体：标题14 坐标轴11）
FONT_TITLE, FONT_AXIS, FONT_TICK = 14, 11, 11
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="white")

# 使用更加高级/低饱和的莫兰迪色系搭配
color_map = {
    'Clos': '#E6B85C',      # 暖黄色
    'FullMesh': '#6AB187',  # 灰绿色
    'SMesh': '#517FA4'      # 雾霾蓝
}

def get_color(name):
    for k, v in color_map.items():
        if k in name: return v
    return '#B0BEC5'

df['Color'] = df['Topology'].apply(get_color)

# 4. 执行绘图: 三列三行
traffic_types = ["7680MB", "60MB", "300KB"]
fig, axes = plt.subplots(3, 3, figsize=(7, 8), sharey='row',
                         gridspec_kw={'wspace': 0.05, 'hspace': 0.28, 'width_ratios': [1, 0.45, 1]})

for i, traffic in enumerate(traffic_types):
    ax_time = axes[i, 0]   
    ax_mid  = axes[i, 1]   
    ax_eff  = axes[i, 2]   
    
    subset = df[df['Traffic'] == traffic].copy()
    y_pos = np.arange(len(subset))

    # --- 左列: Time ---
    bars_t = ax_time.barh(y_pos, subset['Time'],
                          color=subset['Color'], edgecolor='#000',
                          linewidth=1.0, height=0.6, zorder=3)
    time_unit = subset['TimeUnit'].iloc[0]
    # 图(c)即第三行左列，xlim 设为 0, 35
    if i == 0:
        ax_time.set_xlim(0, 48)

    elif i == 1:
        ax_time.set_xlim(0, 220)

    elif i == 2:
        ax_time.set_xlim(0, 40)
    else:
        ax_time.set_xlim(0, subset['Time'].max() * 1.35)
    ax_time.invert_yaxis()
    
    # 标注数值
    for bar in bars_t:
        width = bar.get_width()
        ax_time.text(width + subset['Time'].max() * 0.03, bar.get_y() + bar.get_height()/2, 
                     f'{width:.2f} {time_unit}', ha='left', va='center', fontsize=FONT_TICK, color='#000')
    
    ax_time.tick_params(axis='y', which='both', left=False, labelleft=False)
    ax_time.tick_params(axis='x', labelsize=FONT_TICK, colors='#000')
    
    for spine in ax_time.spines.values():
        spine.set_color('#000')
        
    # --- 中间列: Topology (以及Traffic) ---
    for spine in ax_mid.spines.values():
        spine.set_visible(False)
    ax_mid.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    ax_mid.set_xlim(0, 1)
    
    # 绘制 Topology 文本
    for y, topo in zip(y_pos, subset['Topology']):
        ax_mid.text(0.5, y, topo, ha='center', va='center', fontsize=FONT_AXIS, color='#000')
        
    # 将 Traffic 标签移动到中间的 ax_mid 内的下方
    ax_mid.text(0.5, len(subset) + 0.3, f'Traffic: {traffic}', 
                 ha='center', va='center', fontsize=FONT_AXIS, fontweight='bold', 
                 color='#000', bbox=dict(facecolor='#FFE0E0', edgecolor='none', pad=4.0, alpha=0.9))

    # --- 右列: Port-Normalized Score ---
    bars_e = ax_eff.barh(y_pos, subset['FinalScore'],
                         color=subset['Color'], edgecolor='#000',
                         linewidth=1.0, height=0.6, zorder=3)
    ax_eff.set_xlim(0, 1.25)
    ax_eff.set_xticks([0, 0.5, 1])
    
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

# 整体微调，去掉 constrained_layout() 导致的一些边距问题
fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.08)

# 子图标号 (a)-(f)，仅条形图子图，跳过中间 Topology 列
# 左列 Time、右列 Efficiency 各有3个，共6个子图
subplot_axes = [axes[i, 0] for i in range(3)] + [axes[i, 2] for i in range(3)]
for idx, ax in enumerate(subplot_axes):
    ax.text(0.98, 0.98, f'({chr(ord("a") + idx)})', transform=ax.transAxes,
            fontsize=FONT_TITLE, fontweight='bold', va='top', ha='right', zorder=10)

plt.savefig('fig/table2.png', dpi=300, facecolor='white', bbox_inches='tight', pad_inches=0)
plt.close()