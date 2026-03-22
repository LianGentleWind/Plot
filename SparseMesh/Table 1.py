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
