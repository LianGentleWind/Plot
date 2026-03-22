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

