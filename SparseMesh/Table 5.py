"""
图5: Inference performance comparison
3种rack卡数(64P/128P/256P) × 2种序列长度(4096/8192) × 3种TPOT(20/50/100) = 18个自变量
绘制 Clos 和 SM 各自的 Throughput 值。
"""
import matplotlib.pyplot as plt
import matplotlib.transforms as mtrans
import numpy as np
import seaborn as sns

# 数据: Scale, SeqLen, TPOT, Clos_Thro, SM_Thro
data = [
    # 64P
    (64, 4096, 20, 64, 64),
    (64, 4096, 50, 204, 202),
    (64, 4096, 100, 417, 413),
    (64, 8192, 20, 37, 37),
    (64, 8192, 50, 130, 129),
    (64, 8192, 100, 270, 269),
    # 128P
    (128, 4096, 20, 67, 66),
    (128, 4096, 50, 202, 199),
    (128, 4096, 100, 415, 407),
    (128, 8192, 20, 42, 41),
    (128, 8192, 50, 129, 128),
    (128, 8192, 100, 269, 266),
    # 256P
    (256, 4096, 20, 66, 56),
    (256, 4096, 50, 202, 173),
    (256, 4096, 100, 414, 355),
    (256, 8192, 20, 42, 38),
    (256, 8192, 50, 129, 116),
    (256, 8192, 100, 269, 243),
]

# 统一字体：标题14 坐标轴11；组标签(4096/8192)用更大字号
FONT_TITLE, FONT_AXIS, FONT_TICK = 14, 11, 11
FONT_GROUP = 16  # 4096/8192 组标记字号（更大）
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="white")

# 按 Scale 分三组子图，纵向排列，横排柱状图(barh)
fig, axes = plt.subplots(3, 1, figsize=(7, 7), sharex=True)
scales = [64, 128, 256]
bar_height = 0.35

for ax_idx, scale in enumerate(scales):
    ax = axes[ax_idx]
    subset = [d for d in data if d[0] == scale]

    # 6个配置: Seq4096×3 TPOT + Seq8192×3 TPOT
    y = np.arange(6)
    clos_vals = [d[3] for d in subset]
    sm_vals = [d[4] for d in subset]

    bars1 = ax.barh(y - bar_height/2, clos_vals, bar_height, label='Clos', color='#E6B85C', edgecolor='#000', linewidth=1.0)
    bars2 = ax.barh(y + bar_height/2, sm_vals, bar_height, label='SM', color='#517FA4', edgecolor='#000', linewidth=1.0)

    # 纵坐标只写 TPOT；4096/8192 单独用大字号写在对应三行中间一行的左边
    labels = [f'TPOT {d[2]}' for d in subset]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=FONT_TICK)
    # 组间分隔线（y=2.5 介于 4096 与 8192 之间），颜色加深
    ax.axhline(2.5, color='#000', linewidth=1.0, linestyle='-', alpha=0.8, zorder=1)
    ax.set_xlabel('Throughput' if ax_idx == 2 else '', fontsize=FONT_AXIS)
    ax.tick_params(axis='x', labelsize=FONT_TICK)
    ax.set_title(f'{scale}P', fontsize=FONT_TITLE, fontweight='bold')
    ax.legend(loc='lower right', fontsize=FONT_TICK)
    ax.invert_yaxis()
    for spine in ax.spines.values():
        spine.set_color('#000')

fig.suptitle('Inference Throughput: Clos vs SM', fontsize=FONT_TITLE, fontweight='bold', y=1.02)
fig.tight_layout()
fig.subplots_adjust(left=0.22)  # 左侧留出更多空间给 4096/8192 组标记

# 绘制完成后，在每组三行TPOT的中间一行左侧添加 4096/8192 组标记（更大字号，更靠左）
for ax_idx, ax in enumerate(axes):
    # 混合变换：x 用 axes 坐标（左侧），y 用 data 坐标
    trans = mtrans.blended_transform_factory(ax.transAxes, ax.transData)
    # 4096 组中间行 y=1，8192 组中间行 y=4；x=-0.18 更靠左
    ax.text(-0.18, 1, '4096', transform=trans, fontsize=FONT_GROUP, fontweight='bold',
            va='center', ha='right', color='#000', zorder=10, clip_on=False)
    ax.text(-0.18, 4, '8192', transform=trans, fontsize=FONT_GROUP, fontweight='bold',
            va='center', ha='right', color='#000', zorder=10, clip_on=False)

# 子图标号 (a)-(c)，右上角
for idx, ax in enumerate(axes):
    ax.text(0.98, 0.98, f'({chr(ord("a") + idx)})', transform=ax.transAxes,
            fontsize=FONT_TITLE, fontweight='bold', va='top', ha='right', zorder=10)

plt.savefig('fig/table5.png', dpi=300, facecolor='white', bbox_inches='tight', pad_inches=0)
plt.close()
