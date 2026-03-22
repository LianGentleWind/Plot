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
