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
