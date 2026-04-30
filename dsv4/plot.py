import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch
import os
import re

# ------------------------------
# 全局样式设置
# ------------------------------
for font_path in ['/mnt/c/Windows/Fonts/msyh.ttc', '/mnt/c/Windows/Fonts/simhei.ttf']:
    if os.path.exists(font_path):
        font_manager.fontManager.addfont(font_path)

font_candidates = [
    'Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC', 'Source Han Sans SC',
    'WenQuanYi Micro Hei', 'PingFang SC', 'Heiti SC', 'Arial Unicode MS',
]
available_fonts = {f.name for f in font_manager.fontManager.ttflist}
plt.rcParams['font.sans-serif'] = [
    next((font for font in font_candidates if font in available_fonts), 'DejaVu Sans')
]
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15

# ------------------------------
# 读取数据（请改为你的实际路径）
# ------------------------------
csv_path = "./data/summary_best.csv"
df = pd.read_csv(csv_path)

# 如需使用 exposed 比例，取消下面的注释
# df['compute_time_ratio'] = df['exposed_compute_ratio']
# df['memory_time_ratio'] = df['exposed_memory_ratio']
# df['comm_time_ratio'] = df['exposed_comm_ratio']

# 比例字段转百分数（若原数据是小数则乘100）
ratio_cols = ['compute_time_ratio', 'memory_time_ratio', 'comm_time_ratio']
for col in ratio_cols:
    has_percent = df[col].astype(str).str.contains('%', regex=False).any()
    df[col] = pd.to_numeric(df[col].astype(str).str.rstrip('%'), errors='coerce')
    if not has_percent and df[col].max() < 1.1:
        df[col] = df[col] * 100

# 三部分总和不足 100% 时，按原比例归一化到 100%
ratio_sum = df[ratio_cols].sum(axis=1)
normalize_mask = (ratio_sum > 0) & (ratio_sum < 100)
df.loc[normalize_mask, ratio_cols] = (
    df.loc[normalize_mask, ratio_cols]
    .div(ratio_sum[normalize_mask], axis=0)
    .mul(100)
)


# 序列长度格式化（按1024进制，整数显示）
def format_seq_len(length):
    if length >= 1024 * 1024:
        return f"{int(length / (1024 * 1024))}M"
    elif length >= 1024:
        return f"{int(length / 1024)}K"
    else:
        return str(int(length))


def trailing_p_size(value):
    match = re.search(r'_(\d+)P$', str(value))
    return int(match.group(1)) if match else np.nan


def hardware_type(value):
    return re.sub(r'_\d+P$', '', str(value))


df['supernode_size'] = df['chip'].map(trailing_p_size)
df['supernode_size'] = df['supernode_size'].fillna(df['hardware_name'].map(trailing_p_size))
df = df.dropna(subset=['supernode_size']).copy()
df['supernode_size'] = df['supernode_size'].astype(int)
df['hardware_type'] = df['hardware_name'].map(hardware_type)
all_supernode_sizes = sorted(df['supernode_size'].unique())


# 颜色映射
hardware_types = df['hardware_type'].dropna().unique()
base_colors = ['#4c72b0', '#ff7f0e', '#55a868']
hardware_color_map = {hw: base_colors[i % len(base_colors)] for i, hw in enumerate(hardware_types)}

# 分组绘图
group_cols = ['model', 'latency_constraint_ms', 'seq_len']
grouped = df.groupby(group_cols)

for (model, tpot, seq_len), group in grouped:
    model_name = model[0].upper() + model[1:]
    seq_str = format_seq_len(seq_len)
    title = f"{model_name}@KV={seq_str},TPOT<={tpot}ms"

    supernode_sizes = all_supernode_sizes
    n_sizes = len(supernode_sizes)

    fig, ax = plt.subplots(figsize=(max(9, n_sizes * 0.45 + 2), 6))

    x = np.arange(n_sizes)
    present_hardware_types = group['hardware_type'].dropna().unique()
    num_hardware_types = len(present_hardware_types)

    # 固定柱宽：不同图片或缺少硬件选型时，单根柱子的宽度保持一致
    width = 0.25

    # 根据柱宽动态计算柱内数字字号，确保始终显示
    base_fontsize = 9
    min_fontsize = 6
    # 当柱子变窄时字号按比例缩小
    text_fontsize = max(min_fontsize, base_fontsize * width / 0.15)

    for i, hw_type in enumerate(present_hardware_types):
        hw_data = group[group['hardware_type'] == hw_type]
        hw_data = hw_data.groupby('supernode_size')[ratio_cols].first().reindex(supernode_sizes)
        compute = hw_data['compute_time_ratio'].fillna(0).values
        memory = hw_data['memory_time_ratio'].fillna(0).values
        comm = hw_data['comm_time_ratio'].fillna(0).values

        offset = (i - (num_hardware_types - 1) / 2) * width

        ax.bar(x + offset, compute, width,
               color=hardware_color_map[hw_type], alpha=1.0, edgecolor='white')
        ax.bar(x + offset, memory, width, bottom=compute,
               color=hardware_color_map[hw_type], alpha=0.7, edgecolor='white')
        ax.bar(x + offset, comm, width, bottom=compute + memory,
               color=hardware_color_map[hw_type], alpha=0.4, edgecolor='white')

        # 始终标注数字，根据宽度自适应字号
        for j in range(n_sizes):
            c_val, m_val, k_val = compute[j], memory[j], comm[j]
            center_c = c_val / 2
            center_m = c_val + m_val / 2
            center_k = c_val + m_val + k_val / 2

            if c_val > 0:
                ax.text(x[j] + offset, center_c, f'{c_val:.0f}',
                        ha='center', va='center', fontsize=text_fontsize,
                        color='white', fontweight='bold')
            if m_val > 0:
                ax.text(x[j] + offset, center_m, f'{m_val:.0f}',
                        ha='center', va='center', fontsize=text_fontsize,
                        color='white', fontweight='bold')
            if k_val > 0:
                ax.text(x[j] + offset, center_k, f'{k_val:.0f}',
                        ha='center', va='center', fontsize=text_fontsize,
                        color='black', fontweight='bold')

    # 坐标轴
    ax.set_xticks(x)
    ax.set_xticklabels([f'{size}P' for size in supernode_sizes], fontsize=14)
    ax.set_xlabel('超节点规模', fontsize=15)
    ax.set_ylabel('时间占比 (%)', fontsize=15)
    ax.set_ylim(0, 120)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.set_title(title, fontsize=17, pad=15)

    # ---- 图例 ----
    hardware_legend = [Patch(facecolor=hardware_color_map[hw], edgecolor='white',
                             label=hw) for hw in present_hardware_types]
    ratio_legend_color = hardware_color_map[present_hardware_types[0]]
    ratio_legend = [
        Patch(facecolor=ratio_legend_color, alpha=1.0, label='计算占比'),
        Patch(facecolor=ratio_legend_color, alpha=0.7, label='访存占比'),
        Patch(facecolor=ratio_legend_color, alpha=0.4, label='通信占比')
    ]

    leg1 = ax.legend(handles=hardware_legend, title='硬件选型',
                     loc='upper left',
                     bbox_to_anchor=(0.0, 1.0),
                     ncol=len(present_hardware_types),
                     fontsize=10, title_fontsize=11,
                     framealpha=0.9, edgecolor='grey',
                     handlelength=1.2, handletextpad=0.5, labelspacing=0.3,
                     columnspacing=0.8)
    leg1.get_frame().set_linestyle('--')
    ax.add_artist(leg1)

    # 占比图例（右上角）
    leg2 = ax.legend(handles=ratio_legend, title='占比类型',
                     loc='upper right',
                     bbox_to_anchor=(1.0, 1.0),
                     ncol=3,
                     fontsize=10, title_fontsize=11,
                     framealpha=0.9, edgecolor='grey',
                     handlelength=1.2, handletextpad=0.5, labelspacing=0.3,
                     columnspacing=0.8)
    leg2.get_frame().set_linestyle('--')

    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)
    plt.tight_layout()

    safe_title = title.replace(' ', '_').replace(',', '_').replace('<=', 'LE')
    plt.savefig(f"{safe_title}.png", dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)

print("所有图片已生成，柱内数字已恢复并根据宽度自适应字号。")
