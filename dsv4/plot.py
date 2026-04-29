import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch
import os

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


# 序列长度格式化（按1024进制，整数显示）
def format_seq_len(length):
    if length >= 1024 * 1024:
        return f"{int(length / (1024 * 1024))}M"
    elif length >= 1024:
        return f"{int(length / 1024)}K"
    else:
        return str(int(length))


# 颜色映射
chips = df['chip'].unique()
base_colors = ['#4c72b0', '#ff7f0e', '#55a868', '#d62728', '#9467bd',
               '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
if len(chips) > len(base_colors):
    cmap = plt.cm.get_cmap('tab20', len(chips))
    chip_color_map = {ch: cmap(i) for i, ch in enumerate(chips)}
else:
    chip_color_map = {ch: base_colors[i] for i, ch in enumerate(chips)}

# 分组绘图
group_cols = ['model', 'latency_constraint_ms', 'seq_len']
grouped = df.groupby(group_cols)

for (model, tpot, seq_len), group in grouped:
    model_name = model[0].upper() + model[1:]
    seq_str = format_seq_len(seq_len)
    title = f"{model_name}@KV={seq_str},TPOT<={tpot}ms"

    cards = sorted(group['num_cards'].unique())
    n_cards = len(cards)

    fig, ax = plt.subplots(figsize=(max(12, n_cards * 0.6 + 2), 6))

    x = np.arange(n_cards)
    present_chips = group['chip'].unique()
    num_chip_types = len(present_chips)

    # 动态柱宽：在一个刻度内，所有芯片柱子总宽度占90%，留10%空隙
    width = 0.9 / num_chip_types

    # 根据柱宽动态计算柱内数字字号，确保始终显示
    base_fontsize = 9
    min_fontsize = 6
    # 当柱子变窄时字号按比例缩小
    text_fontsize = max(min_fontsize, base_fontsize * width / 0.15)

    for i, chip in enumerate(present_chips):
        chip_data = group[group['chip'] == chip].set_index('num_cards').reindex(cards)
        compute = chip_data['compute_time_ratio'].fillna(0).values
        memory = chip_data['memory_time_ratio'].fillna(0).values
        comm = chip_data['comm_time_ratio'].fillna(0).values

        offset = (i - (num_chip_types - 1) / 2) * width

        ax.bar(x + offset, compute, width,
               color=chip_color_map[chip], alpha=1.0, edgecolor='white')
        ax.bar(x + offset, memory, width, bottom=compute,
               color=chip_color_map[chip], alpha=0.6, edgecolor='white')
        ax.bar(x + offset, comm, width, bottom=compute + memory,
               color=chip_color_map[chip], alpha=0.8, edgecolor='white')

        # 始终标注数字，根据宽度自适应字号
        for j in range(n_cards):
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
    ax.set_xticklabels([str(c) for c in cards], fontsize=14)
    ax.set_xlabel('num_cards', fontsize=15)
    ax.set_ylabel('时间占比 (%)', fontsize=15)
    ax.set_ylim(0, 120)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.set_title(title, fontsize=17, pad=15)

    # ---- 图例 ----
    chip_legend = [Patch(facecolor=chip_color_map[ch], edgecolor='white',
                         label=ch) for ch in present_chips]
    ratio_legend = [
        Patch(facecolor='grey', alpha=1.0, label='计算占比'),
        Patch(facecolor='grey', alpha=0.6, label='访存占比'),
        Patch(facecolor='grey', alpha=0.8, label='通信占比')
    ]

    # 芯片图例：芯片>4时每行显示4个，字号调小
    max_cols_chip = 4
    ncol_chip = num_chip_types if num_chip_types <= max_cols_chip else max_cols_chip
    chip_fontsize = 9 if num_chip_types <= 4 else 8
    chip_title_fontsize = 10 if num_chip_types <= 4 else 9

    leg1 = ax.legend(handles=chip_legend, title='Chip',
                     loc='upper left',
                     bbox_to_anchor=(0.0, 1.0),
                     ncol=ncol_chip,
                     fontsize=chip_fontsize, title_fontsize=chip_title_fontsize,
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
