#!/usr/bin/env python3
"""把 op_details sidecar CSV 画成算子级横向时间线。

每个算子的水平宽度反映其绝对耗时，纵向高度固定不变。
内部竖向用颜色区分 compute / memory / static latency 的比例。

三行子图分别展示全部/cube/vector 算子，共享时间轴。
底部面板：统计文本 + 饼图。

直接修改下方 OP_DETAILS_CSV / MAIN_CSV / OUTPUT_PNG 三个变量后运行：
    python plot_op_timeline_width.py
"""

import csv
import os
import sys
import re
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle, FancyBboxPatch

# ============================================================
# 输入/输出路径（按需手动修改；相对路径基于运行目录）
# ============================================================
OP_DETAILS_CSV = 'configs/260427_op/output/rubin_ultra/op_details/decoder_rubin_ultra_seq131072_tp1_pp1_ep64_moetp1_mbs366.csv'
MAIN_CSV       = 'configs/260427_op/output/rubin_ultra/pd-split-request-optimal_decoder_best.csv'
OUTPUT_PNG     = 'op_timeline_width.png'
# ============================================================

# 配色 — 统一 bar chart 和 pie chart（engine × bottleneck 四分类）
BAR_COLORS = {
    'cube_compute':   '#1F4E79',   # 深蓝 — cube compute time (flops_time)
    'cube_mem':       '#7FB3D5',   # 浅蓝 — cube memory time (mem_time)
    'vector_compute': '#A12921',   # 深红 — vector compute time
    'vector_mem':     '#F1A8A4',   # 粉色 — vector memory time
}
STATIC_COLOR  = '#999999'   # 灰 — static_latency
DIVIDER_COLOR = '#444444'   # attn / mlp 分界线

# 饼图配色
PIE_COLORS = {
    'cube_compute_bound':  BAR_COLORS['cube_compute'],
    'cube_mem_bound':      BAR_COLORS['cube_mem'],
    'vector_compute_bound':BAR_COLORS['vector_compute'],
    'vector_mem_bound':    BAR_COLORS['vector_mem'],
    'static_latency':      '#888888',
    'comm_exposed':        '#6BAA75',
}

STATS_KEYS_ORDERED = [
    ('decoder_time_per_token(s)',                       'time_per_token (s)'),
    ('decoder_instance_compute_time_used_pct(%)',       '  compute-bound %'),
    ('decoder_instance_mem_time_used_pct(%)',           '  mem-bound %'),
    ('decoder_instance_comm_time_exposed_pct(%)',       '  comm exposed %'),
    ('cube_op_count',                                   'cube op_count'),
    ('vector_op_count',                                 'vector op_count'),
    ('cube_total_flops',                                'cube flops'),
    ('vector_total_flops',                              'vector flops'),
    ('cube_total_mem_accessed',                         'cube mem_accessed'),
    ('vector_total_mem_accessed',                       'vector mem_accessed'),
    ('cube_total_flops_time(s)',                        'cube flops_time/block (s)'),
    ('vector_total_flops_time(s)',                      'vector flops_time/block (s)'),
    ('cv_ratio_count',                                  'C/V ratio count'),
    ('cv_ratio_flops',                                  'C/V ratio flops'),
    ('cv_ratio_mem_accessed',                           'C/V ratio mem_accessed'),
    ('cv_ratio_flops_time',                             'C/V ratio flops_time'),
    ('cube_utilization(%)',                             'cube utilization %'),
    ('vector_utilization(%)',                           'vector utilization %'),
    ('cv_utilization_ratio',                            'C/V utilization ratio'),
    ('cube_compute_time_used_pct(%)',                   'cube compute-bound %'),
    ('cube_mem_time_used_pct(%)',                       'cube mem-bound %'),
    ('vector_compute_time_used_pct(%)',                 'vector compute-bound %'),
    ('vector_mem_time_used_pct(%)',                     'vector mem-bound %'),
    ('matrix_op_static_latency_total_per_block(s)',     'static latency/block (s)'),
    ('matrix_op_static_latency_total_pct(%)',           'static latency %'),
]


# ----------------------------------------------------------------
# 数据读取（与原版一致）
# ----------------------------------------------------------------
def read_op_details(path):
    with open(path, 'r', newline='', encoding='utf-8') as fd:
        return list(csv.DictReader(fd))


def dedup_expert_blocks(records):
    """将重复的 ExpertBlock 组合并为单个代表组。

    Returns
    -------
    (new_records, expert_info)
        expert_info: None 或 dict{'start': int, 'end': int, 'count': int}
        start/end 是去重后列表中保留的 expert 算子的索引范围（含两端）。
    """
    expert_pattern = re.compile(r'^ExpertBlock(\d+)_(.+)$')
    expert_indices = []
    for i, r in enumerate(records):
        m = expert_pattern.match(r['name'])
        if m:
            expert_indices.append((i, int(m.group(1)), m.group(2)))

    if not expert_indices:
        return records, None

    op_suffixes = OrderedDict()
    for idx, enum, suffix in expert_indices:
        op_suffixes.setdefault(suffix, set()).add(enum)

    expert_nums = None
    for suffix, nums in op_suffixes.items():
        if expert_nums is None:
            expert_nums = nums

    n_experts = len(expert_nums) if expert_nums else 1
    if n_experts <= 1:
        return records, None

    first_expert = min(expert_nums)
    remove_indices = set()
    rename_indices = {}
    for idx, enum, suffix in expert_indices:
        if enum == first_expert:
            rename_indices[idx] = f'Expert_{suffix}'
        else:
            remove_indices.add(idx)

    new_records = []
    expert_new_start = None
    expert_new_end = None
    for i, r in enumerate(records):
        if i in remove_indices:
            continue
        new_idx = len(new_records)
        if i in rename_indices:
            r = dict(r)
            r['name'] = rename_indices[i]
            if expert_new_start is None:
                expert_new_start = new_idx
            expert_new_end = new_idx
        new_records.append(r)

    expert_info = {'start': expert_new_start, 'end': expert_new_end,
                   'count': n_experts}
    return new_records, expert_info


def read_main_stats(path, keys):
    wanted = set(keys)
    out = {}
    with open(path, 'r', newline='', encoding='utf-8') as fd:
        reader = csv.reader(fd)
        next(reader, None)
        for row in reader:
            if row and row[0] in wanted and len(row) > 1:
                out[row[0]] = row[1]
    return out


def fmt_value(s):
    if not s:
        return ''
    try:
        v = float(s)
    except ValueError:
        return s
    av = abs(v)
    if av == 0:
        return '0'
    if av < 1e-3 or av >= 1e7:
        return f'{v:.4e}'
    return f'{v:.4f}' if av < 10 else f'{v:,.2f}'


def build_stats_text(stats, sidecar_name, main_name):
    header = [f'Sidecar: {sidecar_name}', f'Main:    {main_name}', '']
    stat_lines = []
    for key, label in STATS_KEYS_ORDERED:
        if key in stats:
            stat_lines.append(f'  {label:<28} {fmt_value(stats[key])}')

    mid = (len(stat_lines) + 1) // 2
    lines = list(header)
    for left, right in zip(stat_lines[:mid], stat_lines[mid:]):
        lines.append(f'{left:<46} {right}')
    if len(stat_lines[:mid]) > len(stat_lines[mid:]):
        lines.append(stat_lines[mid - 1])
    return '\n'.join(lines)


def _safe_pct(stats, key):
    try:
        return float(stats.get(key, 0)) if stats.get(key, '') else 0.0
    except ValueError:
        return 0.0


# ----------------------------------------------------------------
# 横向变宽竖条绘制
# ----------------------------------------------------------------
def draw_op_bars_h(ax, records, subtitle, cum_starts, widths,
                   ft_arr, mt_arr, sl_arr,
                   filter_engine=None, show_xticklabels=False,
                   divider_time=None, bar_height=1.0):
    """为一组 op records 画横向变宽柱条。

    每个算子是一个矩形：
      - x 位置 = cum_starts[i]，宽度 = widths[i]（反映绝对耗时）
      - 纵向固定高度 bar_height，内部按 flops_time/mem_time/static_latency
        的比例分段着色

    filter_engine : None（全部）、'cube'、'vector'（仅绘制匹配引擎的算子）
    """
    n = len(records)
    if n == 0:
        ax.text(0.5, 0.5, '(no ops)', ha='center', va='center',
                transform=ax.transAxes, fontsize=10, color='#888')
        ax.set_yticks([])
        ax.set_title(subtitle, fontsize=11, pad=2)
        return

    total_width = cum_starts[-1] + widths[-1] if n > 0 else 1

    for i, r in enumerate(records):
        if filter_engine and r['engine'] != filter_engine:
            continue
        x0 = cum_starts[i]
        w = widths[i]
        if w <= 0:
            continue

        ft, mt, sl = ft_arr[i], mt_arr[i], sl_arr[i]
        total = ft + mt + sl
        if total <= 0:
            continue

        eng = r['engine']
        ft_frac = ft / total
        mt_frac = mt / total
        sl_frac = sl / total

        ft_color = BAR_COLORS[f'{eng}_compute']
        mt_color = BAR_COLORS[f'{eng}_mem']

        # 底部：compute (flops_time)
        h_ft = ft_frac * bar_height
        ax.add_patch(Rectangle((x0, 0), w, h_ft,
                               facecolor=ft_color, alpha=0.85,
                               edgecolor='white', linewidth=0.3))
        # 中部：memory (mem_time)
        h_mt = mt_frac * bar_height
        ax.add_patch(Rectangle((x0, h_ft), w, h_mt,
                               facecolor=mt_color, alpha=0.85,
                               edgecolor='white', linewidth=0.3))
        # 顶部：static_latency
        if sl_frac > 0:
            h_sl = sl_frac * bar_height
            ax.add_patch(Rectangle((x0, h_ft + h_mt), w, h_sl,
                                   facecolor=STATIC_COLOR, alpha=0.85,
                                   edgecolor='white', linewidth=0.3))

    # 坐标轴
    ax.set_xlim(0, total_width)
    ax.set_ylim(0, bar_height)
    ax.set_yticks([])
    ax.set_title(subtitle, fontsize=11, pad=2)

    # attn / mlp 分界线
    if divider_time is not None:
        ax.axvline(divider_time, color=DIVIDER_COLOR,
                   linestyle='--', linewidth=1.0, alpha=0.7)

    # 图例
    engines_present = set(
        r['engine'] for r in records
        if filter_engine is None or r['engine'] == filter_engine
    )
    legend_handles = []
    if 'cube' in engines_present:
        legend_handles.append(mpatches.Patch(
            facecolor=BAR_COLORS['cube_compute'], alpha=0.85,
            label='cube flops_time'))
        legend_handles.append(mpatches.Patch(
            facecolor=BAR_COLORS['cube_mem'], alpha=0.85,
            label='cube mem_time'))
    if 'vector' in engines_present:
        legend_handles.append(mpatches.Patch(
            facecolor=BAR_COLORS['vector_compute'], alpha=0.85,
            label='vector flops_time'))
        legend_handles.append(mpatches.Patch(
            facecolor=BAR_COLORS['vector_mem'], alpha=0.85,
            label='vector mem_time'))
    has_static = any(sl_arr[i] > 0 for i in range(n))
    if has_static:
        legend_handles.append(mpatches.Patch(
            facecolor=STATIC_COLOR, alpha=0.85, label='static_latency'))
    if legend_handles:
        ax.legend(handles=legend_handles, loc='upper right',
                  fontsize=7, framealpha=0.8)

    # X 轴标签：仅最底行显示算子名称
    if show_xticklabels:
        for i, r in enumerate(records):
            w = widths[i]
            if w <= 0:
                continue
            cx = cum_starts[i] + w / 2
            ax.text(cx, -0.03 * bar_height, r['name'],
                    rotation=90, ha='center', va='top',
                    fontsize=5, clip_on=False)
        ax.set_xlabel('time (s)', fontsize=9)
    else:
        ax.tick_params(axis='x', labelbottom=False)


def draw_utilization_line(ax, records, cum_starts, widths, engine,
                          color, divider_time=None,
                          show_xticklabels=True):
    """画单个 engine 的计算率阶梯线，与主时间线共享 X 轴。"""
    n = len(records)
    if n == 0:
        ax.text(0.5, 0.5, '(no ops)', ha='center', va='center',
                transform=ax.transAxes, fontsize=10, color='#888')
        ax.set_yticks([])
        ax.set_title(f'{engine} Compute Utilization', fontsize=10, pad=2)
        return

    total_width = cum_starts[-1] + widths[-1] if n > 0 else 1
    x_centers = cum_starts + widths / 2
    util = np.array([float(r.get('compute_utilization', 0) or 0)
                     for r in records])
    engines = [r['engine'] for r in records]
    engine_util = np.array([util[i] if engines[i] == engine else 0.0
                            for i in range(n)])

    x_steps = np.repeat(np.r_[cum_starts, total_width], 2)[1:-1]
    util_steps = np.repeat(engine_util, 2)

    ax.plot(x_steps, util_steps, color=color, linewidth=2.2,
            label=f'{engine} compute_utilization')

    ax.set_xlim(0, total_width)
    ymax = max(1.0, float(engine_util.max()) * 1.05)
    ax.set_ylim(0, ymax)
    ax.set_ylabel('utilization', fontsize=9)
    ax.set_title(f'{engine.capitalize()} Compute Utilization',
                 fontsize=10, pad=2)
    ax.grid(axis='y', linestyle=':', linewidth=0.6, alpha=0.5)
    ax.legend(loc='upper right', fontsize=7, framealpha=0.8)

    if divider_time is not None:
        ax.axvline(divider_time, color=DIVIDER_COLOR,
                   linestyle='--', linewidth=1.0, alpha=0.7)

    if show_xticklabels:
        for i, r in enumerate(records):
            if widths[i] <= 0:
                continue
            ax.text(x_centers[i], -0.10 * ymax, r['name'],
                    rotation=90, ha='center', va='top',
                    fontsize=5, clip_on=False)
        ax.set_xlabel('time (s)', fontsize=9)
    else:
        ax.tick_params(axis='x', labelbottom=False)


# ----------------------------------------------------------------
# 饼图
# ----------------------------------------------------------------
def draw_pie(ax, stats):
    slices = [
        ('cube compute-bound',   _safe_pct(stats, 'cube_compute_time_used_pct(%)'),    PIE_COLORS['cube_compute_bound']),
        ('cube mem-bound',       _safe_pct(stats, 'cube_mem_time_used_pct(%)'),        PIE_COLORS['cube_mem_bound']),
        ('vector compute-bound', _safe_pct(stats, 'vector_compute_time_used_pct(%)'),  PIE_COLORS['vector_compute_bound']),
        ('vector mem-bound',     _safe_pct(stats, 'vector_mem_time_used_pct(%)'),      PIE_COLORS['vector_mem_bound']),
        ('static latency',       _safe_pct(stats, 'matrix_op_static_latency_total_pct(%)'), PIE_COLORS['static_latency']),
        ('comm exposed',         _safe_pct(stats, 'decoder_instance_comm_time_exposed_pct(%)'), PIE_COLORS['comm_exposed']),
    ]
    slices = [(lab, v, c) for lab, v, c in slices if v > 0]
    if not slices:
        ax.axis('off')
        ax.text(0.5, 0.5, '(no pct stats)', ha='center', va='center',
                transform=ax.transAxes)
        return
    labels = [lab for lab, _, _ in slices]
    values = [v for _, v, _ in slices]
    colors = [c for _, _, c in slices]
    total = sum(values)
    wedges, _ = ax.pie(
        values, colors=colors, startangle=90, counterclock=False,
        wedgeprops={'edgecolor': 'white', 'linewidth': 1.0},
    )
    legend_labels = [f'{lab}  {v:.2f}%' for lab, v in zip(labels, values)]
    ax.legend(wedges, legend_labels, loc='center left',
              bbox_to_anchor=(1.02, 0.5), fontsize=8, frameon=False)
    ax.set_title(f'time breakdown  (Σ={total:.2f}%)', fontsize=10)
    ax.set_aspect('equal')


# ----------------------------------------------------------------
# 主绘图
# ----------------------------------------------------------------
def plot_timeline(records, stats_text, stats, output_path, title,
                  expert_info=None):
    n_all = len(records)

    # ---- 计算每个算子的耗时数组 ----
    ft = np.array([float(r['flops_time']) for r in records])
    mt = np.array([float(r['mem_time']) for r in records])
    sl = np.array([float(r.get('static_latency', 0)) for r in records])

    # expert 区间乘以重复次数，反映真实总耗时
    if expert_info is not None:
        es, ee, ec = expert_info['start'], expert_info['end'], expert_info['count']
        ft[es:ee+1] *= ec
        mt[es:ee+1] *= ec
        sl[es:ee+1] *= ec

    widths = ft + mt + sl  # 每个算子的宽度 = 总耗时

    # 累计起始位置
    cum_starts = np.zeros(n_all)
    for i in range(1, n_all):
        cum_starts[i] = cum_starts[i-1] + widths[i-1]
    total_time = (cum_starts[-1] + widths[-1]) if n_all > 0 else 0

    # attn / mlp 分界线（时间位置）
    divider_time = None
    attn_idxs = [i for i, r in enumerate(records) if r['block'] == 'attn']
    mlp_idxs  = [i for i, r in enumerate(records) if r['block'] != 'attn']
    if attn_idxs and mlp_idxs:
        last_attn = attn_idxs[-1]
        first_mlp = next((i for i in mlp_idxs if i > last_attn), None)
        if first_mlp is not None:
            divider_time = cum_starts[first_mlp]

    # ---- 画布布局 ----
    fig = plt.figure(figsize=(22, 12))
    outer_gs = fig.add_gridspec(4, 1,
                                height_ratios=[2.0, 0.55, 0.55, 1.0],
                                hspace=0.16)

    # 上部：单行时间线 + cube/vector 计算率折线图
    ax_all = fig.add_subplot(outer_gs[0])
    ax_cube_util = fig.add_subplot(outer_gs[1], sharex=ax_all)
    ax_vector_util = fig.add_subplot(outer_gs[2], sharex=ax_all)

    # 下部：文本 + 饼图
    bottom_gs = outer_gs[3].subgridspec(1, 2, width_ratios=[1.5, 1.0],
                                         wspace=0.18)
    ax_text = fig.add_subplot(bottom_gs[0])
    ax_pie  = fig.add_subplot(bottom_gs[1])

    bar_h = 1.0

    # 全部算子
    draw_op_bars_h(ax_all, records, f'All Ops  ({n_all} ops)',
                   cum_starts, widths, ft, mt, sl,
                   filter_engine=None, show_xticklabels=False,
                   divider_time=divider_time, bar_height=bar_h)
    draw_utilization_line(ax_cube_util, records, cum_starts, widths,
                          'cube', BAR_COLORS['cube_compute'],
                          divider_time=divider_time,
                          show_xticklabels=False)
    draw_utilization_line(ax_vector_util, records, cum_starts, widths,
                          'vector', BAR_COLORS['vector_compute'],
                          divider_time=divider_time,
                          show_xticklabels=True)

    # ---- Expert 标注框 ----
    if expert_info is not None:
        es, ee, ec = expert_info['start'], expert_info['end'], expert_info['count']
        x_start = cum_starts[es]
        x_end   = cum_starts[ee] + widths[ee]

        yl, yh = ax_all.get_ylim()
        rect = FancyBboxPatch(
            (x_start, yl), x_end - x_start, yh - yl,
            boxstyle='round,pad=0', linewidth=1.5,
            edgecolor='#E74C3C', facecolor='#E74C3C', alpha=0.08,
        )
        ax_all.add_patch(rect)
        ax_all.annotate(
            f'repeated ×{ec}',
            xy=((x_start + x_end) / 2, yh), xycoords='data',
            xytext=(0, 4), textcoords='offset points',
            ha='center', va='bottom', fontsize=10, color='#E74C3C',
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='#E74C3C',
                      alpha=0.9, lw=0.8),
        )

    fig.suptitle(title, fontsize=13, y=0.995)

    # ---- 统计文本 ----
    ax_text.axis('off')
    ax_text.text(0.0, 1.0, stats_text, transform=ax_text.transAxes,
                 fontfamily='monospace', fontsize=8.5,
                 verticalalignment='top')

    # ---- 饼图 ----
    draw_pie(ax_pie, stats or {})

    plt.savefig(output_path, dpi=130, bbox_inches='tight')
    print(f'Saved: {output_path}')
    plt.close(fig)


# ----------------------------------------------------------------
# main
# ----------------------------------------------------------------
def main():
    if not os.path.exists(OP_DETAILS_CSV):
        print(f'OP_DETAILS_CSV not found: {OP_DETAILS_CSV}', file=sys.stderr)
        sys.exit(1)

    records_raw = read_op_details(OP_DETAILS_CSV)
    if not records_raw:
        print(f'Empty sidecar: {OP_DETAILS_CSV}', file=sys.stderr)
        sys.exit(1)
    records, expert_info = dedup_expert_blocks(records_raw)
    if len(records) < len(records_raw):
        print(f'Deduplicated expert blocks: {len(records_raw)} -> {len(records)} ops')

    stats = (read_main_stats(MAIN_CSV, [k for k, _ in STATS_KEYS_ORDERED])
             if os.path.exists(MAIN_CSV) else {})
    stats_text = build_stats_text(
        stats,
        os.path.basename(OP_DETAILS_CSV),
        os.path.basename(MAIN_CSV) if stats else '(not loaded)',
    )

    out_dir = os.path.dirname(OUTPUT_PNG)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    title = f'Op timeline (width ∝ time)  —  {os.path.basename(OP_DETAILS_CSV)}'
    plot_timeline(records, stats_text, stats, OUTPUT_PNG, title,
                  expert_info=expert_info)


if __name__ == '__main__':
    main()
