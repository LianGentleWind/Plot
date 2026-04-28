#!/usr/bin/env python3
"""把 op_details sidecar CSV 画成算子级竖条时间线，三行分别展示全部/cube/vector 算子。

每个竖条高度 = process_time，内部用颜色展示：
  - flops_time（蓝）和 mem_time（橙）重叠取 max 的逻辑
  - static_latency（灰）叠加在 max(flops, mem) 之上

直接修改下方 OP_DETAILS_CSV / MAIN_CSV / OUTPUT_PNG 三个变量后运行：
    python plot_op_timeline.py
"""

import csv
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ============================================================
# 输入/输出路径（按需手动修改；相对路径基于运行目录）
# ============================================================
OP_DETAILS_CSV = 'configs/260427_op/output/rubin_ultra/op_details/decoder_rubin_ultra_seq131072_tp1_pp1_ep64_moetp1_mbs366.csv'
MAIN_CSV       = 'configs/260427_op/output/rubin_ultra/pd-split-request-optimal_decoder_best.csv'
OUTPUT_PNG     = 'op_timeline.png'
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
# 数据读取
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
    import re
    expert_pattern = re.compile(r'^ExpertBlock(\d+)_(.+)$')
    expert_indices = []
    for i, r in enumerate(records):
        m = expert_pattern.match(r['name'])
        if m:
            expert_indices.append((i, int(m.group(1)), m.group(2)))

    if not expert_indices:
        return records, None

    from collections import OrderedDict
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
    lines = [f'Sidecar: {sidecar_name}', f'Main:    {main_name}', '']
    for key, label in STATS_KEYS_ORDERED:
        if key in stats:
            lines.append(f'  {label:<28} {fmt_value(stats[key])}')
    return '\n'.join(lines)


def _safe_pct(stats, key):
    try:
        return float(stats.get(key, 0)) if stats.get(key, '') else 0.0
    except ValueError:
        return 0.0


# ----------------------------------------------------------------
# 竖条绘制
# ----------------------------------------------------------------
def draw_op_bars(ax, records, subtitle, x_positions=None, xlim=None,
                show_xticklabels=False, all_names=None, use_log=True,
                repeat_range=None, divider_x=None):
    """为一组 op records 画竖条时间线（分段堆叠）。

    repeat_range : (start, end, count) or None
        若提供，将 records[start:end+1] 的时间值乘以 count，
        用于线性图反映重复 expert 的真实总耗时。
    divider_x : float or None
        attn/mlp 分界线的 x 坐标（由外部统一计算，保证各行对齐）。
    """
    n = len(records)
    if n == 0:
        ax.text(0.5, 0.5, '(no ops)', ha='center', va='center',
                transform=ax.transAxes, fontsize=10, color='#888')
        ax.set_yticks([])
        ax.set_title(subtitle, fontsize=11, pad=2)
        if xlim is not None:
            ax.set_xlim(*xlim)
        return

    x = np.asarray(x_positions) if x_positions is not None else np.arange(n)
    ft = np.array([float(r['flops_time']) for r in records])
    mt = np.array([float(r['mem_time']) for r in records])
    sl = np.array([float(r.get('static_latency', 0)) for r in records])
    engines = [r['engine'] for r in records]  # 'cube' or 'vector'

    # 对重复区间进行缩放
    if repeat_range is not None:
        rs, re, rc = repeat_range
        ft[rs:re+1] *= rc
        mt[rs:re+1] *= rc
        sl[rs:re+1] *= rc

    # 逐条设色：compute 用引擎深色，mem 用引擎浅色
    ft_colors = [BAR_COLORS[f'{eng}_compute'] for eng in engines]
    mt_colors = [BAR_COLORS[f'{eng}_mem']     for eng in engines]

    total = ft + mt + sl

    if use_log:
        # ---- 对数坐标：在 log 空间内按绝对值比例分割每个柱条 ----
        pos_totals = total[total > 0]
        if len(pos_totals) == 0:
            y_base = 1e-12
        else:
            y_base = pos_totals.min() / 3

        ft_frac = np.zeros_like(ft)
        mt_frac = np.zeros_like(mt)
        mask = total > 0
        ft_frac[mask] = ft[mask] / total[mask]
        mt_frac[mask] = mt[mask] / total[mask]
        ratio   = np.where(total > 0, total / y_base, 1.0)

        # 在 log 空间内，让 ft 段占 ft_frac 的视觉高度
        split1 = y_base * np.power(ratio, ft_frac)            # ft ↔ mt 分界
        split2 = y_base * np.power(ratio, ft_frac + mt_frac)  # mt ↔ sl 分界

        h_ft = split1 - y_base
        h_mt = split2 - split1
        h_sl = np.where(total > 0, total - split2, 0)

        ax.bar(x, h_ft, bottom=y_base, width=1.0,
               color=ft_colors, alpha=0.85, edgecolor='none')
        ax.bar(x, h_mt, bottom=split1, width=1.0,
               color=mt_colors, alpha=0.85, edgecolor='none')
        if sl.any():
            ax.bar(x, h_sl, bottom=split2, width=1.0,
                   color=STATIC_COLOR, alpha=0.85, edgecolor='none')

        ax.set_yscale('log')
        ax.set_ylim(y_base, total.max() * 3)
        ax.set_ylabel('time (s, log)', fontsize=9)
    else:
        # ---- 线性坐标：直接堆叠 ----
        ax.bar(x, ft, width=1.0, color=ft_colors, alpha=0.85, edgecolor='none')
        ax.bar(x, mt, bottom=ft, width=1.0,
               color=mt_colors, alpha=0.85, edgecolor='none')
        if sl.any():
            ax.bar(x, sl, bottom=ft + mt, width=1.0,
                   color=STATIC_COLOR, alpha=0.85, edgecolor='none')

        top = total.max()
        ax.set_ylim(0, top * 1.15 if top > 0 else 1)
        ax.set_ylabel('time (s)', fontsize=9)

    # legend（Patch 对象确保颜色正确）
    legend_handles = []
    has_cube = 'cube' in set(engines)
    has_vector = 'vector' in set(engines)
    if has_cube:
        legend_handles.append(mpatches.Patch(facecolor=BAR_COLORS['cube_compute'], alpha=0.85, label='cube flops_time'))
        legend_handles.append(mpatches.Patch(facecolor=BAR_COLORS['cube_mem'],     alpha=0.85, label='cube mem_time'))
    if has_vector:
        legend_handles.append(mpatches.Patch(facecolor=BAR_COLORS['vector_compute'], alpha=0.85, label='vector flops_time'))
        legend_handles.append(mpatches.Patch(facecolor=BAR_COLORS['vector_mem'],     alpha=0.85, label='vector mem_time'))
    if sl.any():
        legend_handles.append(mpatches.Patch(facecolor=STATIC_COLOR, alpha=0.85, label='static_latency'))

    # attn / mlp 分界线
    if divider_x is not None:
        ax.axvline(divider_x, color=DIVIDER_COLOR,
                   linestyle='--', linewidth=1.0, alpha=0.7)

    if xlim is not None:
        ax.set_xlim(*xlim)
    else:
        ax.set_xlim(x.min() - 0.5, x.max() + 0.5)

    ax.set_title(subtitle, fontsize=11, pad=2)
    ax.legend(handles=legend_handles, loc='upper right', fontsize=7, framealpha=0.8)

    # x 轴标签：仅最底行显示，竖排文字
    if show_xticklabels and all_names is not None:
        all_x = np.arange(len(all_names))
        ax.set_xticks(all_x)
        ax.set_xticklabels(all_names, rotation=90, ha='center', fontsize=6)
    elif not show_xticklabels:
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
    """expert_info: dedup_expert_blocks 返回的 bracket 信息。"""
    n_all = len(records)
    all_x = np.arange(n_all)
    all_names = [r['name'] for r in records]
    shared_xlim = (-0.5, n_all - 0.5)

    cube_records, cube_x = [], []
    vector_records, vector_x = [], []
    for i, r in enumerate(records):
        if r['engine'] == 'cube':
            cube_records.append(r)
            cube_x.append(i)
        else:
            vector_records.append(r)
            vector_x.append(i)
    # 计算各子集的 repeat_range（将 expert_info 的全局索引映射到各子列表的本地索引）
    all_repeat = None
    cube_repeat = None
    vector_repeat = None
    if expert_info is not None:
        es, ee, ec = expert_info['start'], expert_info['end'], expert_info['count']
        all_repeat = (es, ee, ec)
        # cube 子列表中，expert 范围内的本地索引
        cube_local = [j for j, orig_i in enumerate(cube_x) if es <= orig_i <= ee]
        if cube_local:
            cube_repeat = (cube_local[0], cube_local[-1], ec)
        # vector 子列表中同理
        vec_local = [j for j, orig_i in enumerate(vector_x) if es <= orig_i <= ee]
        if vec_local:
            vector_repeat = (vec_local[0], vec_local[-1], ec)

    # 从全量 records 计算 attn/mlp 分界线 x 坐标（所有子图共用）
    _divider_x = None
    attn_idxs = [i for i, r in enumerate(records) if r['block'] == 'attn']
    mlp_idxs  = [i for i, r in enumerate(records) if r['block'] != 'attn']
    if attn_idxs and mlp_idxs:
        last_attn = attn_idxs[-1]
        first_mlp = mlp_idxs[0] if mlp_idxs[0] > last_attn else None
        if first_mlp is not None:
            _divider_x = (last_attn + first_mlp) / 2

    fig = plt.figure(figsize=(20, 20))
    # 外层: 上部时间线区 + 下部统计区
    outer_gs = fig.add_gridspec(2, 1, height_ratios=[3.5, 1.2], hspace=0.28)
    # 上部: 4 行时间线（线性概览 + 3 行对数），紧凑排列
    inner_gs = outer_gs[0].subgridspec(4, 1,
                                        height_ratios=[0.5, 1, 1, 1.4],
                                        hspace=0.1)
    ax_linear = fig.add_subplot(inner_gs[0])
    ax_all    = fig.add_subplot(inner_gs[1], sharex=ax_linear)
    ax_cube   = fig.add_subplot(inner_gs[2], sharex=ax_linear)
    ax_vector = fig.add_subplot(inner_gs[3], sharex=ax_linear)
    # 下部: 文本 + 饼图
    bottom_gs = outer_gs[1].subgridspec(1, 2, width_ratios=[1.5, 1.0], wspace=0.18)
    ax_text   = fig.add_subplot(bottom_gs[0])
    ax_pie    = fig.add_subplot(bottom_gs[1])

    # 第 0 行：线性坐标总览（扁），重复 expert 区乘以重复次数
    draw_op_bars(ax_linear, records, f'All Ops — linear  ({n_all} ops)',
                 x_positions=all_x, xlim=shared_xlim,
                 show_xticklabels=False, use_log=False,
                 repeat_range=all_repeat, divider_x=_divider_x)
    # 第 1 行：全部算子（对数）
    draw_op_bars(ax_all, records, f'All Ops — log  ({n_all} ops)',
                 x_positions=all_x, xlim=shared_xlim,
                 show_xticklabels=False,
                 repeat_range=all_repeat, divider_x=_divider_x)
    # 第 2 行：Cube 算子（对数）
    draw_op_bars(ax_cube, cube_records, f'Cube Ops  ({len(cube_records)} ops)',
                 x_positions=cube_x, xlim=shared_xlim,
                 show_xticklabels=False,
                 repeat_range=cube_repeat, divider_x=_divider_x)
    # 第 3 行（最底行）：Vector 算子 + 竖排名称
    draw_op_bars(ax_vector, vector_records, f'Vector Ops  ({len(vector_records)} ops)',
                 x_positions=vector_x, xlim=shared_xlim,
                 show_xticklabels=True, all_names=all_names,
                 repeat_range=vector_repeat, divider_x=_divider_x)

    # 在 ax_linear 上绘制 "重复算子" 标注框（最顶行，最醒目）
    if expert_info is not None:
        es, ee, ec = expert_info['start'], expert_info['end'], expert_info['count']
        from matplotlib.patches import FancyBboxPatch
        y_lo, y_hi = ax_linear.get_ylim()
        rect = FancyBboxPatch(
            (es - 0.5, y_lo), ee - es + 1, y_hi - y_lo,
            boxstyle='round,pad=0', linewidth=1.5,
            edgecolor='#E74C3C', facecolor='#E74C3C', alpha=0.08,
        )
        ax_linear.add_patch(rect)
        ax_linear.annotate(
            f'repeated x{ec}',
            xy=((es + ee) / 2, y_hi), xycoords='data',
            xytext=(0, 4), textcoords='offset points',
            ha='center', va='bottom', fontsize=10, color='#E74C3C',
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='#E74C3C',
                      alpha=0.9, lw=0.8),
        )
        # 其余 3 行也画半透明框（无文字）
        for ax_sub in (ax_all, ax_cube, ax_vector):
            yl, yh = ax_sub.get_ylim()
            r2 = FancyBboxPatch(
                (es - 0.5, yl), ee - es + 1, yh - yl,
                boxstyle='round,pad=0', linewidth=1.0,
                edgecolor='#E74C3C', facecolor='#E74C3C', alpha=0.06,
            )
            ax_sub.add_patch(r2)

    fig.suptitle(title, fontsize=13, y=0.995)

    ax_text.axis('off')
    ax_text.text(0.0, 1.0, stats_text, transform=ax_text.transAxes,
                 fontfamily='monospace', fontsize=10, verticalalignment='top')
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

    title = f'Op timeline (block-level)  —  {os.path.basename(OP_DETAILS_CSV)}'
    plot_timeline(records, stats_text, stats, OUTPUT_PNG, title,
                  expert_info=expert_info)


if __name__ == '__main__':
    main()
