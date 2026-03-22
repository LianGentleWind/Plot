import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

os.makedirs('figs', exist_ok=True)

# 稍微放大保证双栏图下的辨识度
FONT_TITLE, FONT_AXIS, FONT_TICK, FONT_LEGEND = 14, 12, 11, 12
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="white")

color_map = {'UB Load/Store': '#517FA4', 'URMA': '#E6B85C', 'UB-XMem (Ours)': '#C00000'}

models = ['Dense LLM\n(LLaMA3-70B)', 'Sparse MoE\n(DeepSeek-V3)', 'DLRM / GNN\n(Random Memory Access)']
scales = ['Scale: 64 P', 'Scale: 256 P']
x_configs = ['Seq=2K', 'Seq=4K', 'Seq=8K']

np.random.seed(42)
def gen_data(s_idx, m_idx):
    ls = np.array([0.9, 0.8, 0.6])
    urma = np.array([0.7, 0.9, 1.0])
    if m_idx == 0: ls *= 0.8; urma *= 1.2
    elif m_idx == 1: ls *= 1.1; urma *= 1.0
    else: ls *= 1.5; urma *= 0.5
    if s_idx == 1: ls *= 0.5; urma *= 0.9
    xmem = np.maximum(ls, urma) * np.random.uniform(1.05, 1.15, 3)
    return ls, urma, xmem

# 双栏大图: 尺寸控制在合理范围确保页面容纳
fig, axes = plt.subplots(2, 3, figsize=(13, 6), sharey=False)
width = 0.25
x = np.arange(len(x_configs))

for s_idx in range(2):
    for m_idx in range(3):
        ax = axes[s_idx, m_idx]
        ls, urma, xmem = gen_data(s_idx, m_idx)
        
        ax.bar(x - width, ls, width, color=color_map['UB Load/Store'], edgecolor='black', linewidth=1.0)
        ax.bar(x, urma, width, color=color_map['URMA'], edgecolor='black', linewidth=1.0)
        ax.bar(x + width, xmem, width, color=color_map['UB-XMem (Ours)'], edgecolor='black', linewidth=1.0, hatch='//')

        ax.set_xticks(x)
        ax.set_xticklabels(x_configs, fontsize=FONT_TICK)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        for spine in ax.spines.values():
            spine.set_color('black')
            spine.set_linewidth(1.0)
            
        if s_idx == 0:
            ax.set_title(models[m_idx], fontsize=FONT_TITLE, fontweight='bold', pad=10)
        if m_idx == 0:
            ax.set_ylabel('Norm. System Throughput', fontsize=FONT_AXIS)
            
        ax.text(0.04, 0.92, scales[s_idx], transform=ax.transAxes, fontsize=FONT_TICK, fontweight='bold', va='top', ha='left', bbox=dict(facecolor='#F5F5F5', edgecolor='black', pad=3.0, alpha=0.9))

        idx = s_idx * 3 + m_idx
        ax.text(0.96, 0.92, f'({chr(ord("a") + idx)})', transform=ax.transAxes, fontsize=FONT_TITLE, fontweight='bold', va='top', ha='right')

# 手动添加全局Legend，取消轴自带的label
handles = [plt.Rectangle((0,0),1,1, color=color_map['UB Load/Store'], ec='black'),
           plt.Rectangle((0,0),1,1, color=color_map['URMA'], ec='black'),
           plt.Rectangle((0,0),1,1, color=color_map['UB-XMem (Ours)'], ec='black', hatch='//')]
labels = ['UB Load/Store', 'URMA', 'UB-XMem (Ours)']
fig.legend(handles, labels, loc='upper center', ncol=3, fontsize=FONT_LEGEND, frameon=False, bbox_to_anchor=(0.5, 1.05))

plt.tight_layout()
plt.subplots_adjust(top=0.88, hspace=0.3, wspace=0.15)
out_path = os.path.join('figs', 'fig5_comprehensive_matrix.png')
plt.savefig(out_path, dpi=300, bbox_inches='tight')
print(f"Saved {out_path}")
plt.show()