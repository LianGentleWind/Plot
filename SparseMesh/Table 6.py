import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# 1. 原始数据更新
scales = ['256', '512', '1024']
topologies = ['SMesh 128P rack', 'SMesh 64P rack', 'FullMesh', 'CLOS']

# 吞吐数据 (Throughput)
data_tp = {
    'EP16': [[1.72368, 1.7199, 1.69911], [1.64997, 1.64619, 1.61217], [1.674, 1.6668, 1.62], [1, 0.997, 0.97]],
    'EP32': [[1.752079, 1.748298, 1.736958], [1.701047, 1.697267, 1.685927], [1.682805, 1.679205, 1.666607], [1, 0.998, 0.991]],
    'EP64': [[1.86874, 1.86247, 1.85768], [1.81445, 1.81256, 1.80689], [1.758396, 1.756596, 1.751197], [1, 0.998, 0.995]]
}

# DP暴露数据 (DP Exposure)
data_dp = {
    'EP16': [[2.114, 3.431, 5.689], [2.544, 4.117, 6.761], [2.253, 3.813, 6.788], [1.639, 2.778, 4.978]],
    'EP32': [[3.9, 5.75, 8.897], [3.436, 5.077, 7.905], [4.407, 5.962, 9.206], [3.294, 4.519, 6.76]],
    'EP64': [[0.324, 0.361, 0.421], [0.299, 0.329, 0.389], [0.284, 0.313, 0.37], [0.276, 0.304, 0.36]]
}

# 2. 绘图配置
FONT_TITLE, FONT_AXIS, FONT_TICK = 18, 15, 14
FONT_LEGEND = 18  # 增大图例字号
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="white")

# 莫兰迪配色方案
color_map = {
    'SMesh 128P rack': '#517FA4',  
    'SMesh 64P rack': '#92B4D2',   
    'FullMesh': '#6AB187',        
    'CLOS': '#E6B85C'             
}

# 3. 创建画布 (2行3列)
fig, axes = plt.subplots(2, 3, figsize=(15, 7.5))
ep_list = ['EP16', 'EP32', 'EP64']
x = np.arange(len(scales))
width = 0.2 

# 4. 绘制上排：吞吐性能柱状图 (Shared Y)
for i, ep in enumerate(ep_list):
    ax = axes[0, i]
    for j, topo in enumerate(topologies):
        ax.bar(x + (j - 1.5) * width, data_tp[ep][j], width, 
               label=topo, color=color_map[topo], edgecolor='black', linewidth=0.8)
    
    ax.set_title(f'{ep}', fontsize=FONT_TITLE, fontweight='bold', pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(scales, fontsize=FONT_TICK)
    ax.set_ylim(0, 2.4) 
    ax.set_xlabel('Experts', fontsize=FONT_AXIS)
    ax.set_ylabel('Normalized Per-port Throughput', fontsize=FONT_AXIS)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

# 5. 绘制下排：DP暴露折线图
for i, ep in enumerate(ep_list):
    ax = axes[1, i]
    for j, topo in enumerate(topologies):
        ax.plot(scales, data_dp[ep][j], marker='o', markersize=7,  # 增大打点标记
                label=topo, color=color_map[topo], linewidth=2.5, markeredgecolor='black')
    
    ax.set_title(f'{ep}', fontsize=FONT_TITLE, fontweight='bold', pad=10)
    ax.tick_params(labelsize=FONT_TICK)
    ax.set_ylabel('DP comm propoertion (%)', fontsize=FONT_AXIS)
    ax.set_xlabel('Experts', fontsize=FONT_AXIS)
    ax.grid(axis='both', linestyle='--', alpha=0.5)

# 6. 标号 (a)-(f)
for idx, ax in enumerate(axes.flat):
    ax.text(0.1, 0.95, f'({chr(ord("a") + idx)})', transform=ax.transAxes,
            fontsize=FONT_TITLE, fontweight='bold', va='top', ha='right')

# 7. 全局图例 (移至顶部)
handles, labels = axes[0, 0].get_legend_handles_labels()
# 调整 bbox_to_anchor 使图例位于 Title 之上
fig.legend(handles, labels, loc='upper center', ncol=4, fontsize=FONT_LEGEND, 
           frameon=False, bbox_to_anchor=(0.5, 1.02), handlelength=2.5, handleheight=1.5)

# 8. 调整布局
# 预留顶部空间给 Legend (top=0.92)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.subplots_adjust(hspace=0.4, wspace=0.25)

# 保存文件
plt.savefig('SparseMesh/fig/table6_Double.png', dpi=300, bbox_inches='tight')
plt.show()
