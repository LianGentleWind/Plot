import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as transforms
import numpy as np
import networkx as nx
import os

# 样式常量配置
COLOR_RECT_FILL = '#8FAADC'
COLOR_RECT_EDGE = '#000000'
COLOR_LINE = '#C00000'
LINE_WIDTH = 1.5

def generate_dual_outputs(N, K, R=12, rect_w=4.0, rect_h=1.2):
    # 1. 拓扑合法性检查
    if K >= N: K = N - 1
    if (N * K) % 2 != 0:
        print(f"Warning: N={N}, K={K} 无法构成正则图。已调整 K 为 {K-1}")
        K -= 1

    # 2. 均匀分布拓扑构建
    G = nx.Graph()
    G.add_nodes_from(range(N))
    max_s = N // 2
    num_pairs = K // 2
    
    if num_pairs > 0:
        potential_steps = list(range(1, max_s + (1 if K % 2 == 0 else 0)))
        potential_steps.reverse()
        indices = np.linspace(0, len(potential_steps) - 1, num_pairs, dtype=int)
        steps = [potential_steps[i] for i in set(indices)]
        for i in range(N):
            for s in steps:
                G.add_edge(i, (i + s) % N)
            if K % 2 != 0:
                G.add_edge(i, (i + N // 2) % N)

    # 3. 预计算连接点坐标
    # 矩形长边面向圆心，连接点在内侧长边中点
    conn_points = []
    node_centers = []
    for i in range(N):
        theta = 2 * np.pi * i / N
        cx, cy = R * np.cos(theta), R * np.sin(theta)
        r_inner = R - (rect_h / 2)
        px, py = r_inner * np.cos(theta), r_inner * np.sin(theta)
        conn_points.append((px, py))
        node_centers.append((cx, cy, np.degrees(theta)))

    # --- 输出 1: 标准拓扑图 (带矩形) ---
    render_and_save(N, K, G, conn_points, node_centers, rect_w, rect_h, 
                    show_rects=True, filename='topology_full.png', R=R)

    # --- 输出 2: 纯连线图 (无矩形，无装饰) ---
    render_and_save(N, K, G, conn_points, node_centers, rect_w, rect_h, 
                    show_rects=False, filename='topology_lines_only.png', R=R)

def render_and_save(N, K, G, conn_points, node_centers, rect_w, rect_h, show_rects, filename, R):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect('equal')
    ax.axis('off')

    # 绘制连线
    for u, v in G.edges():
        p1, p2 = conn_points[u], conn_points[v]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 
                color=COLOR_LINE, lw=LINE_WIDTH, alpha=0.9, zorder=1)

    # 绘制矩形 (仅在 show_rects 为 True 时)
    if show_rects:
        for i in range(N):
            cx, cy, deg = node_centers[i]
            angle = deg + 90
            rect = patches.FancyBboxPatch(
                (-rect_w/2, -rect_h/2), rect_w, rect_h,
                boxstyle='round,pad=0.1',
                facecolor=COLOR_RECT_FILL, edgecolor=COLOR_RECT_EDGE,
                linewidth=1.5, zorder=3
            )
            t = transforms.Affine2D().rotate_deg(angle).translate(cx, cy) + ax.transData
            rect.set_transform(t)
            ax.add_patch(rect)
        
        # 计算跳数作为标题
        try:
            if nx.is_connected(G):
                dia = nx.diameter(G)
                plt.title(f"N={N} K={K} | Max Hops: {dia}", fontsize=14, pad=20)
        except:
            pass

    # 统一视图范围
    limit = R + rect_w
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)

    # 保存无白边图片
    plt.savefig(filename, format='png', bbox_inches='tight', pad_inches=0, dpi=300, transparent=not show_rects)
    plt.close(fig)
    print(f"已导出: {filename}")

if __name__ == "__main__":
    # N=数量, K=度数
    generate_dual_outputs(N=16, K=7)
