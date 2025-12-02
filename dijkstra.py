import numpy as np
import matplotlib.pyplot as plt

# -----------------------
# 输入点（忽略第三维 Z）
# -----------------------
A = np.array([1700, 1400])
B = np.array([1700, 1600])
C = np.array([1900, 1600])
start = np.array([1750, 1200])

# -----------------------
# “Dijkstra”：欧氏距离当作最短路
# -----------------------
def dist(p, q):
    return np.linalg.norm(p - q)

# -----------------------
# 动态决策：每 30s 重新规划
# -----------------------
def plan_stage(current_pos, available_points):
    """
    current_pos: 当前无人机位置
    available_points: 当前需要访问的目标点列表
    返回访问顺序（最近点优先）
    """
    if len(available_points) == 0:
        return []

    pts = available_points.copy()
    route = []

    pos = current_pos.copy()

    while pts:
        # 找最近的点当作TSP近似
        d = [dist(pos, p) for p in pts]
        idx = np.argmin(d)
        nxt = pts[idx]

        route.append(nxt)
        pos = nxt
        pts.pop(idx)

    return route


# ---------- Stage 1 ---------- #
# 0s 时：A 和 C 可见
stage1_points = [A, C]
stage1_route = plan_stage(start, stage1_points)

# t=30s 后无人机的位置
pos_at_30 = stage1_route[-1]

# ---------- Stage 2 ---------- #
# B 在 30s 出现
stage2_points = [B]
stage2_route = plan_stage(pos_at_30, stage2_points)

# -----------------------
# 合并完整路径
# -----------------------
full_route = [start] + stage1_route + stage2_route

# -----------------------
# 绘图
# -----------------------
plt.figure(figsize=(7, 7))

# 路线
xs = [p[0] for p in full_route]
ys = [p[1] for p in full_route]
plt.plot(xs, ys, "-o", linewidth=2)

# 标记点
plt.text(start[0], start[1], "Start")
plt.text(A[0], A[1], "A")
plt.text(B[0], B[1], "B")
plt.text(C[0], C[1], "C")

plt.scatter([start[0], A[0], B[0], C[0]],
            [start[1], A[1], B[1], C[1]],
            s=100)

plt.grid(True)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Dynamic TSP + Dijkstra Path\n(B appears at 30s)")
plt.show()
