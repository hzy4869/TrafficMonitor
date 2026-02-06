import numpy as np
import heapq
import matplotlib.pyplot as plt

POINT_DATA = {
    "A": np.array([1408, 955, 0]),
    "B": np.array([1817, 1387, 20]),
    "C": np.array([2063, 1053, 0]),
    "D": np.array([1978, 776, 55]),
    "E": np.array([2553, 861, 0])
}
START_COORDS = np.array([1062, 1282])
SPEED = 20.0 
DECISION_PERIOD = 10

def calculate_distance(p1, p2):
    return np.linalg.norm(p1[:2] - p2[:2])

def dijkstra_shortest_path(start_node, target_nodes, nodes_coords):
    pq = [(0, start_node, [start_node])]
    min_dist = {node: float('inf') for node in nodes_coords}
    min_dist[start_node] = 0

    while pq:
        dist, current_node, path = heapq.heappop(pq)
        if current_node in target_nodes:
            time_steps = dist / SPEED
            return path, dist, time_steps
        if dist > min_dist[current_node]:
            continue
        for neighbor_node, neighbor_coord in nodes_coords.items():
            if neighbor_node == current_node: continue
            cost = calculate_distance(nodes_coords[current_node], neighbor_coord)
            new_dist = dist + cost
            if new_dist < min_dist.get(neighbor_node, float('inf')):
                min_dist[neighbor_node] = new_dist
                new_path = path + [neighbor_node]
                heapq.heappush(pq, (new_dist, neighbor_node, new_path))
    return None, float('inf'), float('inf')

current_coords = START_COORDS.astype(float)
current_time_step = 0.0
full_path_coords = [current_coords[:2].tolist()]
remaining_points_names = set(POINT_DATA.keys())

print(f"{'='*60}")
print(f"🚀 模拟开始 | 初始位置: {START_COORDS} | 速度: {SPEED}")
print(f"{'='*60}\n")

while remaining_points_names:
    # 1. 检查当前哪些点已解锁，哪些还在等待
    available_targets_names = set()
    locked_targets = []
    for name, data in POINT_DATA.items():
        if name in remaining_points_names:
            if data[2] <= current_time_step:
                available_targets_names.add(name)
            else:
                locked_targets.append(f"{name}(T-{data[2]})")

    # 打印当前周期状态
    print(f"⏰ [时间步: {current_time_step:.1f}]")
    print(f"📍 当前坐标: {np.round(current_coords, 1)}")
    print(f"📦 剩余目标: {remaining_points_names}")
    if locked_targets:
        print(f"🔒 尚未解锁的点: {locked_targets}")

    # 2. 决策逻辑
    if not available_targets_names:
        print(f"💡 决策: ❌ 当前无可用目标，原地等待 {DECISION_PERIOD}s...")
        current_time_step += DECISION_PERIOD
        print("-" * 30)
        continue 
    
    # 使用Dijkstra寻找最近的已解锁点
    nodes_for_dijkstra = {"CURRENT_POS": current_coords}
    for name in available_targets_names:
        nodes_for_dijkstra[name] = POINT_DATA[name][:2]

    path_names, path_dist, path_time_steps = dijkstra_shortest_path(
        "CURRENT_POS", 
        available_targets_names, 
        nodes_for_dijkstra
    )

    if not path_names:
        break
    
    next_point_name = path_names[1] 
    print(f"✅ 可选目标: {available_targets_names}")
    print(f"🎯 决策: 航向点 {next_point_name} | 距离: {path_dist:.1f}m | 预计耗时: {path_time_steps:.1f}s")

    # 3. 移动逻辑
    time_to_next_decision = DECISION_PERIOD - (current_time_step % DECISION_PERIOD)
    if time_to_next_decision == 0: time_to_next_decision = DECISION_PERIOD
    
    time_to_move = min(path_time_steps, time_to_next_decision)
    
    next_target_coords = POINT_DATA[next_point_name][:2].astype(float)
    total_dist_to_target = calculate_distance(current_coords, next_target_coords)
    
    if travel_distance := (time_to_move * SPEED) >= total_dist_to_target:
        new_coords = next_target_coords
        steps_taken = total_dist_to_target / SPEED
        remaining_points_names.discard(next_point_name)
        print(f"🏁 动作: 到达目标点 {next_point_name}！")
    else:
        steps_taken = time_to_move
        unit_vector = (next_target_coords - current_coords) / total_dist_to_target
        new_coords = current_coords + unit_vector * (steps_taken * SPEED)
        print(f"🚢 动作: 向 {next_point_name} 推进了 {steps_taken * SPEED:.1f}m")

    current_time_step += steps_taken
    current_coords = new_coords
    full_path_coords.append(new_coords.tolist())
    print("-" * 30)

print(f"\n{'='*60}")
print(f"🏁 任务完成！总耗时: {current_time_step:.1f}s")
print(f"{'='*60}")

# 后续的轨迹拟合与绘图代码保持不变...