import numpy as np
import heapq
import matplotlib.pyplot as plt

POINT_DATA = {
    "A": np.array([1700, 1400, 0]),
    "B": np.array([1800, 1500, 30]),
    "C": np.array([1900, 1400, 0]),
    "D": np.array([1900, 1300, 55]),
    "E": np.array([2000, 1300, 0])
}
START_COORDS = np.array([1650, 1550])
SPEED = 10.0 
DECISION_PERIOD = 20

FONT_SIZE = 14
FONT_WEIGHT = 'bold'
LINE_WIDTH = 3 

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
            if neighbor_node == current_node:
                continue

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

print(f"--- Simulation Start (Speed: {SPEED} units/step) ---")

while remaining_points_names:
    available_targets_names = set()
    for name, data in POINT_DATA.items():
        if data[2] <= current_time_step and name in remaining_points_names:
            available_targets_names.add(name)

    if not available_targets_names:
        current_time_step += DECISION_PERIOD
        continue 
    
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
    
    time_to_next_decision = DECISION_PERIOD - (current_time_step % DECISION_PERIOD)
    time_to_move = min(path_time_steps, time_to_next_decision)
    
    next_target_coords = POINT_DATA[next_point_name][:2].astype(float)
    total_dist_to_target = calculate_distance(current_coords, next_target_coords)
    
    direction_vector = next_target_coords - current_coords
    
    if total_dist_to_target == 0:
        travel_distance = 0
    else:
        travel_distance = time_to_move * SPEED

    if travel_distance >= total_dist_to_target:
        new_coords = next_target_coords
        steps_taken = total_dist_to_target / SPEED
        remaining_points_names.discard(next_point_name)
    else:
        steps_taken = time_to_move
        unit_vector = direction_vector / total_dist_to_target
        new_coords = current_coords + unit_vector * travel_distance

    current_time_step += steps_taken
    current_coords = new_coords
    full_path_coords.append(new_coords.tolist())


diag_val = SPEED * np.cos(np.pi/4)
move_options = [
    np.array([SPEED, 0]), np.array([-SPEED, 0]),
    np.array([0, SPEED]), np.array([0, -SPEED]),
    np.array([diag_val, diag_val]), np.array([diag_val, -diag_val]),
    np.array([-diag_val, diag_val]), np.array([-diag_val, -diag_val])
]

fitted_path = [full_path_coords[0]]
curr_fit_pos = np.array(full_path_coords[0])

for i in range(len(full_path_coords) - 1):
    target_pos = np.array(full_path_coords[i+1])
    
    while np.linalg.norm(target_pos - curr_fit_pos) > SPEED * 0.5:
        best_move = None
        min_dist_to_target = float('inf')
        
        for move in move_options:
            candidate_pos = curr_fit_pos + move
            dist = np.linalg.norm(target_pos - candidate_pos)
            if dist < min_dist_to_target:
                min_dist_to_target = dist
                best_move = candidate_pos
        
        curr_fit_pos = best_move
        fitted_path.append(curr_fit_pos.tolist())
    
    curr_fit_pos = target_pos
    fitted_path.append(curr_fit_pos.tolist())

final_steps_count = len(fitted_path)
coords_array = np.array(fitted_path)

plt.figure(figsize=(12, 10))

point_info = [
    (key, val[2])
    for key, val in POINT_DATA.items()
    if val[2] > 0
]
point_info.sort(key=lambda x: x[1])

phase_colors = ['gray', 'orange', 'red', 'green', 'purple', 'brown']
phase_labels = ["Initial Points"]
phase_steps = [0]

for i, (name, step) in enumerate(point_info):
    phase_steps.append(step)
    phase_labels.append(f"After {name} (Step {step})")

for i in range(len(phase_steps)):
    start_step = phase_steps[i]
    end_step = phase_steps[i+1] if i + 1 < len(phase_steps) else final_steps_count

    start_idx = int(min(start_step, final_steps_count - 1))
    end_idx = int(min(end_step, final_steps_count))
    
    if end_idx > start_idx:
        if start_idx > 0:
            current_traj = coords_array[start_idx-1:end_idx]
        else:
            current_traj = coords_array[start_idx:end_idx]

        plt.plot(current_traj[:, 0], current_traj[:, 1], 
                 color=phase_colors[i % len(phase_colors)], 
                 linewidth=LINE_WIDTH, 
                 label=f"Phase {i+1}: {phase_labels[i]}")

start_x, start_y = coords_array[0]
plt.scatter(start_x, start_y, s=200, c='blue', marker='s', zorder=5)
plt.text(start_x + 8, start_y + 8, "Start", fontsize=FONT_SIZE, color='blue', fontweight=FONT_WEIGHT, zorder=5)

end_x, end_y = coords_array[-1]
plt.scatter(end_x, end_y, s=200, c='black', marker='X', zorder=5)
plt.text(end_x + 8, end_y + 8, "End", fontsize=FONT_SIZE, color='black', fontweight=FONT_WEIGHT, zorder=5)

target_points = np.array([v[:2] for v in POINT_DATA.values()])
point_names = list(POINT_DATA.keys())
point_steps = np.array([v[2] for v in POINT_DATA.values()])

colors = plt.cm.get_cmap('tab10', len(target_points))
for i, (pt, name, step) in enumerate(zip(target_points, point_names, point_steps)):
    plt.scatter(pt[0], pt[1], s=250, c=[colors(i)], marker='*', 
                edgecolor='black', linewidth=1.5, zorder=4)
    
    if step == 0:
        label_text = f'{name} (Initial)'
    else:
        label_text = f'{name} (Step {step})'
    
    plt.text(pt[0] + 10, pt[1] + 10, label_text, fontsize=FONT_SIZE, 
             color=colors(i), fontweight=FONT_WEIGHT, zorder=5)

plt.title(f"Drone Trajectory (Total Steps: {final_steps_count})", fontsize=FONT_SIZE, fontweight=FONT_WEIGHT)
plt.xlabel("X/meters", fontsize=FONT_SIZE, fontweight=FONT_WEIGHT)
plt.ylabel("Y/meters", fontsize=FONT_SIZE, fontweight=FONT_WEIGHT)

plt.tick_params(axis='both', which='major', labelsize=FONT_SIZE) 

plt.grid(True, linestyle='--', alpha=0.6)
plt.axis("equal")

legend = plt.legend(loc='lower left', fontsize=FONT_SIZE) 
for text in legend.get_texts():
    text.set_fontweight(FONT_WEIGHT)

plt.tight_layout()
plt.show()