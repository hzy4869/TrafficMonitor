import numpy as np
import matplotlib.pyplot as plt

POINT_DATA = {
    "A": np.array([1408, 955, 0]),
    "B": np.array([1817, 1387, 20]),
    "C": np.array([2063, 1053, 0]),
    "D": np.array([1978, 776, 55]),
    "E": np.array([2553, 861, 0])
}
START_COORDS = np.array([1062, 1282])

FONT_SIZE = 14
FONT_WEIGHT = 'bold'

plt.figure(figsize=(12, 10))

start_x, start_y = START_COORDS
plt.scatter(start_x, start_y, s=200, c='blue', marker='s', zorder=5)
plt.text(start_x + 8, start_y + 8, "Start", fontsize=FONT_SIZE, color='blue', fontweight=FONT_WEIGHT, zorder=5)

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

plt.title("Environment", fontsize=FONT_SIZE, fontweight=FONT_WEIGHT)
plt.xlabel("X/meters", fontsize=FONT_SIZE, fontweight=FONT_WEIGHT)
plt.ylabel("Y/meters", fontsize=FONT_SIZE, fontweight=FONT_WEIGHT)

plt.tick_params(axis='both', which='major', labelsize=FONT_SIZE) 

plt.grid(True, linestyle='--', alpha=0.6)
plt.axis("equal")

plt.tight_layout()
plt.show()

## 保存
plt.tight_layout()
file_name = "environment_map.png"
plt.savefig(file_name, dpi=300, bbox_inches='tight', facecolor='white')
print(f"图片已成功保存至: {file_name}")
plt.close()