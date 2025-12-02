'''
@Author: WANG Maonan
@Date: 2024-05-29 17:27:13
@Description: 对 SNIR 进行可视化
@LastEditTime: 2024-05-29 18:45:01
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.colors import ListedColormap, BoundaryNorm
from mpl_toolkits.mplot3d import Axes3D
from scipy import interpolate
from collections import defaultdict

def render_map(
        veh_trajectories, trajectories, cluster_point, img_path
) -> None:
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot() #111, projection='3d'
    img = mpimg.imread('sumo_envs/LONG_GANG/env/dayun_cropped.png')
    h, w = img.shape[:2]
    true_h = h-100
    true_w = w-100
    extent = (5-true_w, 5+true_w, 70-true_h, 70+true_h)

    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    # ax.set_zlabel('Time Step')
    ax.imshow(img, extent=extent, origin='upper')
    ax.set_title('Cluster Point and Vehicle Trajectories Over Time (every 10 steps)')

    # plot_3d_trajectories(ax, trajectories)
    # plot_3d_cluster_points(ax, veh_trajectories, cluster_point)
    # plot_trajectories(ax, trajectories)
    # plot_uav_dot(ax, trajectories)
    # plot_veh_trajectories(ax, veh_trajectories)
    plot_cluster_points(ax, cluster_point)

    plt.legend()
    plt.savefig(img_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_3d_trajectories(ax, trajectories):
    """绘制无人机轨迹信息（带时间维度）
    """
    for aircraft_id, pos in trajectories.items():
        if len(pos) < 2:  # 至少需要2个点才能插值
            continue
        x_coords, y_coords, _ = zip(*pos)
        time_steps = np.array(range(len(pos)))

        t = np.linspace(0, 1, len(pos))
        tck_x = interpolate.splrep(t, x_coords, s=0)
        tck_y = interpolate.splrep(t, y_coords, s=0)

        t_new = np.linspace(0, 1, 100)

        x_smooth = interpolate.splev(t_new, tck_x, der=0)
        y_smooth = interpolate.splev(t_new, tck_y, der=0)
        time_smooth = np.interp(t_new, [0, 1], [time_steps[0], time_steps[-1]])
        ax.plot(x_smooth, y_smooth, time_smooth, c='red', label="UAV trajectory")

        ax.scatter([x_coords[0]], [y_coords[0]], [time_steps[0]],
                   s=100, c='green', marker='o', label="Start")
        ax.scatter([x_coords[-1]], [y_coords[-1]], [time_steps[-1]],
                   s=100, c='red', marker='x', label="End")

def plot_3d_cluster_points(ax, cluster_point):
    """绘制聚类点（带时间维度）
    """
    if not cluster_point:
        return
    cluster_array = np.array(cluster_point)

    if cluster_array.ndim == 1:
        cluster_array = cluster_array.reshape(1, -1)

    sampled_indices = np.arange(0, len(cluster_array), 10)
    sampled_points = cluster_array[sampled_indices]

    x = sampled_points[:, 0]
    y = sampled_points[:, 1]

    # ax.plot(x, y, sampled_indices, c='red', label="Cluster trajectory")
    ax.scatter(x, y, sampled_indices,
               s=100, c='blue', marker='*',
               label="Cluster points (every 10 steps)")

    for i, (x_val, y_val, t_val) in enumerate(zip(x, y, sampled_indices)):
        ax.text(x_val, y_val, t_val, f't={t_val}', color='purple', fontsize=8)

def plot_trajectories(ax, trajectories):
    """绘制无人机轨迹信息（带时间维度）
    """
    for aircraft_id, pos in trajectories.items():

        x_coords, y_coords, _ = zip(*pos)
        t = np.linspace(0, 1, len(pos))
        tck_x = interpolate.splrep(t, x_coords, s=0)
        tck_y = interpolate.splrep(t, y_coords, s=0)

        t_new = np.linspace(0, 1, 100)

        x_smooth = interpolate.splev(t_new, tck_x, der=0)
        y_smooth = interpolate.splev(t_new, tck_y, der=0)
        ax.plot(x_smooth, y_smooth, c='springgreen', label="UAV trajectory")

        ax.scatter([x_coords[0]], [y_coords[0]],
                   s=100, c='green', marker='o', label="Start")

def plot_uav_dot(ax, trajectories):
    """绘制无人机轨迹信息（带时间维度）
    """
    for aircraft_id, pos in trajectories.items():

        uav_array = np.array(pos)
        sampled_indices = np.arange(0, len(uav_array), 5)
        sampled_points = uav_array[sampled_indices]

        x = sampled_points[:, 0]
        y = sampled_points[:, 1]

        ax.scatter(x, y, marker='^', c="green", label="TrafficMonitor Cluster points")
        for i, (x_val, y_val, t_val) in enumerate(zip(x, y, sampled_indices)):
            ax.text(x_val, y_val, f't={t_val}', color='green', fontsize=5)

def plot_veh_trajectories(ax, veh_trajectories):
    color_map = {
        '1125684496':"orange",
        '1131230259':"cyan",
    }
    grouped = defaultdict(list)

    for vehicle_id, pos in veh_trajectories.items():
        prefix = vehicle_id.split('#')[0]
        grouped[prefix].append(pos)
    for prefix, pos in grouped.items():
        x,y = [],[]
        for pose in pos:
            pos_array = np.array(pose)
            time_steps = np.arange(0, len(pos_array), 10)
            sampled_points = pos_array[time_steps]
            x.extend(sampled_points[:, 0])
            y.extend(sampled_points[:, 1])

        ax.scatter(x, y, marker='3', c = 'orange', label="vehicle position") #color_map.get(prefix)
    #     # for i, (x_val, y_val, t_val) in enumerate(zip(x, y, time_steps)):
    #     #     ax.text(x_val, y_val, f't={t_val}', color='purple', fontsize=8)
    # for vehicle_id, pos in veh_trajectories.items():
    #     cluster_array = np.array(pos)
    #     time_steps = np.arange(0, len(cluster_array), 10)
    #     sampled_points = cluster_array[time_steps]
    #
    #     x = sampled_points[:, 0]
    #     y = sampled_points[:, 1]
    #
    #     ax.scatter(x, y, marker='3', c='orange', label="vehicle position")  # color_map.get(prefix)
        # for i, (x_val, y_val, t_val) in enumerate(zip(x, y, time_steps)):
        #     ax.text(x_val, y_val, f't={t_val}', color='green', fontsize=3)


def plot_cluster_points(ax, cluster_point):
    cluster_array = np.array(cluster_point)
    sampled_indices = np.arange(0, len(cluster_array), 10)
    sampled_points = cluster_array[sampled_indices]

    x = sampled_points[:, 0]
    y = sampled_points[:, 1]

    ax.scatter(x, y, marker='*', c = "red", label="STCA Cluster points")
    for i, (x_val, y_val, t_val) in enumerate(zip(x, y, sampled_indices)):
        ax.text(x_val, y_val, f't={t_val}', color='white', fontsize=5)
