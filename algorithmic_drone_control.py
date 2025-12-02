#!/usr/bin/env python3
"""
Algorithmic Drone Control Script for SUMO Environment
===================================================

This script uses density peaks clustering to automatically calculate optimal
endpoints for the drone and plan paths to reach those endpoints.

The algorithm:
1. Collects vehicle positions from the environment
2. Uses density peaks clustering to find optimal coverage points
3. Plans drone path to reach the calculated endpoints
4. Continuously updates as vehicles move

Author: Assistant
Date: 2024
"""

import os
import sys
import time
import random

import numpy as np
import math
from typing import Dict, Any, List, Tuple, Optional
from collections import deque

from numpy import ndarray

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from env_utils.ac_env import ACEnvironment
from env_utils.ac_wrapper_modified import ACEnvWrapper
from env_utils.vis_snir import render_map
from tshub.utils.get_abs_path import get_abs_path

path_convert = get_abs_path(__file__)

def custom_update_cover_radius(position:List[float], communication_range:float) -> float:
    """自定义的更新地面覆盖半径的方法, 在这里实现您的自定义逻辑

    Args:
        position (List[float]): 飞行器的坐标, (x,y,z)
        communication_range (float): 飞行器的通行范围
    """
    height = position[2]
    cover_radius = height / np.tan(math.radians(75/2))
    return cover_radius

class DensityPeaksClustering:
    """Density Peaks Clustering implementation for drone endpoint calculation"""
    
    def __init__(self, radius: float = 1.5, n_centers: int = 2):
        """
        Initialize the clustering algorithm
        
        Args:
            dc: Cutoff distance for local density calculation
            n_centers: Number of cluster centers to find
        """
        self.dc = radius
        self.n_centers = n_centers
    
    def density_peaks_clustering(self, veh_pos: np.ndarray) -> np.ndarray:
        """
        Density Peaks Clustering implementation
        
        Args:
            veh_pos: (N, 2) array of 2D coordinates

        Returns:
            centers: Array of cluster center coordinates
        """
        N = len(veh_pos)
        dist = np.linalg.norm(veh_pos[:, None, :] - veh_pos[None, :, :], axis=2)
        rho = np.sum(dist < self.dc, axis=1) - 1  # Exclude self
        delta = np.zeros(N)
        for i in range(N):
            higher = np.where(rho > rho[i])[0]
            if higher.size > 0:
                delta[i] = np.min(dist[i, higher])
            else:
                delta[i] = np.max(dist[i])
        gamma = rho * delta
        centers_idx = np.argsort(-gamma)[:1]
        center_point = veh_pos[centers_idx]
        center_point = center_point.reshape(-1)
        return center_point

class DronePathPlanner:
    """Path planning for drone to reach calculated endpoints"""
    
    def __init__(self, drone_speed: int = 10, planning_horizon: int = 10):
        """
        Initialize the path planner
        
        Args:
            drone_speed: Drone movement speed in m/s
            planning_horizon: Number of steps to plan ahead
        """
        self.drone_speed = drone_speed
        self.planning_horizon = planning_horizon
        self.current_target = None
        self.path_history = deque(maxlen=100)
        
    def calculate_target(self, drone_pos: List[float], 
                        vehicle_positions: np.ndarray,
                        cover_radius: float) -> ndarray | None:
        """
        Calculate the best target position for the drone
        
        Args:
            drone_pos: Current drone position [x, y, z]
            vehicle_positions: Array of vehicle positions
            cover_radius: Drone coverage radius
            
        Returns:
            target_pos: Target position [x, y, z] or None if no good target
        """
        if len(vehicle_positions) == 0:
            return None
        
        # Use clustering to find optimal coverage points
        clustering = DensityPeaksClustering(radius=cover_radius, n_centers=1)
        centers = clustering.density_peaks_clustering(vehicle_positions)

        return centers  # Keep same height
    
    def plan_action(self, drone_pos: List[float], target_pos: List[float]) -> int:
        """
        Plan the next action to move towards target
        
        Args:
            drone_pos: Current drone position [x, y, z]
            target_pos: Target position [x, y, z]
            
        Returns:
            action: Discrete action (0-7) to take
        """
        if target_pos is None:
            return random.randint(0,7)
        # Calculate direction vector
        dx = target_pos[0]
        dy = target_pos[1]
        
        # Determine action based on direction
        # Action mapping: 0=Right, 1=Up-Right, 2=Up, 3=Up-Left, 4=Left, 5=Down-Left, 6=Down, 7=Down-Right
        # Convert to angle and determine action
        angle = math.atan2(dy, dx)
        angle_deg = math.degrees(angle)
        
        # Map angle to action
        if -22.5 <= angle_deg < 22.5:
            return 0  # Right
        elif 22.5 <= angle_deg < 67.5:
            return 1  # Up-Right
        elif 67.5 <= angle_deg < 112.5:
            return 2  # Up
        elif 112.5 <= angle_deg < 157.5:
            return 3  # Up-Left
        elif 157.5 <= angle_deg < 202.5 or angle_deg < -157.5:
            return 4  # Left
        elif -157.5 <= angle_deg < -112.5:
            return 5  # Down-Left
        elif -112.5 <= angle_deg < -67.5:
            return 6  # Down
        elif -67.5 <= angle_deg < -22.5:
            return 7  # Down-Right
        else: return 8


class AlgorithmicDroneController:
    """Main controller that integrates clustering and path planning"""
    
    def __init__(self, sumo_cfg: str, aircraft_inits: Dict[str, Any], 
                 num_seconds: int = 3600, use_gui: bool = True):
        """
        Initialize the algorithmic drone controller
        
        Args:
            sumo_cfg: Path to SUMO configuration file
            aircraft_inits: Aircraft initialization parameters
            num_seconds: Simulation duration in seconds
            use_gui: Whether to show SUMO GUI
        """
        self.sumo_cfg = sumo_cfg
        self.aircraft_inits = aircraft_inits
        self.num_seconds = num_seconds
        self.use_gui = use_gui
        
        # Initialize environment
        self.ac_env = ACEnvironment(
            sumo_cfg=sumo_cfg,
            num_seconds=num_seconds,
            aircraft_inits=aircraft_inits,
            use_gui=use_gui
        )
        
        self.ac_wrapper = ACEnvWrapper(
            env=self.ac_env, 
            aircraft_inits=aircraft_inits
        )
        
        # Initialize components
        self.path_planner = DronePathPlanner()
        
        # Performance tracking
        self.episode_rewards = []
        self.episode_cover_counts = []
        self.episode_steps = []
        self.current_episode_reward = 0
        self.current_episode_steps = 0
        
        # Algorithm parameters
        self.current_target = None
        self.steps_since_update = 0
        
    def extract_vehicle_positions(self, state: Dict) -> np.ndarray:
        """
        Extract vehicle positions from environment state
        
        Args:
            state: Environment state dictionary
            
        Returns:
            vehicle_positions: Array of vehicle positions
        """
        vehicle_positions = []
        
        # Extract from wrapper's vehicle tracking
        for vehicle_id, vehicle_pos in self.ac_wrapper.latest_veh_pos.items():
            if len(vehicle_pos) >= 2:
                vehicle_positions.append([vehicle_pos[0], vehicle_pos[1]])
        
        return np.array(vehicle_positions) if vehicle_positions else np.array([])
    
    def get_algorithmic_action(self, state: Dict) -> int:
        """
        Get action using the clustering algorithm
        
        Args:
            state: Current environment state
            
        Returns:
            action: Discrete action (0-7) to take
        """
        # Get current drone position
        drone_pos = self.ac_wrapper.latest_ac_pos.get('drone_1', [0, 0, 0])
        cover_radius = self.ac_wrapper.latest_cover_radius.get('drone_1', 200)
        
        # Extract vehicle positions
        vehicle_positions = self.extract_vehicle_positions(state)

        self.current_target = self.path_planner.calculate_target(
            drone_pos, vehicle_positions, cover_radius
        )
        if self.current_target is not None and len(self.current_target) > 0:
            # Plan action to reach target
            action = self.path_planner.plan_action(drone_pos, self.current_target)
        else:
            action = 0
        return {"drone_1": (10, action)}
    
    def reset_environment(self):
        """Reset the environment and start a new episode"""
        print("\n" + "="*50)
        print("RESETTING ENVIRONMENT")
        print("="*50)
        
        # Record previous episode stats
        if self.current_episode_steps > 0:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_steps.append(self.current_episode_steps)
            print(f"Episode completed:")
            print(f"  Total Reward: {self.current_episode_reward:.2f}")
            print(f"  Steps: {self.current_episode_steps}")
            print(f"  Average Reward per Step: {self.current_episode_reward/self.current_episode_steps:.2f}")
        
        # Reset environment
        state, info = self.ac_wrapper.reset()
        self.current_episode_reward = 0
        self.current_episode_steps = 0
        self.current_target = None
        self.steps_since_update = 0
        
        print("Environment reset complete!")
        print(f"Current drone position: {self.ac_wrapper.latest_ac_pos.get('drone_1', 'Unknown')}")
        return state, info
    
    def display_info(self, state: Dict, reward: float, step: int):
        """Display current algorithm information"""
        drone_pos = self.ac_wrapper.latest_ac_pos.get('drone_1', [0, 0, 0])
        cover_count = state.get('cover_counts', [0])[0] if isinstance(state, dict) else 0
        vehicle_positions = self.extract_vehicle_positions(state)
        
        # Clear screen
        os.system('cls' if os.name == 'nt' else 'clear')
        
        # print("="*60)
        # print("ALGORITHMIC DRONE CONTROL - DENSITY PEAKS CLUSTERING")
        # print("="*60)
        # print(f"Drone Position: ({drone_pos[0]:.1f}, {drone_pos[1]:.1f}, {drone_pos[2]:.1f})")
        # print(f"Current Target: {self.current_target if self.current_target is not None else 'None'}")
        # print(f"Covered Vehicles: {cover_count}")
        # print(f"Total Vehicles: {len(vehicle_positions)}")
        # print(f"Current Reward: {reward:.2f}")
        # print(f"Episode Reward: {self.current_episode_reward:.2f}")
        # print(f"Episode Steps: {self.current_episode_steps}")
        # print()
        # print("Algorithm Status:")
        # print(f"  Steps since target update: {self.steps_since_update}")
        # print(f"  Clustering centers found: {len(vehicle_positions) > 0}")
        # print("="*60)
    
    def run_algorithmic_control(self, max_episodes: int = 1):
        """Main loop for algorithmic drone control"""
        print("Starting Algorithmic Drone Control")
        print("Algorithm: Density Peaks Clustering for Endpoint Calculation")
        print("="*60)
        
        # Initial reset
        state, info = self.reset_environment()
        
        episode = 0
        paused = False
        
        while episode < max_episodes:
            try:
                # Get algorithmic action
                action = self.get_algorithmic_action(state)

                state, reward, truncated, done, info = self.ac_wrapper.step(action)

                # Update episode statistics
                self.current_episode_reward += reward
                self.current_episode_steps += 1

                # Display information
                self.display_info(state, reward, self.current_episode_steps)

                # Check if episode is done
                if done:
                    print(f"\nEpisode {episode + 1} finished! Total reward: {self.current_episode_reward:.2f}")
                    episode += 1
                    if episode < max_episodes:
                        state, info = self.reset_environment()

            except Exception as e:
                print(f"Error during simulation: {e}")
                break
        
        # Print final statistics
        self.print_final_statistics()
        
        # Cleanup
        self.ac_wrapper.close()
    
    def print_final_statistics(self):
        """Print final performance statistics"""
        print("\n" + "="*50)
        print("FINAL ALGORITHM PERFORMANCE STATISTICS")
        print("="*50)
        
        if self.episode_rewards:
            avg_reward = np.mean(self.episode_rewards)
            avg_steps = np.mean(self.episode_steps)
            avg_reward_per_step = np.mean([r/s for r, s in zip(self.episode_rewards, self.episode_steps)])
            
            print(f"Total Episodes: {len(self.episode_rewards)}")
            print(f"Average Episode Reward: {avg_reward:.2f}")
            print(f"Average Episode Steps: {avg_steps:.1f}")
            print(f"Average Reward per Step: {avg_reward_per_step:.2f}")
            print(f"Best Episode Reward: {max(self.episode_rewards):.2f}")
            print(f"Worst Episode Reward: {min(self.episode_rewards):.2f}")
        
        print("\nAlgorithmic control session ended.")


def main():
    """Main function to run algorithmic drone control"""
    
    # Configuration
    sumo_cfg = "./sumo_envs/LONG_GANG/env/osm.sumocfg"
    
    aircraft_inits = {
        'drone_1': {
            "aircraft_type": "drone",
            "action_type": "horizontal_movement",
            "position": (1750, 1000, 50), "speed": 10, "heading": (1, 1, 0), "communication_range": 50,
            "if_sumo_visualization": True, "img_file": path_convert('./asset/drone.png'),
            "custom_update_cover_radius": custom_update_cover_radius
        },
    }
    
    # Check if SUMO configuration file exists
    if not os.path.exists(sumo_cfg):
        print(f"Error: SUMO configuration file not found at {sumo_cfg}")
        return
    
    try:
        # Create and run algorithmic controller
        controller = AlgorithmicDroneController(
            sumo_cfg=sumo_cfg,
            aircraft_inits=aircraft_inits,
            num_seconds=200,
            use_gui=True  # Set to False for faster simulation
        )
        
        controller.run_algorithmic_control(max_episodes=5)
        
    except Exception as e:
        print(f"Error during algorithmic control: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 