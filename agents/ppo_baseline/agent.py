import numpy as np
import os
from stable_baselines3 import PPO
from base import Global, warp_point  # Toll machines are located on the ground and warp_points around the world.


# direction (0 = center, 1 = up, 2 = right, 3 = down, 4 = left)
def direction_to(src, target):
    ds = target - src
    dx = ds[0]
    dy = ds[1]
    if dx == 0 and dy == 0:
        return 0
    if abs(dx) > abs(dy):
        if dx > 0:
            return 2 
        else:
            return 4
    else:
        if dy > 0:
            return 3
        else:
            return 1


class Agent():
    def __init__(self, player: str, env_cfg) -> None:
        self.player = player
        self.env_cfg = env_cfg
        self.team_id = 0 if self.player == "player_0" else 1
        self.opp_team_id = 1 - self.team_id

        # Preservation of relic-related information, etc. (original logic)
        self.relic_node_positions = []
        self.discovered_relic_nodes_ids = set()
        self.unit_explore_locations = dict()

        # Load the trained PPO model
        self.ppo_model = PPO.load("ppo_game_env_model")

    def compute_team_vision(self, tile_map, agent_positions):
        """
        Calculate the team's field of vision based on the location of all your own units:
          -For each unit, within its sensor range (range is -sensor_range ~ sensor_range),
            The contribution to (x+dx, y+dy) is: sensor_range+1-max (|dx|,|dy|).
          -If the target plot is Nebula (tile_map ==1), the contribution is subtracted from nebula_reduction (take 2 here, which can be adjusted according to the actual parameters).
          -After accumulating the contributions of all units, plots with a cumulative value greater than 0 are considered visible.
        """
        sensor_range = Global.UNIT_SENSOR_RANGE
        nebula_reduction = 2  # Can be adjusted according to the actual situation
        vision = np.zeros(tile_map.shape, dtype=np.float32)
        # agent_positions is a list, each element is (x, y) (all integers)
        for (x, y) in agent_positions:
            for dy in range(-sensor_range, sensor_range + 1):
                for dx in range(-sensor_range, sensor_range + 1):
                    new_x, new_y = warp_point(x + dx, y + dy)
                    contrib = sensor_range + 1 - max(abs(dx), abs(dy))
                    # If the plot is a nebula, subtract the visual reduction value
                    if tile_map[new_y, new_x] == 1:
                        contrib -= nebula_reduction
                    vision[new_y, new_x] += contrib
        visible_mask = vision > 0
        return visible_mask

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        # Extract the original Lux observation data
        unit_mask = np.array(obs["units_mask"][self.team_id])          # (max_units,)
        unit_positions = np.array(obs["units"]["position"][self.team_id]) # [x, y] of each unit
        unit_energys = np.array(obs["units"]["energy"][self.team_id])     # Unit energy information
        observed_relic_node_positions = np.array(obs["relic_nodes"])        # (max_relic_nodes, 2)
        observed_relic_nodes_mask = np.array(obs["relic_nodes_mask"])       # (max_relic_nodes,)
        
        # Get local map information from obs：
        # Since the game will not return the full tile_map, we use sensor_mask and map_features (~structure part) tile_map
        map_height = self.env_cfg["map_height"]
        map_width = self.env_cfg["map_width"]
        sensor_mask = np.array(obs["sensor_mask"])  # Boolean matrix, shape (map_height, map_width)
        tile_type_obs = np.array(obs["map_features"]["tile_type"])  # Real tile type, shape (map_height, map_width)
        # For the visible area, tile_map takes the real tile_type; fill in -1 for the invisible area to indicate unknown
        tile_map = np.where(sensor_mask, tile_type_obs, -1)
        
        # Initialization returns an array of actions
        actions = np.zeros((self.env_cfg["max_units"], 3), dtype=np.int32)
        available_unit_ids = np.where(unit_mask)[0]
        
        # For each controllable unit, construct a global observation (shape) consistent with the training time：(map_height, map_width, 3)）
        for unit_id in available_unit_ids:
            unit_pos = unit_positions[unit_id]    # [x, y]
            unit_energy = unit_energys[unit_id]     # Unit energy (not used directly in this example)
            
            # Construct the global observation obs_grid, the shape is(map_height, map_width, 3)
            obs_grid = np.zeros((map_height, map_width, 3), dtype=np.float32)
            
            # --- channel 0：tile_map ---
            # Use the tile_map constructed above (fill in -1 for invisible areas)
            obs_grid[..., 0] = tile_map
            
            # --- channel 1：relic_map ---
            # Initialize relic_map (all 0s), and then update the relic nodes in the field of view
            relic_map = np.zeros((map_height, map_width), dtype=np.int8)
            for i in range(len(observed_relic_node_positions)):
                if observed_relic_nodes_mask[i]:
                    x, y = observed_relic_node_positions[i]
                    x, y = int(x), int(y)
                    if 0 <= x < map_width and 0 <= y < map_height:
                        # Update relic information only in the visible area
                        if sensor_mask[y, x]:
                            relic_map[y, x] = 1
            obs_grid[..., 1] = relic_map
            
            # --- channel 2：agent_position ---
            # Mark only the location of the current unit (if the location is visible)
            agent_layer = np.zeros((map_height, map_width), dtype=np.int8)
            unit_x = int(unit_pos[0])
            unit_y = int(unit_pos[1])
            if 0 <= unit_x < map_width and 0 <= unit_y < map_height and sensor_mask[unit_y, unit_x]:
                agent_layer[unit_y, unit_x] = 1
            obs_grid[..., 2] = agent_layer
            
            # Expand the batch dimension and the shape becomes (1, map_height, map_width, 3)
            state = obs_grid[np.newaxis, ...]
            # Call the PPO model for prediction, deterministic=True to ensure the output of deterministic actions
            action, _ = self.ppo_model.predict(state, deterministic=True)
            action = int(action)
            # Map the model output action to the format required by the competition (here it is simply set to [action, 0, 0])
            actions[unit_id] = [action, 0, 0]
        
        return actions