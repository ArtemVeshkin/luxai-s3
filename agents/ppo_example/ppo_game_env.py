import gym
from gym import spaces
import numpy as np

# Import global constants and helper functions in base
from base import Global, ActionType, SPACE_SIZE, warp_point, get_opposite

class PPOGameEnv(gym.Env):
    """
    An example of a simplified game environment using PPO training, the modified reward mechanism includes:
      1. Exploration reward: based on the agent's own vision (information returned by _get_obs),
         Reward +0.02 for every new (never seen before) plot discovered (instead of +0.1 for the previous full picture).
      2. Relic reward: Do not directly use the full picture information, but use the relic information returned by _get_obs;
         If there is a relic node in the team's field of vision, and the agent's current location shows the relic,:
           -If the agent adopts the center (hold still) action, the reward is +5.0;
           -If the agent moves into the ruins, the reward is +3.0;
         At the same time, each step of occupying the relic is only rewarded once (through self.claimed_relic control) and make the team score self.Score increased by 1.
      3. sap action: After obtaining _get_obs first, check the relic information and enemy units in the obs in the four neighboring domains, and the reward is +3.0 if the conditions are met, otherwise 0.1 will be deducted.
      4. Under the center action: if the agent is on the energy node, +1.0 will be rewarded; if it is on Nebula, 1.0 will be deducted.
      5. Asteroid penalty: -2.0 points will be deducted when moving to an asteroid (tile_map==2).
      
      【Note】:
      -Full picture information inside the environment (self.tile_map, self.relic_map, self.agent_position) is only used to generate _get_obs；
        The agent's decision-making and rewards should be based on the known information of the team returned by _get_obs, not the full picture information.
      -Therefore, both exploration rewards and relic rewards are modified to use _get_obs (see detailed comments below).
    """
    def __init__(self):
        super(PPOGameEnv, self).__init__()
        
        # Define the observation space: shape (SPACE_SIZE, SPACE_SIZE, 3)
        # Channel 0: Map tile type: -1 (unknown), 0 (open space), 1 (nebula), 2 (asteroid)
        # Channel 1: Relic mark: 0 means no relic, 1 means there is a relic
        # Channel 2: the location of the agent, one-hot indicates (only the location of the agent is 1)
        self.obs_shape = (SPACE_SIZE, SPACE_SIZE, 3)
        self.observation_space = spaces.Box(low=-1, high=2, shape=self.obs_shape, dtype=np.int8)
        
        # Define the action space: discrete 6 actions, corresponding to 6 actions in the ActionType
        self.action_space = spaces.Discrete(len(ActionType))
        
        self.max_steps = Global.MAX_STEPS_IN_MATCH
        self.current_step = 0
        
        # Full picture status: internal use (subsequent return to the visible part of the agent through _get_obs)
        self.tile_map = None     # Map tile (channel 0)
        self.relic_map = None    # Relic Mark (Channel 1)
        self.agent_position = None  # Agent location (channel 2)
        
        # The coordinates of the agent (x, y)
        self.agent_x = None
        self.agent_y = None
        
        # The team scores points, and each step of occupying the ruins will make self.score increased by 1
        self.score = 0
        
        # Mark whether the ruins have been occupied in the current step to prevent repeated rewards
        self.claimed_relic = False
        
        # Used to record the plots that the agent has explored (based on the field of view of _get_obs, not the full picture)
        self.visited = None
        
        self._init_state()

    def _init_state(self):
        """Initialize or reset the state of the environment, while initializing exploration records, enemy units, and energy nodes"""
        # Initialize the full map: tile_map randomly sets part of Nebula and Asteroid
        self.tile_map = np.zeros((SPACE_SIZE, SPACE_SIZE), dtype=np.int8)
        num_tiles = SPACE_SIZE * SPACE_SIZE
        num_nebula = int(num_tiles * 0.1)     # About 10% are nebulae
        num_asteroid = int(num_tiles * 0.05)    # About 5% are asteroids
        indices = np.random.choice(num_tiles, num_nebula + num_asteroid, replace=False)
        flat_tiles = self.tile_map.flatten()
        flat_tiles[indices[:num_nebula]] = 1
        flat_tiles[indices[num_nebula:]] = 2
        self.tile_map = flat_tiles.reshape((SPACE_SIZE, SPACE_SIZE))
        
        # Initialize the ruins mark: randomly select 3 locations and set them to have ruins (note: the ruins continue to exist and will not be cleared or scrolled)
        self.relic_map = np.zeros((SPACE_SIZE, SPACE_SIZE), dtype=np.int8)
        relic_indices = np.random.choice(num_tiles, 3, replace=False)
        flat_relic = self.relic_map.flatten()
        flat_relic[relic_indices] = 1
        self.relic_map = flat_relic.reshape((SPACE_SIZE, SPACE_SIZE))
        
        # Initialize the agent location: placed in the center of the map
        self.agent_position = np.zeros((SPACE_SIZE, SPACE_SIZE), dtype=np.int8)
        self.agent_x, self.agent_y = SPACE_SIZE // 2, SPACE_SIZE // 2
        self.agent_position[self.agent_y, self.agent_x] = 1
        
        # Initialize the exploration record: only the (visible) plots that the agent has explored are recorded
        self.visited = np.zeros((SPACE_SIZE, SPACE_SIZE), dtype=bool)
        # The initial position is visible
        self.visited[self.agent_y, self.agent_x] = True
        
        # Initialize enemy units: simply place them in a fixed position (for example, the upper left corner)
        self.enemy_position = np.zeros((SPACE_SIZE, SPACE_SIZE), dtype=np.int8)
        self.enemy_x, self.enemy_y = 0, 0
        self.enemy_position[self.enemy_y, self.enemy_x] = 1
        
        # Initialize the energy node: randomly select 2 positions and assign an energy value (for example, 20)
        self.energy_map = np.zeros((SPACE_SIZE, SPACE_SIZE), dtype=np.int8)
        num_energy_nodes = 2
        indices_energy = np.random.choice(num_tiles, num_energy_nodes, replace=False)
        flat_energy = self.energy_map.flatten()
        for idx in indices_energy:
            flat_energy[idx] = 20
        self.energy_map = flat_energy.reshape((SPACE_SIZE, SPACE_SIZE))
        
        self.current_step = 0
        self.score = 0
        self.claimed_relic = False

    def compute_team_vision(self):
        """
        Calculate the visual intensity of each plot on the entire map based on the field of view of each unit in the team.
        rules:
          -For each unit, within its sensor range (dx, dy range are -sensor_range ~ sensor_range)
            The visual contribution to the position (x+dx, y+dy) is:
                sensor_range + 1 - max(|dx|, |dy|)
          -If the plot is a nebula (tile_map ==1), the contribution value is reduced by nebula_reduction (for example, 2)
          -The contribution of all units to the same plot is accumulated, and plots with a cumulative value greater than 0 are regarded as visible
        Returns a Boolean matrix (full picture size) indicating which plots are in the team's field of vision.
        """
        sensor_range = Global.UNIT_SENSOR_RANGE
        nebula_reduction = 2  # Example parameters
        vision = np.zeros((SPACE_SIZE, SPACE_SIZE), dtype=np.float32)
        
        # Traverse all agents (there is only one agent here)
        unit_positions = np.argwhere(self.agent_position == 1)  # Each element is (y, x)
        for (y, x) in unit_positions:
            for dy in range(-sensor_range, sensor_range + 1):
                for dx in range(-sensor_range, sensor_range + 1):
                    new_x, new_y = warp_point(x + dx, y + dy)
                    contrib = sensor_range + 1 - max(abs(dx), abs(dy))
                    if self.tile_map[new_y, new_x] == 1:
                        contrib -= nebula_reduction
                    vision[new_y, new_x] += contrib
        visible_mask = vision > 0
        return visible_mask

    def _get_obs(self):
        """
        Construct observation data: based on the full picture information and the calculation results of the team's field of vision.
          -For each plot, if the plot is in the team's vision (that is, compute_team_vision is True),
            Then display the real information; otherwise:
              * tile_map channel display -1 (unknown)
              * relic_map and agent_position channels display 0
        Returns the observation data with the shape (SPACE_SIZE, SPACE_SIZE, 3).
        """
        visible_mask = self.compute_team_vision()
        
        obs_tile = np.where(visible_mask, self.tile_map, -1)
        obs_relic = np.where(visible_mask, self.relic_map, 0)
        obs_agent = np.where(visible_mask, self.agent_position, 0)
        
        obs = np.stack([obs_tile, obs_relic, obs_agent], axis=-1)
        return obs

    def step(self, action):
        """
        Update the environment status according to the input action, and return (observation, reward, done, info).
        
        The modified reward logic (all based on the known information of the team returned by _get_obs, not the full picture):
          1. Sap action: First call _get_obs, and then check the relic information in obs in the neighboring domain (Channel 1)
             With enemy units (using the global enemy_position), the reward for meeting the conditions is +3.0, otherwise 0.1 will be deducted;
             The sap action does not change the agent position.
          2. Non-sap actions (move or move):
             a. Update the agent position according to the action (if the target is an asteroid, keep it in place and deduct -2.0).
             b. [Exploration reward]: Use the observation data returned by _get_obs,
                For the team's vision (obs [...,0] ≠ -1) Newly discovered plots (not previously in self.Marked in visited) reward +0.02/grid,
                and update self.visited.
             c. [Relic reward]: Based on the relic information returned by _get_obs (Channel 1) judgment,
                If the current location of the agent shows ruins and has not been occupied before this step (self.claimed_relic is False):
                  -If you use the center (hold still) action, the reward is +5.0;
                  -Otherwise (move into the ruins) reward +3.0;
                At the same time make the team score self.The score is increased by 1, and the self is juxtaposed.claimed_relic = True;
                If the agent is not on the server, reset self.claimed_relic is false.
             d. [Additional rewards for Center actions]: If the agent selects the center action,
                Then check the global energy node (self.energy_map) and Nebula (self.tile_map ==1), reward +1.0 and deduct -1.0 respectively.
          3. Scroll the map as a whole every 20 steps (including tile_map, relic_map, enemy_position, and energy_map).
        
        [Key modification instructions] :
          -All reward judgments are based on the local (team) field of vision information returned by _get_obs,
            Instead of directly using the full picture information (such as self.relic_map == 1) to ensure that the training environment is consistent with the actual environment.
          -The exploration reward has also been modified to be based on the number of newly discovered plots in the agent's own vision (+0.02/grid).
        """
        self.current_step += 1
        reward = 0.0
        action_enum = ActionType(action)
        
        if action_enum == ActionType.sap:
            # Sap action: First get the team observation (obs) instead of directly using the full picture relic_map
            obs = self._get_obs()
            sap_reward = 0.0
            hit = False
            # Check the four neighbors
            for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
                tx, ty = warp_point(self.agent_x + dx, self.agent_y + dy)
                # Use the relic information in obs (Channel 1) to judge
                if self.enemy_position[ty, tx] == 1 and obs[ty, tx, 1] == 1:
                    sap_reward += 3.0  # Sap success rewards
                    hit = True
            if not hit:
                sap_reward -= 0.1  # Minor penalty for missed shots
            reward += sap_reward
            # Sap action does not update the agent location
            obs = self._get_obs()  # Update observation data
        else:
            # Non-sap actions: handle movements (including center actions)
            if action_enum in [ActionType.up, ActionType.right, ActionType.down, ActionType.left]:
                dx, dy = action_enum.to_direction()
            else:
                dx, dy = (0, 0)  # center action
            
            new_x, new_y = warp_point(self.agent_x + dx, self.agent_y + dy)
            
            # If the target is an asteroid (tile_map==2), it will not move and points will be deducted
            if self.tile_map[new_y, new_x] == 2:
                reward -= 2.0
                new_x, new_y = self.agent_x, self.agent_y  # Stay where you are
            else:
                # Update agent location
                self.agent_position[:, :] = 0
                self.agent_x, self.agent_y = new_x, new_y
                self.agent_position[self.agent_y, self.agent_x] = 1
            
            # Get the updated observation data based on the team's vision
            obs = self._get_obs()
            
            # [Exploration reward]: Judge newly discovered plots based on obs (channel 01-1 means visible)
            current_visible = (obs[..., 0] != -1)  # A Boolean matrix that represents the visible plot
            new_tiles = current_visible & (~self.visited)  # Newly discovered plots
            num_new = np.sum(new_tiles)
            if num_new > 0:
                reward += 0.02 * num_new
            # Update exploration record
            self.visited[new_tiles] = True
            
            # [Relic reward]: Based on relic information in obs (Channel 1)
            if obs[self.agent_y, self.agent_x, 1] == 1:
                if not self.claimed_relic:
                    if action_enum == ActionType.center:
                        reward += 5.0  # Stay still and get higher rewards on the ruins
                    else:
                        reward += 3.0  # The reward for moving to the ruins is lower
                    self.score += 1
                    self.claimed_relic = True
            else:
                self.claimed_relic = False
            
            # [Additional rewards for actions]: Check the energy node and Nebula punishment (the full picture information is still used here)
            if action_enum == ActionType.center:
                if self.energy_map[self.agent_y, self.agent_x] > 0:
                    reward += 1.0
                if self.tile_map[self.agent_y, self.agent_x] == 1:
                    reward -= 1.0

        # Scroll the map as a whole every 20 steps (including tile_map, relic_map, enemy_position, and energy_map)
        if (self.current_step % 20) == 0:
            self.tile_map = np.roll(self.tile_map, shift=1, axis=1)
            self.relic_map = np.roll(self.relic_map, shift=1, axis=1)
            self.enemy_position = np.roll(self.enemy_position, shift=1, axis=1)
            self.energy_map = np.roll(self.energy_map, shift=1, axis=1)
        
        done = self.current_step >= self.max_steps
        info = {"score": self.score, "step": self.current_step}
        return self._get_obs(), reward, done, info

    def reset(self):
        """Reset the environment and return to the initial observation"""
        self._init_state()
        return self._get_obs()
    
    def render(self, mode='human'):
        """Simple rendering: print the current number of steps and the map with the agent mark (the agent location is represented by 'A')"""
        display = self.tile_map.astype(str).copy()
        display[self.agent_y, self.agent_x] = 'A'
        print("Step:", self.current_step)
        print(display)
