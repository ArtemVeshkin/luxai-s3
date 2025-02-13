from luxai_s3.state import EnvObs

from .space import Space
from .fleet import Fleet
from .node import Node
from .action_type import ActionType
from .ship import Ship
from .base import (
    Config,
    get_match_step,
    get_match_number,
    LAST_MATCH_WHEN_RELIC_CAN_APPEAR,
    SPACE_SIZE,
    MAX_UNITS
)
import numpy as np


def env_obs_to_dict_obs(env_obs: EnvObs):
    dict_obs = {}
    dict_obs['units'] = {
        'position': np.array(env_obs.units.position),
        'energy': np.array(env_obs.units.energy)
    }
    dict_obs['units_mask'] = np.array(env_obs.units_mask)
    dict_obs['sensor_mask'] = np.array(env_obs.sensor_mask)
    dict_obs['map_features'] = {
        'energy': np.array(env_obs.map_features.energy),
        'tile_type': np.array(env_obs.map_features.tile_type)
    }
    dict_obs['relic_nodes'] = np.array(env_obs.relic_nodes)
    dict_obs['relic_nodes_mask'] = np.array(env_obs.relic_nodes_mask)
    dict_obs['team_points'] = np.array(env_obs.team_points)
    dict_obs['team_wins'] = np.array(env_obs.team_wins)
    dict_obs['steps'] = int(env_obs.steps)
    dict_obs['match_steps'] = int(env_obs.match_steps)
    
    return dict_obs



class State:
    def __init__(self, player: str, env_cfg=None):
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        self.team_id = 0 if self.player == "player_0" else 1
        self.opp_team_id = 1 if self.team_id == 0 else 0

        self._init()
        if env_cfg is not None:
            self.set_config(env_cfg)

    
    def _init(self):
        self.config = Config()
        self.space = Space()
        self.fleet = Fleet(self.team_id)
        self.opp_fleet = Fleet(self.opp_team_id)

        self.match_step = 0
        self.game_num = 0
        self.step = 0
        self.points_gain = 0

    
    def set_config(self, env_cfg):
        self.config.UNIT_MOVE_COST = env_cfg["unit_move_cost"]
        self.config.UNIT_SAP_COST = env_cfg["unit_sap_cost"]
        self.config.UNIT_SAP_RANGE = env_cfg["unit_sap_range"]
        self.config.UNIT_SENSOR_RANGE = env_cfg["unit_sensor_range"]

    
    def update(self, obs):
        if isinstance(obs, EnvObs):
            obs = env_obs_to_dict_obs(obs)

        self.step = obs['steps']
        self.match_step = get_match_step(self.step)
        self.game_num = int(obs['team_wins'].sum())
        match_number = get_match_number(self.step)

        if self.match_step == 0:
            # nothing to do here at the beginning of the match
            # just need to clean up some of the garbage that was left after the previous match
            self.fleet.clear()
            self.opp_fleet.clear()
            self.space.clear()
            self.space.move_obstacles(self.step, self.config)
            if match_number <= LAST_MATCH_WHEN_RELIC_CAN_APPEAR:
                self.space.clear_exploration_info(self.config)
            return

        self.points = int(obs['team_points'][self.team_id])
        self.opp_points = int(obs['team_points'][self.opp_team_id])

        reward = max(0, self.points - self.fleet.points)
        self.points_gain = reward

        self.space.update(self.step, obs, self.team_id, reward, self.config)
        self.fleet.update(obs, self.space, self.config)
        self.opp_fleet.update(obs, self.space, self.config)

        for ship in self.fleet:
            ship.node.visited_times += 1


    def get_obs(self):
        space_map = np.zeros((SPACE_SIZE, SPACE_SIZE), dtype=np.int8)
        for y in range(SPACE_SIZE):
            for x in range(SPACE_SIZE):
                node = self.space.get_node(x, y)
                space_map[x, y] = node.type.value

        relic_map = np.zeros((SPACE_SIZE, SPACE_SIZE), dtype=np.int8)
        for node in self.space.relic_nodes:
            x, y = node.coordinates
            relic_map[x, y] = 1
        
        reward_map = np.zeros((SPACE_SIZE, SPACE_SIZE), dtype=np.int8)
        for node in self.space.reward_nodes:
            x, y = node.coordinates
            reward_map[x, y] = 1

        ship_masks = np.zeros((SPACE_SIZE, SPACE_SIZE, MAX_UNITS))
        for idx, ship in enumerate(self.fleet.ships):
            if ship.node is not None:
                x, y = ship.node.coordinates
                ship_masks[x, y, idx] = 1

        obs = np.stack([
            space_map,
            relic_map,
            reward_map,
            *[ship_masks[:, :, idx] for idx in range(MAX_UNITS)]
        ], axis=-1)

        return obs


    def reset(self, env_cfg):
        self._init()
        self.set_config(env_cfg)