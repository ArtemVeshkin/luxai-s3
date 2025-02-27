from luxai_s3.state import EnvObs

from .space import Space
from .fleet import Fleet
from .node import Node
from .action_type import ActionType, _DIRECTIONS
from .ship import Ship
from .base import (
    Config,
    get_match_step,
    get_match_number,
    get_opposite,
    is_team_sector,
    LAST_MATCH_WHEN_RELIC_CAN_APPEAR,
    SPACE_SIZE,
    MAX_UNITS,
    RELIC_REWARD_RANGE
)
from copy import deepcopy
import numpy as np
from scipy.signal import convolve2d


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
        self.our_wins = 0
        self.opp_wins = 0
        self.points = 0
        self.opp_points = 0

    
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
        self.our_wins = obs['team_wins'][self.team_id]
        self.opp_wins = obs['team_wins'][self.opp_team_id]
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


    def _get_global_info_features(self):
        config: Config = self.config
        ships = self._get_ships(self.fleet)

        # numerical
        match_step = self.match_step
        step = self.step
        points_gain = self.points_gain
        our_wins = self.our_wins
        opp_wins = self.opp_wins
        our_points = self.points
        opp_points = self.opp_points

        alive_ships = 0
        ships_with_energy = 0
        harvesting_ships = 0
        for ship in ships:
            if ship['node'] is not None:
                alive_ships += 1
                if ship['node'].reward:
                    harvesting_ships += 1
                if ship['energy'] > 0:
                    ships_with_energy += 1
        harvesting_ships_percent = harvesting_ships / float(alive_ships) if alive_ships > 0 else 0
        found_relics_count = len(self._get_relic_nodes())
        found_rewards_count = len(self._get_reward_nodes())
        move_cost = config.UNIT_MOVE_COST
        sap_cost  = config.UNIT_SAP_COST
        sap_range = config.UNIT_SAP_RANGE
        unit_sensor_range = config.UNIT_SENSOR_RANGE
        nebula_energy_reduction = config.NEBULA_ENERGY_REDUCTION

        # cat
        game_num = self.game_num
        is_winning_by_games = int(our_wins > opp_wins)
        is_winning_by_points = int(our_points > opp_points)

        relic_can_appear = int(match_step <= 50 and game_num <= 2)

        move_cost_cat = int(move_cost) - 1
        sap_range_cat = int(sap_range) - 3
        unit_sensor_range_cat = int(unit_sensor_range) - 1

        obstacles_movement_period_found = config.OBSTACLE_MOVEMENT_PERIOD_FOUND
        obstacles_movement_period_cat = 0
        if obstacles_movement_period_found:
            if config.OBSTACLE_MOVEMENT_PERIOD == 40:
                obstacles_movement_period_cat = 4
            elif config.OBSTACLE_MOVEMENT_PERIOD == 20:
                obstacles_movement_period_cat = 3
            elif config.OBSTACLE_MOVEMENT_PERIOD == 10:
                obstacles_movement_period_cat = 2
            else:
                obstacles_movement_period_cat = 1

        obstacles_movement_direction_found = config.OBSTACLE_MOVEMENT_DIRECTION_FOUND
        obstacles_movement_direction_cat = 0
        if obstacles_movement_direction_found:
            obstacles_movement_direction_cat = 1 + int(config.OBSTACLE_MOVEMENT_DIRECTION == (1, -1))

        nebula_energy_reduction_cat = 0
        nebula_energy_reduction_found = config.NEBULA_ENERGY_REDUCTION_FOUND
        if nebula_energy_reduction_found:
            if config.NEBULA_ENERGY_REDUCTION <= 5:
                nebula_energy_reduction_cat = config.NEBULA_ENERGY_REDUCTION + 1
            if config.NEBULA_ENERGY_REDUCTION == 25:
                nebula_energy_reduction_cat = 5

        all_relics_found = int(config.ALL_RELICS_FOUND)
        all_rewards_found = int(config.ALL_REWARDS_FOUND)

        # concat features
        num_features = np.array([
            match_step / 100.,
            step / 500.,
            points_gain,
            our_wins / 2.,
            opp_wins / 2.,
            our_points / 200.,
            opp_points / 200.,
            alive_ships / 16.,
            ships_with_energy / 16.,
            harvesting_ships / 16.,
            harvesting_ships_percent,
            found_relics_count / 3.,
            found_rewards_count / 3.,
            move_cost / 3.,
            sap_cost / 30.,
            sap_range / 3.,
            unit_sensor_range / 2.,
            nebula_energy_reduction / 5.,
        ])

        cat_features = np.array([
            match_step, # 101
            step, # 505
            game_num, # 5
            is_winning_by_games, # 2
            is_winning_by_points, # 2
            relic_can_appear, # 2
            move_cost_cat, # 6
            sap_range_cat, # 6
            unit_sensor_range_cat, # 4
            obstacles_movement_period_cat, # 5
            obstacles_movement_direction_cat, # 3
            nebula_energy_reduction_cat, # 7
            all_relics_found, # 2
            all_rewards_found, # 2
        ])

        num_features_packed = np.concat([
            num_features,
            np.zeros((SPACE_SIZE * SPACE_SIZE - len(num_features)))
        ]).reshape((SPACE_SIZE, SPACE_SIZE))

        cat_features_packed = np.concat([
            cat_features,
            np.zeros((SPACE_SIZE * SPACE_SIZE - len(cat_features)))
        ]).reshape((SPACE_SIZE, SPACE_SIZE))

        return num_features_packed, cat_features_packed


    def get_obs(self):
        is_explored = np.zeros((SPACE_SIZE, SPACE_SIZE))
        is_visible = np.zeros((SPACE_SIZE, SPACE_SIZE))
        is_empty = np.zeros((SPACE_SIZE, SPACE_SIZE))
        is_nebula = np.zeros((SPACE_SIZE, SPACE_SIZE))
        is_asteroid = np.zeros((SPACE_SIZE, SPACE_SIZE))
        is_explored_for_relic = np.zeros((SPACE_SIZE, SPACE_SIZE))
        is_explored_for_reward = np.zeros((SPACE_SIZE, SPACE_SIZE))
        real_energy = np.zeros((SPACE_SIZE, SPACE_SIZE))
        is_pos_real_energy_zone = np.zeros((SPACE_SIZE, SPACE_SIZE))
        is_team_sector_map = np.zeros((SPACE_SIZE, SPACE_SIZE))
        can_go_up = np.ones((SPACE_SIZE, SPACE_SIZE))
        can_go_right = np.ones((SPACE_SIZE, SPACE_SIZE))
        can_go_down = np.ones((SPACE_SIZE, SPACE_SIZE))
        can_go_left = np.ones((SPACE_SIZE, SPACE_SIZE))

        space_nodes = self._get_space_nodes()
        for y in range(SPACE_SIZE):
            for x in range(SPACE_SIZE):
                node = space_nodes[x][y]
                is_explored[x, y] = float(node.type.value != -1)
                is_visible[x, y] = float(node.is_visible)
                is_empty[x, y] = float(node.type.value == 0)
                is_nebula[x, y] = float(node.type.value == 1)
                is_asteroid[x, y] = float(node.type.value == 2)
                is_explored_for_relic[x, y] = float(node.explored_for_relic)
                is_explored_for_reward[x, y] = float(node.explored_for_reward)
                real_energy[x, y] = node.energy if node.energy is not None else 0.
                is_pos_real_energy_zone[x, y] = float(node.energy is not None and (node.energy > 0))
                is_team_sector_map[x, y] = float(is_team_sector(0, x, y))

                for action, action_map in zip([1, 2, 3, 4], [
                    can_go_up, can_go_right, can_go_down, can_go_left
                ]):
                    direction = _DIRECTIONS[action]
                    next_x = direction[0] + x
                    next_y = direction[1] + y
                    if next_x > SPACE_SIZE - 1 or next_x < 0 or next_y > SPACE_SIZE - 1 or next_y < 0:
                        action_map[x, y] = 0.
                    elif not node.is_walkable:
                        action_map[x, y] = 0.

        is_relic = np.zeros((SPACE_SIZE, SPACE_SIZE), dtype=np.int8)
        for node in self._get_relic_nodes():
            x, y = node.coordinates
            is_relic[x, y] = 1.

        reward_size = 2 * RELIC_REWARD_RANGE + 1
        relic_area = convolve2d(
            is_relic,
            np.ones((reward_size, reward_size), dtype=np.int32),
            mode="same",
            boundary="fill",
            fillvalue=0,
        )

        is_reward = np.zeros((SPACE_SIZE, SPACE_SIZE), dtype=np.int8)
        for node in self._get_reward_nodes():
            x, y = node.coordinates
            is_reward[x, y] = 1.

        ship_masks = np.zeros((SPACE_SIZE, SPACE_SIZE, MAX_UNITS))

        own_ships = np.zeros((SPACE_SIZE, SPACE_SIZE))
        own_ships_energy = np.zeros((SPACE_SIZE, SPACE_SIZE))
        own_ship_is_harvesting = np.zeros((SPACE_SIZE, SPACE_SIZE))
        for idx, ship in enumerate(self._get_ships(self.fleet)):
            if ship['node'] is not None:
                x, y = ship['node'].coordinates
                ship_masks[x, y, idx] = 1.
                own_ships[x, y] += 1.
                own_ships_energy[x, y] += ship['energy']
                own_ship_is_harvesting[x, y] = float(ship['node'].reward)

        opp_ships = np.zeros((SPACE_SIZE, SPACE_SIZE))
        opp_ships_energy = np.zeros((SPACE_SIZE, SPACE_SIZE))
        opp_ship_is_harvesting = np.zeros((SPACE_SIZE, SPACE_SIZE))
        for idx, ship in enumerate(self._get_ships(self.opp_fleet)):
            if ship['node'] is not None:
                x, y = ship['node'].coordinates
                opp_ships[x, y] += 1.
                opp_ships_energy[x, y] += ship['energy']
                opp_ship_is_harvesting[x, y] = float(ship['node'].reward)

        dist_to_center_x = np.zeros((SPACE_SIZE, SPACE_SIZE))
        dist_to_center_y = np.zeros((SPACE_SIZE, SPACE_SIZE))
        for x in range(SPACE_SIZE // 2):
            dist_to_center_x[SPACE_SIZE // 2 + x, :] = x
            dist_to_center_x[SPACE_SIZE // 2 - x, :] = x
        for y in range(SPACE_SIZE // 2):
            dist_to_center_y[:, SPACE_SIZE // 2 + y] = y
            dist_to_center_y[:, SPACE_SIZE // 2 - y] = y

        num_features, cat_features = self._get_global_info_features()

        obs = np.stack([
            *[ship_masks[:, :, idx] for idx in range(MAX_UNITS)],
            num_features,
            cat_features,
            is_explored,
            is_visible,
            is_empty,
            is_nebula,
            is_asteroid,
            is_explored_for_relic,
            is_explored_for_reward,
            real_energy / 10.,
            is_pos_real_energy_zone,
            is_team_sector_map,
            can_go_up,
            can_go_right,
            can_go_down,
            can_go_left,
            is_relic,
            relic_area,
            is_reward,
            dist_to_center_x / 6.,
            dist_to_center_y / 6.,
            own_ships,
            own_ships_energy / 100.,
            own_ship_is_harvesting,
            opp_ships,
            opp_ships_energy / 100.,
            opp_ship_is_harvesting
        ], axis=0)

        return obs
    
    
    @staticmethod
    def _get_ships(fleet: Fleet):
        ships = []
        needs_mirroring = fleet.team_id == 1
        for ship in fleet.ships:
            node = None
            if ship.node is not None:
                ship_node = ship.node
                if needs_mirroring:
                    ship_node = deepcopy(ship.node)
                    x, y = get_opposite(*ship_node.coordinates)
                    ship_node.x = x
                    ship_node.y = y
                node = ship_node
            ships.append({
                'unit_id': ship.unit_id,
                'energy': ship.energy,
                'node': node
            })
        return ships
    

    def _get_space_nodes(self):
        nodes = np.zeros((SPACE_SIZE, SPACE_SIZE), dtype=object)
        for y in range(SPACE_SIZE):
            for x in range(SPACE_SIZE):
                if self.team_id == 1:
                    x, y = get_opposite(x, y)
                node = self.space.get_node(x, y)
                if self.team_id == 1:
                    node = deepcopy(self.space.get_node(x, y))
                    x, y = get_opposite(*node.coordinates)
                    node.x = x
                    node.y = y
                nodes[x][y] = node
        return nodes


    def _get_relic_nodes(self):
        relic_nodes = self.space.relic_nodes
        if self.team_id == 1:
            relic_nodes = deepcopy(self.space.relic_nodes)
            for node in relic_nodes:
                x, y = get_opposite(*node.coordinates)
                node.x = x
                node.y = y
        return relic_nodes


    def _get_reward_nodes(self):
        reward_nodes = self.space.reward_nodes
        if self.team_id == 1:
            reward_nodes = deepcopy(self.space.reward_nodes)
            for node in reward_nodes:
                x, y = get_opposite(*node.coordinates)
                node.x = x
                node.y = y
        return reward_nodes

    
    def to_dict(self):
        game_state = {
            'match_step': self.match_step,
            'game_num': self.game_num,
            'step': self.step,
            'points_gain': self.points_gain,
            'our_wins': self.our_wins,
            'opp_wins': self.opp_wins,
            'points': self.points,
            'opp_points': self.opp_points
        }
        space = {
            'nodes': self._get_space_nodes(),
            'relic_nodes': self._get_relic_nodes(),
            'reward_nodes': self._get_reward_nodes()
        }
        
        return {
            'game_state': game_state,
            'config': self.config,
            'ships': self._get_ships(self.fleet),
            'opp_ships': self._get_ships(self.opp_fleet),
            'space': space
        }
        


    def reset(self, env_cfg):
        self._init()
        self.set_config(env_cfg)