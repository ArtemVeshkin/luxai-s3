from pathlib import Path
import torch
from torch.utils.data import Dataset
import numpy as np
import os
import pickle
from state.base import SPACE_SIZE, MAX_UNITS, RELIC_REWARD_RANGE, Config, is_team_sector
from scipy.signal import convolve2d
from state.action_type import _DIRECTIONS


class StatesDataset(Dataset):
    def __init__(self, data_path: Path):
        super().__init__()
        self.data_path = data_path
        files = os.listdir(data_path)
        # files = files[:int(len(files) * 0.05)]
        self.files = self.filter_files(files)


    def __len__(self):
        return len(self.files)


    @staticmethod
    def filter_files(files):
        result_files = []
        for file in files:
            game, player, step = map(int, (file.split('.')[0]).split('_'))
            if step % 101 == 0:
                continue
            result_files.append(file)

        return result_files


    @staticmethod
    def _get_global_info_features(state: dict):
        game_state = state['game_state']
        config: Config = state['config']
        ships = state['ships']

        # numerical
        match_step = game_state['match_step']
        step = game_state['step']
        points_gain = game_state['points_gain']
        our_wins = game_state['our_wins']
        opp_wins = game_state['opp_wins']
        our_points = game_state['points']
        opp_points = game_state['opp_points']

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
        found_relics_count = len(state['space']['relic_nodes'])
        found_rewards_count = len(state['space']['reward_nodes'])
        move_cost = config.UNIT_MOVE_COST
        sap_cost  = config.UNIT_SAP_COST
        sap_range = config.UNIT_SAP_RANGE
        unit_sensor_range = config.UNIT_SENSOR_RANGE
        nebula_energy_reduction = config.NEBULA_ENERGY_REDUCTION

        # cat
        game_num = game_state['game_num']
        is_winning_by_games = int(our_wins > opp_wins)
        is_winning_by_points = int(our_points > opp_points)

        relic_can_appear = int(match_step <= 50 and game_state['game_num'] <= 2)

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


    def _state_to_obs(self, state: dict):
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

        space_nodes = state['space']['nodes']
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
        for node in state['space']['relic_nodes']:
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
        for node in state['space']['reward_nodes']:
            x, y = node.coordinates
            is_reward[x, y] = 1.

        ship_masks = np.zeros((SPACE_SIZE, SPACE_SIZE, MAX_UNITS))

        own_ships = np.zeros((SPACE_SIZE, SPACE_SIZE))
        own_ships_energy = np.zeros((SPACE_SIZE, SPACE_SIZE))
        own_ship_is_harvesting = np.zeros((SPACE_SIZE, SPACE_SIZE))
        for idx, ship in enumerate(state['ships']):
            if ship['node'] is not None:
                x, y = ship['node'].coordinates
                ship_masks[x, y, idx] = 1.
                own_ships[x, y] += 1.
                own_ships_energy[x, y] += ship['energy']
                own_ship_is_harvesting[x, y] = float(ship['node'].reward)

        opp_ships = np.zeros((SPACE_SIZE, SPACE_SIZE))
        opp_ships_energy = np.zeros((SPACE_SIZE, SPACE_SIZE))
        opp_ship_is_harvesting = np.zeros((SPACE_SIZE, SPACE_SIZE))
        for idx, ship in enumerate(state['opp_ships']):
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

        num_features, cat_features = self._get_global_info_features(state)

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

    def __getitem__(self, index):
        file = self.files[index]
        game, player, step = map(int, (file.split('.')[0]).split('_'))
        with open(self.data_path / file, 'rb') as fh:
            data = pickle.load(fh)

        state = data['state']
        actions = data['actions'][:, 0]

        obs = self._state_to_obs(state)
        alive_ships = ','.join(str(ship_idx) for ship_idx in range(16) if (state['ships'][ship_idx]['node'] is not None) and (state['ships'][ship_idx]['energy'] > 0))

        info = {
            'game': game,
            'player': player,
            'step': step,
            'alive_ships': alive_ships
        }

        return {
            'obs': torch.Tensor(obs),
            'actions': torch.Tensor(actions).long(),
            'info': info
        }
