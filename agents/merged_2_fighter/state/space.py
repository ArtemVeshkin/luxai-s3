import copy
import numpy as np
from luxai_s3.state import EnvObs
from sys import stderr
from scipy.signal import convolve2d
from .base import (
    Config,
    SPACE_SIZE,
    RELIC_REWARD_RANGE,
    LAST_MATCH_WHEN_RELIC_CAN_APPEAR,
    LAST_MATCH_STEP_WHEN_RELIC_CAN_APPEAR,
    warp_point,
    get_opposite,
    get_match_number,
    get_match_step
)
from .node import Node, NodeType
from .energy_predictor import EnergyPredictor


def nearby_positions(x, y, distance):
    for x_ in range(max(0, x - distance), min(SPACE_SIZE, x + distance + 1)):
        for y_ in range(max(0, y - distance), min(SPACE_SIZE, y + distance + 1)):
            yield x_, y_


class Space:
    def __init__(self):
        # self.energy_predictor = EnergyPredictor()

        self._nodes: list[list[Node]] = []
        for y in range(SPACE_SIZE):
            row = [Node(x, y) for x in range(SPACE_SIZE)]
            self._nodes.append(row)

        # set of nodes with a relic
        self._relic_nodes: set[Node] = set()

        # set of nodes that provide points
        self._reward_nodes: set[Node] = set()

    def __repr__(self) -> str:
        return f"Space({SPACE_SIZE}x{SPACE_SIZE})"

    def __iter__(self):
        for row in self._nodes:
            yield from row

    @property
    def relic_nodes(self) -> set[Node]:
        return self._relic_nodes

    @property
    def reward_nodes(self) -> set[Node]:
        return self._reward_nodes

    def get_node(self, x, y) -> Node:
        return self._nodes[y][x]

    def get_energy_field(self):
        space_size = SPACE_SIZE
        energy_field = np.zeros((space_size, space_size, 2))
        for y in range(space_size):
            for x in range(space_size):
                node: Node = self.get_node(x, y)
                if node.energy is None:
                    energy_field[x, y, :] = (0., 0)
                else:
                    energy_field[x, y, :] = (node.energy, 1)
        return energy_field

    def update(self, step, obs: EnvObs, team_id, team_reward, config: Config):
        self.move_obstacles(step, config)
        self._update_map(obs, config, team_id)
        self._update_relic_map(step, obs, team_id, team_reward, config)
        # self.energy_predictor.predict_hidden_energy(self)


    def _update_relic_map(self, step, obs, team_id, team_reward, config: Config):
        for mask, xy in zip(obs["relic_nodes_mask"], obs["relic_nodes"]):
            if mask and not self.get_node(*xy).relic:
                # We have found a new relic.
                self._update_relic_status(*xy, status=True)
                for x, y in nearby_positions(*xy, RELIC_REWARD_RANGE):
                    if not self.get_node(x, y).reward:
                        self._update_reward_status(x, y, status=None)

        all_relics_found = True
        all_rewards_found = True
        for node in self:
            if node.is_visible and not node.explored_for_relic:
                self._update_relic_status(*node.coordinates, status=False)

            if not node.explored_for_relic:
                all_relics_found = False

            if not node.explored_for_reward:
                all_rewards_found = False

        config.ALL_RELICS_FOUND = all_relics_found
        config.ALL_REWARDS_FOUND = all_rewards_found

        match = get_match_number(step)
        match_step = get_match_step(step)
        num_relics_th = 2 * min(match, LAST_MATCH_WHEN_RELIC_CAN_APPEAR) + 1

        if not config.ALL_RELICS_FOUND:
            if len(self._relic_nodes) >= num_relics_th:
                # all relics found, mark all nodes as explored for relics
                config.ALL_RELICS_FOUND = True
                for node in self:
                    if not node.explored_for_relic:
                        self._update_relic_status(*node.coordinates, status=False)

        if not config.ALL_REWARDS_FOUND:
            if (
                match_step > LAST_MATCH_STEP_WHEN_RELIC_CAN_APPEAR
                or len(self._relic_nodes) >= num_relics_th
            ):
                self._update_reward_status_from_relics_distribution()
                self._update_reward_results(obs, team_id, team_reward, config)
                self._update_reward_status_from_reward_results(config)

    def _update_reward_status_from_reward_results(self, config: Config):
        # We will use config.REWARD_RESULTS to identify which nodes yield points
        for result in config.REWARD_RESULTS:

            unknown_nodes = set()
            known_reward = 0
            for n in result["nodes"]:
                if n.explored_for_reward and not n.reward:
                    continue

                if n.reward:
                    known_reward += 1
                    continue

                unknown_nodes.add(n)

            if not unknown_nodes:
                # all nodes already explored, nothing to do here
                continue

            reward = result["reward"] - known_reward  # reward from unknown_nodes

            if reward == 0:
                # all nodes are empty
                for node in unknown_nodes:
                    self._update_reward_status(*node.coordinates, status=False)

            elif reward == len(unknown_nodes):
                # all nodes yield points
                for node in unknown_nodes:
                    self._update_reward_status(*node.coordinates, status=True)

            elif reward > len(unknown_nodes):
                # we shouldn't be here
                pass
                # print(
                #     f"Something wrong with reward result: {result}"
                #     ", this result will be ignored.",
                #     file=stderr,
                # )

    def _update_reward_results(self, obs, team_id, team_reward, config: Config):
        ship_nodes = set()
        for active, energy, position in zip(
            obs["units_mask"][team_id],
            obs["units"]["energy"][team_id],
            obs["units"]["position"][team_id],
        ):
            if active and energy >= 0:
                # Only units with non-negative energy can give points
                ship_nodes.add(self.get_node(*position))

        config.REWARD_RESULTS.append({"nodes": ship_nodes, "reward": team_reward})

    def _update_reward_status_from_relics_distribution(self):
        # Rewards can only occur near relics.
        # Therefore, if there are no relics near the node
        # we can infer that the node does not contain a reward.

        relic_map = np.zeros((SPACE_SIZE, SPACE_SIZE), np.int32)
        for node in self:
            if node.relic or not node.explored_for_relic:
                relic_map[node.y][node.x] = 1

        reward_size = 2 * RELIC_REWARD_RANGE + 1

        reward_map = convolve2d(
            relic_map,
            np.ones((reward_size, reward_size), dtype=np.int32),
            mode="same",
            boundary="fill",
            fillvalue=0,
        )

        for node in self:
            if reward_map[node.y][node.x] == 0:
                # no relics in range RELIC_REWARD_RANGE
                node.update_reward_status(False)

    def _update_relic_status(self, x, y, status):
        node = self.get_node(x, y)
        node.update_relic_status(status)

        # relics are symmetrical
        opp_node = self.get_node(*get_opposite(x, y))
        opp_node.update_relic_status(status)

        if status:
            self._relic_nodes.add(node)
            self._relic_nodes.add(opp_node)

    def _update_reward_status(self, x, y, status):
        node = self.get_node(x, y)
        node.update_reward_status(status)

        # rewards are symmetrical
        opp_node = self.get_node(*get_opposite(x, y))
        opp_node.update_reward_status(status)

        if status:
            self._reward_nodes.add(node)
            self._reward_nodes.add(opp_node)

    def _update_map(self, obs: EnvObs, config: Config, team_id):
        sensor_mask = obs["sensor_mask"]
        obs_energy = obs["map_features"]["energy"]
        obs_tile_type = obs["map_features"]["tile_type"]
        enemies = obs["units"]["position"][abs(team_id - 1)]

        obstacles_shifted = False
        energy_nodes_shifted = False
        for node in self:
            x, y = node.coordinates
            is_visible = sensor_mask[x, y]

            if (
                is_visible
                and not node.is_unknown
                and node.type.value != obs_tile_type[x, y]
            ):
                obstacles_shifted = True

            if (
                is_visible
                and node.energy is not None
                and node.energy != obs_energy[x, y]
            ):
                energy_nodes_shifted = True

        config.OBSTACLES_MOVEMENT_STATUS.append(obstacles_shifted)
        config.ENERGY_NODES_MOVEMENT_STATUS.append(energy_nodes_shifted)

        def clear_map_info():
            for n in self:
                n.type = NodeType.unknown

        if not config.OBSTACLE_MOVEMENT_DIRECTION_FOUND and obstacles_shifted:
            direction = self._find_obstacle_movement_direction(obs)
            if direction:
                config.OBSTACLE_MOVEMENT_DIRECTION_FOUND = True
                config.OBSTACLE_MOVEMENT_DIRECTION = direction

                self.move(*config.OBSTACLE_MOVEMENT_DIRECTION, inplace=True)
            else:
                clear_map_info()


        if not config.ENERGY_NODE_MOVEMENT_PERIOD_FOUND:
            period = self._find_energy_movement_period(
                config.ENERGY_NODES_MOVEMENT_STATUS
            )
            if period is not None:
                config.ENERGY_NODE_MOVEMENT_PERIOD_FOUND = True
                config.ENERGY_NODE_MOVEMENT_PERIOD = period


        if not config.OBSTACLE_MOVEMENT_PERIOD_FOUND:
            period = self._find_obstacle_movement_period(
                config.OBSTACLES_MOVEMENT_STATUS
            )
            if period is not None:
                config.OBSTACLE_MOVEMENT_PERIOD_FOUND = True
                config.OBSTACLE_MOVEMENT_PERIOD = period

            if obstacles_shifted:
                clear_map_info()

        if (
            obstacles_shifted
            and config.OBSTACLE_MOVEMENT_PERIOD_FOUND
            and config.OBSTACLE_MOVEMENT_DIRECTION_FOUND
        ):
            # maybe something is wrong
            clear_map_info()

        # The energy field has changed
        # I cannot predict what the new energy field will be like.
        if energy_nodes_shifted:
            # self.energy_predictor.update_prev_energy_fields(self)

            for node in self:
                node.energy = None

        for node in self:
            x, y = node.coordinates
            is_visible = bool(sensor_mask[x, y])

            # update enemies
            if any(map(lambda coord: coord[0] == x and coord[1] == y, enemies)):
                node.is_enemy = True
                # print('Spotted enemy')
            else:
                node.is_enemy = False

            node.is_visible = is_visible

            if is_visible and node.is_unknown:
                node.type = NodeType(int(obs_tile_type[x, y]))

                # we can also update the node type on the other side of the map
                # because the map is symmetrical
                self.get_node(*get_opposite(x, y)).type = node.type

            if is_visible:
                node.energy = int(obs_energy[x, y])

                # the energy field should be symmetrical
                self.get_node(*get_opposite(x, y)).energy = node.energy

    @staticmethod
    def _find_obstacle_movement_period(obstacles_movement_status):
        if len(obstacles_movement_status) < 81:
            return

        num_movements = sum(obstacles_movement_status)

        if num_movements <= 2:
            return 40
        elif num_movements <= 4:
            return 20
        elif num_movements <= 8:
            return 10
        else:
            return 20 / 3
        
    @staticmethod
    def _find_energy_movement_period(energies_movement_status):
        if sum(energies_movement_status) >= 2:
            first = None
            second = None
            for idx, i in enumerate(energies_movement_status):
                if i and first is None:
                    first = idx
                elif i and first and second is None:
                    second = idx
            return second - first
        else:
            return None

    def _find_obstacle_movement_direction(self, obs):
        sensor_mask = obs["sensor_mask"]
        obs_tile_type = obs["map_features"]["tile_type"]

        suitable_directions = []
        for direction in [(1, -1), (-1, 1)]:
            moved_space = self.move(*direction, inplace=False)

            match = True
            for node in moved_space:
                x, y = node.coordinates
                if (
                    sensor_mask[x, y]
                    and not node.is_unknown
                    and obs_tile_type[x, y] != node.type.value
                ):
                    match = False
                    break

            if match:
                suitable_directions.append(direction)

        if len(suitable_directions) == 1:
            return suitable_directions[0]

    def clear(self):
        for node in self:
            node.is_visible = False

    def move_obstacles(self, step, config: Config):
        if (
            config.OBSTACLE_MOVEMENT_PERIOD_FOUND
            and config.OBSTACLE_MOVEMENT_DIRECTION_FOUND
            and config.OBSTACLE_MOVEMENT_PERIOD > 0
        ):
            speed = 1 / config.OBSTACLE_MOVEMENT_PERIOD
            if (step - 2) * speed % 1 > (step - 1) * speed % 1:
                self.move(*config.OBSTACLE_MOVEMENT_DIRECTION, inplace=True)

    def move(self, dx: int, dy: int, *, inplace=False) -> "Space":
        if not inplace:
            new_space = copy.deepcopy(self)
            for node in self:
                x, y = warp_point(node.x + dx, node.y + dy)
                new_space.get_node(x, y).type = node.type
            return new_space
        else:
            types = [n.type for n in self]
            for node, node_type in zip(self, types):
                x, y = warp_point(node.x + dx, node.y + dy)
                self.get_node(x, y).type = node_type
            return self
        
    def clear_exploration_info(self, config: Config):
        config.REWARD_RESULTS = []
        config.ALL_RELICS_FOUND = False
        config.ALL_REWARDS_FOUND = False
        for node in self:
            if not node.relic:
                self._update_relic_status(node.x, node.y, status=None)
