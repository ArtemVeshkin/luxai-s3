import random

import numpy as np
from sys import stderr

from base import (
    Global,
    get_match_step,
    is_team_sector,
    get_match_number
)
from space import Space
from fleet import Fleet
from node import Node
from action_type import ActionType
from ship import Ship
from debug import show_map, show_energy_field, show_exploration_map
from pathfinding import (
    astar,
    find_closest_target,
    nearby_positions,
    create_weights,
    estimate_energy_cost,
    path_to_actions,
    manhattan_distance,
)

random.seed(42)


class Agent:

    def __init__(self, player: str, env_cfg) -> None:
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        self.team_id = 0 if self.player == "player_0" else 1
        self.opp_team_id = 1 if self.team_id == 0 else 0
        self.env_cfg = env_cfg

        Global.MAX_UNITS = env_cfg["max_units"]
        Global.UNIT_MOVE_COST = env_cfg["unit_move_cost"]
        Global.UNIT_SAP_COST = env_cfg["unit_sap_cost"]
        Global.UNIT_SAP_RANGE = env_cfg["unit_sap_range"]
        Global.UNIT_SENSOR_RANGE = env_cfg["unit_sensor_range"]

        self.space = Space()
        self.fleet = Fleet(self.team_id)
        self.opp_fleet = Fleet(self.opp_team_id)

        self.match_step = 0
        self.game_num = 0
        self.step = 0

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        self.step = step
        self.match_step = get_match_step(step)
        self.game_num = int(obs["team_wins"].sum())
        match_number = get_match_number(step)
        # print(f"start step={self.match_step}({step})", file=stderr)

        if self.match_step == 0:
            # nothing to do here at the beginning of the match
            # just need to clean up some of the garbage that was left after the previous match
            self.fleet.clear()
            self.opp_fleet.clear()
            self.space.clear()
            self.space.move_obstacles(step)
            if match_number <= Global.LAST_MATCH_WHEN_RELIC_CAN_APPEAR:
                self.space.clear_exploration_info()
            return self.create_actions_array()

        points = int(obs["team_points"][self.team_id])
        opp_points = int(obs["team_points"][self.opp_team_id])

        # how many points did we score in the last step
        reward = max(0, points - self.fleet.points)

        self.space.update(step, obs, self.team_id, reward)
        self.fleet.update(obs, self.space)
        self.opp_fleet.update(obs, self.space)

        for ship in self.fleet:
            ship.node.visited_times += 1

        self.find_relics()
        self.find_rewards()
        self.old_harvest()
        # self.harvest()
        self.fight()
        self.employ_unemployed()
        self.harvest_if_losing(points, opp_points)
        self.optimize_harvesting()

        # self.show_explored_energy_field()
        # if step >= 405 and step <= 430:
        #     print(f'step={step}:', file=stderr)
        #     for ship in self.fleet:
        #         print(f'ship={ship}, task={ship.task}, target={ship.target}, action={ship.action}', file=stderr)

        return self.create_actions_array()

    def create_actions_array(self):
        ships = self.fleet.ships
        actions = np.zeros((len(ships), 3), dtype=int)

        for i, ship in enumerate(ships):
            if ship.action is not None:
                sap_x, sap_y = ship.sap_direction if ship.action == ActionType.sap else (0, 0)
                actions[i] = ship.action, sap_x, sap_y
                ship.sap_direction = (0, 0)

        return actions

    def employ_unemployed(self):
        unexplored_relics = self.get_unexplored_relics()
        if len(unexplored_relics) == 0:
            return
        for ship in self.fleet:
            if ship.task is not None or ship.target is not None:
                continue
            relic_coordinates, _ = find_closest_target(ship.node.coordinates,
                                             [relic.coordinates for relic in unexplored_relics])
            targets = []
            for x, y in nearby_positions(*relic_coordinates, Global.RELIC_REWARD_RANGE):
                node = self.space.get_node(x, y)
                if not node.explored_for_reward and node.is_walkable:
                    targets.append((x, y))
            target, _ = find_closest_target(ship.coordinates, targets)
            if not target:
                return
            path = astar(create_weights(self.space), ship.coordinates, target)
            energy = estimate_energy_cost(self.space, path)
            actions = path_to_actions(path)
            if actions and ship.energy >= energy:
                ship.task = "find_rewards"
                ship.target = self.space.get_node(*target)
                ship.action = actions[0]
        ship_coords = [ship.coordinates for ship in self.fleet]
        for ship in self.fleet:
            if ship.task is not None or ship.target is not None:
                continue
            target, _ = find_closest_target(ship.node.coordinates,
                                            [rew.coordinates for rew in self.space.reward_nodes
                                                if rew.is_walkable and rew.coordinates not in ship_coords])
            if not target:
                return
            path = astar(create_weights(self.space), ship.coordinates, target)
            energy = estimate_energy_cost(self.space, path)
            actions = path_to_actions(path)
            if actions and ship.energy >= energy:
                ship.task = "harvest"
                ship.target = self.space.get_node(*target)
                ship.action = actions[0]

    def find_relics(self):
        if Global.ALL_RELICS_FOUND:
            for ship in self.fleet:
                if ship.task == "find_relics":
                    ship.task = None
                    ship.target = None
            return

        targets = set()
        for node in self.space:
            if not node.explored_for_relic:
                # We will only find relics in our part of the map
                # because relics are symmetrical.
                if is_team_sector(self.fleet.team_id, *node.coordinates):
                    targets.add(node.coordinates)

        def set_task(ship):
            if ship.task and ship.task != "find_relics":
                return False

            if ship.energy < Global.UNIT_MOVE_COST:
                return False

            target, _ = find_closest_target(ship.coordinates, targets)
            if not target:
                return False

            path = astar(create_weights(self.space), ship.coordinates, target)
            energy = estimate_energy_cost(self.space, path)
            actions = path_to_actions(path)
            if actions and ship.energy >= energy:
                ship.task = "find_relics"
                ship.target = self.space.get_node(*target)
                ship.action = actions[0]

                for x, y in path:
                    for xy in nearby_positions(x, y, Global.UNIT_SENSOR_RANGE):
                        if xy in targets:
                            targets.remove(xy)

                return True

            return False

        for ship in self.fleet:
            if set_task(ship):
                continue

            if ship.task == "find_relics":
                ship.task = None
                ship.target = None

    def find_rewards(self):
        if Global.ALL_REWARDS_FOUND:
            for ship in self.fleet:
                if ship.task == "find_rewards":
                    ship.task = None
                    ship.target = None
            return

        unexplored_relics = self.get_unexplored_relics()

        relic_node_to_ship = {}
        for ship in self.fleet:
            if ship.task == "find_rewards":
                if ship.target is None:
                    ship.task = None
                    continue

                if (
                    ship.target in unexplored_relics
                    and ship.energy > Global.UNIT_MOVE_COST * 5
                ):
                    relic_node_to_ship[ship.target] = ship
                else:
                    ship.task = None
                    ship.target = None

        for relic in unexplored_relics:
            if relic not in relic_node_to_ship:

                # find the closest ship to the relic node
                min_distance, closes_ship = float("inf"), None
                for ship in self.fleet:
                    if ship.task and ship.task != "find_rewards":
                        continue

                    if ship.energy < Global.UNIT_MOVE_COST * 5:
                        continue

                    distance = manhattan_distance(ship.coordinates, relic.coordinates)
                    if distance < min_distance:
                        min_distance, closes_ship = distance, ship

                if closes_ship:
                    relic_node_to_ship[relic] = closes_ship

        def set_task(ship, relic_node, can_pause):
            targets = []
            for x, y in nearby_positions(
                *relic_node.coordinates, Global.RELIC_REWARD_RANGE
            ):
                node = self.space.get_node(x, y)
                if not node.explored_for_reward and node.is_walkable:
                    targets.append((x, y))

            target, _ = find_closest_target(ship.coordinates, targets)

            if target == ship.coordinates and not can_pause:
                target, _ = find_closest_target(
                    ship.coordinates,
                    [
                        n.coordinates
                        for n in self.space
                        if n.explored_for_reward and n.is_walkable
                    ],
                )

            if not target:
                return

            path = astar(create_weights(self.space), ship.coordinates, target)
            energy = estimate_energy_cost(self.space, path)
            actions = path_to_actions(path)

            if actions and ship.energy >= energy:
                ship.task = "find_rewards"
                ship.target = self.space.get_node(*target)
                ship.action = actions[0]

        can_pause = True
        for n, s in sorted(
            list(relic_node_to_ship.items()), key=lambda _: _[1].unit_id
        ):
            if set_task(s, n, can_pause):
                if s.target == s.node:
                    # If one ship is stationary, we will move all the other ships.
                    # This will help generate more useful data in Global.REWARD_RESULTS.
                    can_pause = False
            else:
                if s.task == "find_rewards":
                    s.task = None
                    s.target = None

    def get_unexplored_relics(self) -> list[Node]:
        relic_nodes = []
        for relic_node in self.space.relic_nodes:
            if not is_team_sector(self.team_id, *relic_node.coordinates):
                continue

            explored = True
            for x, y in nearby_positions(
                *relic_node.coordinates, Global.RELIC_REWARD_RANGE
            ):
                node = self.space.get_node(x, y)
                if not node.explored_for_reward and node.is_walkable:
                    explored = False
                    break

            if explored:
                continue

            relic_nodes.append(relic_node)

        return relic_nodes


    def old_harvest(self):

        def set_task(ship, target_node):
            if ship.node == target_node:
                ship.task = "harvest"
                ship.target = target_node
                ship.action = ActionType.center
                return True

            path = astar(
                create_weights(self.space),
                start=ship.coordinates,
                goal=target_node.coordinates,
            )
            energy = estimate_energy_cost(self.space, path)
            actions = path_to_actions(path)

            if not actions or ship.energy < energy:
                return False

            ship.task = "harvest"
            ship.target = target_node
            ship.action = actions[0]
            return True

        booked_nodes = set()
        for ship in self.fleet:
            if ship.task == "harvest":
                if ship.target is None:
                    ship.task = None
                    continue

                if set_task(ship, ship.target):
                    booked_nodes.add(ship.target)
                else:
                    ship.task = None
                    ship.target = None

        targets = set()
        for n in self.space.reward_nodes:
            if n.is_walkable and n not in booked_nodes:
                targets.add(n.coordinates)
        if not targets:
            return

        for ship in self.fleet:
            if ship.task:
                continue

            target, _ = find_closest_target(ship.coordinates, targets)

            if target and set_task(ship, self.space.get_node(*target)):
                targets.remove(target)
            else:
                ship.task = None
                ship.target = None


    def _get_ships_centroid(self, ship_indices):
        ships_centroid_pos = np.array([0, 0])
        for ship_idx in ship_indices:
            x, y = self.fleet.ships[ship_idx].coordinates
            ships_centroid_pos[0] += x
            ships_centroid_pos[1] += y
        if len(ship_indices) > 0:
            ships_centroid_pos = np.clip(
                (ships_centroid_pos / len(ship_indices))
            , 0, 23).astype('int')
        else:
            ships_centroid_pos = np.array((0, 0) if self.team_id == 0 else (23, 23))
        return ships_centroid_pos


    def _get_sorter_rewards(self, space_weights, remaining_ship_indices):
        remaining_ships_centroid_pos = self._get_ships_centroid(remaining_ship_indices)

        def rewards_sort_key(reward_node: Node):
            path = astar(
                space_weights,
                start=tuple(remaining_ships_centroid_pos),
                goal=reward_node.coordinates,
            )
            actions = path_to_actions(path)
            if actions:
                return len(actions), -estimate_energy_cost(self.space, path)
            return float('inf'), float('inf')

        sorted_reward_nodes = sorted(
            self.space.reward_nodes,
            key=rewards_sort_key
        )
        return sorted_reward_nodes
    

    def _ship_relevance_to_node(self, ship: Ship, node: Node, space_weights):
        if node.coordinates == ship.coordinates:
            return 0, ActionType.center

        path = astar(
            space_weights,
            start=ship.coordinates,
            goal=node.coordinates,
        )
        path_energy_cost = estimate_energy_cost(self.space, path)
        actions = path_to_actions(path)
        path_len = len(path)
        if (ship.energy - path_energy_cost) >= 0 and actions and path_len <= (100 - self.match_step):
            return path_len, actions[0]

        return None, None


    def harvest(self):
        space_weights = create_weights(self.space)
        booked_nodes = set()
        for ship in self.fleet:
            if ship.task == 'harvest':
                relevance, action = self._ship_relevance_to_node(ship, ship.target, space_weights)
                if relevance is None:
                    ship.task = None
                    ship.target = None
                    ship.action = ActionType.center
                else:
                    booked_nodes.add(ship.target)
                    ship.action = action


        remaining_ship_indices = set(
            ship_idx for ship_idx, ship in enumerate(self.fleet.ships)
            if ship.node is not None and ship.task is None
        )

        sorted_reward_nodes = self._get_sorter_rewards(space_weights, remaining_ship_indices)
        for reward_node in sorted_reward_nodes:
            if reward_node in booked_nodes:
                continue

            ship_relevances = [None] * Global.MAX_UNITS
            ship_actions = [None] * Global.MAX_UNITS
            for ship_idx in remaining_ship_indices:
                ship = self.fleet.ships[ship_idx]
                ship_relevances[ship_idx], ship_actions[ship_idx] = self._ship_relevance_to_node(ship, reward_node, space_weights)

            best_ship_idx = None
            for ship_idx in remaining_ship_indices:
                if ship_relevances[ship_idx] is not None:
                    if best_ship_idx is None:
                        best_ship_idx = ship_idx

                    if ship_relevances[ship_idx] <= ship_relevances[best_ship_idx]:
                        if self.fleet.ships[ship_idx].energy < self.fleet.ships[best_ship_idx].energy:
                            best_ship_idx = ship_idx

            if best_ship_idx is not None:
                best_ship = self.fleet.ships[best_ship_idx]
                best_ship.target = reward_node
                best_ship.task = "harvest"
                best_ship.action = ship_actions[best_ship_idx]
                remaining_ship_indices.remove(best_ship_idx)


    def harvest_if_losing(self, points, opp_points):
        remaining_steps = Global.MAX_STEPS_IN_MATCH - self.match_step

        reward = max(0, points - self.fleet.points)
        opp_reward = max(0, points - self.fleet.points)

        predicted_points = points + remaining_steps * reward
        opp_predicted_points = opp_points + remaining_steps * opp_reward

        if opp_predicted_points > predicted_points and remaining_steps > 0:
            # HARVEST AS FAST AS YOU CAN!!!

            missing_harvesters = int((opp_predicted_points - predicted_points) / remaining_steps) + 1

            space_weights = create_weights(self.space)
            booked_nodes = set()
            for ship in self.fleet:
                if ship.task == 'harvest':
                    booked_nodes.add(ship.target)

            remaining_ship_indices = set(
                ship_idx for ship_idx, ship in enumerate(self.fleet.ships)
                if ship.node is not None and ship.task != 'harvest'
            )

            sorted_reward_nodes = self._get_sorter_rewards(space_weights, remaining_ship_indices)
            new_harvesters = []
            for reward_node in sorted_reward_nodes:
                if reward_node in booked_nodes:
                    continue

                ship_relevances = [None] * Global.MAX_UNITS
                ship_actions = [None] * Global.MAX_UNITS
                for ship_idx in remaining_ship_indices:
                    ship = self.fleet.ships[ship_idx]
                    ship_relevances[ship_idx], ship_actions[ship_idx] = self._ship_relevance_to_node(ship, reward_node, space_weights)

                best_ship_idx = None
                for ship_idx in remaining_ship_indices:
                    if ship_relevances[ship_idx] is not None:
                        if best_ship_idx is None:
                            best_ship_idx = ship_idx

                        if ship_relevances[ship_idx] <= ship_relevances[best_ship_idx]:
                            if self.fleet.ships[ship_idx].energy < self.fleet.ships[best_ship_idx].energy:
                                best_ship_idx = ship_idx

                if best_ship_idx is not None:
                    new_harvesters.append({
                        'ship': self.fleet.ships[best_ship_idx],
                        'target': reward_node,
                        'action': ship_actions[best_ship_idx]
                    })
                    remaining_ship_indices.remove(best_ship_idx)

            if len(new_harvesters) >= missing_harvesters:
                for new_harvester in new_harvesters[:missing_harvesters]:
                    new_harvester['ship'].target = new_harvester['target']
                    new_harvester['ship'].task = "harvest"
                    new_harvester['ship'].action = new_harvester['action']


    def optimize_harvesting(self):
        cur_harvesting_ships = [
            ship for ship in self.fleet
            if ship.task == 'harvest' and ship.action in (ActionType.center, ActionType.sap) and ship.target == ship.node
        ]
        space_weights = create_weights(self.space)
        for ship in self.fleet:
            if ship not in cur_harvesting_ships and ship.task == 'harvest' and ship.action not in (ActionType.center, ActionType.sap):
                for cur_harvesting_ship in cur_harvesting_ships:
                    if ship.next_coords() == cur_harvesting_ship.coordinates and cur_harvesting_ship.energy > Global.UNIT_MOVE_COST:
                        path = astar(
                            space_weights,
                            start=ship.coordinates,
                            goal=ship.target.coordinates,
                        )
                        actions = path_to_actions(path)
                        if len(actions) == 2:
                            cur_harvesting_ship.action = actions[1]
                            cur_harvesting_ship.target, ship.target = ship.target, cur_harvesting_ship.target

                        


    def fight(self):
        def find_best_target(ship, targets):
            #todo: optimizirovat!!!!! schitat cherez dp
            if self.game_num == 0 and random.random() < 0.7:
                return None
            best_target = None
            max_opp_ship_count = 0
            for x, y in nearby_positions(*ship.coordinates, Global.UNIT_SAP_RANGE):
                opp_ship_count = 0
                for dx in range(-1, 2):
                    for dy in range(-1, 2):
                        if (x + dx, y + dy) in targets:
                            opp_ship_count += 1 if dx == 0 and dy == 0 else 0.25 #todo: pomenyat'!!!!
                if opp_ship_count > max_opp_ship_count and manhattan_distance(ship.coordinates, (x, y)) <= Global.UNIT_SAP_RANGE:
                    max_opp_ship_count = opp_ship_count
                    best_target = (x, y)
            if max_opp_ship_count <= 1.5 and random.random() < 0.5:
                return None
            return best_target

        targets = {opp_ship.coordinates for opp_ship in self.opp_fleet}
        for x, y in targets:
            for ship in self.fleet:
                if manhattan_distance((x, y), ship.coordinates) <= Global.UNIT_SAP_RANGE:
                    for relic_node in self.space.relic_nodes:
                        if manhattan_distance((x, y), relic_node.coordinates) <= Global.RELIC_REWARD_RANGE\
                                or manhattan_distance(ship.coordinates, relic_node.coordinates) <= Global.RELIC_REWARD_RANGE: #todo: mb ubrat' eto uslovie
                            if ship.node.energy\
                                    and ship.energy + (ship.node.energy * 2 if ship.node.energy < 0 else 0) > Global.UNIT_SAP_COST:
                                best_target = find_best_target(ship, targets)
                                if best_target:
                                    ship.sap_direction = (best_target[0] - ship.node.x, best_target[1] - ship.node.y)
                                    ship.action = ActionType.sap
                                    break




    def show_visible_energy_field(self):
        print("Visible energy field:", file=stderr)
        show_energy_field(self.space)

    def show_explored_energy_field(self):
        print("Explored energy field:", file=stderr)
        show_energy_field(self.space, only_visible=False)

    def show_visible_map(self):
        print("Visible map:", file=stderr)
        show_map(self.space, self.fleet, self.opp_fleet)

    def show_explored_map(self):
        print("Explored map:", file=stderr)
        show_map(self.space, self.fleet, only_visible=False)

    def show_exploration_map(self):
        print("Exploration map:", file=stderr)
        show_exploration_map(self.space)
