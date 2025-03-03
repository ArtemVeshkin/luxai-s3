from state.state import State
from state.action_type import ActionType
from state.base import (
    is_team_sector,
    RELIC_REWARD_RANGE,
    MAX_UNITS,
    MAX_STEPS_IN_MATCH
)
from state.space import Space
from state.fleet import Fleet
from state.node import Node
from state.action_type import ActionType
from state.ship import Ship
from pathfinding import (
    astar,
    find_closest_target,
    nearby_positions,
    create_weights,
    estimate_energy_cost,
    path_to_actions,
    manhattan_distance,
)
import numpy as np
import random


class Rulebased:
    def __init__(self):
        pass


    def act(self, state: State):
        if state.match_step == 0:
            return self.create_actions_array(state)
        
        self.find_relics(state)
        self.find_rewards(state)
        self.harvest(state)
        self.fight(state)
        self.employ_unemployed(state)
        self.harvest_if_losing(state)
        self.optimize_harvesting(state)

        return self.create_actions_array(state)

    

    def create_actions_array(self, state: State):
        ships = state.fleet.ships
        actions = np.zeros((len(ships), 3), dtype=int)

        for i, ship in enumerate(ships):
            if ship.action is not None:
                sap_x, sap_y = ship.sap_direction if ship.action == ActionType.sap else (0, 0)
                actions[i] = ship.action, sap_x, sap_y
                ship.sap_direction = (0, 0)

        return actions

    def employ_unemployed(self, state: State):
        unexplored_relics = self.get_unexplored_relics(state)
        if len(unexplored_relics) == 0:
            return
        for ship in state.fleet:
            if ship.task is not None or ship.target is not None:
                continue
            relic_coordinates, _ = find_closest_target(ship.node.coordinates,
                                             [relic.coordinates for relic in unexplored_relics])
            targets = []
            for x, y in nearby_positions(*relic_coordinates, RELIC_REWARD_RANGE):
                node = state.space.get_node(x, y)
                if not node.explored_for_reward and node.is_walkable:
                    targets.append((x, y))
            target, _ = find_closest_target(ship.coordinates, targets)
            if not target:
                return
            path = astar(create_weights(state.space, state.config.ALL_REWARDS_FOUND, state.config.NEBULA_ENERGY_REDUCTION), ship.coordinates, target)
            energy = estimate_energy_cost(state.space, path, state.config.NEBULA_ENERGY_REDUCTION, state.config.UNIT_MOVE_COST)
            actions = path_to_actions(path)
            if actions and ship.energy >= energy:
                ship.task = "find_rewards"
                ship.target = state.space.get_node(*target)
                ship.action = actions[0]
        ship_coords = [ship.coordinates for ship in state.fleet]
        for ship in state.fleet:
            if ship.task is not None or ship.target is not None:
                continue
            target, _ = find_closest_target(ship.node.coordinates,
                                            [rew.coordinates for rew in state.space.reward_nodes
                                                if rew.is_walkable and rew.coordinates not in ship_coords])
            if not target:
                return
            path = astar(create_weights(state.space, state.config.ALL_REWARDS_FOUND, state.config.NEBULA_ENERGY_REDUCTION), ship.coordinates, target)
            energy = estimate_energy_cost(state.space, path, state.config.NEBULA_ENERGY_REDUCTION, state.config.UNIT_MOVE_COST)
            actions = path_to_actions(path)
            if actions and ship.energy >= energy:
                ship.task = "harvest"
                ship.target = state.space.get_node(*target)
                ship.action = actions[0]

    def find_relics(self, state: State):
        if state.config.ALL_RELICS_FOUND:
            for ship in state.fleet:
                if ship.task == "find_relics":
                    ship.task = None
                    ship.target = None
            return

        targets = set()
        for node in state.space:
            if not node.explored_for_relic:
                # We will only find relics in our part of the map
                # because relics are symmetrical.
                if is_team_sector(state.fleet.team_id, *node.coordinates):
                    targets.add(node.coordinates)

        def set_task(ship):
            if ship.task and ship.task != "find_relics":
                return False

            if ship.energy < state.config.UNIT_MOVE_COST:
                return False

            target, _ = find_closest_target(ship.coordinates, targets)
            if not target:
                return False

            path = astar(create_weights(state.space, state.config.ALL_REWARDS_FOUND, state.config.NEBULA_ENERGY_REDUCTION), ship.coordinates, target)
            energy = estimate_energy_cost(state.space, path, state.config.NEBULA_ENERGY_REDUCTION, state.config.UNIT_MOVE_COST)
            actions = path_to_actions(path)
            if actions and ship.energy >= energy:
                ship.task = "find_relics"
                ship.target = state.space.get_node(*target)
                ship.action = actions[0]

                for x, y in path:
                    for xy in nearby_positions(x, y, state.config.UNIT_SENSOR_RANGE):
                        if xy in targets:
                            targets.remove(xy)

                return True

            return False

        for ship in state.fleet:
            if set_task(ship):
                continue

            if ship.task == "find_relics":
                ship.task = None
                ship.target = None

    def find_rewards(self, state: State):
        if state.config.ALL_REWARDS_FOUND:
            for ship in state.fleet:
                if ship.task == "find_rewards":
                    ship.task = None
                    ship.target = None
            return

        unexplored_relics = self.get_unexplored_relics(state)

        relic_node_to_ship = {}
        for ship in state.fleet:
            if ship.task == "find_rewards":
                if ship.target is None:
                    ship.task = None
                    continue

                if (
                    ship.target in unexplored_relics
                    and ship.energy > state.config.UNIT_MOVE_COST * 5
                ):
                    relic_node_to_ship[ship.target] = ship
                else:
                    ship.task = None
                    ship.target = None

        for relic in unexplored_relics:
            if relic not in relic_node_to_ship:

                # find the closest ship to the relic node
                min_distance, closes_ship = float("inf"), None
                for ship in state.fleet:
                    if ship.task and ship.task != "find_rewards":
                        continue

                    if ship.energy < state.config.UNIT_MOVE_COST * 5:
                        continue

                    distance = manhattan_distance(ship.coordinates, relic.coordinates)
                    if distance < min_distance:
                        min_distance, closes_ship = distance, ship

                if closes_ship:
                    relic_node_to_ship[relic] = closes_ship

        def set_task(ship, relic_node, can_pause):
            targets = []
            for x, y in nearby_positions(
                *relic_node.coordinates, RELIC_REWARD_RANGE
            ):
                node = state.space.get_node(x, y)
                if not node.explored_for_reward and node.is_walkable:
                    targets.append((x, y))

            target, _ = find_closest_target(ship.coordinates, targets)

            if target == ship.coordinates and not can_pause:
                target, _ = find_closest_target(
                    ship.coordinates,
                    [
                        n.coordinates
                        for n in state.space
                        if n.explored_for_reward and n.is_walkable
                    ],
                )

            if not target:
                return

            path = astar(create_weights(state.space, state.config.ALL_REWARDS_FOUND, state.config.NEBULA_ENERGY_REDUCTION), ship.coordinates, target)
            energy = estimate_energy_cost(state.space, path, state.config.NEBULA_ENERGY_REDUCTION, state.config.UNIT_MOVE_COST)
            actions = path_to_actions(path)

            if actions and ship.energy >= energy:
                ship.task = "find_rewards"
                ship.target = state.space.get_node(*target)
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

    def get_unexplored_relics(self, state: State) -> list[Node]:
        relic_nodes = []
        for relic_node in state.space.relic_nodes:
            if not is_team_sector(state.team_id, *relic_node.coordinates):
                continue

            explored = True
            for x, y in nearby_positions(
                *relic_node.coordinates, RELIC_REWARD_RANGE
            ):
                node = state.space.get_node(x, y)
                if not node.explored_for_reward and node.is_walkable:
                    explored = False
                    break

            if explored:
                continue

            relic_nodes.append(relic_node)

        return relic_nodes


    def harvest(self, state: State):

        def set_task(ship, target_node):
            if ship.node == target_node:
                ship.task = "harvest"
                ship.target = target_node
                ship.action = ActionType.center
                return True

            path = astar(
                create_weights(state.space, state.config.ALL_REWARDS_FOUND, state.config.NEBULA_ENERGY_REDUCTION),
                start=ship.coordinates,
                goal=target_node.coordinates,
            )
            energy = estimate_energy_cost(state.space, path, state.config.NEBULA_ENERGY_REDUCTION, state.config.UNIT_MOVE_COST)
            actions = path_to_actions(path)

            if not actions or ship.energy < energy:
                return False

            ship.task = "harvest"
            ship.target = target_node
            ship.action = actions[0]
            return True

        booked_nodes = set()
        for ship in state.fleet:
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
        for n in state.space.reward_nodes:
            if n.is_walkable and n not in booked_nodes:
                targets.add(n.coordinates)
        if not targets:
            return

        for ship in state.fleet:
            if ship.task:
                continue

            target, _ = find_closest_target(ship.coordinates, targets)

            if target and set_task(ship, state.space.get_node(*target)):
                targets.remove(target)
            else:
                ship.task = None
                ship.target = None


    def _get_ships_centroid(self, ship_indices, state: State):
        ships_centroid_pos = np.array([0, 0])
        for ship_idx in ship_indices:
            x, y = state.fleet.ships[ship_idx].coordinates
            ships_centroid_pos[0] += x
            ships_centroid_pos[1] += y
        if len(ship_indices) > 0:
            ships_centroid_pos = np.clip(
                (ships_centroid_pos / len(ship_indices))
            , 0, 23).astype('int')
        else:
            ships_centroid_pos = np.array((0, 0) if state.team_id == 0 else (23, 23))
        return ships_centroid_pos


    def _get_sorter_rewards(self, space_weights, remaining_ship_indices, state: State):
        remaining_ships_centroid_pos = self._get_ships_centroid(remaining_ship_indices, state)

        def rewards_sort_key(reward_node: Node):
            path = astar(
                space_weights,
                start=tuple(remaining_ships_centroid_pos),
                goal=reward_node.coordinates,
            )
            actions = path_to_actions(path)
            if actions:
                return len(actions), -estimate_energy_cost(state.space, path, state.config.NEBULA_ENERGY_REDUCTION, state.config.UNIT_MOVE_COST)
            return float('inf'), float('inf')

        sorted_reward_nodes = sorted(
            state.space.reward_nodes,
            key=rewards_sort_key
        )
        return sorted_reward_nodes
    

    def _ship_relevance_to_node(self, ship: Ship, node: Node, space_weights, state: State):
        if node.coordinates == ship.coordinates:
            return 0, ActionType.center

        path = astar(
            space_weights,
            start=ship.coordinates,
            goal=node.coordinates,
        )
        path_energy_cost = estimate_energy_cost(state.space, path, state.config.NEBULA_ENERGY_REDUCTION, state.config.UNIT_MOVE_COST)
        actions = path_to_actions(path)
        path_len = len(path)
        if (ship.energy - path_energy_cost) >= 0 and actions and path_len <= (100 - state.match_step):
            return path_len, actions[0]

        return None, None


    def harvest_if_losing(self, state: State):
        points = state.points
        opp_points = state.opp_points
        remaining_steps = MAX_STEPS_IN_MATCH - state.match_step

        reward = max(0, points - state.fleet.points)
        opp_reward = max(0, points - state.fleet.points)

        predicted_points = points + remaining_steps * reward
        opp_predicted_points = opp_points + remaining_steps * opp_reward

        if opp_predicted_points > predicted_points and remaining_steps > 0:
            # HARVEST AS FAST AS YOU CAN!!!

            missing_harvesters = int((opp_predicted_points - predicted_points) / remaining_steps) + 1

            space_weights = create_weights(state.space, state.config.ALL_REWARDS_FOUND, state.config.NEBULA_ENERGY_REDUCTION)
            booked_nodes = set()
            for ship in state.fleet:
                if ship.task == 'harvest':
                    booked_nodes.add(ship.target)

            remaining_ship_indices = set(
                ship_idx for ship_idx, ship in enumerate(state.fleet.ships)
                if ship.node is not None and ship.task != 'harvest'
            )

            sorted_reward_nodes = self._get_sorter_rewards(space_weights, remaining_ship_indices, state)
            new_harvesters = []
            for reward_node in sorted_reward_nodes:
                if reward_node in booked_nodes:
                    continue

                ship_relevances = [None] * MAX_UNITS
                ship_actions = [None] * MAX_UNITS
                for ship_idx in remaining_ship_indices:
                    ship = state.fleet.ships[ship_idx]
                    ship_relevances[ship_idx], ship_actions[ship_idx] = self._ship_relevance_to_node(ship, reward_node, space_weights, state)

                best_ship_idx = None
                for ship_idx in remaining_ship_indices:
                    if ship_relevances[ship_idx] is not None:
                        if best_ship_idx is None:
                            best_ship_idx = ship_idx

                        if ship_relevances[ship_idx] <= ship_relevances[best_ship_idx]:
                            if state.fleet.ships[ship_idx].energy < state.fleet.ships[best_ship_idx].energy:
                                best_ship_idx = ship_idx

                if best_ship_idx is not None:
                    new_harvesters.append({
                        'ship': state.fleet.ships[best_ship_idx],
                        'target': reward_node,
                        'action': ship_actions[best_ship_idx]
                    })
                    remaining_ship_indices.remove(best_ship_idx)

            if len(new_harvesters) >= missing_harvesters:
                for new_harvester in new_harvesters[:missing_harvesters]:
                    new_harvester['ship'].target = new_harvester['target']
                    new_harvester['ship'].task = "harvest"
                    new_harvester['ship'].action = new_harvester['action']


    def optimize_harvesting(self, state: State):
        cur_harvesting_ships = [
            ship for ship in state.fleet
            if ship.task == 'harvest' and ship.action in (ActionType.center, ActionType.sap) and ship.target == ship.node
        ]
        space_weights = create_weights(state.space, state.config.ALL_REWARDS_FOUND, state.config.NEBULA_ENERGY_REDUCTION)
        for ship in state.fleet:
            if ship not in cur_harvesting_ships and ship.task == 'harvest' and ship.action not in (ActionType.center, ActionType.sap):
                for cur_harvesting_ship in cur_harvesting_ships:
                    if ship.next_coords() == cur_harvesting_ship.coordinates and cur_harvesting_ship.energy > state.config.UNIT_MOVE_COST:
                        path = astar(
                            space_weights,
                            start=ship.coordinates,
                            goal=ship.target.coordinates,
                        )
                        actions = path_to_actions(path)
                        if len(actions) == 2:
                            cur_harvesting_ship.action = actions[1]
                            cur_harvesting_ship.target, ship.target = ship.target, cur_harvesting_ship.target

                        


    def fight(self, state: State):
        def find_best_target(ship, targets):
            #todo: optimizirovat!!!!! schitat cherez dp
            if state.game_num == 0 and random.random() < 0.7:
                return None
            best_target = None
            max_opp_ship_count = 0
            for x, y in nearby_positions(*ship.coordinates, state.config.UNIT_SAP_RANGE):
                opp_ship_count = 0
                for dx in range(-1, 2):
                    for dy in range(-1, 2):
                        if (x + dx, y + dy) in targets:
                            opp_ship_count += 1 if dx == 0 and dy == 0 else 0.25 #todo: pomenyat'!!!!
                if opp_ship_count > max_opp_ship_count and manhattan_distance(ship.coordinates, (x, y)) <= state.config.UNIT_SAP_RANGE:
                    max_opp_ship_count = opp_ship_count
                    best_target = (x, y)
            if max_opp_ship_count <= 1.5 and random.random() < 0.5:
                return None
            return best_target

        targets = {opp_ship.coordinates for opp_ship in state.opp_fleet}
        for x, y in targets:
            for ship in state.fleet:
                if manhattan_distance((x, y), ship.coordinates) <= state.config.UNIT_SAP_RANGE:
                    for relic_node in state.space.relic_nodes:
                        if manhattan_distance((x, y), relic_node.coordinates) <= RELIC_REWARD_RANGE\
                                or manhattan_distance(ship.coordinates, relic_node.coordinates) <= RELIC_REWARD_RANGE: #todo: mb ubrat' eto uslovie
                            if ship.node.energy\
                                    and ship.energy + (ship.node.energy * 2 if ship.node.energy < 0 else 0) > state.config.UNIT_SAP_COST:
                                best_target = find_best_target(ship, targets)
                                if best_target:
                                    ship.sap_direction = (best_target[0] - ship.node.x, best_target[1] - ship.node.y)
                                    ship.action = ActionType.sap
                                    break