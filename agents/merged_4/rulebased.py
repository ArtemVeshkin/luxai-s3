from scipy.signal import convolve2d

from state.state import State
from state.action_type import ActionType
from state.base import (
    is_team_sector,
    RELIC_REWARD_RANGE,
    MAX_UNITS,
    MAX_STEPS_IN_MATCH,
    SPACE_SIZE
)
from state.space import Space
from state.fleet import Fleet
from state.node import Node, NodeType
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
    reward_distance_matrix
)
import numpy as np
import random
from collections import defaultdict
from sys import stderr
import math


class Rulebased:
    def __init__(self):
        pass

    def act(self, state: State):
        if state.match_step == 0:
            return self.create_actions_array(state)
        
        self.find_relics(state)
        self.find_rewards(state)
        self.harvest(state)
        self.destroy(state)
        self.estimate_nebula_energy(state)
        self.build_barrier(state)
        self.harvest_if_losing(state)
        self.optimize_harvesting(state)

        return self.create_actions_array(state)

    def create_actions_array(self, state: State):
        ships = state.fleet.ships
        actions = np.zeros((len(ships), 3), dtype=int)

        for i, ship in enumerate(ships):
            if ship.action is not None:
                if ship.action == ActionType.sap:
                    # print("Sapping to", ship.action, ship.action_sap_info[0], ship.action_sap_info[1])
                    actions[i] = ship.action, int(ship.action_sap_info[0]), int(ship.action_sap_info[1])
                else:
                    # print("Action", ship.action)
                    actions[i] = ship.action, 0, 0

        return actions


    def estimate_nebula_energy(self, state: State):
        ships = state.fleet.ships
        for ship in ships:
            if ship.active:
                ship.projected_energy = self.project_energy(ship, state)


    def project_energy(self, ship: Ship, state: State):
        # get needed info
        ship_coords = ship.coordinates
        direction = ship.action.to_direction() if ship.action else (0, 0)
        cur_tile = state.space.get_node(ship_coords[0], ship_coords[1])
        target_tile = state.space.get_node(ship_coords[0] + direction[0],
                                          ship_coords[1] + direction[1])
        cur_tile_energy = cur_tile.energy

        # ship steps from nebula on empty or oposite
        if (cur_tile.type == NodeType.nebula):  # is not (ship.prev_node_type == NodeType.nebula)
            nebula_diff = True
        else:
            nebula_diff = False

        if nebula_diff and not state.config.ENERGY_NODES_MOVEMENT_STATUS[-1] and not state.config.OBSTACLES_MOVEMENT_STATUS[-1]:
            if not ship.expected_is_enemy and len(ship.track) > 0 and len(ship.energies) > 0 and ship.energy != 0:
                move_cost = (0 if ship.track[-1][0] == ship_coords[0] and ship.track[-1][1] == ship_coords[
                    1] else state.config.UNIT_MOVE_COST)
                est = ship.energy - ship.energies[-1] - cur_tile_energy + move_cost
                if sum(state.config.NEBULA_ENERGY_REDUCTION_SAMPLES.values()) < 10:
                    state.config.NEBULA_ENERGY_REDUCTION_SAMPLES.update([est])

        # update ship logs
        ship.energies.append(ship.energy)
        ship.track.append(ship_coords)
        # ship.predicted_energies.append(predicted_energy)

        ship.expected_is_enemy = target_tile.is_enemy
        ship.prev_node_type = cur_tile.type
        ship.prev_node_energy = cur_tile_energy

        return cur_tile_energy
    

    def get_relic_barrier(self, state: State, relic_node: Node):
        relic_border_coords = set()
        for shift in range(7):
            relic_border_coords.add((relic_node.x + 3, relic_node.y - 3 + shift))
            relic_border_coords.add((relic_node.x - 3, relic_node.y - 3 + shift))
        for shift in range(7):
            relic_border_coords.add((relic_node.x - 3 + shift, relic_node.y + 3))
            relic_border_coords.add((relic_node.x - 3 + shift, relic_node.y - 3))
        
        def coord_filter(coords):
            if coords[0] < 0 or coords[0] > 23 or coords[1] < 0 or coords[1] > 23:
                return False
            return state.space.get_node(*coords).is_walkable

        relic_border_coords = np.array(list(filter(coord_filter, relic_border_coords)))
        distances = np.array([
            -manhattan_distance((0, 0) if state.team_id == 0 else (23, 23), coord) for coord in relic_border_coords
        ])
        return relic_border_coords[np.argsort(distances)][:5]


    def get_barrier_coords(self, state: State):
        barrier_coords = []
        for relic_node in state.space.relic_nodes:
            if not is_team_sector(state.team_id, *relic_node.coordinates):
                continue

            for barrier_coord in self.get_relic_barrier(state, relic_node):
                barrier_coords.append(barrier_coord)
        barrier_coords = [(coord[0], coord[1]) for coord in barrier_coords]

        def coord_filter(coords):
            if coords[0] < 0 or coords[0] > 23 or coords[1] < 0 or coords[1] > 23:
                return False
            return state.space.get_node(*coords).is_walkable

        barrier_coords = set(filter(coord_filter, barrier_coords))
        if len(barrier_coords) == 0:
            center_barrier_coords = [
                (10, 10),
                (7, 13),
                (13, 7),
                (4, 16),
                (16, 4),
                (1, 19),
                (19, 1),
                (7, 7),
                (4, 10),
                (10, 4),
                (4, 4),
            ] if state.team_id == 0 else [
                (13, 13),
                (10, 16),
                (16, 10),
                (7, 19),
                (19, 7),
                (4, 22),
                (22, 4),
                (16, 16),
                (13, 19),
                (19, 13),
                (19, 19)
            ]
            barrier_coords = list(filter(coord_filter, center_barrier_coords))

        return set(barrier_coords)


    def build_barrier(self, state: State):
        barrier_coords = list(self.get_barrier_coords(state))
        unemployed_ship_ids = set()
        for ship in state.fleet:
            if ship.task is not None or ship.target is not None or ship.action == ActionType.sap or ship.energy < state.config.UNIT_MOVE_COST:
                continue
            unemployed_ship_ids.add(ship.unit_id)
        n_unemployed_ships = len(unemployed_ship_ids)

        barrier_coords = np.array(barrier_coords)

        distances = np.array([
            -manhattan_distance((0, 0) if state.team_id == 0 else (23, 23), coord) for coord in barrier_coords
        ])
        barrier_coords = barrier_coords[np.argsort(distances)]

        for i in range(n_unemployed_ships):
            target = tuple(barrier_coords[i % len(barrier_coords)])

            closest_ship_id, min_distance = None, float("inf")
            for ship_id in unemployed_ship_ids:
                d = manhattan_distance(target, state.fleet.ships[ship_id].coordinates)
                if d < min_distance:
                    closest_ship_id = ship_id
                    min_distance = d
            if closest_ship_id is None:
                continue
            
            closest_ship = state.fleet.ships[closest_ship_id]
            path = astar(create_weights(
                state.space,
                state.config.ALL_REWARDS_FOUND,
                state.config.NEBULA_ENERGY_REDUCTION,
                closest_ship.coordinates,
                max_dist=MAX_STEPS_IN_MATCH + 1 - state.match_step,
                energy_update_in=state.match_step % state.config.ENERGY_NODE_MOVEMENT_PERIOD + 1
            ), closest_ship.coordinates, target)
            actions = path_to_actions(path)
            if actions or ship.coordinates == target:
                unemployed_ship_ids.remove(closest_ship_id)
            if actions:
                closest_ship.action = actions[0]


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

            path = astar(create_weights(
                state.space,
                state.config.ALL_REWARDS_FOUND,
                state.config.NEBULA_ENERGY_REDUCTION,
                ship.coordinates,
                max_dist=MAX_STEPS_IN_MATCH + 1 - state.match_step,
                energy_update_in=state.match_step % state.config.ENERGY_NODE_MOVEMENT_PERIOD + 1
            ), ship.coordinates, target)
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
            
            if target == ship.coordinates:
                ship.task = "find_reward"
                ship.target = state.space.get_node(*target)
                ship.action == ActionType.center
                return True
            else:
                path = astar(create_weights(
                    state.space,
                    state.config.ALL_REWARDS_FOUND,
                    state.config.NEBULA_ENERGY_REDUCTION,
                    ship.coordinates,
                    max_dist=MAX_STEPS_IN_MATCH + 1 - state.match_step,
                    energy_update_in=state.match_step % state.config.ENERGY_NODE_MOVEMENT_PERIOD + 1
                ), ship.coordinates, target)
                energy = estimate_energy_cost(state.space, path, state.config.NEBULA_ENERGY_REDUCTION, state.config.UNIT_MOVE_COST)
                actions = path_to_actions(path)

                if actions and ship.energy >= energy:
                    ship.task = "find_rewards"
                    ship.target = state.space.get_node(*target)
                    ship.action = actions[0]
                    return True

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
                create_weights(
                    state.space,
                    state.config.ALL_REWARDS_FOUND,
                    state.config.NEBULA_ENERGY_REDUCTION,
                    ship.coordinates,
                    max_dist=MAX_STEPS_IN_MATCH + 1 - state.match_step,
                    energy_update_in=state.match_step % state.config.ENERGY_NODE_MOVEMENT_PERIOD + 1
                ), start=ship.coordinates, goal=target_node.coordinates,
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


    def _get_sorter_rewards(self, remaining_ship_indices, state: State):
        remaining_ships_centroid_pos = tuple(self._get_ships_centroid(remaining_ship_indices, state))

        def rewards_sort_key(reward_node: Node):
            space_weights = create_weights(
                state.space,
                state.config.ALL_REWARDS_FOUND,
                state.config.NEBULA_ENERGY_REDUCTION,
                remaining_ships_centroid_pos,
                max_dist=MAX_STEPS_IN_MATCH + 1 - state.match_step,
                energy_update_in=state.match_step % state.config.ENERGY_NODE_MOVEMENT_PERIOD + 1
            )
            path = astar(
                space_weights,
                start=remaining_ships_centroid_pos,
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
    

    def _ship_relevance_to_node(self, ship: Ship, node: Node, state: State):
        if node.coordinates == ship.coordinates:
            return 0, ActionType.center

        space_weights = create_weights(
            state.space,
            state.config.ALL_REWARDS_FOUND,
            state.config.NEBULA_ENERGY_REDUCTION,
            ship.coordinates,
            max_dist=MAX_STEPS_IN_MATCH + 1 - state.match_step,
            energy_update_in=state.match_step % state.config.ENERGY_NODE_MOVEMENT_PERIOD + 1
        )
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

            booked_nodes = set()
            for ship in state.fleet:
                if ship.task == 'harvest':
                    booked_nodes.add(ship.target)

            remaining_ship_indices = set(
                ship_idx for ship_idx, ship in enumerate(state.fleet.ships)
                if ship.node is not None and ship.task != 'harvest'
            )

            sorted_reward_nodes = self._get_sorter_rewards(remaining_ship_indices, state)
            new_harvesters = []
            for reward_node in sorted_reward_nodes:
                if reward_node in booked_nodes:
                    continue

                ship_relevances = [None] * MAX_UNITS
                ship_actions = [None] * MAX_UNITS
                for ship_idx in remaining_ship_indices:
                    ship = state.fleet.ships[ship_idx]
                    ship_relevances[ship_idx], ship_actions[ship_idx] = self._ship_relevance_to_node(ship, reward_node, state)

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
        for ship in state.fleet:
            if ship not in cur_harvesting_ships and ship.task == 'harvest' and ship.action not in (ActionType.center, ActionType.sap):
                for cur_harvesting_ship in cur_harvesting_ships:
                    if ship.next_coords() == cur_harvesting_ship.coordinates and cur_harvesting_ship.energy > state.config.UNIT_MOVE_COST:
                        space_weights = create_weights(
                            state.space,
                            state.config.ALL_REWARDS_FOUND,
                            state.config.NEBULA_ENERGY_REDUCTION,
                            ship.coordinates,
                            max_dist=MAX_STEPS_IN_MATCH + 1 - state.match_step,
                            energy_update_in=state.match_step % state.config.ENERGY_NODE_MOVEMENT_PERIOD + 1
                        )
                        path = astar(
                            space_weights,
                            start=ship.coordinates,
                            goal=ship.target.coordinates,
                        )
                        actions = path_to_actions(path)
                        if len(actions) == 2:
                            cur_harvesting_ship.action = actions[1]
                            cur_harvesting_ship.target, ship.target = ship.target, cur_harvesting_ship.target


    def destroy(self, state: State):
        team_spawn_positions = {0: (0, 0), 1: (23, 23)}
        if state.team_id in team_spawn_positions:
            state.spawn = team_spawn_positions[state.team_id]
            state.enemy_spawn = team_spawn_positions[abs(state.team_id-1)]
        else:
            raise ValueError(f'Unknown team {state.team_id}')

        def predict_ship_position(ship):
            """Predict the next ship position based on movement history."""
            if ship.node.reward:
                return ship.coordinates
            if len(ship.track) >= 2:
                dx = ship.track[-1][0] - ship.track[-2][0]
                dy = ship.track[-1][1] - ship.track[-2][1]
                if (ship.coordinates[0] + dx >= 0) and (ship.coordinates[0] + dx <= 23) and (ship.coordinates[1] + dy >= 0) and (ship.coordinates[1] + dy <= 23) and state.space.get_node(ship.coordinates[0] + dx, ship.coordinates[1] + dy).is_walkable:
                    return (ship.coordinates[0] + dx, ship.coordinates[1] + dy)
                else:
                    return (ship.coordinates[0], ship.coordinates[1])
            return ship.coordinates

        def get_energy_matrices(state, reward_nodes_pos):
            predicted_energy_matrix = np.zeros((SPACE_SIZE, SPACE_SIZE), dtype=int)
            current_energy_matrix = np.zeros((SPACE_SIZE, SPACE_SIZE), dtype=int)
            current_allied_energy_matrix = np.zeros((SPACE_SIZE, SPACE_SIZE), dtype=int)

            for ship in state.opp_fleet:
                current_x, current_y = ship.coordinates
                current_energy_matrix[current_x, current_y] += ship.energy

                if ship.energy > state.config.UNIT_MOVE_COST and ship.coordinates not in reward_nodes_pos:
                    predicted_x, predicted_y = predict_ship_position(ship)
                    if 0 <= predicted_x < SPACE_SIZE and 0 <= predicted_y < SPACE_SIZE:
                        predicted_energy_matrix[predicted_x, predicted_y] += ship.energy
                else:
                    predicted_energy_matrix[current_x, current_y] += ship.energy

            for ship in state.fleet:
                current_x, current_y = ship.next_coords() if ship.action else ship.coordinates
                current_allied_energy_matrix[current_x, current_y] += ship.energy

            return predicted_energy_matrix, current_energy_matrix, current_allied_energy_matrix

        def filter_ships(state):
            for ship in state.fleet:
                if ship.energy >= state.config.UNIT_SAP_COST and (
                        (ship.task == 'harvest' and ship.action == ActionType.center) or (ship.task != 'harvest')
                ):
                    yield ship

        def find_best_sap_node(matrix):
            kernel = np.full((3, 3), state.config.UNIT_SAP_DROPOFF_FACTOR, dtype=int)
            kernel[1, 1] = 1  # Central node weight

            convolved = convolve2d(matrix, kernel, mode='same', boundary='fill', fillvalue=0)

            return convolved

        reward_nodes_pos = set(node.coordinates for node in state.space.relic_nodes)
        prior_matrix = reward_distance_matrix(reward_nodes_pos)
        predicted_energy_matrix, current_energy_matrix, current_allied_energy_matrix = get_energy_matrices(state, reward_nodes_pos)
        dmg_matrix = np.zeros(shape=(SPACE_SIZE, SPACE_SIZE))

        for ship in filter_ships(state):
            valid_nodes = [node for node in state.space if ship.can_sap(node, state.config)]
            if valid_nodes:
                enemy_priors = 1 / np.exp(prior_matrix + 1)
                edjusted_energy_matix = (predicted_energy_matrix - dmg_matrix - current_allied_energy_matrix - state.config.UNIT_MOVE_COST) * enemy_priors
                max_dmg_matrix = find_best_sap_node(edjusted_energy_matix)
                best_node_idx = np.argmax(list(max_dmg_matrix[node.coordinates[0]][node.coordinates[1]] for node in valid_nodes))
                max_score = np.max(list(max_dmg_matrix[node.coordinates[0]][node.coordinates[1]] for node in valid_nodes))
                max_node = valid_nodes[best_node_idx]

                ax, yx = max_node.coordinates
                dmg_matrix[ax-1:ax+1, yx-1:yx+1] = state.config.UNIT_SAP_DROPOFF_FACTOR * state.config.UNIT_SAP_COST
                dmg_matrix[ax, yx] = state.config.UNIT_SAP_COST

                if max_score > 0:
                    ship.action = ActionType.sap
                    ship.action_sap_info = (
                        max_node.coordinates[0] - ship.coordinates[0],
                        max_node.coordinates[1] - ship.coordinates[1],
                    )

