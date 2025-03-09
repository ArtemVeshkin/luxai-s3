import heapq
import numpy as np

from state.base import (
    SPACE_SIZE,
    HIDDEN_NODE_ENERGY,
    MAX_ENERGY_PER_TILE
)
from state.action_type import ActionType
from state.node import NodeType

CARDINAL_DIRECTIONS = [(0, 1), (0, -1), (1, 0), (-1, 0)]


def astar(weights, start, goal):
    # A* algorithm
    # returns the shortest path form start to goal

    applicable_weighs = weights[np.where(weights >= 0)]
    if len(applicable_weighs) > 0:
        min_weight = applicable_weighs.min()
    else:
        return []

    def heuristic(p1, p2):
        return min_weight * manhattan_distance(p1, p2)

    queue = []

    # nodes: [x, y, (parent.x, parent.y, distance, f)]
    nodes = np.zeros((*weights.shape, 4), dtype=np.float32)
    nodes[:] = -1

    heapq.heappush(queue, (0, start))
    nodes[start[0], start[1], :] = (*start, 0, heuristic(start, goal))

    while queue:
        f, (x, y) = heapq.heappop(queue)

        if (x, y) == goal:
            return reconstruct_path(nodes, start, goal)

        if f > nodes[x, y, 3]:
            continue

        distance = nodes[x, y, 2]
        for x_, y_ in get_neighbors(x, y):
            cost = weights[y_, x_]
            if cost < 0:
                continue

            new_distance = distance + cost
            if nodes[x_, y_, 2] < 0 or nodes[x_, y_, 2] > new_distance:
                new_f = new_distance + heuristic((x_, y_), goal)
                nodes[x_, y_, :] = x, y, new_distance, new_f
                heapq.heappush(queue, (new_f, (x_, y_)))

    return []


def manhattan_distance(a, b) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def get_neighbors(x, y):
    for dx, dy in CARDINAL_DIRECTIONS:
        x_ = x + dx
        if x_ < 0 or x_ >= SPACE_SIZE:
            continue

        y_ = y + dy
        if y_ < 0 or y_ >= SPACE_SIZE:
            continue

        yield x_, y_


def reconstruct_path(nodes, start, goal):
    p = goal
    path = [p]
    while p != start:
        x = int(nodes[p[0], p[1], 0])
        y = int(nodes[p[0], p[1], 1])
        p = x, y
        path.append(p)
    return path[::-1]


def nearby_positions(x, y, distance):
    for x_ in range(max(0, x - distance), min(SPACE_SIZE, x + distance + 1)):
        for y_ in range(max(0, y - distance), min(SPACE_SIZE, y + distance + 1)):
            yield x_, y_


def create_weights(space, all_rewards_found, nebula_energy_reduction, coords=None, max_dist=None, energy_update_in=None):
    # create weights for AStar algorithm

    weights = np.zeros((SPACE_SIZE, SPACE_SIZE), dtype=np.float32)
    for node in space:

        distance = manhattan_distance(coords, node.coordinates) if coords else None
        if distance is not None and max_dist is not None and distance > max_dist:
            weight = -1
        elif not node.is_walkable:
            weight = -1
        else:
            node_energy = node.predicted_energy
            if node_energy is None:
                node_energy = HIDDEN_NODE_ENERGY
    
            node_energy = node.predicted_energy if node.predicted_energy is not None else HIDDEN_NODE_ENERGY
            if distance is not None and energy_update_in is not None and distance > energy_update_in:
                node_energy = HIDDEN_NODE_ENERGY

            # pathfinding can't deal with negative weight
            weight = MAX_ENERGY_PER_TILE + 1 - node_energy + (1 if node.visited_times > 0 and not all_rewards_found else 0)
            #todo: uchest' uvelichenie seen nodes

        if node.type == NodeType.nebula:
            weight += nebula_energy_reduction * 3

        weights[node.y, node.x] = weight

    return weights


def find_closest_target(start, targets):
    target, min_distance = None, float("inf")
    for t in targets:
        d = manhattan_distance(start, t)
        if d < min_distance:
            target, min_distance = t, d

    return target, min_distance


def estimate_energy_cost(space, path, nebula_energy_reduction, unit_move_cost):
    if len(path) <= 1:
        return 0

    energy = 0
    last_position = path[0]
    for x, y in path[1:]:
        node = space.get_node(x, y)
        if node.predicted_energy is not None:
            energy -= node.predicted_energy
        else:
            energy -= HIDDEN_NODE_ENERGY

        if node.type == NodeType.nebula:
            energy += nebula_energy_reduction

        if (x, y) != last_position:
            energy += unit_move_cost

    return energy


def path_to_actions(path):
    actions = []
    if not path:
        return actions

    last_position = path[0]
    for x, y in path[1:]:
        direction = ActionType.from_coordinates(last_position, (x, y))
        actions.append(direction)
        last_position = (x, y)

    return actions


import numpy as np

SPACE_SIZE = 24

def reward_distance_matrix(reward_nodes_pos, size=SPACE_SIZE):
    if not reward_nodes_pos:
        # No reward nodes; return a large-distance matrix (or np.inf)
        return np.full((size, size), np.inf, dtype=float)

    grid_x, grid_y = np.indices((size, size))
    reward_coords = np.array(list(reward_nodes_pos)).reshape(-1, 2)

    distances = np.abs(grid_x[..., np.newaxis] - reward_coords[:, 0]) + \
                np.abs(grid_y[..., np.newaxis] - reward_coords[:, 1])

    return distances.min(axis=2)
