from typing import Dict
import numpy as np
from sys import stderr

from base import (
    Global,
    get_match_step,
    is_team_sector,
)
from space import Space
from fleet import Fleet
from node import Node
from ship import Ship
from action_type import ActionType
from pathfinding import (
    astar,
    find_closest_target,
    nearby_positions,
    create_weights,
    estimate_energy_cost,
    path_to_actions,
    manhattan_distance,
)
from debug import (
    show_map,
    show_energy_field,
    show_exploration_map
)


def check_target_is_reachable(ship: Ship, space: Space, target: Node):
    if not target:
        return False
    
    if ship.node == target:
        return True

    if ship.energy < Global.UNIT_MOVE_COST:
        return False
    
    path = astar(create_weights(space), ship.coordinates, target)
    actions = path_to_actions(path)
    if not actions:
        return False
    
    energy = estimate_energy_cost(space, path)
    if ship.energy < energy:
        return False

    return True


def go_to_target(ship: Ship, space: Space, target: Node):
    if ship.node == target:
        return ActionType.center

    path = astar(create_weights(space), ship.coordinates, target)
    actions = path_to_actions(path)

    return actions[0]


### SLEEP ###

### RUN ###
def sleep_run(params: Dict) -> ActionType:
    return ActionType.center


### CHECK ###
def sleep_check(params: Dict) -> bool:
    return True


### FIND_RELICS ###

### CHECK ###
def find_relics_check(params: Dict) -> bool:
    if Global.ALL_RELICS_FOUND:
        return False

    ship: Ship = params['ship']
    space: Space = params['space']
    target: Node = params['target']

    return check_target_is_reachable(
        ship=ship,
        space=space,
        target=target
    )


### RUN ###
def find_relics_run(params: Dict) -> ActionType:
    ship: Ship = params['ship']
    space: Space = params['space']
    target: Node = params['target']

    return go_to_target(
        ship=ship,
        space=space,
        target=target
    )


### FIND_REWARDS ###

### CHECK ###
def find_rewards_check(params: Dict) -> bool:
    raise NotImplementedError()


### RUN ###
def find_rewards_run(params: Dict) -> ActionType:
    raise NotImplementedError()


### HARVEST ###

### CHECK ###
def harvest_check(params: Dict) -> bool:
    ship: Ship = params['ship']
    space: Space = params['space']
    target: Node = params['target']

    if not target.reward:
        return False

    return check_target_is_reachable(
        ship=ship,
        space=space,
        target=target
    )


### RUN ###
def harvest_run(params: Dict) -> ActionType:
    raise NotImplementedError()


### ATTACK ###

### CHECK ###
def attack_check(params: Dict) -> bool:
    raise NotImplementedError()


### RUN ###
def attack_run(params: Dict) -> ActionType:
    raise NotImplementedError()