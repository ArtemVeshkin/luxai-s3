from enum import IntEnum
from collections import Counter


# Game related constants:
SPACE_SIZE = 24
MAX_STEPS_IN_MATCH = 100
MAX_UNITS = 16
RELIC_REWARD_RANGE = 2
MAX_ENERGY_PER_TILE = 20
MAX_RELIC_NODES = 6
LAST_MATCH_STEP_WHEN_RELIC_CAN_APPEAR = 50
LAST_MATCH_WHEN_RELIC_CAN_APPEAR = 2

# Others:

# The energy on the unknown tiles will be used in the pathfinding
HIDDEN_NODE_ENERGY = 0


class Config:
    def __init__(self):
        # We will find the exact value of these constants during the game
        self.UNIT_MOVE_COST = 3  # OPTIONS: list(range(1, 6))
        self.UNIT_SAP_COST = int((50 + 30) / 2)  # OPTIONS: list(range(30, 51))
        self.UNIT_SAP_RANGE = 3  # OPTIONS: list(range(3, 8))
        self.UNIT_SENSOR_RANGE = 2  # OPTIONS: [1, 2, 3, 4]
        self.OBSTACLE_MOVEMENT_PERIOD = 20  # OPTIONS: 6.67, 10, 20, 40
        self.ENERGY_NODE_MOVEMENT_PERIOD = 20  # OPTIONS: 100, 50, 33.3, 25, 20
        self.OBSTACLE_MOVEMENT_DIRECTION = (0, 0)  # OPTIONS: [(1, -1), (-1, 1)]
        self.NEBULA_ENERGY_REDUCTION_SAMPLES = Counter()
        self.UNIT_SAP_DROPOFF_FACTOR = (0.25 + 0.5 + 1) / 3

        # We will NOT find the exact value of these constants during the game
        self.NEBULA_ENERGY_REDUCTION = 5  # OPTIONS: [0, 1, 2, 3, 5, 25]

        # Exploration flags:

        self.ALL_RELICS_FOUND = False
        self.ALL_REWARDS_FOUND = False
        self.OBSTACLE_MOVEMENT_PERIOD_FOUND = False
        self.OBSTACLE_MOVEMENT_DIRECTION_FOUND = False
        self.NEBULA_ENERGY_REDUCTION_FOUND = False
        self.ENERGY_NODE_MOVEMENT_PERIOD_FOUND = False

        # Game logs:

        # REWARD_RESULTS: [{"nodes": Set[Node], "points": int}, ...]
        # A history of reward events, where each entry contains:
        # - "nodes": A set of nodes where our ships were located.
        # - "points": The number of points scored at that location.
        # This data will help identify which nodes yield points.
        self.REWARD_RESULTS = []
        self.ENERGY_NODES_MOVEMENT_STATUS = []

        # obstacles_movement_status: list of bool
        # A history log of obstacle (asteroids and nebulae) movement events.
        # - `True`: The ships' sensors detected a change in the obstacles' positions at this step.
        # - `False`: The sensors did not detect any changes.
        # This information will be used to determine the speed and direction of obstacle movement.
        self.OBSTACLES_MOVEMENT_STATUS = []


def get_match_step(step: int) -> int:
    return step % (MAX_STEPS_IN_MATCH + 1)


def get_match_number(step: int) -> int:
    return step // (MAX_STEPS_IN_MATCH + 1)


def warp_int(x):
    if x >= SPACE_SIZE:
        x -= SPACE_SIZE
    elif x < 0:
        x += SPACE_SIZE
    return x


def warp_point(x, y) -> tuple:
    return warp_int(x), warp_int(y)


def get_opposite(x, y) -> tuple:
    # Returns the mirrored point across the diagonal
    return SPACE_SIZE - y - 1, SPACE_SIZE - x - 1


def is_upper_sector(x, y) -> bool:
    return SPACE_SIZE - x - 1 >= y


def is_lower_sector(x, y) -> bool:
    return SPACE_SIZE - x - 1 <= y


def is_team_sector(team_id, x, y) -> bool:
    return is_upper_sector(x, y) if team_id == 0 else is_lower_sector(x, y)
