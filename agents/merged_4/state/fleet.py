from .ship import Ship
from .base import (
    Config,
    MAX_UNITS
)
from .space import Space
from .node import NodeType


class Fleet:
    def __init__(self, team_id):
        self.team_id: int = team_id
        self.points: int = 0  # how many points have we scored in this match so far
        self.ships = [Ship(unit_id) for unit_id in range(MAX_UNITS)]

    def __repr__(self):
        return f"Fleet({self.team_id})"

    def __iter__(self):
        for ship in self.ships:
            if ship.node is not None:
                yield ship

    def clear(self):
        self.points = 0
        for ship in self.ships:
            ship.clean()

    def update(self, obs, space: Space, config: Config):
        self.points = int(obs["team_points"][self.team_id])

        for ship, active, position, energy in zip(
            self.ships,
            obs["units_mask"][self.team_id],
            obs["units"]["position"][self.team_id],
            obs["units"]["energy"][self.team_id],
        ):
            ship.active = active
            if active:
                ship.node = space.get_node(*position)
                ship.track.append(position)
                ship.action_sap_info = (0, 0)
                if ship.node.type == NodeType.nebula and not config.NEBULA_ENERGY_REDUCTION_FOUND:
                    if ship.node.energy is not None:
                        nebula_energy_reduction = ship.energy + ship.node.energy - config.UNIT_MOVE_COST - int(energy)
                        if nebula_energy_reduction in (0, 1, 2, 3, 5, 25):
                            config.NEBULA_ENERGY_REDUCTION_FOUND = True
                            config.NEBULA_ENERGY_REDUCTION = nebula_energy_reduction
                ship.energy = int(energy)
                ship.action = None
            else:
                ship.clean()