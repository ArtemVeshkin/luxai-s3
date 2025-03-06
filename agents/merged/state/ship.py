from .node import Node
from .base import Config
from .action_type import ActionType


class Ship:
    def __init__(self, unit_id: int):
        self.unit_id = unit_id
        self.energy = 0
        self.energies = []
        self.predicted_energies = []
        self.track = []
        self.node: Node | None = None

        self.task: str | None = None
        self.target: Node | None = None
        self.action: ActionType | None = None
        self.action_sap_info: tuple = (0, 0)
        self.observe_where_steps: bool = True
        self.projected_energy = None
        self.expected_node_type = None
        self.prev_node_type = None
        self.prev_node_energy = None
        self.active = False
        self.expected_is_enemy = False

    def __repr__(self):
        return (
            f"Ship({self.unit_id}, node={self.node.coordinates}, energy={self.energy})"
        )

    @property
    def coordinates(self):
        return self.node.coordinates if self.node else None
    
    def next_coords(self):
        ship_move_direction = self.action.to_direction()
        x, y = self.node.coordinates
        return x + ship_move_direction[0], y + ship_move_direction[1]

    def clean(self):
        self.energy = 0
        self.node = None
        self.task = None
        self.target = None
        self.action = None
        self.energies = []
        self.predicted_energies = []
        self.track = []
        self.action_sap_info: tuple = (0, 0)
        self.observe_where_steps = True
        self.projected_energy = None
        self.expected_is_enemy = False
        self.active = False
        self.prev_node_energy = None
        self.prev_node_type = None
        self.expected_node_type = None


    def can_sap(self, node: Node, config: Config):
        if self.node:
            coords = self.coordinates
            if abs(node.coordinates[0] - coords[0]) <= config.UNIT_SAP_RANGE and abs(
                    node.coordinates[1] - coords[1]) <= config.UNIT_SAP_RANGE:
                return True
            else:
                return False
        else:
            return False