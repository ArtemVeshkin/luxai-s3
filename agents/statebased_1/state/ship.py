from .node import Node
from .action_type import ActionType


class Ship:
    def __init__(self, unit_id: int):
        self.unit_id = unit_id
        self.energy = 0
        self.node: Node | None = None

        self.task: str | None = None
        self.target: Node | None = None
        self.sap_direction = None
        self.action: ActionType | None = None

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