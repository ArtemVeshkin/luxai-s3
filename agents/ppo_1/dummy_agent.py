from state.state import State
import numpy as np


class DummyAgent:
    def __init__(self):
        pass

    def act(self, state: State):
        return np.zeros((16, 3), dtype=np.uint8)