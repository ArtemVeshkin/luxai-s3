from state.state import State
from rulebased import Rulebased
import pickle
from log_states import try_get_luxai_root_path
from pathlib import Path
from copy import deepcopy
from sys import stderr


ACTIONS_INVERSE = {
    0: 0,
    1: 3,
    2: 4,
    3: 1,
    4: 2,
    5: 5
}


class Agent:
    def __init__(self, player: str, env_cfg) -> None:
        self.state = State(player, env_cfg)
        self.agent = Rulebased()

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        self.state.update(obs)
        actions = self.agent.act(self.state)

        team_id = self.state.team_id
        log_actions = deepcopy(actions)
        if team_id == 1:
            for i in range(log_actions.shape[0]):
                log_actions[i, 0] = ACTIONS_INVERSE[log_actions[i, 0]]

        step = self.state.step
        LUXAI_ROOT_PATH: Path = try_get_luxai_root_path()
        with open(LUXAI_ROOT_PATH / f'state_logs/processed_states/{team_id}_{step}.pickle', 'wb') as fh:
            pickle.dump({
                'state': self.state.to_dict(),
                'actions': log_actions
            }, fh, protocol=pickle.HIGHEST_PROTOCOL)
        
        return actions