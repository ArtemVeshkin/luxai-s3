from state.state import State
from ppo_agent import PPOAgent
from pretrained_agent import PretrainedAgent

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
        # self.agent = PPOAgent()
        self.agent = PretrainedAgent()

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        self.state.update(obs)
        actions = self.agent.act(self.state)
        if self.state.team_id == 1:
            for i in range(actions.shape[0]):
                actions[i, 0] = ACTIONS_INVERSE[actions[i, 0]]
        
        return actions