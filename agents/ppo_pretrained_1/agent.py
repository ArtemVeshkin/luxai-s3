from state.state import State
from ppo_agent import PPOAgent

class Agent:
    def __init__(self, player: str, env_cfg) -> None:
        self.state = State(player, env_cfg)
        self.agent = PPOAgent()

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        self.state.update(obs)
        return self.agent.act(self.state)