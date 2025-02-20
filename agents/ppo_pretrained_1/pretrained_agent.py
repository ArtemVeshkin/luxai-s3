from state.state import State
from ppo_policy import ActorNet
import numpy as np
from sys import stderr
import torch


class PretrainedAgent:
    def __init__(self):
        input_channles = 21
        self.model = ActorNet({
            'input_channels': input_channles,
            'n_res_blocks': 8,
            'all_channel': input_channles * 2,
            'n_actions': 5
        })
        self.model.load_state_dict(torch.load(
            './pretrained.pt'
        , weights_only=True))
        self.model.eval()


    def act(self, state: State):
        obs = torch.Tensor(state.get_obs()).unsqueeze(0)
        model_out = self.model(obs).reshape(-1, 16, 5).detach().numpy()
        model_actions = np.argmax(model_out, axis=2)[0]
        action = np.array([[a, 0, 0] for a in model_actions], dtype=np.int8)
        return action