from state.state import State
from ppo_policy import ActorNet
import numpy as np
from sys import stderr
import torch


class PretrainedAgent:
    def __init__(self):
        self.model = ActorNet({
            'input_channels': 44,
            'n_res_blocks': 2,
            'all_channel': 88,
            'n_actions': 5
        })
        self.model.load_state_dict(torch.load(
            '/home/artemveshkin/dev/luxai-s3/agents/ppo_pretrained_1/pretrained.pt'
        , weights_only=True))
        self.model.eval()


    def act(self, state: State):
        obs = torch.Tensor(state.get_obs()).unsqueeze(0)
        model_out = self.model(obs)[0]
        action = torch.argmax(model_out.reshape((5, 16)), dim=0)
        action = np.array([[a, 0, 0] for a in action], dtype=np.int8)
        return action