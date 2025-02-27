from state.state import State
from ppo_policy import ActorNet
import numpy as np
from sys import stderr
import torch


class PretrainedAgent:
    def __init__(self):
        self.model = ActorNet({
            'input_channels': 25,
            'n_res_blocks': 8,
            'all_channel': 48,
            'n_actions': 5,
            'num_features_count': 18,
            'cat_features_count': 14,
            'emb_dim': 8,
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