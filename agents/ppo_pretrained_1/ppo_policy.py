from typing import Callable, Tuple

from gymnasium import spaces
import gym
import torch
from torch import nn
import torch.nn.functional as F
from sys import stderr
from state.base import MAX_UNITS, SPACE_SIZE

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3 import PPO
from copy import deepcopy


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=5, padding=2):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=1, padding=padding, bias=False),
            nn.LeakyReLU(),
            nn.Conv2d(out_channel, out_channel, kernel_size=kernel_size, stride=1, padding=padding, bias=False),
            nn.LeakyReLU(),
        )
        # self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=1, padding=padding, bias=False)
        # self.leaky_relu = nn.LeakyReLU()
        self.shortcut = nn.Sequential()
        self.selayer = SELayer(out_channel)
        self._init_w_b()

    def forward(self, x):
        out = self.left(x)
        out = self.selayer(out)
        out += self.shortcut(x)
        out = F.leaky_relu(out)
        return out

    def _init_w_b(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

def _init_w_b(layers):
    # for layer in layers:
    #     nn.init.kaiming_normal_(layer.weight)
    #     #nn.init.zeros_(layer.bias)
    for layer in layers:
        if isinstance(layer, nn.Conv2d):
            nn.init.kaiming_normal_(layer.weight)
        if isinstance(layer, nn.Linear):
            nn.init.xavier_normal_(layer.weight)


class IdentityFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 0) -> None:
        super().__init__(observation_space=observation_space, features_dim=1)
        self._features_dim = 0

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return observations


class ActorNet(nn.Module):
    def __init__(self, model_params):
        super().__init__()

        n_res_blocks = model_params["n_res_blocks"]
        all_channel = model_params["all_channel"]
        n_actions = model_params['n_actions']
        input_channels = model_params['input_channels']
        self.num_features_count = model_params['num_features_count']
        self.ohe_features_count = model_params['ohe_features_count']
        emb_dim = model_params['emb_dim']

        self.input_channels = input_channels

        self.embedding_layer = nn.Linear(self.ohe_features_count, emb_dim)
        self.conv1 = nn.Conv2d(emb_dim, emb_dim, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(self.num_features_count, self.num_features_count, kernel_size=1, stride=1, padding=0, bias=False)
        global_channels = emb_dim + self.num_features_count
        self.conv3 = nn.Conv2d(global_channels, global_channels, kernel_size=1, stride=1, padding=0, bias=False)

        self.input_conv1 = nn.Conv2d(input_channels + global_channels, all_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.res_blocks = nn.ModuleList([
            ResidualBlock(all_channel, all_channel) for _ in range(n_res_blocks)
        ])
        self.spectral_norm = nn.utils.spectral_norm(nn.Conv2d(all_channel, all_channel, kernel_size=1, stride=1, padding=0, bias=False))

        self.actions_conv = nn.Conv2d(all_channel, n_actions, kernel_size=1, stride=1, padding=0, bias=False)

        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.kaiming_normal_(self.conv3.weight)
        nn.init.kaiming_normal_(self.actions_conv.weight)
        nn.init.kaiming_normal_(self.input_conv1.weight)


    def get_x_and_ships_mask(self, x: torch.Tensor):
        if x.shape[1] == self.input_channels:
            ships_mask = torch.zeros(
                (x.shape[0], MAX_UNITS, SPACE_SIZE, SPACE_SIZE)
            , device=x.device)
            return x, ships_mask

        ships_mask = x[:, :MAX_UNITS, :, :]
        ships_mask = ships_mask.flatten(start_dim=2,end_dim=3)
        ships_mask = ships_mask.transpose(1, 2)
        return x[:, MAX_UNITS:, :, :], ships_mask


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, ships_mask = self.get_x_and_ships_mask(x)

        x = self.get_embed(x)
        x = self.get_actions_from_embed(x, ships_mask)
        return x

    def get_global_info_embed(self, x: torch.Tensor) -> torch.Tensor:
        num_features = x[:, 0, :, :].flatten(1)[:, :self.num_features_count]
        ohe_features = x[:, 1, :, :].flatten(1)[:, :self.ohe_features_count]

        ohe_features = self.embedding_layer(ohe_features)
        ohe_features = ohe_features.view(-1, ohe_features.shape[1], 1, 1).expand(-1, ohe_features.shape[1], SPACE_SIZE, SPACE_SIZE)
        ohe_features = self.conv1(ohe_features)
        ohe_features = F.leaky_relu(ohe_features)

        num_features = num_features.view(-1, num_features.shape[1], 1, 1).expand(-1, num_features.shape[1], SPACE_SIZE, SPACE_SIZE)
        num_features = self.conv2(num_features)
        num_features = F.leaky_relu(num_features)

        global_features = torch.cat([ohe_features, num_features], dim=1)
        global_features = self.conv3(global_features)
        return global_features



    def get_embed(self, x: torch.Tensor) -> torch.Tensor:
        global_info = x[:, :2, :, :]
        map_features = x[:, 2:, :, :]

        global_info_embed = self.get_global_info_embed(global_info)

        x = torch.cat([global_info_embed, map_features], dim=1)
        x = self.input_conv1(x)
        for block in self.res_blocks:
            x = block(x)
        x = self.spectral_norm(x)
        return x
    
    def get_actions_from_embed(self, x: torch.Tensor, ships_mask) -> torch.Tensor:
        actions_map = self.actions_conv(x)
        actions_map = actions_map.flatten(start_dim=2,end_dim=3)
        actions_map = actions_map.transpose(1,2)

        result_actions_map = torch.cat([
            (actions_map * ships_mask[:, :, idx:idx+1]).sum(dim=1) for idx in range(MAX_UNITS)
        ], dim=1)

        force_center_mask = torch.tile(torch.Tensor([1000., 0., 0., 0., 0.,]),
                    (result_actions_map.shape[0], MAX_UNITS)).to(torch.device(result_actions_map.device))
        ships_mask = 1. - ships_mask.sum(dim=1, keepdim=True)
        for ship_idx in range(MAX_UNITS):
            force_center_mask[:, ship_idx * 5] *= ships_mask[:, 0, ship_idx]
        result_actions_map += force_center_mask
            

        return result_actions_map

class ActorCriticNet(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the features extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(self, model_params):
        super().__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = model_params['n_actions'] * 16
        self.latent_dim_vf = 1

        self.actor_net = ActorNet(model_params)
        if model_params['acton_net_path'] is not None:
            self.actor_net.load_state_dict(torch.load(
                model_params['acton_net_path']
            , weights_only=True))
        self.all_channel = model_params["all_channel"]
        self.critic_fc = nn.Linear(self.all_channel, 1)
        nn.init.xavier_normal_(self.critic_fc.weight)


    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        x, ships_mask = self.actor_net.get_x_and_ships_mask(x)
        x = self.actor_net.get_embed(x)
        fleet_actions = self.actor_net.get_actions_from_embed(x, ships_mask)

        critic_x = torch.flatten(x, start_dim=-2, end_dim=-1).sum(dim=-1) / (24 * 24)
        critic_value = self.critic_fc(critic_x.view(-1, self.all_channel)).view(-1)

        return fleet_actions, critic_value


    def forward_actor(self, x: torch.Tensor) -> torch.Tensor:
        return self.actor_net(x)


    def forward_critic(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.actor_net.get_x_and_ships_mask(x)
        x = self.actor_net.get_embed(x)
        critic_x = torch.flatten(x, start_dim=-2, end_dim=-1).sum(dim=-1) / (24 * 24)
        critic_value = self.critic_fc(critic_x.view(-1, self.all_channel)).view(-1)
        return critic_value



class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        **kwargs,
    ):
        # Disable orthogonal initialization
        kwargs["ortho_init"] = False
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            # Pass remaining arguments to base class
            features_extractor_class=IdentityFeatureExtractor,
            *args,
            **kwargs,
        )
        self.action_net = nn.Identity()
        self.value_net = nn.Identity()


    def _build_mlp_extractor(self) -> None:
        model_param = {
            'input_channels': 22,
            'n_res_blocks': 8,
            'all_channel': 64,
            'n_actions': 5,
            'num_features_count': 18,
            'ohe_features_count': 49,
            'emb_dim': 9,
            'acton_net_path': './pretrained.pt'
            # 'acton_net_path': None
        }
        self.mlp_extractor = ActorCriticNet(model_param)
