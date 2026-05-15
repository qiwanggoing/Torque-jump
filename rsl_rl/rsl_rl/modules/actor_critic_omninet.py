# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch
import torch.nn as nn
from torch.distributions import Normal

from .actor_critic import ActorCritic, get_activation


class ActorCriticOmniNet(ActorCritic):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=None,
        critic_hidden_dims=None,
        estimator_hidden_dims=None,
        estimator_target_dim=10,
        single_obs_dim=46,
        history_length=20,
        activation="elu",
        estimator_activation="relu",
        init_noise_std=1.0,
        estimator_loss_coef=0.5,
        **kwargs,
    ):
        nn.Module.__init__(self)
        if kwargs:
            print(
                "ActorCriticOmniNet.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )

        actor_hidden_dims = actor_hidden_dims or [512, 256, 128]
        critic_hidden_dims = critic_hidden_dims or [512, 256, 128]
        estimator_hidden_dims = estimator_hidden_dims or [258, 128]

        self.single_obs_dim = single_obs_dim
        self.history_length = history_length
        self.estimator_target_dim = estimator_target_dim
        self.estimator_loss_coef = estimator_loss_coef

        actor_activation = get_activation(activation)
        estimator_act = get_activation(estimator_activation)

        estimator_layers = []
        estimator_input_dim = num_actor_obs
        prev_dim = estimator_input_dim
        for hidden_dim in estimator_hidden_dims:
            estimator_layers.append(nn.Linear(prev_dim, hidden_dim))
            estimator_layers.append(estimator_act)
            prev_dim = hidden_dim
        estimator_layers.append(nn.Linear(prev_dim, estimator_target_dim))
        self.estimator = nn.Sequential(*estimator_layers)

        actor_input_dim = single_obs_dim + estimator_target_dim
        actor_layers = [nn.Linear(actor_input_dim, actor_hidden_dims[0]), actor_activation]
        for layer_idx in range(len(actor_hidden_dims)):
            if layer_idx == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_idx], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_idx], actor_hidden_dims[layer_idx + 1]))
                actor_layers.append(actor_activation)
        self.actor = nn.Sequential(*actor_layers)

        critic_layers = [nn.Linear(num_critic_obs, critic_hidden_dims[0]), actor_activation]
        for layer_idx in range(len(critic_hidden_dims)):
            if layer_idx == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_idx], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_idx], critic_hidden_dims[layer_idx + 1]))
                critic_layers.append(actor_activation)
        self.critic = nn.Sequential(*critic_layers)

        print(f"Estimator MLP: {self.estimator}")
        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        Normal.set_default_validate_args = False

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    def _get_current_single_obs(self, observations):
        return observations[:, -self.single_obs_dim :]

    def _predict_estimates(self, observations):
        return self.estimator(observations)

    def _build_actor_input(self, observations):
        current_single_obs = self._get_current_single_obs(observations)
        estimator_pred = self._predict_estimates(observations)
        return torch.cat((estimator_pred, current_single_obs), dim=-1)

    def update_distribution(self, observations):
        actor_input = self._build_actor_input(observations)
        mean = self.actor(actor_input)
        self.distribution = Normal(mean, mean * 0.0 + self.std)

    def act_inference(self, observations):
        actor_input = self._build_actor_input(observations)
        return self.actor(actor_input)

    def evaluate(self, critic_observations, **kwargs):
        return self.critic(critic_observations)

    def compute_auxiliary_loss(self, observations, critic_observations):
        pred = self._predict_estimates(observations)
        target = critic_observations[:, : self.estimator_target_dim]
        estimator_loss = torch.mean(torch.square(pred - target))
        total_aux_loss = self.estimator_loss_coef * estimator_loss
        return total_aux_loss, {"estimator_loss": estimator_loss.detach()}
