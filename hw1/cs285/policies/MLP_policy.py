import abc
import itertools
from typing import Any
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch import distributions
import torchinfo

from cs285.infrastructure import pytorch_util as ptu
from cs285.policies.base_policy import BasePolicy


class MLPPolicy(BasePolicy, nn.Module, metaclass=abc.ABCMeta):

    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
                 discrete=False,
                 learning_rate=1e-4,
                 training=True,
                 nn_baseline=False,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        # init vars
        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.n_layers = n_layers
        self.discrete = discrete
        self.size = size
        self.learning_rate = learning_rate
        self.training = training
        self.nn_baseline = nn_baseline

        if self.discrete:
            self.logits_na = ptu.build_mlp(
                input_size=self.ob_dim,
                output_size=self.ac_dim,
                n_layers=self.n_layers,
                size=self.size,
            )
            self.logits_na.to(ptu.device)
            self.mean_net = None
            self.logstd = None
            self.optimizer = optim.Adam(self.logits_na.parameters(),
                                        self.learning_rate)
        else:
            self.logits_na = None
            self.mean_net = ptu.build_mlp(
                input_size=self.ob_dim,
                output_size=self.ac_dim,
                n_layers=self.n_layers, size=self.size,
            )
            self.mean_net.to(ptu.device)
            self.logstd = nn.Parameter(
                torch.zeros(self.ac_dim, dtype=torch.float32, device=ptu.device)
            )
            self.logstd.to(ptu.device)
            self.optimizer = optim.Adam(
                itertools.chain([self.logstd], self.mean_net.parameters()),
                self.learning_rate
            )
        
        print("=== Model Stats ===")
        print(f"Using {'discrete' if self.discrete else 'continuous'} MLP")
        print(f"Using input sz {self.ob_dim}, output sz {self.ac_dim}, n layers {self.n_layers}, hidden sz {self.size}")
        col_names = ("input_size", "output_size", "num_params")#, "kernel_size", "mult_adds",)
        if self.discrete:
            torchinfo.summary(self.logits_na, (1, self.ob_dim), col_names=col_names, col_width=18)
        else:
            torchinfo.summary(self.mean_net, (1, self.ob_dim), col_names=col_names, col_width=18)

    ##################################

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    ##################################

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        if len(obs.shape) > 1:
            observation = obs
        else:
            observation = obs[None]

        # TODO return the action that the policy prescribes
        observation_t = torch.tensor(observation, dtype=torch.float32)
        if self.discrete:
            policy_logits_t = self.forward(observations_t)
            action_t = torch.max(policy_logits_t, 1)
        else:
            policy_dists_t = self.forward(observation_t)
            action_t = policy_dists_t.sample()
        return action_t.cpu().detach().numpy()

    # update/train this policy
    def update(self, observations, actions, **kwargs):
        raise NotImplementedError

    # This function defines the forward pass of the network.
    # You can return anything you want, but you should be able to differentiate
    # through it. For example, you can return a torch.FloatTensor. You can also
    # return more flexible objects, such as a
    # `torch.distributions.Distribution` object. It's up to you!
    def forward(self, observation: torch.FloatTensor) -> Any:
        observation = observation
        if self.discrete:
            return self.logits_na(observation)
        else:
            means = self.mean_net(observation)
            return torch.distributions.normal.Normal(means, torch.exp(self.logstd))


#####################################################
#####################################################

class MLPPolicySL(MLPPolicy):
    def __init__(self, ac_dim, ob_dim, n_layers, size, **kwargs):
        super().__init__(ac_dim, ob_dim, n_layers, size, **kwargs)
        # self.loss = nn.MSELoss()
        if self.discrete:
            self.loss = nn.CrossEntropyLoss()
        else:
            self.loss = nn.MSELoss()

    def update(
            self, observations, actions,
            adv_n=None, acs_labels_na=None, qvals=None
    ):
        # TODO: update the policy and return the loss
        observations_t = torch.tensor(observations, dtype=torch.float32)
        expert_actions_t = torch.tensor(actions, dtype=torch.float32)
        loss = None
        if self.discrete:
            policy_logits_t = self.forward(observations_t)
            loss = self.loss(policy_logits_t, expert_actions_t)
        else:
            policy_dists_t = self.forward(observations_t)
            policy_actions_t = policy_dists_t.sample()
            loss = self.loss(policy_actions_t, expert_actions_t)
        return {
            # You can add extra logging information here, but keep this line
            'Training Loss': ptu.to_numpy(loss),
        }
