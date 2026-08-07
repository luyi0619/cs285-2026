import itertools
from torch import nn
from torch.nn import functional as F
import torch.distributions as D
from torch import optim

import numpy as np
import torch
from torch import distributions

from infrastructure import pytorch_util as ptu


class MLPPolicy(nn.Module):
    """Base MLP policy, which can take an observation and output a distribution over actions.

    This class should implement the `forward` and `get_action` methods. The `update` method should be written in the
    subclasses, since the policy update rule differs for different algorithms.
    """

    def __init__(
        self,
        ac_dim: int,
        ob_dim: int,
        discrete: bool,
        n_layers: int,
        layer_size: int,
        learning_rate: float,
    ):
        super().__init__()

        if discrete:
            self.logits_net = ptu.build_mlp(
                input_size=ob_dim,
                output_size=ac_dim,
                n_layers=n_layers,
                size=layer_size,
            ).to(ptu.device)
            parameters = self.logits_net.parameters()
        else:
            self.mean_net = ptu.build_mlp(
                input_size=ob_dim,
                output_size=ac_dim,
                n_layers=n_layers,
                size=layer_size,
            ).to(ptu.device)
            self.logstd = nn.Parameter(
                torch.zeros(ac_dim, dtype=torch.float32, device=ptu.device)
            )
            parameters = itertools.chain([self.logstd], self.mean_net.parameters())

        self.optimizer = optim.Adam(
            parameters,
            learning_rate,
        )

        self.discrete = discrete

    @torch.no_grad()
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """Takes a single observation (as a numpy array) and returns a single action (as a numpy array)."""
        # TODO: implement get_action
        
        # 1. Convert numpy array observation to a torch tensor
        obs_tensor = ptu.from_numpy(obs)

        # Ensure obs has a batch dimension (e.g., shape (1, ob_dim) if passed a single vector)
        if len(obs_tensor.shape) == 1:
            obs_tensor = obs_tensor.unsqueeze(0)

        dist = self.forward(obs_tensor)
        action_tensor = dist.sample()

        action = ptu.to_numpy(action_tensor).squeeze(0)
        return action

    def forward(self, obs: torch.FloatTensor):
        """
        This function defines the forward pass of the network.  You can return anything you want, but you should be
        able to differentiate through it. For example, you can return a torch.FloatTensor. You can also return more
        flexible objects, such as a `torch.distributions.Distribution` object. It's up to you!
        """
        if self.discrete:
            logits = self.logits_net(obs)
            return torch.distributions.Categorical(logits=logits)
        else:
            mean = self.mean_net(obs)
            std = torch.exp(self.logstd)
            return torch.distributions.Normal(mean, std)

    def update(self, obs: np.ndarray, actions: np.ndarray, *args, **kwargs) -> dict:
        """
        Performs one iteration of gradient descent on the provided batch of data. You don't need to implement this
        method in the base class, but you do need to implement it in the subclass.
        """
        raise NotImplementedError


class MLPPolicyPG(MLPPolicy):
    """Policy subclass for the policy gradient algorithm."""

    def update(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        advantages: np.ndarray,
    ) -> dict:
        """Implements the policy gradient actor update."""
        obs = ptu.from_numpy(obs) # [B, D]
        actions = ptu.from_numpy(actions) # [B]
        advantages = ptu.from_numpy(advantages) # [B]

        if self.discrete:
            actions = actions.long()

        # TODO: compute the policy gradient actor loss
        dist = self.forward(obs) 
        log_prob = dist.log_prob(actions) # [B]
        assert self.discrete
        loss = -(log_prob * advantages).mean()

        # TODO: perform an optimizer step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "Actor Loss": loss.item(),
        }
