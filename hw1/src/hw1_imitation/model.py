"""Model definitions for Push-T imitation policies."""

from __future__ import annotations

import abc
from typing import Literal, TypeAlias

import torch
from torch import nn


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def compute_loss(
        self, state: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        """Compute training loss for a batch."""

    @abc.abstractmethod
    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,  # only applicable for flow policy
    ) -> torch.Tensor:
        """Generate a chunk of actions with shape (batch, chunk_size, action_dim)."""


class MSEPolicy(BasePolicy):
    """Predicts action chunks with an MSE loss."""

    ### TODO: IMPLEMENT MSEPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)

        # Target output dimension for chunked action prediction
        output_dim = action_dim * chunk_size

        layers: list[nn.Module] = []
        in_dim = state_dim

        # Dynamically build hidden layers
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim

        # Final projection to action chunk space
        layers.append(nn.Linear(in_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        actions = self.net(x)
        return actions.view(-1, self.chunk_size, self.action_dim)

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        pred = self.forward(state)
        return torch.mean((pred - action_chunk) ** 2)

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        y = self.forward(state)
        return y


class FlowMatchingPolicy(BasePolicy):
    """Predicts action chunks with a flow matching loss."""

    ### TODO: IMPLEMENT FlowMatchingPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)

        # Target output dimension for chunked action prediction
        output_dim = action_dim * chunk_size

        layers: list[nn.Module] = []
        in_dim = state_dim + action_dim * chunk_size + 1

        # Dynamically build hidden layers
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim

        # Final projection to action chunk space
        layers.append(nn.Linear(in_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        actions = self.net(x)
        return actions.view(-1, self.chunk_size, self.action_dim)


    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:

        batch_size, chunk_size, action_dim = action_chunk.shape
        A_t_0 = torch.randn(
            (batch_size, chunk_size, action_dim), 
            device=action_chunk.device,
            dtype=action_chunk.dtype,
        )

        # 1. Sample tau uniformly in [0, 1] per batch element: shape (batch_size, 1, 1)
        tau = torch.rand(
            (batch_size, 1, 1), 
            device=action_chunk.device, 
            dtype=action_chunk.dtype,
        )

        # 2. Linear interpolation / Flow Matching step:
        A_t_i = tau * action_chunk + (1 - tau) * A_t_0

        # [B, chunk_size * action_dim]
        A_t_i_flat = A_t_i.view(-1, self.chunk_size * self.action_dim)

        # tau as shape [B, 1]
        tau_flat = tau.squeeze(1)

        # Concatenate: [B, 5] + [B, 16] + [B, 1] -> [B, 22]
        network_input = torch.cat([state, A_t_i_flat, tau_flat], dim=-1)
        # [B, chunk_size * action_dim]
        pred = self.forward(network_input)
        return torch.mean((pred - (action_chunk - A_t_0)) ** 2)

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:

        batch_size = state.shape[0]
        dt = 1.0 / num_steps

        x = torch.randn(
            (state.shape[0], self.chunk_size, self.action_dim), 
            device=state.device,
            dtype=state.dtype,
        )

        for i in range(num_steps):
            tau_val = i * dt

            tau_flat = torch.full(
                (batch_size, 1),
                fill_value=tau_val,
                device=state.device,
                dtype=state.dtype,
            )

            x_flat = x.view(batch_size, -1)
            # Concatenate: [B, 5] + [B, 16] + [B, 1] -> [B, 22]
            network_input = torch.cat([state, x_flat, tau_flat], dim=-1)
            pred = self.forward(network_input)
            # Euler step: x_{tau + dt} = x_{tau} + v * dt
            x = x + pred * dt
        
        return x


PolicyType: TypeAlias = Literal["mse", "flow"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    hidden_dims: tuple[int, ...] = (128, 128),
) -> BasePolicy:
    if policy_type == "mse":
        return MSEPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    if policy_type == "flow":
        return FlowMatchingPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    raise ValueError(f"Unknown policy type: {policy_type}")
