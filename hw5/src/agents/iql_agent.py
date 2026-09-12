from typing import Optional
import torch
from torch import nn
import numpy as np
import infrastructure.pytorch_util as ptu

from typing import Callable, Optional, Sequence, Tuple, List


class IQLAgent(nn.Module):
    def __init__(
        self,
        observation_shape: Sequence[int],
        action_dim: int,

        make_actor,
        make_actor_optimizer,
        make_critic,
        make_critic_optimizer,
        make_value,
        make_value_optimizer,

        discount: float,
        target_update_rate: float,
        alpha: float,
        expectile: float,
    ):
        super().__init__()

        self.actor = make_actor(observation_shape, action_dim)
        self.critic = make_critic(observation_shape, action_dim)
        self.target_critic = make_critic(observation_shape, action_dim)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.value = make_value(observation_shape)

        self.actor_optimizer = make_actor_optimizer(self.actor.parameters())
        self.critic_optimizer = make_critic_optimizer(self.critic.parameters())
        self.value_optimizer = make_value_optimizer(self.value.parameters())

        self.discount = discount
        self.target_update_rate = target_update_rate
        self.alpha = alpha
        self.expectile = expectile

    def get_action(self, observation: np.ndarray):
        """
        Used for evaluation.
        """
        observation = ptu.from_numpy(np.asarray(observation))[None]
        action = self.actor(observation).mode  # Take the mean (mode) action
        action = torch.clamp(action, -1, 1)
        return ptu.to_numpy(action[0])

    @staticmethod
    def iql_expectile_loss(
        adv: torch.Tensor, expectile: float,
    ) -> torch.Tensor:
        """
        Compute the expectile loss for IQL
        """
        w = torch.where(adv > 0, expectile, 1 - expectile)
        return (w * (adv ** 2)).mean()

    @torch.compile
    def update_v(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ):
        """
        Update V(s) with expectile regression
        """
        # TODO(student): Compute the value loss
        with torch.no_grad():
            # [2, B]
            q = self.target_critic(observations, actions)
            # [B]
            min_q = q.min(dim=0).values

        # [B]
        v = self.value(observations)
        loss = self.iql_expectile_loss(min_q - v, self.expectile)

        self.value_optimizer.zero_grad()
        loss.backward()
        self.value_optimizer.step()

        return {
            "v_loss": loss,
            "v_mean": v.mean(),
            "v_max": v.max(),
            "v_min": v.min(),
        }

    @torch.compile
    def update_q(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
    ) -> dict:
        """
        Update Q(s, a)
        """
        # TODO(student): Compute the Q loss
        # [2, B]
        q = self.critic(observations, actions)
        # [B]
        with torch.no_grad():
            q_target = rewards + self.discount * (1 - dones) * self.value(next_observations)
        loss = ((q - q_target[None, :]) ** 2).mean()

        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()

        return {
            "q_loss": loss,
            "q_mean": q.mean(),
            "q_max": q.max(),
            "q_min": q.min(),
        }

    #@torch.compile
    def update_actor(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ):
        """
        Update the actor using advantage-weighted regression
        """
        # TODO(student): Compute the actor loss
        dist = self.actor(observations)
        with torch.no_grad():
            # [B]
            adv = self.critic(observations, actions).min(dim=0).values - self.value(observations)
            exp_adv = torch.clamp(torch.exp(self.alpha * adv), max=100.0)

        log_prob = dist.log_prob(actions)
        loss = - (exp_adv * log_prob).mean()

        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()

        return {
            "actor_loss": loss,
            "mse": torch.mean((dist.mode - actions) ** 2),
        }

    def update(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
        step: int,
    ):
        metrics_v = self.update_v(observations, actions)
        metrics_q = self.update_q(observations, actions, rewards, next_observations, dones)
        metrics_actor = self.update_actor(observations, actions)
        metrics = {
            **{f"value/{k}": v.item() for k, v in metrics_v.items()},
            **{f"critic/{k}": v.item() for k, v in metrics_q.items()},
            **{f"actor/{k}": v.item() for k, v in metrics_actor.items()},
        }

        self.update_target_critic()

        return metrics

    def update_target_critic(self) -> None:
        # TODO(student): Update target_critic using Polyak averaging with self.target_update_rate
        with torch.no_grad():
            # target <- target + tau * (critic - target) 
            #        <- (1 - tau) * target + tau * critic

            for target_param, critic_param in zip(
                self.target_critic.parameters(),
                self.critic.parameters(),
            ):
                target_param.mul_(1 - self.target_update_rate)
                target_param.add_(self.target_update_rate * critic_param)
