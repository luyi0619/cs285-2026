from typing import Optional
import torch
from torch import nn
import numpy as np
import infrastructure.pytorch_util as ptu

from typing import Callable, Optional, Sequence, Tuple, List


class FQLAgent(nn.Module):
    def __init__(
        self,
        observation_shape: Sequence[int],
        action_dim: int,

        make_bc_actor,
        make_bc_actor_optimizer,
        make_onestep_actor,
        make_onestep_actor_optimizer,
        make_critic,
        make_critic_optimizer,

        discount: float,
        target_update_rate: float,
        flow_steps: int,
        alpha: float,
    ):
        super().__init__()

        self.action_dim = action_dim

        self.bc_actor = make_bc_actor(observation_shape, action_dim)
        self.onestep_actor = make_onestep_actor(observation_shape, action_dim)
        self.critic = make_critic(observation_shape, action_dim)
        self.target_critic = make_critic(observation_shape, action_dim)
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.bc_actor_optimizer = make_bc_actor_optimizer(self.bc_actor.parameters())
        self.onestep_actor_optimizer = make_onestep_actor_optimizer(self.onestep_actor.parameters())
        self.critic_optimizer = make_critic_optimizer(self.critic.parameters())

        self.discount = discount
        self.target_update_rate = target_update_rate
        self.flow_steps = flow_steps
        self.alpha = alpha

    def get_action(self, observation: np.ndarray):
        """
        Used for evaluation.
        """
        observation = ptu.from_numpy(np.asarray(observation))[None]
        # TODO(student): Compute the action for evaluation
        # Hint: Unlike SAC+BC and IQL, the evaluation action is *sampled* (i.e., not the mode or mean) from the policy
        z = torch.randn(observation.shape[0], self.action_dim, device=observation.device, dtype=observation.dtype)        action = self.onestep_actor(observation, z)
        action = torch.clamp(action, -1, 1)
        return ptu.to_numpy(action)[0]

    @torch.compile
    def get_bc_action(self, observation: torch.Tensor, noise: torch.Tensor):
        """
        Used for training.
        """
        # TODO(student): Compute the BC flow action using the Euler method for `self.flow_steps` steps
        # Hint: This function should *only* be used in `update_onestep_actor`
        batch_size = observation.shape[0]
        action = noise
        step_size = 1.0 / self.flow_steps
        for i in range(self.flow_steps):
            t = torch.full(
                (batch_size, 1),
                i * step_size,
                device=observation.device,
                dtype=observation.dtype,
            )            
            action = action + step_size * self.bc_actor(observation, action, t)
        action = torch.clamp(action, -1, 1)
        return action

    @torch.compile
    def update_q(
        self,
        observations: torch.Tensor, # [B, O] 
        actions: torch.Tensor, # [B, A]
        rewards: torch.Tensor, # [B]
        next_observations: torch.Tensor, # [B, O]
        dones: torch.Tensor, # [B]
    ) -> dict:
        """
        Update Q(s, a)
        """
        # TODO(student): Compute the Q loss
        # Hint: Use the one-step actor to compute next actions
        # Hint: Remember to clamp the actions to be in [-1, 1] when feeding them to the critic!
        with torch.no_grad():
            # [B, A]
            z = torch.randn(next_observations.shape[0], self.action_dim, device=next_observations.device, dtype=next_observations.dtype) 
            # [B, A]
            one_step_actions = self.onestep_actor(next_observations, z)
            one_step_actions = torch.clamp(one_step_actions, -1, 1)
            # [B]
            q = rewards + self.discount * (1-dones.float()) * self.target_critic(next_observations, one_step_actions).mean(dim=0)
        
        q_pred = self.critic(observations, actions) # [2, B]

        loss = ((q_pred - q[None, :]) ** 2).mean()

        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()

        return {
            "q_loss": loss,
            "q_mean": q.mean(),
            "q_max": q.max(),
            "q_min": q.min(),
        }

    @torch.compile
    def update_bc_actor(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ):
        """
        Update the BC actor
        """
        # TODO(student): Compute the BC flow loss
        # [B, A]
        z = torch.randn(observations.shape[0], self.action_dim, device=observations.device, dtype=observations.dtype) 
        # [B]
        t = torch.rand(z.shape[0], 1, device=z.device, dtype=z.dtype)
        bc_actions = (1 - t) * z + t * actions


        loss = ((self.bc_actor(observations, bc_actions, t) - (actions - z)) ** 2).mean()

        self.bc_actor_optimizer.zero_grad()
        loss.backward()
        self.bc_actor_optimizer.step()

        return {
            "loss": loss,
        }

    @torch.compile
    def update_onestep_actor(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ):
        """
        Update the one-step actor
        """
        # TODO(student): Compute the one-step actor loss
        # Hint: Do *not* clip the one-step actor actions when computing the distillation loss
        z = torch.randn(observations.shape[0], self.action_dim, device=observations.device, dtype=observations.dtype) 
        with torch.no_grad():
            # [B, A]
            bc_actions = self.get_bc_action(observations, z)
        # [B, A]
        one_step_actions = self.onestep_actor(observations, z)

        distill_loss = self.alpha * ((one_step_actions - bc_actions) ** 2).mean()

        # Hint: *Do* clip the one-step actor actions when feeding them to the critic
        clamped_one_step_actions = torch.clamp(one_step_actions, -1, 1)
        
        q_loss = - self.critic(observations, clamped_one_step_actions).mean()

        # Total loss.
        loss = distill_loss + q_loss

        # Additional metrics for logging.
        mse = ((one_step_actions - actions) ** 2).mean()

        self.onestep_actor_optimizer.zero_grad()
        loss.backward()
        self.onestep_actor_optimizer.step()

        return {
            "total_loss": loss,
            "distill_loss": distill_loss,
            "q_loss": q_loss,
            "mse": mse,
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
        metrics_q = self.update_q(observations, actions, rewards, next_observations, dones)
        metrics_bc_actor = self.update_bc_actor(observations, actions)
        metrics_onestep_actor = self.update_onestep_actor(observations, actions)
        metrics = {
            **{f"critic/{k}": v.item() for k, v in metrics_q.items()},
            **{f"bc_actor/{k}": v.item() for k, v in metrics_bc_actor.items()},
            **{f"onestep_actor/{k}": v.item() for k, v in metrics_onestep_actor.items()},
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
