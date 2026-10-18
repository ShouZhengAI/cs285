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
        layers: list[nn.Module] = []
        in_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.extend((nn.Linear(in_dim, hidden_dim), nn.ReLU()))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, chunk_size * action_dim))
        self.network = nn.Sequential(*layers)

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        prediction = self.network(state).view(-1, self.chunk_size, self.action_dim)
        return nn.functional.mse_loss(prediction, action_chunk)

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        prediction = self.network(state)
        return prediction.view(-1, self.chunk_size, self.action_dim)


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
        layers: list[nn.Module] = []
        in_dim = state_dim + 1 + chunk_size * action_dim
        for hidden_dim in hidden_dims:
            layers.extend((nn.Linear(in_dim, hidden_dim), nn.ReLU()))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, chunk_size * action_dim))
        self.network = nn.Sequential(*layers)

    def _velocity(
        self, state: torch.Tensor, action: torch.Tensor, time: torch.Tensor
    ) -> torch.Tensor:
        batch_size = state.shape[0]
        features = torch.cat((state, time.reshape(batch_size, 1), action.flatten(1)), dim=-1)
        return self.network(features).view(batch_size, self.chunk_size, self.action_dim)

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = state.shape[0]
        target = action_chunk
        noise = torch.randn_like(target)
        time = torch.rand(batch_size, device=state.device, dtype=state.dtype)
        t = time.view(batch_size, 1, 1)
        interpolated = (1.0 - t) * noise + t * target
        target_velocity = target - noise
        velocity = self._velocity(state, interpolated, time)
        return nn.functional.mse_loss(velocity, target_velocity)

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        batch_size = state.shape[0]
        action = torch.randn(
            batch_size, self.chunk_size, self.action_dim,
            device=state.device, dtype=state.dtype,
        )
        steps = max(1, num_steps)
        dt = 1.0 / steps
        for index in range(steps):
            time = torch.full(
                (batch_size,), index / steps, device=state.device, dtype=state.dtype
            )
            action = action + dt * self._velocity(state, action, time)
        return action


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
