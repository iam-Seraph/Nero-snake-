"""Dueling Deep Q-Network used by the Snake agent."""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class DuelingDQN(nn.Module):
    """Shared trunk → value head V(s) + advantage head A(s,a).

    Q(s,a) = V(s) + A(s,a) - mean_a' A(s,a')
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_sizes: Sequence[int] = (256, 256),
    ) -> None:
        super().__init__()
        trunk: list[nn.Module] = []
        prev = state_size
        for h in hidden_sizes:
            trunk.append(nn.Linear(prev, h))
            trunk.append(nn.ReLU(inplace=True))
            prev = h
        self.feature = nn.Sequential(*trunk)

        self.value = nn.Sequential(
            nn.Linear(prev, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )
        self.advantage = nn.Sequential(
            nn.Linear(prev, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, action_size),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.feature(x)
        v = self.value(f)                                  # (B, 1)
        a = self.advantage(f)                              # (B, A)
        return v + (a - a.mean(dim=1, keepdim=True))       # (B, A)