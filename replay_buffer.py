"""Prioritized Experience Replay (proportional, SumTree-free).

Uses a numpy priorities array + np.random.choice for sampling. Fast
enough for Snake-scale training; swap in a SumTree if you push past
~500k transitions.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


class PrioritizedReplayBuffer:
    """Fixed-capacity circular buffer with proportional prioritization."""

    def __init__(
        self,
        capacity: int,
        state_size: int,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_steps: int = 200_000,
        eps: float = 1e-6,
        seed: Optional[int] = None,
    ) -> None:
        self.capacity = int(capacity)
        self.state_size = int(state_size)
        self.alpha = float(alpha)
        self.beta_start = float(beta_start)
        self.beta_steps = int(beta_steps)
        self.eps = float(eps)

        self.states = np.zeros((self.capacity, self.state_size), dtype=np.float32)
        self.next_states = np.zeros((self.capacity, self.state_size), dtype=np.float32)
        self.actions = np.zeros(self.capacity, dtype=np.int64)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.dones = np.zeros(self.capacity, dtype=np.float32)
        self.priorities = np.zeros(self.capacity, dtype=np.float32)

        self.ptr = 0
        self.size = 0
        self.max_priority = 1.0
        self._samples_drawn = 0
        self._rng = np.random.default_rng(seed)

    # ── mutation ─────────────────────────────────────────────────────────
    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        i = self.ptr
        self.states[i] = state
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_states[i] = next_state
        self.dones[i] = 1.0 if done else 0.0
        self.priorities[i] = self.max_priority
        self.ptr = (self.ptr + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        p = (np.abs(td_errors) + self.eps) ** self.alpha
        self.priorities[indices] = p
        self.max_priority = max(self.max_priority, float(p.max()))

    # ── sampling ─────────────────────────────────────────────────────────
    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """Return (idx, states, actions, rewards, next_states, dones, weights)."""
        if self.size < batch_size:
            raise ValueError(
                f"Cannot sample {batch_size} from buffer of size {self.size}"
            )
        prios = self.priorities[: self.size].astype(np.float64)
        total = prios.sum()
        if total <= 0.0:
            prios = np.ones(self.size, dtype=np.float64) / self.size
        else:
            prios = prios / total

        idx = self._rng.choice(self.size, size=batch_size, p=prios)

        beta = min(
            1.0,
            self.beta_start
            + (1.0 - self.beta_start) * (self._samples_drawn / max(1, self.beta_steps)),
        )
        self._samples_drawn += batch_size

        probs = prios[idx]
        weights = (self.size * probs) ** (-beta)
        weights = weights / weights.max()
        weights = weights.astype(np.float32)

        return (
            idx,
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_states[idx],
            self.dones[idx],
            weights,
        )

    # ── introspection ────────────────────────────────────────────────────
    def __len__(self) -> int:
        return self.size

    def can_sample(self, batch_size: int) -> bool:
        return self.size >= batch_size

    def mean_priority(self) -> float:
        if self.size == 0:
            return 0.0
        return float(self.priorities[: self.size].mean())