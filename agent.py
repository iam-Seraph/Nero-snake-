"""Double Dueling DQN agent with PER, N-step returns, and Polyak targets."""
from __future__ import annotations

import logging
import os
import random
from collections import deque
from pathlib import Path
from typing import Deque, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import config
from model import DuelingDQN
from replay_buffer import PrioritizedReplayBuffer


class _NStepAccumulator:
    """Holds the last N (s, a, r, s', done) tuples and emits N-step returns."""

    def __init__(self, n: int, gamma: float) -> None:
        self.n = n
        self.gamma = gamma
        self.buf: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=n)

    def push(
        self,
        s: np.ndarray,
        a: int,
        r: float,
        ns: np.ndarray,
        done: bool,
    ) -> Optional[Tuple[np.ndarray, int, float, np.ndarray, bool]]:
        """Push 1-step; return an N-step transition once the window is full."""
        self.buf.append((s, a, r, ns, done))
        if len(self.buf) < self.n:
            return None

        s0, a0 = self.buf[0][0], self.buf[0][1]
        G = 0.0
        for i, (_, _, ri, _, di) in enumerate(self.buf):
            G += (self.gamma ** i) * ri
            if di:
                # Episode ended early — truncate the return here.
                return s0, a0, G, self.buf[i][3], True
        return s0, a0, G, self.buf[-1][3], False

    def flush(self) -> List[Tuple[np.ndarray, int, float, np.ndarray, bool]]:
        """Emit remaining (shorter) N-step transitions at episode end."""
        out: List[Tuple[np.ndarray, int, float, np.ndarray, bool]] = []
        buf = list(self.buf)
        for j in range(len(buf)):
            s0, a0 = buf[j][0], buf[j][1]
            G = 0.0
            terminal = False
            last_ns = buf[j][3]
            for i in range(j, len(buf)):
                _, _, ri, nsi, di = buf[i]
                G += (self.gamma ** (i - j)) * ri
                last_ns = nsi
                if di:
                    terminal = True
                    break
            out.append((s0, a0, G, last_ns, terminal))
        self.buf.clear()
        return out


class DQNAgent:
    """Double Dueling DQN with prioritized replay and N-step returns."""

    def __init__(
        self,
        state_size: int,
        action_size: int,
        device: torch.device,
        learning_rate: float = config.LEARNING_RATE,
        gamma: float = config.GAMMA,
        hidden_sizes: Sequence[int] = config.HIDDEN_SIZES,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.gamma = gamma
        self.logger = logger or logging.getLogger("snake_ai.agent")

        self.online = DuelingDQN(state_size, action_size, hidden_sizes).to(device)
        self.target = DuelingDQN(state_size, action_size, hidden_sizes).to(device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        for p in self.target.parameters():
            p.requires_grad_(False)

        self.optimizer = optim.Adam(self.online.parameters(), lr=learning_rate)
        self.loss_fn = nn.SmoothL1Loss(reduction="none")

        self.train_steps = 0
        self.episode = 0
        self.best_score = 0
        self.epsilon = config.EPSILON_START
        self.reward_history: List[float] = []

        self.nstep = _NStepAccumulator(config.N_STEP, gamma)

    # ── action selection ─────────────────────────────────────────────────
    def select_action(
        self,
        state: np.ndarray,
        epsilon: Optional[float] = None,
    ) -> int:
        """Epsilon-greedy action. 0=straight, 1=right, 2=left."""
        eps = self.epsilon if epsilon is None else epsilon
        if random.random() < eps:
            return random.randrange(self.action_size)
        with torch.no_grad():
            t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q = self.online(t)
            return int(q.argmax(dim=1).item())

    def q_values(self, state: np.ndarray) -> List[float]:
        """Expose Q-values for UI."""
        with torch.no_grad():
            t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            return self.online(t).squeeze(0).tolist()

    # ── observation pipeline ─────────────────────────────────────────────
    def observe(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        buffer: PrioritizedReplayBuffer,
    ) -> None:
        """Feed a 1-step transition; internally emits N-step transitions."""
        nstep = self.nstep.push(state, action, reward, next_state, done)
        if nstep is not None:
            buffer.push(*nstep)
        if done:
            for t in self.nstep.flush():
                buffer.push(*t)

    # ── training ─────────────────────────────────────────────────────────
    def train_step(
        self,
        buffer: PrioritizedReplayBuffer,
        batch_size: int,
    ) -> Optional[float]:
        if not buffer.can_sample(max(batch_size, config.MIN_REPLAY_TO_TRAIN)):
            return None

        idx, states, actions, rewards, next_states, dones, weights = buffer.sample(batch_size)

        s = torch.as_tensor(states, device=self.device)
        a = torch.as_tensor(actions, device=self.device, dtype=torch.long)
        r = torch.as_tensor(rewards, device=self.device)
        ns = torch.as_tensor(next_states, device=self.device)
        d = torch.as_tensor(dones, device=self.device)
        w = torch.as_tensor(weights, device=self.device)

        q_values = self.online(s).gather(1, a.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # Double DQN: online picks argmax action, target evaluates it.
            next_actions = self.online(ns).argmax(dim=1, keepdim=True)
            next_q = self.target(ns).gather(1, next_actions).squeeze(1)
            targets = r + (self.gamma ** config.N_STEP) * next_q * (1.0 - d)

        td_errors = (q_values - targets).detach().cpu().numpy()
        per_sample = self.loss_fn(q_values, targets) * w
        loss = per_sample.mean()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), config.GRAD_CLIP)
        self.optimizer.step()

        buffer.update_priorities(idx, td_errors)

        # Polyak averaging — smoother than periodic hard sync.
        if config.TAU > 0:
            with torch.no_grad():
                for tp, op in zip(self.target.parameters(), self.online.parameters()):
                    tp.data.mul_(1.0 - config.TAU).add_(op.data, alpha=config.TAU)
                for tb, ob in zip(self.target.buffers(), self.online.buffers()):
                    tb.data.copy_(ob.data)
        elif config.HARD_TARGET_SYNC > 0:
            self.train_steps += 1
            if self.train_steps % config.HARD_TARGET_SYNC == 0:
                self.target.load_state_dict(self.online.state_dict())

        return float(loss.item())

    # ── epsilon schedule ─────────────────────────────────────────────────
    def update_epsilon(self) -> None:
        if config.EPSILON_MODE == "linear":
            frac = min(1.0, self.episode / max(1, config.EPSILON_LINEAR_EPISODES))
            self.epsilon = (
                config.EPSILON_START
                + frac * (config.EPSILON_MIN - config.EPSILON_START)
            )
        else:
            self.epsilon = max(config.EPSILON_MIN, self.epsilon * config.EPSILON_DECAY)

    # ── persistence ──────────────────────────────────────────────────────
    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "online": self.online.state_dict(),
            "target": self.target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "episode": self.episode,
            "best_score": self.best_score,
            "train_steps": self.train_steps,
            "reward_history": self.reward_history[-500:],
            "state_size": self.state_size,
            "action_size": self.action_size,
            "n_step": config.N_STEP,
            "tau": config.TAU,
        }
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, tmp)
        os.replace(tmp, path)

    def load(self, path: Path) -> bool:
        """Load a checkpoint. Returns True on success; never raises on corruption."""
        path = Path(path)
        if not path.exists():
            return False
        try:
            payload = torch.load(path, map_location=self.device, weights_only=False)
            if (
                payload.get("state_size") != self.state_size
                or payload.get("action_size") != self.action_size
            ):
                self.logger.warning("Checkpoint architecture mismatch — starting fresh.")
                return False
            self.online.load_state_dict(payload["online"])
            self.target.load_state_dict(payload.get("target", payload["online"]))
            if "optimizer" in payload:
                try:
                    self.optimizer.load_state_dict(payload["optimizer"])
                except Exception:
                    self.logger.warning("Optimizer state skipped.")
            self.epsilon = float(payload.get("epsilon", config.EPSILON_START))
            self.episode = int(payload.get("episode", 0))
            self.best_score = int(payload.get("best_score", 0))
            self.train_steps = int(payload.get("train_steps", 0))
            self.reward_history = list(payload.get("reward_history", []))
            return True
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("Checkpoint load failed (%s) — starting fresh.", exc)
            return False

    def average_recent_reward(self, n: int = 100) -> float:
        hist = self.reward_history[-n:]
        return float(sum(hist) / len(hist)) if hist else 0.0