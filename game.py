"""Snake environment and Pygame renderer."""
from __future__ import annotations

import math
import random
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import pygame

import config

# Direction vectors: 0=up, 1=right, 2=down, 3=left
DIRS: Tuple[Tuple[int, int], ...] = ((0, -1), (1, 0), (0, 1), (-1, 0))


# ─────────────────────────────────────────────────────────────────────────
#  Environment
# ─────────────────────────────────────────────────────────────────────────
class SnakeGame:
    """Snake environment exposing reset / step / get_state and rich info dicts."""

    def __init__(
        self,
        width: int = config.BOARD_WIDTH,
        height: int = config.BOARD_HEIGHT,
        seed: Optional[int] = None,
    ) -> None:
        self.width = width
        self.height = height
        self._rng = random.Random(seed)
        self.snake: Deque[Tuple[int, int]] = deque()
        self.direction = 1
        self.food: Tuple[int, int] = (-1, -1)
        self.steps_since_food = 0
        self.total_steps = 0
        self.score = 0
        self.done = False
        self.reset()

    def reset(self) -> np.ndarray:
        cx, cy = self.width // 2, self.height // 2
        self.snake = deque([(cx, cy), (cx - 1, cy), (cx - 2, cy)])
        self.direction = 1
        self.food = self._spawn_food()
        self.steps_since_food = 0
        self.total_steps = 0
        self.score = 0
        self.done = False
        return self.get_state()

    def _spawn_food(self) -> Tuple[int, int]:
        occupied = set(self.snake)
        free = [
            (x, y)
            for x in range(self.width)
            for y in range(self.height)
            if (x, y) not in occupied
        ]
        if not free:
            return (-1, -1)
        return self._rng.choice(free)

    @staticmethod
    def _distance(a: Tuple[int, int], b: Tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        assert action in (0, 1, 2), f"invalid action {action}"

        if action == 1:
            self.direction = (self.direction + 1) % 4
        elif action == 2:
            self.direction = (self.direction - 1) % 4

        dx, dy = DIRS[self.direction]
        hx, hy = self.snake[0]
        new_head = (hx + dx, hy + dy)

        body = set(self.snake)
        will_eat = new_head == self.food
        if not will_eat:
            body.discard(self.snake[-1])

        collision = (
            new_head[0] < 0
            or new_head[0] >= self.width
            or new_head[1] < 0
            or new_head[1] >= self.height
            or new_head in body
        )

        prev_dist = self._distance(self.snake[0], self.food) if self.food[0] >= 0 else 0
        info: Dict[str, Any] = {
            "food_eaten": False, "collision": False,
            "distance_to_food": prev_dist, "length": len(self.snake),
            "score": self.score, "food_reward": 0.0, "survival_reward": 0.0,
            "distance_reward": 0.0, "collision_penalty": 0.0, "total_reward": 0.0,
        }

        if collision:
            self.done = True
            info["collision"] = True
            info["collision_penalty"] = config.COLLISION_PENALTY
            info["total_reward"] = config.COLLISION_PENALTY
            return self.get_state(), config.COLLISION_PENALTY, True, info

        self.snake.appendleft(new_head)
        food_reward = 0.0
        distance_reward = 0.0

        if will_eat:
            self.score += 1
            self.steps_since_food = 0
            food_reward = config.FOOD_REWARD
            info["food_eaten"] = True
            self.food = self._spawn_food()
            if self.food[0] < 0:
                self.done = True
                total = food_reward + config.SURVIVAL_REWARD
                info["food_reward"] = food_reward
                info["survival_reward"] = config.SURVIVAL_REWARD
                info["total_reward"] = total
                return self.get_state(), total, True, info
        else:
            self.snake.pop()
            self.steps_since_food += 1
            new_dist = self._distance(self.snake[0], self.food)
            if new_dist < prev_dist:
                distance_reward = config.DISTANCE_REWARD_SCALE
            elif new_dist > prev_dist:
                distance_reward = -config.DISTANCE_REWARD_SCALE

        self.total_steps += 1
        survival_reward = config.SURVIVAL_REWARD
        total_reward = food_reward + survival_reward + distance_reward

        info["food_reward"] = food_reward
        info["survival_reward"] = survival_reward
        info["distance_reward"] = distance_reward
        info["total_reward"] = total_reward
        info["length"] = len(self.snake)
        info["distance_to_food"] = (
            self._distance(self.snake[0], self.food) if self.food[0] >= 0 else 0
        )

        if self.steps_since_food > config.MAX_STEPS_NO_FOOD:
            self.done = True

        return self.get_state(), total_reward, self.done, info

    def get_state(self) -> np.ndarray:
        """Egocentric 5×5 grid + 5 scalars (80 features total)."""
        hx, hy = self.snake[0]
        fwd = DIRS[self.direction]
        right = DIRS[(self.direction + 1) % 4]
        body = set(self.snake)

        out: List[float] = []
        for ry in range(-config.VISION, config.VISION + 1):
            for rx in range(-config.VISION, config.VISION + 1):
                cx = hx + fwd[0] * ry + right[0] * rx
                cy = hy + fwd[1] * ry + right[1] * rx
                if not (0 <= cx < self.width and 0 <= cy < self.height):
                    out.extend((1.0, 0.0, 0.0))
                else:
                    is_body = 1.0 if (cx, cy) in body else 0.0
                    is_food = 1.0 if self.food == (cx, cy) else 0.0
                    out.extend((0.0, is_body, is_food))

        if self.food[0] >= 0:
            fx, fy = self.food
            d = abs(hx - fx) + abs(hy - fy)
            dist_norm = d / (self.width + self.height)
            dx, dy = fx - hx, fy - hy
            fx_fwd = dx * fwd[0] + dy * fwd[1]
            fx_rgt = dx * right[0] + dy * right[1]
            norm = max(1e-6, (fx_fwd * fx_fwd + fx_rgt * fx_rgt) ** 0.5)
            sin_a = fx_rgt / norm
            cos_a = fx_fwd / norm
        else:
            dist_norm = 1.0
            sin_a = cos_a = 0.0

        len_norm = len(self.snake) / (self.width * self.height)
        hunger = min(1.0, self.steps_since_food / max(1, config.MAX_STEPS_NO_FOOD))

        out.extend((dist_norm, sin_a, cos_a, len_norm, hunger))
        return np.array(out, dtype=np.float32)

    @property
    def state_size(self) -> int:
        return (2 * config.VISION + 1) ** 2 * 3 + 5


# ─────────────────────────────────────────────────────────────────────────
#  Renderer
# ─────────────────────────────────────────────────────────────────────────
class SnakeRenderer:
    """Polished board renderer: smooth snake body, eyes, pulsing food, glow."""

    BG_TOP      = (18, 20, 30)
    BG_BOTTOM   = (10, 11, 17)
    GRID_COLOR  = (30, 33, 48)
    BORDER_COL  = (60, 66, 92)

    SNAKE_HEAD  = (130, 250, 220)
    SNAKE_MID   = (70, 200, 180)
    SNAKE_TAIL  = (40, 130, 120)
    SNAKE_GLOW  = (110, 240, 210)

    FOOD_CORE   = (255, 130, 170)
    FOOD_HILITE = (255, 210, 230)
    FOOD_GLOW   = (240, 90, 130)

    def __init__(
        self,
        x: int, y: int, width: int, height: int, cell: int,
    ) -> None:
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.cell = cell
        self._bg_cache: Optional[pygame.Surface] = None
        self._grid_cache: Optional[pygame.Surface] = None
        self.rect = pygame.Rect(
            x, y, width * cell, height * cell,
        )

    # ── cached backgrounds ───────────────────────────────────────────
    def _bg(self) -> pygame.Surface:
        if self._bg_cache is None:
            surf = pygame.Surface(self.rect.size)
            for yy in range(self.rect.height):
                t = yy / self.rect.height
                c = (
                    int(self.BG_TOP[0] + (self.BG_BOTTOM[0] - self.BG_TOP[0]) * t),
                    int(self.BG_TOP[1] + (self.BG_BOTTOM[1] - self.BG_TOP[1]) * t),
                    int(self.BG_TOP[2] + (self.BG_BOTTOM[2] - self.BG_TOP[2]) * t),
                )
                pygame.draw.line(surf, c, (0, yy), (self.rect.width, yy))
            self._bg_cache = surf
        return self._bg_cache

    def _grid(self) -> pygame.Surface:
        if self._grid_cache is None:
            surf = pygame.Surface(self.rect.size, pygame.SRCALPHA)
            for gx in range(self.width + 1):
                px = gx * self.cell
                pygame.draw.line(
                    surf, self.GRID_COLOR,
                    (px, 0), (px, self.rect.height), 1,
                )
            for gy in range(self.height + 1):
                py = gy * self.cell
                pygame.draw.line(
                    surf, self.GRID_COLOR,
                    (0, py), (self.rect.width, py), 1,
                )
            self._grid_cache = surf
        return self._grid_cache

    # ── public draw ──────────────────────────────────────────────────
    def draw(
        self,
        surface: pygame.Surface,
        game: SnakeGame,
        tick: float = 0.0,
        paused: bool = False,
    ) -> None:
        surface.blit(self._bg(), self.rect.topleft)
        surface.blit(self._grid(), self.rect.topleft)
        self._draw_food(surface, game, tick)
        self._draw_snake(surface, game, tick)
        self._draw_border(surface)

        if paused:
            overlay = pygame.Surface(self.rect.size, pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            surface.blit(overlay, self.rect.topleft)

    def _draw_border(self, surface: pygame.Surface) -> None:
        pygame.draw.rect(surface, self.BORDER_COL, self.rect, 2, border_radius=8)

    # ── snake ────────────────────────────────────────────────────────
    def _draw_snake(
        self, surface: pygame.Surface, game: SnakeGame, tick: float,
    ) -> None:
        cells = list(game.snake)
        if not cells:
            return

        cell = self.cell
        cx = self.rect.x
        cy = self.rect.y

        centers = [
            (cx + gx * cell + cell // 2, cy + gy * cell + cell // 2)
            for (gx, gy) in cells
        ]
        n = len(centers)
        head_r = (cell - 6) // 2
        tail_r = max(5, int(head_r * 0.55))

        # Body segments (drawn back-to-front so head sits on top)
        for i in range(n - 1, 0, -1):
            t = i / max(1, n - 1)
            color = self._seg_color(t)
            r = int(head_r + (tail_r - head_r) * t)
            pygame.draw.line(
                surface, color, centers[i], centers[i - 1], r * 2,
            )
            pygame.draw.circle(surface, color, centers[i], r)

        # Head glow
        head = centers[0]
        glow_r = head_r + 10
        glow = pygame.Surface((glow_r * 2, glow_r * 2), pygame.SRCALPHA)
        for gr in range(glow_r, head_r, -1):
            a = int(60 * (1 - (gr - head_r) / (glow_r - head_r)))
            pygame.draw.circle(
                glow, (*self.SNAKE_GLOW, a), (glow_r, glow_r), gr,
            )
        surface.blit(glow, (head[0] - glow_r, head[1] - glow_r))

        # Head base
        pygame.draw.circle(surface, self.SNAKE_HEAD, head, head_r)

        # Eyes
        dx, dy = DIRS[game.direction]
        rx, ry = -dy, dx
        eye_dist = max(3, head_r // 2)
        eye_fwd = max(2, head_r // 3)
        pupil_r = max(2, head_r // 4)

        for side in (-1, 1):
            ex = int(head[0] + dx * eye_fwd + rx * side * eye_dist)
            ey = int(head[1] + dy * eye_fwd + ry * side * eye_dist)
            pygame.draw.circle(surface, (14, 16, 24), (ex, ey), pupil_r + 2)
            pygame.draw.circle(surface, (240, 250, 255), (ex, ey), pupil_r)
            pygame.draw.circle(
                surface, (10, 12, 20),
                (ex + dx, ey + dy), max(1, pupil_r // 2),
            )

    def _seg_color(self, t: float):
        if t < 0.5:
            return self._lerp_color(self.SNAKE_HEAD, self.SNAKE_MID, t * 2)
        return self._lerp_color(self.SNAKE_MID, self.SNAKE_TAIL, (t - 0.5) * 2)

    @staticmethod
    def _lerp_color(a, b, t):
        return (
            int(a[0] + (b[0] - a[0]) * t),
            int(a[1] + (b[1] - a[1]) * t),
            int(a[2] + (b[2] - a[2]) * t),
        )

    # ── food ─────────────────────────────────────────────────────────
    def _draw_food(
        self, surface: pygame.Surface, game: SnakeGame, tick: float,
    ) -> None:
        if game.food[0] < 0:
            return
        cell = self.cell
        fx, fy = game.food
        hx = self.rect.x + fx * cell + cell // 2
        hy = self.rect.y + fy * cell + cell // 2

        base_r = max(3, cell // 3)
        pulse = 1.0 + 0.12 * math.sin(tick * 5.0)
        r = int(base_r * pulse)

        glow_r = r + 12
        glow = pygame.Surface((glow_r * 2, glow_r * 2), pygame.SRCALPHA)
        for gr in range(glow_r, 0, -1):
            a = int(80 * (1 - gr / glow_r))
            pygame.draw.circle(
                glow, (*self.FOOD_GLOW, a), (glow_r, glow_r), gr,
            )
        surface.blit(glow, (hx - glow_r, hy - glow_r))

        pygame.draw.circle(surface, self.FOOD_CORE, (hx, hy), r)
        pygame.draw.circle(
            surface, self.FOOD_HILITE,
            (hx - r // 3, hy - r // 3), max(1, r // 3),
        )