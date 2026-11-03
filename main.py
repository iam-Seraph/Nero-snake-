"""Entry point for the Neuro-Snake AI ecosystem."""
from __future__ import annotations

import sys
import time
from typing import Any, Dict, List, Optional

import pygame

import config
from agent import DQNAgent
from chatbot import ChatbotWorker
from game import DIRS, SnakeGame, SnakeRenderer
from replay_buffer import PrioritizedReplayBuffer
from ui import (
    ACCENT, ACCENT_DEEP, ACCENT_DIM, AMBER, BAD, BG_APP, BG_CARD, BG_FOOTER,
    BG_HEADER, BG_INPUT, BG_PANEL, BORDER, BORDER_SOFT, BORDER_STRONG, Button,
    ChatPanel, Fonts, GOOD, PINK, Sparkline, StatCard, TEXT, TEXT_DIM,
    TEXT_MUTED, VIOLET, draw_badge, draw_bar, draw_card,
)


# ─── Layout ───────────────────────────────────────────────────────────────
PAD = 20
HEADER_H = 60
FOOTER_H = 40

BOARD_PX_W = config.BOARD_WIDTH * config.CELL_SIZE
BOARD_PX_H = config.BOARD_HEIGHT * config.CELL_SIZE
BOARD_X = PAD
BOARD_Y = HEADER_H + PAD

RIGHT_X = BOARD_X + BOARD_PX_W + PAD
RIGHT_W = config.WINDOW_WIDTH - RIGHT_X - PAD
RIGHT_Y = BOARD_Y

STAT_GRID_H = 200
SPARK_H = 110
CONTROLS_H = 76
GAP = 12

STAT_GRID_Y = RIGHT_Y
SPARK_Y = STAT_GRID_Y + STAT_GRID_H + GAP
CONTROLS_Y = SPARK_Y + SPARK_H + GAP

CHAT_X = PAD
CHAT_Y = BOARD_Y + BOARD_PX_H + PAD
CHAT_W = config.WINDOW_WIDTH - 2 * PAD
CHAT_H = config.WINDOW_HEIGHT - CHAT_Y - FOOTER_H - PAD

FOOTER_Y = config.WINDOW_HEIGHT - FOOTER_H


# ─── Application ──────────────────────────────────────────────────────────
class GameApp:
    MENU = "menu"
    TRAINING = "training"
    WATCHING = "watching"
    HUMAN = "human"

    def __init__(self) -> None:
        self.logger = config.configure_logging()
        self.logger.info("Starting Neuro-Snake AI (v3 UI)")
        self.logger.info("Hardware: %s", config.hardware_diagnostics())

        pygame.init()
        pygame.display.set_caption("Neuro-Snake AI")
        self.screen = pygame.display.set_mode(
            (config.WINDOW_WIDTH, config.WINDOW_HEIGHT)
        )
        self.clock = pygame.time.Clock()
        self.fonts = Fonts()

        # Device
        self.device = config.detect_device()
        self.device_label = config.device_name(self.device)
        print(f"Device: {self.device_label}")

        # Agent + buffer
        self.agent = DQNAgent(
            state_size=config.STATE_SIZE,
            action_size=config.ACTION_SIZE,
            device=self.device,
            logger=self.logger,
        )
        self.buffer = PrioritizedReplayBuffer(
            capacity=config.REPLAY_CAPACITY,
            state_size=config.STATE_SIZE,
            alpha=config.PER_ALPHA,
            beta_start=config.PER_BETA_START,
            beta_steps=config.PER_BETA_STEPS,
            eps=config.PER_EPS,
        )
        self.loaded_checkpoint = self.agent.load(config.CHECKPOINT_FILE)

        # Game + renderer
        self.game = SnakeGame()
        self.state = self.game.reset()
        self.renderer = SnakeRenderer(
            BOARD_X, BOARD_Y,
            config.BOARD_WIDTH, config.BOARD_HEIGHT,
            config.CELL_SIZE,
        )

        # UI components
        self._build_stat_cards()
        self.sparkline = Sparkline(max_points=200, color=ACCENT)
        self.chat_panel = ChatPanel()
        self.chat_panel.place(
            pygame.Rect(CHAT_X, CHAT_Y, CHAT_W, CHAT_H)
        )
        self._build_buttons()

        # Chatbot
        self.chatbot = ChatbotWorker()
        self.chatbot.set_state_provider(self._chat_state)
        self.chatbot.set_comment_cooldown(config.CHAT_COMMENT_COOLDOWN)
        self.chatbot.start()
        self.chat_panel.add(
            "system",
            "Offline AI ready. Ask about status, learning, deaths, epsilon.",
        )
        if self.loaded_checkpoint:
            self.chat_panel.add(
                "system",
                f"Loaded checkpoint · ep {self.agent.episode} · "
                f"best {self.agent.best_score} · eps {self.agent.epsilon:.4f}",
            )
            self.chatbot.post_event(
                "model_loaded",
                {
                    "episode": self.agent.episode,
                    "best_score": self.agent.best_score,
                },
            )
        else:
            self.chat_panel.add("system", "No checkpoint on disk — starting fresh.")

        # Runtime state
        self.mode = self.MENU
        self.running = True
        self.episode_reward = 0.0
        self.last_loss = 0.0
        self.avg_reward = 0.0
        self.fps = 0.0
        self._accum_ai = 0.0
        self._accum_human = 0.0
        self._pending_action: Optional[int] = None
        self._tick = 0.0
        self._flash_until = 0.0
        self._flash_color = GOOD
        self._flash_message = ""

        # Menu items
        self.menu_items = [
            ("START TRAINING", "Train fast. Render at 60 fps."),
            ("WATCH AI", "Render every move at 30 steps/sec."),
            ("HUMAN PLAY", "Steer with arrows or WASD."),
            ("SAVE MODEL", "Write the current checkpoint."),
            ("LOAD MODEL", "Reload checkpoint from disk."),
            ("QUIT", "Exit the application."),
        ]
        self.menu_selected = 0

    # ─────────────────────────────────────────────────────────────────
    #  Construction helpers
    # ─────────────────────────────────────────────────────────────────
    def _build_stat_cards(self) -> None:
        self.stat_cards = {
            "episode": StatCard("Episode", accent=ACCENT),
            "best": StatCard("Best Score", accent=PINK),
            "length": StatCard("Length", accent=VIOLET),
            "epsilon": StatCard("Epsilon", accent=AMBER),
            "avg": StatCard("Avg Reward", accent=GOOD),
            "loss": StatCard("Loss", accent=BAD),
        }
        # Fixed grid: 3 columns × 2 rows
        cols, rows = 3, 2
        cell_w = (RIGHT_W - (cols - 1) * GAP) // cols
        cell_h = (STAT_GRID_H - (rows - 1) * GAP) // rows
        self._stat_rects: List[pygame.Rect] = []
        order = ["episode", "best", "length", "epsilon", "avg", "loss"]
        for idx, _ in enumerate(order):
            r, c = divmod(idx, cols)
            self._stat_rects.append(pygame.Rect(
                RIGHT_X + c * (cell_w + GAP),
                STAT_GRID_Y + r * (cell_h + GAP),
                cell_w, cell_h,
            ))
        self._stat_order = order

    def _build_buttons(self) -> None:
        labels = [
            ("TRAIN", "1", ACCENT),
            ("WATCH", "2", ACCENT),
            ("HUMAN", "3", VIOLET),
            ("SAVE", "F5", AMBER),
            ("MENU", "ESC", ACCENT_DIM),
        ]
        n = len(labels)
        btn_w = (RIGHT_W - (n - 1) * 8) // n
        self.buttons: List[Button] = []
        for i, (label, key, color) in enumerate(labels):
            b = Button(label, shortcut=key, accent=color)
            b.place(pygame.Rect(
                RIGHT_X + i * (btn_w + 8), CONTROLS_Y,
                btn_w, CONTROLS_H,
            ))
            self.buttons.append(b)

    # ─────────────────────────────────────────────────────────────────
    #  Modes
    # ─────────────────────────────────────────────────────────────────
    def _enter_mode(self, mode: str) -> None:
        self.mode = mode
        self.logger.info("Mode → %s", mode)
        if mode == self.TRAINING:
            self.chat_panel.add("system", "Training mode engaged.")
            self.chatbot.post_event("ai_control")
        elif mode == self.WATCHING:
            self.chat_panel.add("system", "Watching AI at 30 steps/sec.")
            self.chatbot.post_event("ai_control")
        elif mode == self.HUMAN:
            self.chat_panel.add("system", "Manual control. Arrow keys / WASD.")
            self.chatbot.post_event("human_control")

    # ─────────────────────────────────────────────────────────────────
    #  Event handling
    # ─────────────────────────────────────────────────────────────────
    def _handle_events(self) -> None:
        mouse_pos = pygame.mouse.get_pos()

        for b in self.buttons:
            b.update(mouse_pos)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._quit()
                return

            # Menu
            if self.mode == self.MENU:
                self._handle_menu_event(event)
                continue

            # Chat capture takes priority
            if self.chat_panel.active:
                self._handle_chat_key(event)
                continue

            # Buttons
            clicked = False
            for b in self.buttons:
                if b.handle_event(event):
                    clicked = True
                    self._activate_button(b)
                    break
            if clicked:
                continue

            if event.type == pygame.KEYDOWN:
                self._handle_game_key(event)

    def _handle_menu_event(self, event) -> None:
        if event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_UP, pygame.K_w):
                self.menu_selected = (self.menu_selected - 1) % len(self.menu_items)
            elif event.key in (pygame.K_DOWN, pygame.K_s):
                self.menu_selected = (self.menu_selected + 1) % len(self.menu_items)
            elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                self._activate_menu(self.menu_selected)
            elif event.key == pygame.K_ESCAPE:
                self._quit()
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            for i in range(len(self.menu_items)):
                if self._menu_rect(i).collidepoint(event.pos):
                    self.menu_selected = i
                    self._activate_menu(i)
                    break
        elif event.type == pygame.MOUSEMOTION:
            for i in range(len(self.menu_items)):
                if self._menu_rect(i).collidepoint(event.pos):
                    self.menu_selected = i
                    break

    def _handle_chat_key(self, event) -> None:
        if event.type != pygame.KEYDOWN:
            return
        if event.key == pygame.K_RETURN:
            text = self.chat_panel.input_text.strip()
            self.chat_panel.input_text = ""
            self.chat_panel.active = False
            if text:
                self.chatbot.post_user_message(text)
        elif event.key == pygame.K_ESCAPE:
            self.chat_panel.input_text = ""
            self.chat_panel.active = False
        elif event.key == pygame.K_BACKSPACE:
            self.chat_panel.input_text = self.chat_panel.input_text[:-1]
        else:
            ch = event.unicode
            if ch and ch.isprintable() and len(self.chat_panel.input_text) < 200:
                self.chat_panel.input_text += ch

    def _handle_game_key(self, event) -> None:
        if event.key == pygame.K_ESCAPE:
            self._enter_mode(self.MENU)
            return
        if event.key == pygame.K_RETURN:
            self.chat_panel.active = True
            self.chat_panel.input_text = ""
            return
        if event.key == pygame.K_F5:
            self._save_checkpoint()
            return
        if event.key == pygame.K_1:
            self._enter_mode(self.TRAINING); return
        if event.key == pygame.K_2:
            self._enter_mode(self.WATCHING); return
        if event.key == pygame.K_3:
            self._enter_mode(self.HUMAN); return

        if self.mode == self.HUMAN:
            action = self._key_to_action(event.key)
            if action is not None:
                self._pending_action = action

    def _key_to_action(self, key: int) -> Optional[int]:
        desired = {
            pygame.K_UP: (0, -1), pygame.K_w: (0, -1),
            pygame.K_RIGHT: (1, 0), pygame.K_d: (1, 0),
            pygame.K_DOWN: (0, 1), pygame.K_s: (0, 1),
            pygame.K_LEFT: (-1, 0), pygame.K_a: (-1, 0),
        }.get(key)
        if desired is None:
            return None
        cur = DIRS[self.game.direction]
        if desired == cur:
            return 0
        right = DIRS[(self.game.direction + 1) % 4]
        left = DIRS[(self.game.direction - 1) % 4]
        if desired == right:
            return 1
        if desired == left:
            return 2
        return None

    def _activate_button(self, button: Button) -> None:
        label = button.label
        if label == "TRAIN":
            self._enter_mode(self.TRAINING)
        elif label == "WATCH":
            self._enter_mode(self.WATCHING)
        elif label == "HUMAN":
            self._enter_mode(self.HUMAN)
        elif label == "SAVE":
            self._save_checkpoint()
        elif label == "MENU":
            self._enter_mode(self.MENU)

    def _activate_menu(self, idx: int) -> None:
        if idx == 0:
            self._enter_mode(self.TRAINING)
        elif idx == 1:
            self._enter_mode(self.WATCHING)
        elif idx == 2:
            self._enter_mode(self.HUMAN)
        elif idx == 3:
            self._save_checkpoint()
            self._flash("Checkpoint saved", GOOD)
        elif idx == 4:
            self._reload_checkpoint()
        elif idx == 5:
            self._quit()

    # ─────────────────────────────────────────────────────────────────
    #  Simulation stepping
    # ─────────────────────────────────────────────────────────────────
    def _env_steps(self) -> int:
        return self.agent.train_steps * config.TRAIN_EVERY_STEPS

    def _ai_step(self, train: bool) -> None:
        action = self.agent.select_action(self.state, self.agent.epsilon)
        next_state, reward, done, info = self.game.step(action)
        self.episode_reward += reward

        if train:
            self.agent.observe(
                self.state, action, reward, next_state, done, self.buffer
            )
            if self._env_steps() % config.TRAIN_EVERY_STEPS == 0:
                loss = self.agent.train_step(self.buffer, config.BATCH_SIZE)
                if loss is not None:
                    self.last_loss = loss

        self.state = next_state
        if done:
            self._end_episode(info)

    def _human_step(self) -> None:
        if self._pending_action is None:
            return
        action = self._pending_action
        self._pending_action = None
        next_state, reward, done, info = self.game.step(action)
        self.episode_reward += reward
        self.state = next_state
        if done:
            self._end_episode(info)

    def _end_episode(self, info: Dict[str, Any]) -> None:
        score = self.game.score
        self.agent.episode += 1
        self.agent.reward_history.append(self.episode_reward)
        self.agent.update_epsilon()
        self.avg_reward = self.agent.average_recent_reward(100)

        self.sparkline.push(self.episode_reward)

        new_best = score > self.agent.best_score
        if new_best:
            self.agent.best_score = score
            self.chatbot.post_event("new_high_score", {"score": score})
            self._flash(f"NEW BEST  ·  {score}", PINK)

        if info.get("collision"):
            self.chatbot.post_event(
                "collision",
                {"episode": self.agent.episode, "score": score},
            )
            self._flash(f"COLLISION  ·  score {score}", BAD)

        self.logger.info(
            "Episode %d ended: score=%d reward=%.2f eps=%.4f",
            self.agent.episode, score,
            self.episode_reward, self.agent.epsilon,
        )

        if new_best and config.SAVE_ON_BEST:
            self._save_checkpoint(silent=True)
        elif self.agent.episode % config.CHECKPOINT_INTERVAL == 0:
            self._save_checkpoint(silent=True)

        self.episode_reward = 0.0
        self.state = self.game.reset()

    def _flash(self, message: str, color) -> None:
        self._flash_message = message
        self._flash_color = color
        self._flash_until = time.perf_counter() + 1.2

    # ─────────────────────────────────────────────────────────────────
    #  Checkpointing
    # ─────────────────────────────────────────────────────────────────
    def _save_checkpoint(self, silent: bool = False) -> None:
        try:
            self.agent.save(config.CHECKPOINT_FILE)
            if not silent:
                self.chat_panel.add("system", "Checkpoint saved.")
            self.chatbot.post_event("model_saved")
            self.logger.info(
                "Saved checkpoint at episode %d (best=%d)",
                self.agent.episode, self.agent.best_score,
            )
        except Exception as exc:
            self.logger.error("Checkpoint save failed: %s", exc)
            if not silent:
                self.chat_panel.add("system", f"Save failed: {exc}")

    def _reload_checkpoint(self) -> None:
        if self.agent.load(config.CHECKPOINT_FILE):
            self.chat_panel.add(
                "system",
                f"Reloaded · ep {self.agent.episode} · "
                f"best {self.agent.best_score} · eps {self.agent.epsilon:.4f}",
            )
            self.chatbot.post_event(
                "model_loaded",
                {
                    "episode": self.agent.episode,
                    "best_score": self.agent.best_score,
                },
            )
            self._flash("Checkpoint loaded", GOOD)
        else:
            self.chat_panel.add("system", "No usable checkpoint on disk.")
            self._flash("No checkpoint found", AMBER)

    def _quit(self) -> None:
        self._save_checkpoint(silent=True)
        self.chatbot.stop()
        self.running = False

    # ─────────────────────────────────────────────────────────────────
    #  Chatbot state
    # ─────────────────────────────────────────────────────────────────
    def _chat_state(self) -> Dict[str, Any]:
        return {
            "episode": self.agent.episode,
            "length": len(self.game.snake),
            "best_score": self.agent.best_score,
            "epsilon": self.agent.epsilon,
            "avg_reward": self.avg_reward,
            "loss": self.last_loss,
            "device": self.device_label,
            "last_death": None,
        }

    # ─────────────────────────────────────────────────────────────────
    #  Rendering: header
    # ─────────────────────────────────────────────────────────────────
    def _draw_header(self) -> None:
        hdr = pygame.Rect(0, 0, config.WINDOW_WIDTH, HEADER_H)
        pygame.draw.rect(self.screen, BG_HEADER, hdr)
        pygame.draw.line(
            self.screen, BORDER,
            (0, HEADER_H - 1), (config.WINDOW_WIDTH, HEADER_H - 1),
        )

        # Logo
        lf_big = self.fonts.get(24, bold=True)
        neuro = lf_big.render("NEURO", True, ACCENT)
        dot = lf_big.render("·", True, TEXT_MUTED)
        snake = lf_big.render("SNAKE", True, PINK)

        x = PAD
        y = 14
        self.screen.blit(neuro, (x, y))
        x += neuro.get_width() + 8
        self.screen.blit(dot, (x, y))
        x += dot.get_width() + 8
        self.screen.blit(snake, (x, y))

        # Tagline
        sf = self.fonts.get(10)
        tag = sf.render(
            "DOUBLE DUELING DQN · PRIORITIZED REPLAY · N-STEP RETURNS",
            True, TEXT_MUTED,
        )
        self.screen.blit(tag, (x + snake.get_width() + 16, y + 8))

        # Right side: mode pill + device badge
        right_x = config.WINDOW_WIDTH - PAD

        # Device badge
        dev_f = self.fonts.get(11, bold=True)
        dev_surf = dev_f.render(self.device_label, True, TEXT_DIM)
        dev_w = dev_surf.get_width() + 20
        dev_rect = pygame.Rect(
            right_x - dev_w, 18, dev_w, 24,
        )
        pygame.draw.rect(self.screen, BG_CARD, dev_rect, border_radius=12)
        pygame.draw.rect(self.screen, BORDER_SOFT, dev_rect, 1, border_radius=12)
        self.screen.blit(dev_surf, (dev_rect.x + 10, dev_rect.y + 5))
        right_x = dev_rect.x - 12

        # Mode pill
        mode_label = {
            self.MENU: "MENU",
            self.TRAINING: "TRAINING",
            self.WATCHING: "WATCHING",
            self.HUMAN: "HUMAN",
        }.get(self.mode, "—")
        mode_color = {
            self.MENU: TEXT_DIM,
            self.TRAINING: ACCENT,
            self.WATCHING: GOOD,
            self.HUMAN: VIOLET,
        }.get(self.mode, TEXT_DIM)
        mf = self.fonts.get(11, bold=True)
        m_surf = mf.render(mode_label, True, mode_color)
        m_w = m_surf.get_width() + 20
        m_rect = pygame.Rect(
            right_x - m_w, 18, m_w, 24,
        )
        pygame.draw.rect(
            self.screen, (mode_color[0] // 6, mode_color[1] // 6, mode_color[2] // 6),
            m_rect, border_radius=12,
        )
        pygame.draw.rect(self.screen, mode_color, m_rect, 1, border_radius=12)
        self.screen.blit(m_surf, (m_rect.x + 10, m_rect.y + 5))

    # ─────────────────────────────────────────────────────────────────
    #  Rendering: footer
    # ─────────────────────────────────────────────────────────────────
    def _draw_footer(self) -> None:
        ft = pygame.Rect(0, FOOTER_Y, config.WINDOW_WIDTH, FOOTER_H)
        pygame.draw.rect(self.screen, BG_FOOTER, ft)
        pygame.draw.line(self.screen, BORDER, (0, FOOTER_Y), (config.WINDOW_WIDTH, FOOTER_Y))

        f = self.fonts.get(11)
        hints = [
            ("1", "train"), ("2", "watch"), ("3", "human"),
            ("ESC", "menu"), ("ENTER", "chat"), ("F5", "save"),
        ]
        x = PAD
        y = FOOTER_Y + 14
        for key, label in hints:
            k_surf = f.render(key, True, ACCENT)
            l_surf = f.render(" " + label, True, TEXT_MUTED)
            self.screen.blit(k_surf, (x, y))
            x += k_surf.get_width()
            self.screen.blit(l_surf, (x, y))
            x += l_surf.get_width() + 24

        # Right side: FPS
        fps_surf = f.render(f"{self.fps:.0f} fps", True, TEXT_MUTED)
        self.screen.blit(
            fps_surf,
            (config.WINDOW_WIDTH - fps_surf.get_width() - PAD, y),
        )

    # ─────────────────────────────────────────────────────────────────
    #  Rendering: board overlay badges
    # ─────────────────────────────────────────────────────────────────
    def _draw_board_overlays(self) -> None:
        board_rect = self.renderer.rect

        # Score card in top-left
        score_f = self.fonts.get(20, bold=True)
        score_label_f = self.fonts.get(9, bold=True)
        score_val = f"{self.game.score}"

        card_w = 110
        card_h = 46
        sc = pygame.Rect(board_rect.x + 12, board_rect.y + 12, card_w, card_h)
        bg = pygame.Surface(sc.size, pygame.SRCALPHA)
        bg.fill((0, 0, 0, 150))
        self.screen.blit(bg, sc.topleft)
        pygame.draw.rect(self.screen, (60, 66, 92), sc, 1, border_radius=6)

        lbl = score_label_f.render("SCORE", True, TEXT_MUTED)
        self.screen.blit(lbl, (sc.x + 10, sc.y + 6))
        val = score_f.render(score_val, True, ACCENT)
        self.screen.blit(val, (sc.x + 10, sc.y + 20))

        # Best score in top-right
        bc = pygame.Rect(
            board_rect.right - 12 - card_w, board_rect.y + 12,
            card_w, card_h,
        )
        bg2 = pygame.Surface(bc.size, pygame.SRCALPHA)
        bg2.fill((0, 0, 0, 150))
        self.screen.blit(bg2, bc.topleft)
        pygame.draw.rect(self.screen, (60, 66, 92), bc, 1, border_radius=6)

        lbl2 = score_label_f.render("BEST", True, TEXT_MUTED)
        self.screen.blit(lbl2, (bc.x + 10, bc.y + 6))
        val2 = score_f.render(str(self.agent.best_score), True, PINK)
        self.screen.blit(val2, (bc.x + 10, bc.y + 20))

    # ─────────────────────────────────────────────────────────────────
    #  Rendering: stat cards + sparkline + buttons
    # ─────────────────────────────────────────────────────────────────
    def _update_stat_cards(self) -> None:
        c = self.stat_cards
        c["episode"].set(f"{self.agent.episode:,}")
        c["best"].set(self.agent.best_score)
        c["length"].set(len(self.game.snake))
        c["epsilon"].set(
            f"{self.agent.epsilon:.4f}",
            bar=float(min(1.0, self.agent.epsilon / config.EPSILON_START)),
        )
        avg = self.avg_reward
        # Normalise avg reward to [0,1] using a soft ceiling of 30.
        bar = max(0.0, min(1.0, (avg + 5) / 35))
        c["avg"].set(f"{avg:.2f}", bar=bar)
        c["loss"].set(f"{self.last_loss:.4f}")

    def _draw_right_column(self) -> None:
        self._update_stat_cards()
        for i, key in enumerate(self._stat_order):
            self.stat_cards[key].draw(self.screen, self._stat_rects[i], self.fonts)

        self.sparkline.draw(
            self.screen,
            pygame.Rect(RIGHT_X, SPARK_Y, RIGHT_W, SPARK_H),
            self.fonts,
        )

        for b in self.buttons:
            b.active = (
                (b.label == "TRAIN" and self.mode == self.TRAINING)
                or (b.label == "WATCH" and self.mode == self.WATCHING)
                or (b.label == "HUMAN" and self.mode == self.HUMAN)
            )
            b.draw(self.screen, self.fonts)

    # ─────────────────────────────────────────────────────────────────
    #  Rendering: flash banner
    # ─────────────────────────────────────────────────────────────────
    def _draw_flash(self) -> None:
        if time.perf_counter() >= self._flash_until:
            return
        alpha = int(255 * max(0.0, (self._flash_until - time.perf_counter()) / 1.2))
        f = self.fonts.get(18, bold=True)
        surf = f.render(self._flash_message, True, self._flash_color)
        rect = surf.get_rect(center=(
            self.renderer.rect.centerx,
            self.renderer.rect.y + 40,
        ))
        bg = pygame.Surface(
            (rect.width + 32, rect.height + 16), pygame.SRCALPHA,
        )
        bg.fill((*self._flash_color, min(60, alpha // 2)))
        self.screen.blit(bg, (rect.x - 16, rect.y - 8))
        pygame.draw.rect(
            self.screen, self._flash_color,
            pygame.Rect(rect.x - 16, rect.y - 8, rect.width + 32, rect.height + 16),
            2, border_radius=8,
        )
        # apply alpha to the text
        text_surf = surf.copy()
        text_surf.set_alpha(alpha)
        self.screen.blit(text_surf, rect.topleft)

    # ─────────────────────────────────────────────────────────────────
    #  Rendering: menu
    # ─────────────────────────────────────────────────────────────────
    def _menu_rect(self, idx: int) -> pygame.Rect:
        w = 480
        h = 56
        x = (config.WINDOW_WIDTH - w) // 2
        y0 = 260
        return pygame.Rect(x, y0 + idx * (h + 12), w, h)

    def _draw_menu(self) -> None:
        self.screen.fill(BG_APP)

        # Accent glow behind title
        t = self._tick
        glow_r = 260 + int(20 * (0.5 + 0.5 * pygame.math.Vector2(1, 0).length()))
        glow = pygame.Surface((glow_r * 2, glow_r * 2), pygame.SRCALPHA)
        for gr in range(glow_r, 0, -6):
            a = int(28 * (1 - gr / glow_r))
            pygame.draw.circle(
                glow, (*ACCENT, a),
                (glow_r, glow_r), gr,
            )
        self.screen.blit(
            glow,
            (
                config.WINDOW_WIDTH // 2 - glow_r,
                180 - glow_r,
            ),
        )

        # Title
        title_f = self.fonts.get(72, bold=True)
        neuro = title_f.render("NEURO", True, ACCENT)
        snake = title_f.render("SNAKE", True, PINK)
        gap = 24
        total = neuro.get_width() + snake.get_width() + gap
        x0 = (config.WINDOW_WIDTH - total) // 2
        self.screen.blit(neuro, (x0, 100))
        self.screen.blit(snake, (x0 + neuro.get_width() + gap, 100))

        # Subtitle
        sub_f = self.fonts.get(13)
        sub = sub_f.render(
            "DOUBLE DUELING DQN  ·  PRIORITIZED REPLAY  ·  N-STEP RETURNS",
            True, TEXT_DIM,
        )
        self.screen.blit(
            sub,
            (config.WINDOW_WIDTH // 2 - sub.get_width() // 2, 190),
        )

        # Device / checkpoint status
        dev_f = self.fonts.get(11, bold=True)
        dev_surf = dev_f.render(f"device · {self.device_label}", True, TEXT_MUTED)
        self.screen.blit(
            dev_surf,
            (config.WINDOW_WIDTH // 2 - dev_surf.get_width() // 2, 214),
        )
        if self.loaded_checkpoint:
            ck_f = self.fonts.get(11, bold=True)
            ck = ck_f.render(
                f"checkpoint · ep {self.agent.episode} · "
                f"best {self.agent.best_score} · "
                f"eps {self.agent.epsilon:.4f}",
                True, GOOD,
            )
            self.screen.blit(
                ck,
                (config.WINDOW_WIDTH // 2 - ck.get_width() // 2, 232),
            )

        # Menu cards
        for i, (label, desc) in enumerate(self.menu_items):
            rect = self._menu_rect(i)
            sel = i == self.menu_selected

            if sel:
                draw_card(
                    self.screen, rect,
                    bg=(26, 40, 40),
                    border=ACCENT,
                    radius=10, border_width=2, shadow=True,
                )
                # selection bar
                pygame.draw.rect(
                    self.screen, ACCENT,
                    pygame.Rect(rect.x, rect.y + 8, 4, rect.height - 16),
                    border_radius=2,
                )
            else:
                draw_card(
                    self.screen, rect,
                    bg=BG_CARD,
                    border=BORDER_SOFT,
                    radius=10,
                )

            lf = self.fonts.get(15, bold=True)
            label_surf = lf.render(label, True, TEXT if sel else TEXT_DIM)
            self.screen.blit(label_surf, (rect.x + 22, rect.y + 10))

            df = self.fonts.get(11)
            desc_surf = df.render(desc, True, TEXT_MUTED)
            self.screen.blit(desc_surf, (rect.x + 22, rect.y + 32))

            if sel:
                arrow = self.fonts.get(15, bold=True).render("›", True, ACCENT)
                self.screen.blit(
                    arrow,
                    (rect.right - 24, rect.y + 18),
                )

        # Footer hints
        ff = self.fonts.get(11)
        hint = ff.render(
            "↑ ↓ navigate   ·   ENTER select   ·   ESC quit",
            True, TEXT_MUTED,
        )
        self.screen.blit(
            hint,
            (config.WINDOW_WIDTH // 2 - hint.get_width() // 2, config.WINDOW_HEIGHT - 40),
        )

    # ─────────────────────────────────────────────────────────────────
    #  Main loop
    # ─────────────────────────────────────────────────────────────────
    def run(self) -> None:
        last = time.perf_counter()
        while self.running:
            now = time.perf_counter()
            dt = min(0.1, now - last)
            last = now
            self._tick += dt
            self.fps = 1.0 / max(1e-6, dt)

            self._handle_events()
            if not self.running:
                break

            # Simulation
            if self.mode == self.TRAINING:
                for _ in range(config.FAST_AI_STEPS_PER_FRAME):
                    self._ai_step(train=True)
            elif self.mode == self.WATCHING:
                self._accum_ai += dt
                step_dt = 1.0 / config.AI_STEPS_PER_SECOND
                while self._accum_ai >= step_dt:
                    self._ai_step(train=False)
                    self._accum_ai -= step_dt
            elif self.mode == self.HUMAN:
                self._accum_human += dt
                step_dt = 1.0 / config.HUMAN_STEPS_PER_SECOND
                while self._accum_human >= step_dt:
                    self._human_step()
                    self._accum_human -= step_dt

            # Chatbot
            for msg in self.chatbot.poll():
                self.chat_panel.add(msg.who, msg.text)

            # Draw
            if self.mode == self.MENU:
                self._draw_menu()
            else:
                self.screen.fill(BG_APP)
                self._draw_header()
                self.renderer.draw(
                    self.screen, self.game,
                    tick=self._tick,
                    paused=False,
                )
                self._draw_board_overlays()
                self._draw_right_column()
                self.chat_panel.draw(self.screen, self.fonts)
                self._draw_flash()
                self._draw_footer()

            pygame.display.flip()
            self.clock.tick(config.FPS)

        pygame.quit()

    @staticmethod
    def main() -> int:
        try:
            app = GameApp()
            app.run()
            return 0
        except KeyboardInterrupt:
            print("\nInterrupted.")
            return 1
        except Exception:
            import traceback
            traceback.print_exc()
            return 2


if __name__ == "__main__":
    sys.exit(GameApp.main())