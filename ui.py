"""Reusable UI primitives for the Neuro-Snake interface."""
from __future__ import annotations

from typing import Optional

import pygame


# ─── Palette ──────────────────────────────────────────────────────────────
BG_APP        = (10, 11, 17)
BG_HEADER     = (14, 16, 24)
BG_FOOTER     = (12, 13, 20)
BG_PANEL      = (18, 20, 30)
BG_CARD       = (24, 27, 40)
BG_CARD_HOVER = (33, 37, 54)
BG_INPUT      = (14, 16, 24)

BORDER        = (36, 40, 58)
BORDER_SOFT   = (28, 31, 46)
BORDER_STRONG = (58, 64, 88)

TEXT          = (232, 234, 244)
TEXT_DIM      = (150, 156, 178)
TEXT_MUTED    = (98, 104, 128)

ACCENT        = (110, 240, 210)
ACCENT_DIM    = (55, 155, 140)
ACCENT_DEEP   = (20, 60, 55)

PINK          = (240, 96, 130)
AMBER         = (255, 180, 80)
VIOLET        = (170, 130, 255)
GOOD          = (110, 230, 150)
BAD           = (240, 90, 110)


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def lerp_color(c1, c2, t):
    return (
        int(lerp(c1[0], c2[0], t)),
        int(lerp(c1[1], c2[1], t)),
        int(lerp(c1[2], c2[2], t)),
    )


# ─── Fonts ────────────────────────────────────────────────────────────────
class Fonts:
    """Lazy font cache keyed by (size, bold)."""

    DEFAULT_FAMILY = "consolas,dejavusansmono,monospace"

    def __init__(self, family: Optional[str] = None) -> None:
        self.family = family or self.DEFAULT_FAMILY
        self._cache: dict = {}

    def get(self, size: int, bold: bool = False) -> pygame.font.Font:
        key = (size, bold)
        f = self._cache.get(key)
        if f is None:
            f = pygame.font.SysFont(self.family, size, bold=bold)
            self._cache[key] = f
        return f


# ─── Shadow helper ────────────────────────────────────────────────────────
def draw_shadow(
    surface: pygame.Surface,
    rect: pygame.Rect,
    radius: int = 8,
    spread: int = 12,
    alpha: int = 45,
) -> None:
    """Blurred drop shadow beneath a rounded rect."""
    shadow = pygame.Surface(
        (rect.width + spread * 2, rect.height + spread * 2), pygame.SRCALPHA
    )
    for i in range(spread, 0, -1):
        a = int(alpha * (1 - i / spread))
        pygame.draw.rect(
            shadow,
            (0, 0, 0, a),
            pygame.Rect(
                spread - i, spread - i,
                rect.width + i * 2, rect.height + i * 2,
            ),
            border_radius=radius + i,
        )
    surface.blit(shadow, (rect.x - spread, rect.y - spread))


# ─── Card ─────────────────────────────────────────────────────────────────
def draw_card(
    surface: pygame.Surface,
    rect: pygame.Rect,
    *,
    bg=BG_CARD,
    border=BORDER_SOFT,
    radius: int = 10,
    border_width: int = 1,
    shadow: bool = False,
) -> None:
    if shadow:
        draw_shadow(surface, rect, radius=radius, spread=10, alpha=30)
    pygame.draw.rect(surface, bg, rect, border_radius=radius)
    if border_width > 0:
        pygame.draw.rect(surface, border, rect, border_width, border_radius=radius)


# ─── Badge ────────────────────────────────────────────────────────────────
def draw_badge(
    surface: pygame.Surface,
    rect: pygame.Rect,
    text: str,
    fonts: Fonts,
    color=ACCENT,
) -> None:
    bg = (color[0] // 6, color[1] // 6, color[2] // 6)
    pygame.draw.rect(surface, bg, rect, border_radius=rect.height // 2)
    pygame.draw.rect(surface, color, rect, 1, border_radius=rect.height // 2)
    f = fonts.get(11, bold=True)
    s = f.render(text, True, color)
    surface.blit(
        s,
        (rect.centerx - s.get_width() // 2, rect.centery - s.get_height() // 2),
    )


# ─── Bar ──────────────────────────────────────────────────────────────────
def draw_bar(
    surface: pygame.Surface,
    rect: pygame.Rect,
    value: float,
    color=ACCENT,
    bg=BG_INPUT,
) -> None:
    pygame.draw.rect(surface, bg, rect, border_radius=rect.height // 2)
    fill_w = int(rect.width * max(0.0, min(1.0, value)))
    if fill_w > 0:
        pygame.draw.rect(
            surface,
            color,
            pygame.Rect(rect.x, rect.y, fill_w, rect.height),
            border_radius=rect.height // 2,
        )


# ─── Stat card ────────────────────────────────────────────────────────────
class StatCard:
    """Compact label / value card with an optional bar below the value."""

    def __init__(self, label: str, accent=ACCENT) -> None:
        self.label = label
        self.accent = accent
        self.value_str = "—"
        self.bar_value: Optional[float] = None

    def set(self, value, bar: Optional[float] = None) -> None:
        self.value_str = str(value)
        self.bar_value = bar

    def draw(self, surface: pygame.Surface, rect: pygame.Rect, fonts: Fonts) -> None:
        draw_card(surface, rect)

        lf = fonts.get(10, bold=True)
        label_surf = lf.render(self.label.upper(), True, TEXT_MUTED)
        surface.blit(label_surf, (rect.x + 14, rect.y + 12))

        vf = fonts.get(22, bold=True)
        value_surf = vf.render(self.value_str, True, self.accent)
        surface.blit(value_surf, (rect.x + 14, rect.y + 30))

        if self.bar_value is not None:
            bar_rect = pygame.Rect(
                rect.x + 14, rect.bottom - 16, rect.width - 28, 6,
            )
            draw_bar(surface, bar_rect, self.bar_value, color=self.accent)


# ─── Sparkline ────────────────────────────────────────────────────────────
class Sparkline:
    """Filled area chart for a rolling series."""

    def __init__(self, max_points: int = 200, color=ACCENT) -> None:
        self.max_points = max_points
        self.color = color
        self.values: list[float] = []

    def push(self, v: float) -> None:
        self.values.append(float(v))
        if len(self.values) > self.max_points:
            self.values = self.values[-self.max_points:]

    def draw(
        self,
        surface: pygame.Surface,
        rect: pygame.Rect,
        fonts: Optional[Fonts] = None,
        title: str = "REWARD · last 200",
    ) -> None:
        draw_card(surface, rect)

        # Title
        if fonts is not None:
            tf = fonts.get(10, bold=True)
            title_surf = tf.render(title, True, TEXT_MUTED)
            surface.blit(title_surf, (rect.x + 14, rect.y + 10))

        inner = pygame.Rect(
            rect.x + 14, rect.y + 30,
            rect.width - 28, rect.height - 42,
        )

        if len(self.values) < 2:
            if fonts is not None:
                f = fonts.get(11)
                hint = f.render("waiting for episodes…", True, TEXT_MUTED)
                surface.blit(
                    hint,
                    (
                        inner.centerx - hint.get_width() // 2,
                        inner.centery - hint.get_height() // 2,
                    ),
                )
            return

        vmin = min(self.values)
        vmax = max(self.values)
        if vmax == vmin:
            vmax = vmin + 1.0
        span = vmax - vmin

        # Zero baseline if the range crosses zero
        if vmin < 0 < vmax:
            zy = inner.bottom - (0 - vmin) / span * inner.height
            pygame.draw.line(
                surface, BORDER, (inner.left, zy), (inner.right, zy), 1
            )

        n = len(self.values)
        pts = []
        for i, v in enumerate(self.values):
            x = inner.left + inner.width * i / (n - 1)
            y = inner.bottom - (v - vmin) / span * inner.height
            pts.append((x, y))

        # Filled area
        area = pts + [(inner.right, inner.bottom), (inner.left, inner.bottom)]
        surf = pygame.Surface(rect.size, pygame.SRCALPHA)
        offset = [(p[0] - rect.left, p[1] - rect.top) for p in area]
        pygame.draw.polygon(surf, (*self.color, 34), offset)
        surface.blit(surf, rect.topleft)

        # Line
        pygame.draw.lines(surface, self.color, False, pts, 2)
        pygame.draw.circle(surface, self.color, pts[-1], 3)


# ─── Button ───────────────────────────────────────────────────────────────
class Button:
    """Rounded button with hover / pressed / active / disabled states."""

    def __init__(self, label: str, shortcut: str = "", accent=ACCENT) -> None:
        self.rect = pygame.Rect(0, 0, 0, 0)
        self.label = label
        self.shortcut = shortcut
        self.accent = accent
        self.hovered = False
        self.pressed = False
        self.active = False
        self.enabled = True

    def place(self, rect: pygame.Rect) -> None:
        self.rect = rect

    def update(self, mouse_pos) -> None:
        self.hovered = self.rect.collidepoint(mouse_pos) and self.enabled

    def handle_event(self, event) -> bool:
        """Return True if the button was clicked this event."""
        if not self.enabled:
            return False
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.pressed = True
        if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            was_pressed = self.pressed
            self.pressed = False
            if was_pressed and self.rect.collidepoint(event.pos):
                return True
        return False

    def draw(self, surface: pygame.Surface, fonts: Fonts) -> None:
        if self.active:
            pygame.draw.rect(surface, ACCENT_DEEP, self.rect, border_radius=8)
            pygame.draw.rect(surface, self.accent, self.rect, 2, border_radius=8)
            text_color = self.accent
        elif not self.enabled:
            pygame.draw.rect(surface, BG_CARD, self.rect, border_radius=8)
            pygame.draw.rect(surface, BORDER_SOFT, self.rect, 1, border_radius=8)
            text_color = TEXT_MUTED
        elif self.pressed:
            pygame.draw.rect(surface, BORDER_STRONG, self.rect, border_radius=8)
            pygame.draw.rect(surface, self.accent, self.rect, 2, border_radius=8)
            text_color = self.accent
        elif self.hovered:
            pygame.draw.rect(surface, BG_CARD_HOVER, self.rect, border_radius=8)
            pygame.draw.rect(surface, BORDER_STRONG, self.rect, 1, border_radius=8)
            text_color = TEXT
        else:
            pygame.draw.rect(surface, BG_CARD, self.rect, border_radius=8)
            pygame.draw.rect(surface, BORDER_SOFT, self.rect, 1, border_radius=8)
            text_color = TEXT_DIM

        lf = fonts.get(12, bold=True)
        label_surf = lf.render(self.label.upper(), True, text_color)
        surface.blit(
            label_surf,
            (
                self.rect.centerx - label_surf.get_width() // 2,
                self.rect.centery - label_surf.get_height() // 2,
            ),
        )

        if self.shortcut:
            sf = fonts.get(9, bold=True)
            ss = sf.render(self.shortcut, True, TEXT_MUTED)
            surface.blit(ss, (self.rect.x + 7, self.rect.y + 5))


# ─── Chat panel ───────────────────────────────────────────────────────────
class ChatPanel:
    """Message log with bubble styling and a single-line input."""

    AI_COLOR   = (232, 200, 224)
    USER_COLOR = (180, 240, 230)
    SYS_COLOR  = (170, 172, 200)
    AI_BG      = (32, 22, 34)
    USER_BG    = (18, 34, 32)
    SYS_BG     = (22, 24, 34)

    def __init__(self) -> None:
        self.rect = pygame.Rect(0, 0, 0, 0)
        self.messages: list[tuple[str, str]] = []
        self.input_text = ""
        self.active = False
        self._max = 200

    def place(self, rect: pygame.Rect) -> None:
        self.rect = rect

    def add(self, who: str, text: str) -> None:
        self.messages.append((who, text))
        if len(self.messages) > self._max:
            self.messages = self.messages[-self._max:]

    # ── drawing ──────────────────────────────────────────────────────
    def draw(self, surface: pygame.Surface, fonts: Fonts) -> None:
        draw_card(surface, self.rect)

        # Title bar
        tf = fonts.get(10, bold=True)
        title = tf.render("CONVERSATION", True, TEXT_MUTED)
        surface.blit(title, (self.rect.x + 14, self.rect.y + 10))

        hint = fonts.get(9)
        hs = hint.render(
            "ENTER to type · ESC to blur" if not self.active else "typing…",
            True, TEXT_MUTED,
        )
        surface.blit(
            hs, (self.rect.right - hs.get_width() - 14, self.rect.y + 12)
        )

        # Body
        input_h = 30
        body_top = self.rect.y + 30
        body_bottom = self.rect.bottom - input_h - 14
        body_h = max(0, body_bottom - body_top)
        body_rect = pygame.Rect(
            self.rect.x + 10, body_top, self.rect.width - 20, body_h
        )

        self._draw_messages(surface, body_rect, fonts)

        # Input bar
        input_rect = pygame.Rect(
            self.rect.x + 10,
            self.rect.bottom - input_h - 8,
            self.rect.width - 20,
            input_h,
        )
        bg = BG_CARD_HOVER if self.active else BG_INPUT
        border = ACCENT if self.active else BORDER
        pygame.draw.rect(surface, bg, input_rect, border_radius=6)
        pygame.draw.rect(surface, border, input_rect, 1, border_radius=6)

        # Prompt
        pf = fonts.get(12, bold=True)
        prompt = pf.render("›", True, ACCENT if self.active else TEXT_MUTED)
        surface.blit(prompt, (input_rect.x + 10, input_rect.y + 7))

        # Text + cursor
        tf2 = fonts.get(12)
        cursor = ""
        if self.active and (pygame.time.get_ticks() // 500) % 2 == 0:
            cursor = "│"
        text_color = TEXT if self.active else TEXT_DIM
        text_surf = tf2.render(self.input_text + cursor, True, text_color)
        surface.blit(text_surf, (input_rect.x + 28, input_rect.y + 7))

        if not self.active and not self.input_text:
            ph = fonts.get(12)
            ph_surf = ph.render("ask the AI anything…", True, TEXT_MUTED)
            surface.blit(ph_surf, (input_rect.x + 28, input_rect.y + 7))

    def _draw_messages(
        self, surface: pygame.Surface, rect: pygame.Rect, fonts: Fonts
    ) -> None:
        body = pygame.Surface(rect.size, pygame.SRCALPHA)
        font = fonts.get(12)
        line_h = 16
        pad = 8

        # Build wrapped lines with alignment metadata
        lines: list[tuple[str, str, int, int]] = []  # (who, text, w, h)
        max_w = rect.width - 30

        for who, text in self.messages:
            label = {"ai": "AI", "user": "YOU", "system": "SYS"}.get(who, who.upper())
            prefix = f"{label}  "
            wrapped = self._wrap(font, prefix + text, max_w)
            for i, line in enumerate(wrapped):
                lines.append((who, line, font.size(line)[0], line_h))

        # If too many, keep last N
        max_visible = max(1, rect.height // line_h)
        if len(lines) > max_visible:
            lines = lines[-max_visible:]

        y = 2
        for who, text, w, h in lines:
            color = {
                "ai": self.AI_COLOR,
                "user": self.USER_COLOR,
                "system": self.SYS_COLOR,
            }.get(who, self.SYS_COLOR)
            bg = {
                "ai": self.AI_BG,
                "user": self.USER_BG,
                "system": self.SYS_BG,
            }.get(who, self.SYS_BG)

            bubble_w = min(max_w, w + pad * 2)
            bubble = pygame.Surface((bubble_w, h + pad), pygame.SRCALPHA)
            pygame.draw.rect(
                bubble, (*bg, 220),
                pygame.Rect(0, 0, bubble_w, h + pad),
                border_radius=6,
            )
            text_surf = font.render(text, True, color)
            bubble.blit(text_surf, (pad, pad // 2))

            if who == "user":
                x = rect.width - bubble_w - 6
            elif who == "system":
                x = (rect.width - bubble_w) // 2
            else:
                x = 6

            body.blit(bubble, (x, y))
            y += h + pad + 2
            if y > rect.height:
                break

        surface.blit(body, rect.topleft)

    @staticmethod
    def _wrap(font: pygame.font.Font, text: str, max_w: int) -> list[str]:
        words = text.split(" ")
        out: list[str] = []
        cur = ""
        for w in words:
            test = (cur + " " + w).strip()
            if font.size(test)[0] <= max_w:
                cur = test
            else:
                if cur:
                    out.append(cur)
                cur = w
        if cur:
            out.append(cur)
        return out or [""]