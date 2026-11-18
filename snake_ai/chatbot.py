"""Asynchronous, offline-first chatbot worker.

Runs on a background thread so it never blocks the Pygame event loop.
Communication is via two queues: inbox for events/user messages, outbox
for generated replies. Fully offline; the state provider is called every
time a reply is needed so answers reflect live training statistics.
"""
from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class ChatMessage:
    who: str                     # "ai" | "user" | "system"
    text: str
    ts: float = field(default_factory=time.time)


class ChatbotWorker:
    """Background worker producing contextual commentary and answers."""

    def __init__(self, max_messages: int = 200) -> None:
        self.inbox: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self.outbox: "queue.Queue[ChatMessage]" = queue.Queue()
        self.max_messages = max_messages
        self._state_provider: Optional[Callable[[], Dict[str, Any]]] = None
        self._comment_cooldown = 3.0
        self._last_comment_ts = 0.0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._started = False

    # ── public API ───────────────────────────────────────────────────────
    def start(self) -> None:
        if self._started:
            return
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="chatbot-worker"
        )
        self._thread.start()
        self._started = True

    def stop(self) -> None:
        self._stop.set()
        try:
            self.inbox.put_nowait({"type": "stop"})
        except queue.Full:
            pass

    def set_state_provider(self, fn: Callable[[], Dict[str, Any]]) -> None:
        self._state_provider = fn

    def set_comment_cooldown(self, seconds: float) -> None:
        self._comment_cooldown = float(seconds)

    def post_user_message(self, text: str) -> None:
        self.inbox.put({"type": "user", "text": text})

    def post_event(self, event: str, info: Optional[Dict[str, Any]] = None) -> None:
        self.inbox.put({"type": "event", "event": event, "info": info or {}})

    def poll(self) -> List[ChatMessage]:
        out: List[ChatMessage] = []
        while True:
            try:
                out.append(self.outbox.get_nowait())
            except queue.Empty:
                break
        return out

    # ── worker loop ──────────────────────────────────────────────────────
    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                item = self.inbox.get(timeout=0.25)
            except queue.Empty:
                continue
            if item.get("type") == "stop":
                break
            try:
                self._handle(item)
            except Exception:
                # A chatbot failure must never crash the game.
                pass

    def _handle(self, item: Dict[str, Any]) -> None:
        state: Dict[str, Any] = {}
        if self._state_provider is not None:
            try:
                state = self._state_provider() or {}
            except Exception:
                state = {}

        kind = item.get("type")
        if kind == "user":
            text = str(item.get("text", "")).strip()
            if not text:
                return
            self.outbox.put(ChatMessage("user", text))
            reply = self._answer(text, state)
            if reply:
                self.outbox.put(ChatMessage("ai", reply))
        elif kind == "event":
            now = time.time()
            if now - self._last_comment_ts < self._comment_cooldown:
                return
            comment = self._commentary(
                item.get("event", ""), item.get("info", {}), state
            )
            if comment:
                self.outbox.put(ChatMessage("ai", comment))
                self._last_comment_ts = now

    # ── response generation ──────────────────────────────────────────────
    def _answer(self, text: str, s: Dict[str, Any]) -> str:
        t = text.lower()

        if any(k in t for k in ("hello", "hi ", "hey")) or t.strip() == "hi":
            return "Online. The board is live and my policy is ready."

        if "how is the ai" in t or "how are you" in t or "status" in t:
            return (
                f"Episode {s.get('episode', 0):,}, length {s.get('length', 0)}, "
                f"best score {s.get('best_score', 0)}, epsilon "
                f"{s.get('epsilon', 0.0):.3f}, avg reward "
                f"{s.get('avg_reward', 0.0):.2f}."
            )

        if "why" in t and ("die" in t or "death" in t or "died" in t or "collision" in t):
            cause = s.get("last_death") or "unknown"
            return (
                f"Last death: {cause}. That transition stays in the prioritized "
                f"replay buffer and keeps getting sampled."
            )

        if "learn" in t:
            return (
                "I minimise the temporal-difference error between predicted Q "
                "values and rewards plus the discounted best next value. Every "
                "death contributes a terminal transition."
            )

        if "epsilon" in t:
            return (
                f"Epsilon is {s.get('epsilon', 0.0):.3f}. It decays as episodes "
                f"complete, shifting the policy from exploration to exploitation."
            )

        if "improv" in t or "getting better" in t:
            return (
                f"Recent average reward is {s.get('avg_reward', 0.0):.2f}. "
                f"Improvement is noisy but generally trends upward over hundreds "
                f"of episodes."
            )

        if "device" in t or "gpu" in t or "cuda" in t or "mps" in t or "cpu" in t:
            return f"Running on {s.get('device', 'CPU')}."

        if "high score" in t or "best" in t:
            return f"High score is {s.get('best_score', 0)}."

        if "help" in t or t.strip() == "?":
            return (
                "Ask me about: status, learning, deaths, epsilon, improvement, "
                "device, high score."
            )

        return (
            "Noted. Ask me about my status, learning, deaths, epsilon, "
            "improvement, or the device in use."
        )

    def _commentary(
        self,
        event: str,
        info: Dict[str, Any],
        s: Dict[str, Any],
    ) -> Optional[str]:
        if event == "new_high_score":
            return f"New high score: {info.get('score')}."
        if event == "long_survival":
            return f"Length {info.get('length')} reached. The policy is holding together."
        if event == "collision":
            return (
                f"Collision. Episode {info.get('episode')} ended at "
                f"{info.get('score')}. That mistake is now training data."
            )
        if event == "model_saved":
            return "Checkpoint saved."
        if event == "model_loaded":
            return (
                f"Loaded checkpoint at episode {info.get('episode')}, "
                f"best {info.get('best_score')}."
            )
        if event == "training_improved":
            return f"Average reward moved to {info.get('avg_reward', 0.0):.2f}."
        if event == "human_control":
            return "Manual control engaged. I'll watch what you do."
        if event == "ai_control":
            return "AI control resumed."
        return None