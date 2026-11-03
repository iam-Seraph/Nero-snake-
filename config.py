"""Central configuration for the Neuro-Snake AI ecosystem.

Every tunable parameter lives here so the rest of the codebase stays
free of magic numbers. Paths, hyperparameters, UI dimensions, and
hardware discovery helpers are all defined in this module.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import torch

# ─── Paths ────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
LOG_DIR = BASE_DIR / "logs"
REPLAY_DIR = BASE_DIR / "replays"

for _d in (CHECKPOINT_DIR, LOG_DIR, REPLAY_DIR):
    _d.mkdir(parents=True, exist_ok=True)

CHECKPOINT_FILE = CHECKPOINT_DIR / "snake_dqn.pth"
LOG_FILE = LOG_DIR / "training.log"

# ─── Board / window ───────────────────────────────────────────────────────
BOARD_WIDTH = 20
BOARD_HEIGHT = 20
CELL_SIZE = 20

WINDOW_WIDTH = 760
WINDOW_HEIGHT = 720
FPS = 60

# ─── DQN ──────────────────────────────────────────────────────────────────
# Egocentric 5×5 grid × 3 channels = 75 features, plus 5 scalars = 80.
VISION = 2
STATE_SIZE = (2 * VISION + 1) ** 2 * 3 + 5
ACTION_SIZE = 3
HIDDEN_SIZES = (256, 256)

GAMMA = 0.99
LEARNING_RATE = 5e-4
GRAD_CLIP = 10.0

# N-step returns propagate sparse food rewards back faster.
N_STEP = 3

# Soft (Polyak) target updates — smoother than periodic hard sync.
TAU = 0.005
HARD_TARGET_SYNC = 0        # set >0 to use hard sync instead of Polyak

REPLAY_CAPACITY = 150_000
BATCH_SIZE = 128
MIN_REPLAY_TO_TRAIN = 2_000
TRAIN_EVERY_STEPS = 2

# ─── Prioritized Experience Replay ────────────────────────────────────────
PER_ALPHA = 0.6
PER_BETA_START = 0.4
PER_BETA_STEPS = 200_000
PER_EPS = 1e-6

# ─── Exploration ──────────────────────────────────────────────────────────
EPSILON_START = 1.0
EPSILON_MIN = 0.02
EPSILON_DECAY = 0.9975
EPSILON_MODE = "exponential"       # "linear" | "exponential"
EPSILON_LINEAR_EPISODES = 5_000

# ─── Rewards ──────────────────────────────────────────────────────────────
FOOD_REWARD = 10.0
COLLISION_PENALTY = -10.0
SURVIVAL_REWARD = 0.02
DISTANCE_REWARD_SCALE = 0.15
MAX_STEPS_NO_FOOD = BOARD_WIDTH * BOARD_HEIGHT

# ─── Checkpointing ────────────────────────────────────────────────────────
CHECKPOINT_INTERVAL = 25
SAVE_ON_BEST = True

# ─── Simulation ───────────────────────────────────────────────────────────
AI_STEPS_PER_SECOND = 30
HUMAN_STEPS_PER_SECOND = 8
FAST_AI_STEPS_PER_FRAME = 200

# ─── Chatbot ──────────────────────────────────────────────────────────────
CHAT_MAX_MESSAGES = 200
CHAT_COMMENT_COOLDOWN = 3.0


# ─── Hardware ─────────────────────────────────────────────────────────────
def detect_device() -> torch.device:
    """Best available device: CUDA > MPS > XPU > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")

    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")

    try:
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return torch.device("xpu")
    except Exception:
        pass

    return torch.device("cpu")


def device_name(device: torch.device) -> str:
    t = device.type
    if t == "cuda":
        try:
            return f"NVIDIA CUDA ({torch.cuda.get_device_name(device)})"
        except Exception:
            return "NVIDIA CUDA"
    if t == "mps":
        return "Apple MPS"
    if t == "xpu":
        try:
            return f"Intel XPU ({torch.xpu.get_device_name(device)})"
        except Exception:
            return "Intel XPU"
    return "CPU"


def hardware_diagnostics() -> Dict[str, Any]:
    """Return a dict summarising the runtime hardware environment."""
    info: Dict[str, Any] = {
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "mps_available": bool(
            getattr(torch.backends, "mps", None)
            and torch.backends.mps.is_available()
        ),
    }
    if torch.cuda.is_available():
        try:
            info["cuda_device"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            info["cuda_memory_gb"] = round(props.total_memory / (1024 ** 3), 2)
        except Exception:
            pass
    return info


def configure_logging() -> logging.Logger:
    """Return the shared application logger with file + console handlers."""
    logger = logging.getLogger("snake_ai")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    sh.setLevel(logging.WARNING)
    logger.addHandler(sh)
    return logger