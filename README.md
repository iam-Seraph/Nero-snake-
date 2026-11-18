# Neuro-Snake AI

A Pygame Snake game with a PyTorch Double Dueling DQN training agent, prioritized replay buffer and offline chatbot.

## Layout

```text
Nero-snake-/
├── main.py                # Start the game
├── diagnose.py            # Hardware diagnostics
├── snake_ai/              # Game, AI, models, UI and configuration
│   ├── main.py
│   ├── agent.py
│   ├── model.py
│   ├── replay_buffer.py
│   ├── game.py
│   ├── ui.py
│   ├── chatbot.py
│   ├── config.py
│   └── diagnose.py
├── requirements.txt
└── README.md
```

## Install and run

```bash
python -m pip install -r requirements.txt
python diagnose.py
python main.py
```

Start training, watch the AI or play manually using the menu. Arrow keys / WASD steer the snake; F5 saves the model. The project stores training checkpoints in `checkpoints/`, logs in `logs/`, and replay files in `replays/`, automatically created at the repository root. These runtime directories are excluded by `.gitignore`.

Requires Python, PyTorch, NumPy and Pygame, and a desktop graphical display. Headless Codespaces typically cannot display the Pygame window without extra setup. CPU is supported; compatible CUDA/MPS/DirectML backends may be used depending on installed PyTorch and hardware.
