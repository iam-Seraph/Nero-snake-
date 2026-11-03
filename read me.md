\# Neuro-Snake AI



A production-quality, self-learning Snake AI built on \*\*PyTorch\*\*.



\*\*Learning stack:\*\* Double Dueling DQN · Prioritized Experience Replay ·

3-step returns · Polyak (soft) target updates · egocentric grid state ·

potential-style reward shaping.



\## Install



```bash

python -m venv .venv

\# Windows

.venv\\Scripts\\activate

\# macOS / Linux

source .venv/bin/activate



pip install -r requirements.txt

```



\## Run



```bash

python main.py

```



A start menu appears. Choose \*\*START TRAINING\*\*, \*\*WATCH AI\*\*, or \*\*HUMAN PLAY\*\*.



\## Controls



| Key | Action |

|-----|--------|

| ↑ ↓ / W S | Menu navigation |

| Enter | Confirm / open chat input |

| Esc | Back to menu / cancel chat |

| Arrows / WASD | Human steering |

| F5 | Save checkpoint now |



\## Files



```

snake\_ai/

├── main.py            # entry point + Pygame loop

├── game.py            # environment + renderer

├── agent.py           # Double Dueling DQN + PER + N-step

├── model.py           # PyTorch network

├── replay\_buffer.py   # prioritized replay

├── chatbot.py         # async offline commentary

├── config.py          # all tunables + hardware detection

├── requirements.txt

├── checkpoints/       # snake\_dqn.pth

├── logs/              # training.log

└── replays/

```



\## Hardware verification



```bash

python -c "

import torch

print('torch:', torch.\_\_version\_\_)

print('cuda available:', torch.cuda.is\_available())

if torch.cuda.is\_available():

&#x20;   print('cuda device:', torch.cuda.get\_device\_name(0))

print('mps available:', getattr(torch.backends, 'mps', None) and torch.backends.mps.is\_available())

"

```



The menu displays the selected device at the top. To verify training actually

runs on the GPU, watch `nvidia-smi` (CUDA) or Activity Monitor → GPU (MPS)

while \*\*START TRAINING\*\* is active — utilization should be non-trivial.



\## Persistent learning



The checkpoint is written automatically:



\- On a new high score

\- Every `CHECKPOINT\_INTERVAL` episodes (default 25)

\- On graceful shutdown

\- When you press \*\*F5\*\* or choose \*\*SAVE MODEL\*\*



Restart the app; the menu shows the loaded episode, best score, and epsilon,

and training continues from the saved weights.



\## Expected learning curve



| Episodes | Mean score | Behaviour |

|----------|-----------|-----------|

| 0 – 100 | 1 – 3 | random wandering |

| 100 – 500 | 4 – 10 | finds nearby food |

| 500 – 2000 | 10 – 22 | path-follows well |

| 2000+ | 25 – 55+ | long efficient games |



DQN learning is stochastic. Judge progress by the 100-episode rolling average

reward, not by any single episode.

