"""macluster — low-communication distributed training on a personal MacBook
cluster (AWDL/Wi-Fi), emulated on a single machine.

Axes (per the project proposal):
  1. link role separation  -> emulation/ (AWDL vs Wi-Fi profiles)
  2. dense vs low-comm      -> algorithms/ (dense, diloco, sparseloco)
  3. adaptive sync policy   -> algorithms/adaptive.py (novel contribution)
"""

from .train import TrainConfig, run_training

__all__ = ["TrainConfig", "run_training"]
__version__ = "0.1.0"
