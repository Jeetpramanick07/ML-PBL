"""Global seeding utility so every experiment in this project is reproducible."""

import os
import random

import numpy as np


def set_seed(seed: int) -> None:
    """Set the seed for python, numpy, and torch (CPU + CUDA, if available).

    Call this once at the start of every training/evaluation script, using the
    seed value from the active config (configs/base.yaml: seed).
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
