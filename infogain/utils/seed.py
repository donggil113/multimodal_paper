"""Deterministic seeding across numpy / torch / python-random."""
from __future__ import annotations

import os
import random
from contextlib import contextmanager

import numpy as np


def set_seed(seed: int, deterministic_torch: bool = True) -> None:
    """Seed every RNG we rely on.

    ``deterministic_torch`` also pins cuDNN/oneDNN into deterministic mode; this
    matters because the multi-seed variance estimator in
    :mod:`infogain.vinfo.estimators` treats between-seed spread as *estimator*
    variance, which is only meaningful if a fixed seed reproduces exactly.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:  # pragma: no cover - torch is a hard dep in practice
        pass


def rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


@contextmanager
def temp_seed(seed: int):
    """Temporarily seed numpy's legacy global RNG (sklearn still consults it)."""
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)
