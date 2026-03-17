from __future__ import annotations

from pathlib import Path
import numpy as np


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def train_val_test_split(n: int, train_frac: float = 0.7, val_frac: float = 0.15, seed: int = 42):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_train = int(train_frac * n)
    n_val = int(val_frac * n)
    return idx[:n_train], idx[n_train:n_train+n_val], idx[n_train+n_val:]
