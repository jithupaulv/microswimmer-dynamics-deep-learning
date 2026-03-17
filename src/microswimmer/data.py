from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class SummaryDataset(Dataset):
    def __init__(self, frame: pd.DataFrame):
        self.x = frame[["beta", "gamma", "omega", "theta0", "phi0"]].to_numpy(dtype=np.float32)
        self.y_reg = frame[["mean_theta", "mean_phi", "mean_speed"]].to_numpy(dtype=np.float32)
        self.y_cls = frame[["regime"]].to_numpy(dtype=np.float32)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y_reg[idx], self.y_cls[idx]


class SequenceDataset(Dataset):
    def __init__(self, manifest: pd.DataFrame, seq_len: int = 30, pred_len: int = 100, root_dir: str | Path | None = None):
        self.manifest = manifest.reset_index(drop=True)
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.root_dir = Path(root_dir) if root_dir is not None else None

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int):
        row = self.manifest.iloc[idx]
        path = Path(row["trajectory_path"])
        if not path.is_absolute() and self.root_dir is not None:
            path = self.root_dir / path
        arr = np.load(path)
        theta = arr[:, 0].astype(np.float32)
        phi = arr[:, 1].astype(np.float32)
        t = arr[:, 2].astype(np.float32)
        params = np.array([row["beta"], row["gamma"], row["omega"]], dtype=np.float32)

        x_hist = np.stack([theta[: self.seq_len], phi[: self.seq_len]], axis=-1)
        y_future = np.stack([
            theta[self.seq_len : self.seq_len + self.pred_len],
            phi[self.seq_len : self.seq_len + self.pred_len],
        ], axis=-1)
        t_future = t[self.seq_len : self.seq_len + self.pred_len]
        ic = np.array([theta[0], phi[0]], dtype=np.float32)
        return x_hist, params, y_future, t_future, ic


def save_manifest(manifest_rows: List[Dict], out_dir: Path) -> pd.DataFrame:
    frame = pd.DataFrame(manifest_rows)
    frame.to_csv(out_dir / "manifest.csv", index=False)
    frame[["beta", "gamma", "omega", "theta0", "phi0", "mean_theta", "mean_phi", "mean_speed", "regime"]].to_csv(
        out_dir / "summary.csv", index=False
    )
    return frame


def load_manifest(data_dir: str | Path) -> pd.DataFrame:
    return pd.read_csv(Path(data_dir) / "manifest.csv")
