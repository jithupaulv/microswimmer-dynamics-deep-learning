from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from microswimmer.utils import ensure_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--n", type=int, default=3)
    args = parser.parse_args()

    out_dir = ensure_dir(args.out_dir)
    manifest = pd.read_csv(Path(args.data_dir) / "manifest.csv")
    sample = manifest.sample(min(args.n, len(manifest)), random_state=42)

    for i, row in enumerate(sample.itertuples(index=False)):
        arr = np.load(row.trajectory_path)
        theta, phi, t, xdot = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]

        plt.figure(figsize=(7, 4))
        plt.plot(t, theta, label="theta")
        plt.plot(t, phi, label="phi")
        plt.xlabel("t")
        plt.ylabel("state")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"trajectory_{i}.png", dpi=200)
        plt.close()

        plt.figure(figsize=(5, 5))
        plt.plot(theta, phi)
        plt.xlabel("theta")
        plt.ylabel("phi")
        plt.tight_layout()
        plt.savefig(out_dir / f"phase_{i}.png", dpi=200)
        plt.close()

        plt.figure(figsize=(7, 4))
        plt.plot(t, xdot)
        plt.xlabel("t")
        plt.ylabel("xdot")
        plt.tight_layout()
        plt.savefig(out_dir / f"speed_{i}.png", dpi=200)
        plt.close()


if __name__ == "__main__":
    main()
