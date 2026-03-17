from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from microswimmer.utils import ensure_dir


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--baseline_dir", type=str, required=True)
    parser.add_argument("--lstm_dir", type=str, required=True)
    parser.add_argument("--pinn_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()

    out_dir = ensure_dir(args.out_dir)
    baseline = load_json(Path(args.baseline_dir) / "metrics.json")
    lstm = load_json(Path(args.lstm_dir) / "metrics.json")
    pinn = load_json(Path(args.pinn_dir) / "metrics.json")

    rows = [
        {"model": "MLP baseline", **baseline},
        {"model": "LSTM", **lstm},
        {"model": "PINN", **pinn},
    ]
    frame = pd.DataFrame(rows)
    frame.to_csv(out_dir / "comparison.csv", index=False)

    cols = [c for c in frame.columns if c != "model"]
    for col in cols:
        plt.figure(figsize=(6, 4))
        plt.bar(frame["model"], frame[col])
        plt.ylabel(col)
        plt.title(f"Model comparison: {col}")
        plt.tight_layout()
        plt.savefig(out_dir / f"{col}.png", dpi=200)
        plt.close()

    print(frame)


if __name__ == "__main__":
    main()
