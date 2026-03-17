from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from microswimmer.data import save_manifest
from microswimmer.dynamics import simulate
from microswimmer.utils import ensure_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--n_trajectories", type=int, default=400)
    parser.add_argument("--t_final", type=float, default=15.0)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = ensure_dir(args.out_dir)
    traj_dir = ensure_dir(out_dir / "trajectories")
    rng = np.random.default_rng(args.seed)

    rows = []
    for i in tqdm(range(args.n_trajectories), desc="simulate"):
        beta = float(rng.uniform(0.3, 3.0))
        gamma = float(rng.uniform(0.0, 1.0))
        omega = float(rng.uniform(1.5, 20.0))
        theta0 = float(rng.uniform(-0.2, 2.9))
        phi0 = float(rng.uniform(-0.8, 0.8))

        sim = simulate(beta, gamma, omega, theta0, phi0, t_final=args.t_final, dt=args.dt)
        arr = np.stack([sim.theta, sim.phi, sim.t, sim.xdot], axis=-1)
        path = traj_dir / f"traj_{i:05d}.npy"
        np.save(path, arr)

        row = dict(sim.summary)
        row["trajectory_path"] = str(path.resolve())
        rows.append(row)

    save_manifest(rows, out_dir)
    print(f"Saved dataset to {out_dir}")


if __name__ == "__main__":
    main()
