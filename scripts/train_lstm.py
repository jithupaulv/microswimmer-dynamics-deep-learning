from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from microswimmer.data import SequenceDataset
from microswimmer.models import TrajectoryLSTM
from microswimmer.train_utils import save_json, set_seed
from microswimmer.utils import ensure_dir, train_val_test_split


def eval_model(model, loader, device):
    model.eval()
    total = 0.0
    n = 0
    mse = nn.MSELoss(reduction="sum")
    with torch.no_grad():
        for x_hist, params, y_future, _, _ in loader:
            x_hist = x_hist.to(device)
            params = params.to(device)
            y_future = y_future.to(device)
            pred = model(x_hist, params, pred_len=y_future.shape[1])
            total += mse(pred, y_future).item()
            n += y_future.numel()
    return {"rollout_rmse": float((total / n) ** 0.5)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=30)
    parser.add_argument("--pred_len", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    out_dir = ensure_dir(args.out_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    manifest = pd.read_csv(Path(args.data_dir) / "manifest.csv")
    ds = SequenceDataset(manifest, seq_len=args.seq_len, pred_len=args.pred_len, root_dir=Path(args.data_dir).parent)
    tr, va, te = train_val_test_split(len(ds), seed=args.seed)
    train_loader = DataLoader(Subset(ds, tr), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(Subset(ds, va), batch_size=args.batch_size)
    test_loader = DataLoader(Subset(ds, te), batch_size=args.batch_size)

    model = TrajectoryLSTM().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    best = float("inf")
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for x_hist, params, y_future, _, _ in train_loader:
            x_hist = x_hist.to(device)
            params = params.to(device)
            y_future = y_future.to(device)
            pred = model(x_hist, params, pred_len=y_future.shape[1])
            loss = loss_fn(pred, y_future)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(x_hist)
        val_metrics = eval_model(model, val_loader, device)
        if val_metrics["rollout_rmse"] < best:
            best = val_metrics["rollout_rmse"]
            torch.save(model.state_dict(), out_dir / "lstm.pt")
        print(f"epoch={epoch+1} train_loss={total/len(tr):.4f} val_rollout_rmse={val_metrics['rollout_rmse']:.4f}")

    model.load_state_dict(torch.load(out_dir / "lstm.pt", map_location=device))
    metrics = eval_model(model, test_loader, device)
    save_json(metrics, out_dir / "metrics.json")
    print(metrics)


if __name__ == "__main__":
    main()
