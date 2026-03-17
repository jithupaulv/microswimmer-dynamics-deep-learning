from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from torch import nn
from torch.utils.data import DataLoader, Subset

from microswimmer.data import SummaryDataset
from microswimmer.models import SummaryMLP
from microswimmer.train_utils import save_json, set_seed
from microswimmer.utils import ensure_dir, train_val_test_split


def evaluate(model, loader, device):
    model.eval()
    ys_reg, ps_reg = [], []
    ys_cls, ps_cls = [], []
    with torch.no_grad():
        for x, y_reg, y_cls in loader:
            x = x.to(device)
            reg, cls = model(x)
            ys_reg.append(y_reg.numpy())
            ps_reg.append(reg.cpu().numpy())
            ys_cls.append(y_cls.numpy())
            ps_cls.append((torch.sigmoid(cls).cpu().numpy() > 0.5).astype(float))
    import numpy as np
    ys_reg = np.concatenate(ys_reg)
    ps_reg = np.concatenate(ps_reg)
    ys_cls = np.concatenate(ys_cls)
    ps_cls = np.concatenate(ps_cls)
    return {
        "rmse": float(mean_squared_error(ys_reg, ps_reg) ** 0.5),
        "mae": float(mean_absolute_error(ys_reg, ps_reg)),
        "accuracy": float(accuracy_score(ys_cls, ps_cls)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    out_dir = ensure_dir(args.out_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    frame = pd.read_csv(Path(args.data_dir) / "summary.csv")
    ds = SummaryDataset(frame)
    tr, va, te = train_val_test_split(len(ds), seed=args.seed)
    train_loader = DataLoader(Subset(ds, tr), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(Subset(ds, va), batch_size=args.batch_size)
    test_loader = DataLoader(Subset(ds, te), batch_size=args.batch_size)

    model = SummaryMLP().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()

    best = float("inf")
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for x, y_reg, y_cls in train_loader:
            x = x.to(device)
            y_reg = y_reg.to(device)
            y_cls = y_cls.to(device)
            pred_reg, pred_cls = model(x)
            loss = mse(pred_reg, y_reg) + 0.5 * bce(pred_cls, y_cls)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(x)
        val_metrics = evaluate(model, val_loader, device)
        if val_metrics["rmse"] < best:
            best = val_metrics["rmse"]
            torch.save(model.state_dict(), out_dir / "baseline.pt")
        print(f"epoch={epoch+1} train_loss={total/len(tr):.4f} val_rmse={val_metrics['rmse']:.4f} val_acc={val_metrics['accuracy']:.4f}")

    model.load_state_dict(torch.load(out_dir / "baseline.pt", map_location=device))
    metrics = evaluate(model, test_loader, device)
    save_json(metrics, out_dir / "metrics.json")
    print(metrics)


if __name__ == "__main__":
    main()
