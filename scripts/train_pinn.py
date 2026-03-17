from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from microswimmer.data import SequenceDataset
from microswimmer.models import PINNNet
from microswimmer.train_utils import save_json, set_seed
from microswimmer.utils import ensure_dir, train_val_test_split


def reduced_rhs_torch(t, theta, phi, beta, gamma, omega):
    forcing = gamma * torch.sin(theta) + phi - beta * torch.cos(theta) * torch.sin(t * omega)
    denom = torch.cos(2.0 * phi) - 17.0
    theta_dot = (3.0 * torch.cos(phi) ** 2 * forcing - 3.0 * (torch.sin(phi) ** 2 - 19.0) * forcing + 36.0 * phi * torch.cos(phi)) / denom
    phi_dot = 6.0 * (torch.cos(phi) + 3.0) ** 2 * (gamma * torch.sin(theta) + 2.0 * phi - beta * torch.cos(theta) * torch.sin(t * omega)) / denom
    return theta_dot, phi_dot


def pinn_loss(model, batch, device):
    _, params, y_future, t_future, ic = batch
    params = params.to(device)
    y_future = y_future.to(device)
    t_future = t_future.to(device)
    ic = ic.to(device)

    B, T = t_future.shape
    t_flat = t_future.reshape(-1, 1).clone().detach().requires_grad_(True)
    params_flat = params.unsqueeze(1).repeat(1, T, 1).reshape(-1, 3)
    ic_flat = ic.unsqueeze(1).repeat(1, T, 1).reshape(-1, 2)

    pred = model(t_flat, params_flat, ic_flat)
    theta = pred[:, 0:1]
    phi = pred[:, 1:2]

    dtheta_dt = torch.autograd.grad(theta, t_flat, grad_outputs=torch.ones_like(theta), create_graph=True)[0]
    dphi_dt = torch.autograd.grad(phi, t_flat, grad_outputs=torch.ones_like(phi), create_graph=True)[0]

    theta_rhs, phi_rhs = reduced_rhs_torch(
        t_flat,
        theta,
        phi,
        params_flat[:, 0:1],
        params_flat[:, 1:2],
        params_flat[:, 2:3],
    )
    residual = (dtheta_dt - theta_rhs).pow(2).mean() + (dphi_dt - phi_rhs).pow(2).mean()
    data = (pred.reshape(B, T, 2) - y_future).pow(2).mean()
    ic_pred = model(torch.zeros((params.shape[0], 1), device=device), params, ic)
    ic_loss = (ic_pred - ic).pow(2).mean()
    total = data + 0.1 * residual + 0.2 * ic_loss
    return total, data.detach(), residual.detach(), ic_loss.detach()


def eval_model(model, loader, device):
    model.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for _, params, y_future, t_future, ic in loader:
            params = params.to(device)
            y_future = y_future.to(device)
            t_future = t_future.to(device)
            ic = ic.to(device)
            B, T = t_future.shape
            pred = model(
                t_future.reshape(-1, 1),
                params.unsqueeze(1).repeat(1, T, 1).reshape(-1, 3),
                ic.unsqueeze(1).repeat(1, T, 1).reshape(-1, 2),
            ).reshape(B, T, 2)
            total += ((pred - y_future) ** 2).sum().item()
            n += y_future.numel()
    return {"rollout_rmse": float((total / n) ** 0.5)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=16)
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

    model = PINNNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best = float("inf")
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for batch in train_loader:
            loss, data_loss, res_loss, ic_loss = pinn_loss(model, batch, device)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item()
        val_metrics = eval_model(model, val_loader, device)
        if val_metrics["rollout_rmse"] < best:
            best = val_metrics["rollout_rmse"]
            torch.save(model.state_dict(), out_dir / "pinn.pt")
        print(f"epoch={epoch+1} train_loss={running/len(train_loader):.4f} val_rollout_rmse={val_metrics['rollout_rmse']:.4f}")

    model.load_state_dict(torch.load(out_dir / "pinn.pt", map_location=device))
    metrics = eval_model(model, test_loader, device)
    save_json(metrics, out_dir / "metrics.json")
    print(metrics)


if __name__ == "__main__":
    main()
