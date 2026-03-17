from __future__ import annotations

import torch
from torch import nn


class SummaryMLP(nn.Module):
    def __init__(self, in_dim: int = 5, hidden: int = 128):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.reg_head = nn.Linear(hidden, 3)
        self.cls_head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.backbone(x)
        return self.reg_head(h), self.cls_head(h)


class TrajectoryLSTM(nn.Module):
    def __init__(self, input_dim: int = 2, param_dim: int = 3, hidden_dim: int = 128, num_layers: int = 2):
        super().__init__()
        self.param_proj = nn.Linear(param_dim, hidden_dim)
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.decoder_cell = nn.LSTMCell(input_dim + param_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 2)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

    def forward(self, x_hist, params, pred_len: int):
        _, (h_n, c_n) = self.encoder(x_hist)
        h = h_n[-1] + self.param_proj(params)
        c = c_n[-1]
        cur = x_hist[:, -1, :]
        preds = []
        for _ in range(pred_len):
            inp = torch.cat([cur, params], dim=-1)
            h, c = self.decoder_cell(inp, (h, c))
            cur = self.out(h)
            preds.append(cur.unsqueeze(1))
        return torch.cat(preds, dim=1)


class PINNNet(nn.Module):
    def __init__(self, hidden: int = 128, depth: int = 4):
        super().__init__()
        layers = []
        in_dim = 6  # t, beta, gamma, omega, theta0, phi0
        dims = [in_dim] + [hidden] * depth + [2]
        for i in range(len(dims) - 2):
            layers += [nn.Linear(dims[i], dims[i + 1]), nn.Tanh()]
        layers += [nn.Linear(dims[-2], dims[-1])]
        self.net = nn.Sequential(*layers)

    def forward(self, t, params, ic):
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        x = torch.cat([t, params, ic], dim=-1)
        return self.net(x)
