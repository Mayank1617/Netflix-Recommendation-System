"""Model Training Pipeline.

Four recommendation models of increasing sophistication, sharing a common
index space (see ``data.build_index``):

1. ``fit_baseline``  — global mean + regularized user/item biases
2. ``ItemCF``        — adjusted-cosine item-based collaborative filtering (GPU)
3. ``MF``            — biased matrix factorization (SVD-style), trained with SGD
4. ``NCF``           — neural collaborative filtering (MLP over embeddings)

Each model can produce a full (n_users x n_items) score matrix used by the
evaluation and recommendation modules.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# Model 1 — Bias baseline
# --------------------------------------------------------------------------- #
def fit_baseline(tr_u, tr_i, tr_r, n_users, n_items, lam: float = 10.0):
    """Closed-form regularized biases. Returns (mu, b_u, b_i)."""
    mu = float(tr_r.mean())
    tdf = pd.DataFrame({"u": tr_u, "i": tr_i, "r": tr_r})
    it = tdf.groupby("i")["r"]
    b_i = ((it.sum() - mu * it.count()) / (lam + it.count())).reindex(range(n_items)).fillna(0.0).to_numpy()
    tdf["res"] = tdf["r"].to_numpy() - mu - b_i[tr_i]
    ut = tdf.groupby("u")["res"]
    b_u = (ut.sum() / (lam + ut.count())).reindex(range(n_users)).fillna(0.0).to_numpy()
    return mu, b_u.astype(np.float32), b_i.astype(np.float32)


def baseline_score_matrix(mu, b_u, b_i, device="cpu"):
    bu = torch.tensor(b_u, device=device).unsqueeze(1)
    bi = torch.tensor(b_i, device=device).unsqueeze(0)
    return mu + bu + bi


# --------------------------------------------------------------------------- #
# Model 2 — Item-based CF (adjusted cosine)
# --------------------------------------------------------------------------- #
class ItemCF:
    """Adjusted-cosine item-based CF. Builds an item-item similarity matrix and a
    full predicted-rating matrix on the GPU. ``SIM`` also powers explanations."""

    def __init__(self, device="cpu"):
        self.device = device
        self.SIM = None
        self.PRED = None
        self.user_mean = None

    def fit(self, tr_u, tr_i, tr_r, n_users, n_items):
        dev = self.device
        user_sum = np.zeros(n_users, np.float32); np.add.at(user_sum, tr_u, tr_r)
        user_cnt = np.bincount(tr_u, minlength=n_users).astype(np.float32)
        self.user_mean = (user_sum / np.maximum(user_cnt, 1)).astype(np.float32)
        centered = (tr_r - self.user_mean[tr_u]).astype(np.float32)

        Iu = torch.zeros((n_items, n_users), device=dev)
        Iu[torch.tensor(tr_i), torch.tensor(tr_u)] = torch.tensor(centered, device=dev)
        norm = Iu.norm(dim=1, keepdim=True).clamp_min(1e-8)
        Iu_n = Iu / norm
        self.SIM = Iu_n @ Iu_n.t()
        self.SIM.fill_diagonal_(0)
        del Iu_n

        P = Iu.t().contiguous(); del Iu
        mask = (P != 0).float()
        num = P @ self.SIM
        den = (mask @ self.SIM.abs()).clamp_min(1e-8)
        self.PRED = torch.tensor(self.user_mean, device=dev).unsqueeze(1) + num / den
        del P, mask, num, den
        if dev == "cuda":
            torch.cuda.empty_cache()
        return self

    def score_matrix(self):
        return self.PRED


# --------------------------------------------------------------------------- #
# Model 3 — Matrix Factorization (biased SVD)
# --------------------------------------------------------------------------- #
class MF(nn.Module):
    def __init__(self, n_users, n_items, mu, k=64):
        super().__init__()
        self.P = nn.Embedding(n_users, k); self.Q = nn.Embedding(n_items, k)
        self.bu = nn.Embedding(n_users, 1); self.bi = nn.Embedding(n_items, 1)
        nn.init.normal_(self.P.weight, std=0.05); nn.init.normal_(self.Q.weight, std=0.05)
        nn.init.zeros_(self.bu.weight); nn.init.zeros_(self.bi.weight)
        self.mu = mu

    def forward(self, u, i):
        return self.mu + self.bu(u).squeeze(1) + self.bi(i).squeeze(1) + (self.P(u) * self.Q(i)).sum(1)

    def score_matrix(self):
        with torch.no_grad():
            return self.mu + self.bu.weight + self.bi.weight.t() + self.P.weight @ self.Q.weight.t()


# --------------------------------------------------------------------------- #
# Model 4 — Neural Collaborative Filtering
# --------------------------------------------------------------------------- #
class NCF(nn.Module):
    def __init__(self, n_users, n_items, mu, k=32):
        super().__init__()
        self.uemb = nn.Embedding(n_users, k); self.iemb = nn.Embedding(n_items, k)
        self.mlp = nn.Sequential(nn.Linear(2 * k, 128), nn.ReLU(), nn.Dropout(0.2),
                                 nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
        nn.init.normal_(self.uemb.weight, std=0.05); nn.init.normal_(self.iemb.weight, std=0.05)
        self.mu = mu

    def forward(self, u, i):
        x = torch.cat([self.uemb(u), self.iemb(i)], 1)
        return self.mu + self.mlp(x).squeeze(1)

    def score_matrix(self, n_users, n_items, device, chunk=512):
        S = torch.empty((n_users, n_items), device=device)
        items = torch.arange(n_items, device=device)
        with torch.no_grad():
            for u0 in range(0, n_users, chunk):
                us = torch.arange(u0, min(u0 + chunk, n_users), device=device)
                uu = us.repeat_interleave(n_items); ii = items.repeat(len(us))
                S[u0:u0 + len(us)] = self(uu, ii).view(len(us), n_items)
        return S


def train_torch_model(model, tr_u, tr_i, tr_r, device, lr=0.01, weight_decay=2e-5,
                      epochs=15, batch_size=100_000, seed=42, verbose=True):
    """Generic mini-batch SGD (Adam) trainer for MF / NCF. Reproducible via seed."""
    torch.manual_seed(seed); np.random.seed(seed)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    U = torch.tensor(tr_u, device=device); I = torch.tensor(tr_i, device=device)
    R = torch.tensor(tr_r, device=device); n = len(R)
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        for s in range(0, n, batch_size):
            idx = perm[s:s + batch_size]
            opt.zero_grad()
            loss = ((model(U[idx], I[idx]) - R[idx]) ** 2).mean()
            loss.backward(); opt.step()
        if verbose and ((ep + 1) % 5 == 0 or ep == 0):
            print(f"  epoch {ep + 1}/{epochs}  train MSE {loss.item():.4f}")
    return model
