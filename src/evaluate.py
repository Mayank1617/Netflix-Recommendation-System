"""Evaluation Scripts.

Mandatory metrics:
  * RMSE   — rating-prediction accuracy
  * MAP@10 — ranking quality (a movie is relevant if its true rating >= 3.5)

Plus Precision@K, Recall@K, NDCG@K and catalog Coverage.

The ranking procedure for every user: score all movies, mask out movies already
seen in train, take the Top-K, and compare against the relevant test movies.
"""
from __future__ import annotations
import numpy as np
import torch


def rmse(pred, true):
    return float(np.sqrt(np.mean((np.clip(pred, 1, 5) - true) ** 2)))


def build_relevance(te_u, te_i, te_r, n_users, threshold: float = 3.5):
    """Per-user set of relevant (rating >= threshold) held-out item indices."""
    rel_sets = [set() for _ in range(n_users)]
    m = te_r >= threshold
    for u, i in zip(te_u[m], te_i[m]):
        rel_sets[int(u)].add(int(i))
    return rel_sets


def evaluate_ranking(score_matrix, seen_u, seen_i, rel_sets, n_users, n_items, K=10):
    """Top-K ranking metrics from a full (n_users x n_items) score matrix.

    `seen_u`/`seen_i` are the train (user, item) index tensors used to mask
    already-seen movies so they are never recommended.
    """
    S = score_matrix.clone()
    S[seen_u, seen_i] = -1e9
    topk = torch.topk(S, K, dim=1).indices.cpu().numpy()
    del S
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    discounts = 1.0 / np.log2(np.arange(2, K + 2))
    idcg = np.cumsum(discounts)
    aps, precs, recs, ndcgs, recommended = [], [], [], [], set()
    for u in range(n_users):
        recommended.update(int(x) for x in topk[u])
        rs = rel_sets[u]
        if not rs:
            continue
        hits = ap = dcg = 0.0
        for k, item in enumerate(topk[u]):
            if int(item) in rs:
                hits += 1; ap += hits / (k + 1); dcg += discounts[k]
        denom = min(len(rs), K)
        aps.append(ap / denom); precs.append(hits / K)
        recs.append(hits / len(rs)); ndcgs.append(dcg / idcg[denom - 1])
    return dict(MAP10=float(np.mean(aps)), Prec10=float(np.mean(precs)),
                Recall10=float(np.mean(recs)), NDCG10=float(np.mean(ndcgs)),
                Coverage=len(recommended) / n_items)
