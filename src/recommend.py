"""Recommendation Generation Module.

Generates Top-K recommendations for a user from any model's score matrix, and
produces human-readable explanations using the item-item similarity matrix
(``ItemCF.SIM``): *"recommended because you liked A and B."*
"""
from __future__ import annotations
from collections import defaultdict
import numpy as np
import torch


def build_user_history(tr_u, tr_i, tr_r):
    """user_idx -> list of (item_idx, rating) from the training set."""
    hist = defaultdict(list)
    for u, i, r in zip(tr_u, tr_i, tr_r):
        hist[int(u)].append((int(i), float(r)))
    return hist


def top_k(score_matrix, seen_u, seen_i, K=10):
    """Top-K item indices per user, with train-seen movies masked out."""
    S = score_matrix.clone()
    S[seen_u, seen_i] = -1e9
    return torch.topk(S, K, dim=1).indices.cpu().numpy()


def explain(u_idx, i_idx, SIM, user_hist, m_ids, title_of,
            global_mean, topn=3, min_rating=4):
    """Return the user's highly-rated movies most responsible for recommending i.

    Scores each previously-liked movie j by sim(i, j) * (rating - mean) and
    returns the top contributors as (title, similarity, rating).
    """
    sims = SIM[i_idx]
    contrib = [(float(sims[j]) * (r - global_mean), j, r)
               for j, r in user_hist[u_idx] if r >= min_rating and float(sims[j]) > 0]
    contrib.sort(reverse=True)
    out = []
    for _, j, r in contrib[:topn]:
        title = title_of.get(int(m_ids[j]), f"movie#{int(m_ids[j])}")
        out.append((title, round(float(SIM[i_idx][j]), 3), r))
    return out


def recommend_for_user(u_idx, topk_idx, rel_sets, SIM, user_hist, m_ids, title_of, global_mean):
    """Assemble a readable Top-K block for one user (titles + why + relevance hit)."""
    lines = []
    for rank, i_idx in enumerate(topk_idx[u_idx], 1):
        title = title_of.get(int(m_ids[i_idx]), f"movie#{int(m_ids[i_idx])}")
        why = explain(u_idx, int(i_idx), SIM, user_hist, m_ids, title_of, global_mean)
        why_s = "; ".join(f"{nm} (sim {s})" for nm, s, _ in why) or "popular among similar users"
        hit = "  <-- relevant in test" if int(i_idx) in rel_sets[u_idx] else ""
        lines.append(f"  {rank:2d}. {title:45.45s}  because: {why_s}{hit}")
    return "\n".join(lines)
