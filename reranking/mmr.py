"""Maximal Marginal Relevance (MMR) reranking for diversity.

The core tension in recommendations: users want relevant items, but they also
want DIVERSE items. Showing 10 nearly identical movies is accurate but useless.

MMR solves this by iteratively selecting items that are both relevant AND
different from items already selected. The lambda parameter controls the tradeoff.

Reference: Carbonell & Goldstein, 1998 — "The Use of MMR, Diversity-Based
Reranking for Reordering Documents and Producing Summaries"
"""

import numpy as np


def mmr_rerank(candidate_items, relevance_scores, item_features, top_k=10, lambda_param=0.5):
    """Rerank candidates using Maximal Marginal Relevance.

    MMR(i) = λ * Relevance(i) - (1-λ) * max_j∈S Similarity(i, j)

    At each step, select the item that maximizes the MMR score:
    high relevance AND low similarity to already-selected items.

    Args:
        candidate_items: Array of item IDs.
        relevance_scores: Array of relevance scores (from base model).
        item_features: Dict mapping item_id -> feature vector.
        top_k: Number of items to select.
        lambda_param: Trade-off (0=pure diversity, 1=pure relevance).

    Returns:
        List of reranked item IDs.
    """
    if len(candidate_items) == 0:
        return []

    # Normalize relevance scores to [0, 1]
    scores = np.array(relevance_scores, dtype=float)
    score_min, score_max = scores.min(), scores.max()
    if score_max > score_min:
        scores = (scores - score_min) / (score_max - score_min)
    else:
        scores = np.ones_like(scores)

    # Build feature matrix for candidates
    feature_vecs = {}
    for item in candidate_items:
        if item in item_features:
            feature_vecs[item] = np.array(item_features[item], dtype=float)

    selected = []
    remaining = list(range(len(candidate_items)))

    for _ in range(min(top_k, len(candidate_items))):
        if not remaining:
            break

        best_idx = None
        best_mmr = -float("inf")

        for idx in remaining:
            item = candidate_items[idx]
            relevance = scores[idx]

            # Compute max similarity to already-selected items
            if selected and item in feature_vecs:
                max_sim = 0.0
                item_vec = feature_vecs[item]
                item_norm = np.linalg.norm(item_vec)

                for sel_item in selected:
                    if sel_item in feature_vecs:
                        sel_vec = feature_vecs[sel_item]
                        sel_norm = np.linalg.norm(sel_vec)
                        if item_norm > 0 and sel_norm > 0:
                            sim = np.dot(item_vec, sel_vec) / (item_norm * sel_norm)
                            max_sim = max(max_sim, sim)
            else:
                max_sim = 0.0

            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim

            if mmr_score > best_mmr:
                best_mmr = mmr_score
                best_idx = idx

        if best_idx is not None:
            selected.append(candidate_items[best_idx])
            remaining.remove(best_idx)

    return selected
