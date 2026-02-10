"""Recommendation evaluation metrics — standard and beyond-accuracy.

Standard metrics tell you if the model is accurate.
Beyond-accuracy metrics tell you if the recommendations are actually USEFUL.
FAANG systems optimize for both simultaneously.
"""

import numpy as np
from collections import Counter


# ── Standard Ranking Metrics ──────────────────────────────────────────────


def precision_at_k(recommended, relevant, k=10):
    """Fraction of top-K recommendations that are relevant."""
    rec_set = set(recommended[:k])
    rel_set = set(relevant)
    if len(rec_set) == 0:
        return 0.0
    return len(rec_set & rel_set) / len(rec_set)


def recall_at_k(recommended, relevant, k=10):
    """Fraction of relevant items captured in top-K."""
    rec_set = set(recommended[:k])
    rel_set = set(relevant)
    if len(rel_set) == 0:
        return 0.0
    return len(rec_set & rel_set) / len(rel_set)


def ndcg_at_k(recommended, relevant, k=10):
    """Normalized Discounted Cumulative Gain at K.

    Accounts for the position of relevant items — finding a relevant item
    at position 1 is worth more than finding it at position 10.
    """
    rel_set = set(relevant)
    dcg = 0.0
    for i, item in enumerate(recommended[:k]):
        if item in rel_set:
            dcg += 1.0 / np.log2(i + 2)  # +2 because i is 0-indexed

    # Ideal DCG: all relevant items at the top
    ideal_hits = min(len(rel_set), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))

    if idcg == 0:
        return 0.0
    return dcg / idcg


def hit_rate_at_k(recommended, relevant, k=10):
    """Binary: did at least one relevant item appear in top-K?"""
    return 1.0 if len(set(recommended[:k]) & set(relevant)) > 0 else 0.0


def mrr(recommended, relevant):
    """Mean Reciprocal Rank — position of the first relevant item."""
    rel_set = set(relevant)
    for i, item in enumerate(recommended):
        if item in rel_set:
            return 1.0 / (i + 1)
    return 0.0


# ── Beyond-Accuracy Metrics ──────────────────────────────────────────────
# These are what separate a tutorial project from a production-grade system.


def catalog_coverage(all_recommendations, total_items):
    """Fraction of the item catalog that appears in any recommendation list.

    Low coverage means the system only recommends a small set of items —
    the "blockbuster" problem. FAANG systems target >30% coverage.
    """
    recommended_items = set()
    for rec_list in all_recommendations:
        recommended_items.update(rec_list)
    return len(recommended_items) / total_items if total_items > 0 else 0.0


def gini_index(all_recommendations):
    """Gini coefficient of item recommendation frequency.

    0 = perfectly equal exposure across items (every item recommended equally)
    1 = maximum inequality (one item gets all recommendations)

    FAANG systems monitor this to prevent "winner-take-all" dynamics.
    """
    item_counts = Counter()
    for rec_list in all_recommendations:
        item_counts.update(rec_list)

    if len(item_counts) == 0:
        return 0.0

    counts = sorted(item_counts.values())
    n = len(counts)
    total = sum(counts)

    if total == 0:
        return 0.0

    numerator = sum((2 * (i + 1) - n - 1) * count for i, count in enumerate(counts))
    return numerator / (n * total)


def intra_list_similarity(recommended, item_features):
    """Average pairwise similarity within a recommendation list.

    Lower ILS = more diverse recommendations.
    item_features: dict mapping item_id -> feature vector (e.g., genre vector).
    """
    if len(recommended) < 2:
        return 0.0

    vectors = []
    for item in recommended:
        if item in item_features:
            vectors.append(item_features[item])

    if len(vectors) < 2:
        return 0.0

    vectors = np.array(vectors)
    # Cosine similarity
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normalized = vectors / norms
    sim_matrix = normalized @ normalized.T

    # Average upper triangle (exclude diagonal)
    n = len(vectors)
    upper_sum = (sim_matrix.sum() - n) / 2  # Subtract diagonal, halve
    n_pairs = n * (n - 1) / 2
    return upper_sum / n_pairs if n_pairs > 0 else 0.0


def novelty(recommended, item_popularity, total_users):
    """Average self-information of recommended items.

    Recommending popular items is easy. Recommending relevant AND unpopular
    items demonstrates the model actually learned user preferences.

    Higher novelty = recommending items users wouldn't find on their own.
    """
    if len(recommended) == 0 or total_users == 0:
        return 0.0

    scores = []
    for item in recommended:
        pop = item_popularity.get(item, 1)
        # Self-information: -log2(popularity)
        scores.append(-np.log2(pop / total_users))

    return np.mean(scores)


def popularity_bias(recommended, item_popularity, popularity_bins=5):
    """Distribution of recommendations across item popularity groups.

    Returns dict mapping popularity_group -> fraction of recommendations.
    A fair system would have roughly equal representation across groups.
    """
    if len(item_popularity) == 0:
        return {}

    # Create popularity bins
    all_items = sorted(item_popularity.keys())
    pop_values = np.array([item_popularity[i] for i in all_items])
    thresholds = np.percentile(pop_values, np.linspace(0, 100, popularity_bins + 1))

    item_to_bin = {}
    for item in all_items:
        pop = item_popularity[item]
        for b in range(popularity_bins):
            if pop <= thresholds[b + 1] or b == popularity_bins - 1:
                item_to_bin[item] = b
                break

    bin_counts = Counter()
    for item in recommended:
        if item in item_to_bin:
            bin_counts[item_to_bin[item]] += 1

    total = sum(bin_counts.values())
    if total == 0:
        return {b: 0.0 for b in range(popularity_bins)}

    return {b: bin_counts.get(b, 0) / total for b in range(popularity_bins)}


# ── Aggregated Evaluation ─────────────────────────────────────────────────


def evaluate_recommendations(
    user_recommendations,
    user_relevant_items,
    all_recommendations,
    total_items,
    item_popularity,
    total_users,
    item_features=None,
    k=10,
):
    """Compute all metrics for a set of user recommendations.

    Args:
        user_recommendations: Dict mapping user_id -> list of recommended item IDs.
        user_relevant_items: Dict mapping user_id -> list of relevant item IDs.
        all_recommendations: List of all recommendation lists.
        total_items: Total number of items in catalog.
        item_popularity: Dict mapping item_id -> interaction count.
        total_users: Total number of users.
        item_features: Optional dict mapping item_id -> feature vector.
        k: Top-K cutoff.

    Returns:
        Dict of metric_name -> value.
    """
    precisions, recalls, ndcgs, hits, mrrs = [], [], [], [], []
    novelties, ilss = [], []

    for user_id, recommended in user_recommendations.items():
        relevant = user_relevant_items.get(user_id, [])
        if len(relevant) == 0:
            continue

        precisions.append(precision_at_k(recommended, relevant, k))
        recalls.append(recall_at_k(recommended, relevant, k))
        ndcgs.append(ndcg_at_k(recommended, relevant, k))
        hits.append(hit_rate_at_k(recommended, relevant, k))
        mrrs.append(mrr(recommended, relevant))
        novelties.append(novelty(recommended[:k], item_popularity, total_users))

        if item_features:
            ilss.append(intra_list_similarity(recommended[:k], item_features))

    results = {
        f"Precision@{k}": np.mean(precisions) if precisions else 0.0,
        f"Recall@{k}": np.mean(recalls) if recalls else 0.0,
        f"NDCG@{k}": np.mean(ndcgs) if ndcgs else 0.0,
        f"HitRate@{k}": np.mean(hits) if hits else 0.0,
        "MRR": np.mean(mrrs) if mrrs else 0.0,
        "Coverage": catalog_coverage(all_recommendations, total_items),
        "Gini": gini_index(all_recommendations),
        "Novelty": np.mean(novelties) if novelties else 0.0,
    }

    if ilss:
        results["ILS (lower=diverse)"] = np.mean(ilss)

    return results
