"""Fairness metrics for recommendation systems.

In a marketplace (Amazon, Spotify, Uber), recommendations don't just affect users.
They determine which SUPPLIERS get visibility and revenue. A fair system ensures
that item providers receive proportional exposure.

This module implements supplier-side fairness metrics that FAANG companies
actively monitor and optimize for.
"""

import numpy as np
from collections import Counter


def exposure_fairness(all_recommendations, item_groups, k=10):
    """Measure how fairly exposure is distributed across item groups.

    Item groups could represent:
    - Artists on Spotify (indie vs. major label)
    - Sellers on Amazon (small business vs. large retailer)
    - Content creators on YouTube (new vs. established)

    Args:
        all_recommendations: List of recommendation lists.
        item_groups: Dict mapping item_id -> group_id.
        k: Top-K cutoff.

    Returns:
        Dict with group exposure shares and fairness metrics.
    """
    group_exposure = Counter()
    total_exposure = 0.0

    for rec_list in all_recommendations:
        for rank, item in enumerate(rec_list[:k]):
            # Exposure weighted by position (position bias model)
            exposure = 1.0 / np.log2(rank + 2)
            group = item_groups.get(item, "unknown")
            group_exposure[group] += exposure
            total_exposure += exposure

    if total_exposure == 0:
        return {"group_shares": {}, "max_disparity": 0.0, "entropy": 0.0}

    group_shares = {
        g: exp / total_exposure for g, exp in group_exposure.items()
    }

    # Max disparity: ratio of most-exposed to least-exposed group
    shares = list(group_shares.values())
    max_disparity = max(shares) / min(shares) if min(shares) > 0 else float("inf")

    # Entropy: higher = more equal distribution
    entropy = -sum(s * np.log2(s) for s in shares if s > 0)
    max_entropy = np.log2(len(shares)) if len(shares) > 1 else 1.0

    return {
        "group_shares": group_shares,
        "max_disparity": max_disparity,
        "normalized_entropy": entropy / max_entropy if max_entropy > 0 else 0.0,
    }


def calibration_error(user_recommendations, user_genre_dist, item_genres, k=10):
    """Measure how well recommendations reflect user's genre preferences.

    Calibrated recommendations match the distribution of genres a user
    has historically engaged with. If a user watches 70% action and 30% comedy,
    their recommendations should roughly follow that distribution.

    Reference: Steck, 2018 — "Calibrated Recommendations"

    Args:
        user_recommendations: Dict mapping user_id -> recommended item list.
        user_genre_dist: Dict mapping user_id -> {genre: proportion}.
        item_genres: Dict mapping item_id -> list of genres.
        k: Top-K cutoff.

    Returns:
        Average KL-divergence between user preference and recommendation distributions.
    """
    kl_divs = []

    for user_id, rec_list in user_recommendations.items():
        if user_id not in user_genre_dist:
            continue

        target_dist = user_genre_dist[user_id]
        all_genres = set(target_dist.keys())

        # Build recommendation genre distribution
        rec_genre_counts = Counter()
        for item in rec_list[:k]:
            genres = item_genres.get(item, [])
            for g in genres:
                rec_genre_counts[g] += 1
                all_genres.add(g)

        total = sum(rec_genre_counts.values())
        if total == 0:
            continue

        rec_dist = {g: rec_genre_counts.get(g, 0) / total for g in all_genres}

        # KL divergence: D_KL(target || rec)
        # Add small epsilon for smoothing
        eps = 1e-10
        kl = sum(
            target_dist.get(g, eps) * np.log(target_dist.get(g, eps) / (rec_dist.get(g, eps)))
            for g in all_genres
        )
        kl_divs.append(max(0, kl))

    return np.mean(kl_divs) if kl_divs else 0.0


def equal_opportunity_gap(user_recommendations, user_relevant, user_groups, k=10):
    """Measure if the system performs equally well for different user groups.

    User groups could represent demographics, activity levels, etc.
    A fair system should have similar hit rates across groups.

    Args:
        user_recommendations: Dict user_id -> recommended items.
        user_relevant: Dict user_id -> relevant items.
        user_groups: Dict user_id -> group_id.
        k: Top-K cutoff.

    Returns:
        Dict with per-group hit rates and the gap between best and worst.
    """
    group_hits = {}

    for user_id, rec_list in user_recommendations.items():
        group = user_groups.get(user_id, "default")
        relevant = user_relevant.get(user_id, [])

        if len(relevant) == 0:
            continue

        hit = 1.0 if len(set(rec_list[:k]) & set(relevant)) > 0 else 0.0

        if group not in group_hits:
            group_hits[group] = []
        group_hits[group].append(hit)

    group_rates = {g: np.mean(hits) for g, hits in group_hits.items()}

    if len(group_rates) < 2:
        return {"group_hit_rates": group_rates, "opportunity_gap": 0.0}

    rates = list(group_rates.values())
    return {
        "group_hit_rates": group_rates,
        "opportunity_gap": max(rates) - min(rates),
    }
