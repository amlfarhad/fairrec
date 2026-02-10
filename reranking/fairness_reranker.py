"""Fairness-aware reranking to ensure equitable supplier exposure.

After a base model generates a ranked list, this module adjusts the ranking
to meet fairness constraints while minimizing loss of relevance.

The key insight: you can often improve fairness significantly with minimal
accuracy cost. A small change in ranking (e.g., swapping positions 5 and 8)
barely affects user experience but can dramatically improve supplier equity.

Reference: Singh & Joachims, 2018 — "Fairness of Exposure in Rankings"
"""

import numpy as np
from collections import Counter


def fairness_constrained_rerank(
    candidate_items,
    relevance_scores,
    item_groups,
    target_distribution,
    top_k=10,
    tolerance=0.1,
):
    """Rerank to match target group exposure distribution.

    Uses a greedy approach: at each position, select the highest-scoring item
    whose group is most under-represented relative to the target.

    Args:
        candidate_items: Array of item IDs.
        relevance_scores: Array of relevance scores.
        item_groups: Dict mapping item_id -> group_id.
        target_distribution: Dict mapping group_id -> target proportion.
        top_k: Number of items to select.
        tolerance: How far from target proportions is acceptable.

    Returns:
        List of reranked item IDs.
    """
    if len(candidate_items) == 0:
        return []

    # Sort candidates by relevance (descending)
    sorted_indices = np.argsort(relevance_scores)[::-1]
    sorted_items = [candidate_items[i] for i in sorted_indices]
    sorted_scores = [relevance_scores[i] for i in sorted_indices]

    selected = []
    group_counts = Counter()
    remaining = list(range(len(sorted_items)))

    for pos in range(min(top_k, len(sorted_items))):
        if not remaining:
            break

        # Find which group is most under-represented
        current_total = pos + 1
        group_deficits = {}
        for group, target_prop in target_distribution.items():
            actual_prop = group_counts.get(group, 0) / current_total
            group_deficits[group] = target_prop - actual_prop

        # Among remaining candidates, find the best one
        best_idx = None
        best_score = -float("inf")

        for idx in remaining:
            item = sorted_items[idx]
            group = item_groups.get(item, "unknown")
            deficit = group_deficits.get(group, 0)

            # Composite score: relevance + fairness bonus for under-represented groups
            composite = sorted_scores[idx] + deficit * (1.0 / (pos + 1))

            if composite > best_score:
                best_score = composite
                best_idx = idx

        if best_idx is not None:
            item = sorted_items[best_idx]
            selected.append(item)
            group = item_groups.get(item, "unknown")
            group_counts[group] += 1
            remaining.remove(best_idx)

    return selected


def proportional_rerank(
    candidate_items,
    relevance_scores,
    item_groups,
    top_k=10,
):
    """Rerank using proportional representation.

    Ensures each group gets representation proportional to its share
    of the candidate pool, while maximizing relevance within each slot.

    Args:
        candidate_items: Array of item IDs.
        relevance_scores: Array of relevance scores.
        item_groups: Dict mapping item_id -> group_id.
        top_k: Number of items to select.

    Returns:
        List of reranked item IDs.
    """
    # Group candidates by their group
    groups = {}
    for i, item in enumerate(candidate_items):
        group = item_groups.get(item, "unknown")
        if group not in groups:
            groups[group] = []
        groups[group].append((item, relevance_scores[i]))

    # Sort within each group by relevance
    for group in groups:
        groups[group].sort(key=lambda x: x[1], reverse=True)

    # Calculate proportional slots
    total_candidates = len(candidate_items)
    group_slots = {}
    for group, items in groups.items():
        proportion = len(items) / total_candidates
        group_slots[group] = max(1, round(proportion * top_k))

    # Round-robin selection, respecting proportions
    selected = []
    group_pointers = {g: 0 for g in groups}

    # First pass: fill proportional slots
    for group, n_slots in group_slots.items():
        items = groups[group]
        for _ in range(min(n_slots, len(items))):
            ptr = group_pointers[group]
            if ptr < len(items):
                selected.append(items[ptr][0])
                group_pointers[group] += 1

    # Fill remaining slots with highest-scoring remaining items
    if len(selected) < top_k:
        all_remaining = []
        for group, items in groups.items():
            ptr = group_pointers[group]
            all_remaining.extend(items[ptr:])
        all_remaining.sort(key=lambda x: x[1], reverse=True)
        for item, _ in all_remaining:
            if len(selected) >= top_k:
                break
            if item not in selected:
                selected.append(item)

    return selected[:top_k]
