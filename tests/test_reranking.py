"""Tests for reranking modules."""

import pytest
import numpy as np

from reranking.mmr import mmr_rerank
from reranking.fairness_reranker import fairness_constrained_rerank, proportional_rerank


class TestMMR:

    def test_pure_relevance(self):
        """lambda=1.0 should return items in relevance order."""
        items = np.array([0, 1, 2, 3, 4])
        scores = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        features = {i: [float(i == j) for j in range(5)] for i in range(5)}

        result = mmr_rerank(items, scores, features, top_k=3, lambda_param=1.0)
        assert result == [0, 1, 2]

    def test_diversity_changes_order(self):
        """lambda<1.0 should change the ranking for diversity."""
        items = np.array([0, 1, 2, 3])
        scores = np.array([4.0, 3.9, 3.8, 1.0])
        # Items 0, 1, 2 are very similar; item 3 is different
        features = {
            0: [1.0, 0.0],
            1: [0.99, 0.01],
            2: [0.98, 0.02],
            3: [0.0, 1.0],
        }

        result = mmr_rerank(items, scores, features, top_k=3, lambda_param=0.3)
        # Item 3 should be promoted despite lower relevance
        assert 3 in result

    def test_empty_input(self):
        result = mmr_rerank(np.array([]), np.array([]), {}, top_k=5)
        assert result == []

    def test_k_larger_than_candidates(self):
        items = np.array([0, 1])
        scores = np.array([2.0, 1.0])
        features = {0: [1, 0], 1: [0, 1]}

        result = mmr_rerank(items, scores, features, top_k=10)
        assert len(result) == 2


class TestFairnessReranking:

    def test_constrained_rerank_balances_groups(self):
        items = np.array([0, 1, 2, 3, 4, 5])
        scores = np.array([6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
        groups = {0: "A", 1: "A", 2: "A", 3: "B", 4: "B", 5: "B"}
        target = {"A": 0.5, "B": 0.5}

        result = fairness_constrained_rerank(items, scores, groups, target, top_k=4)

        group_a = sum(1 for i in result if groups.get(i) == "A")
        group_b = sum(1 for i in result if groups.get(i) == "B")
        assert group_a == 2
        assert group_b == 2

    def test_proportional_rerank(self):
        items = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        scores = np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1], dtype=float)
        # 7 items from group A, 3 from group B
        groups = {i: "A" if i < 7 else "B" for i in range(10)}

        result = proportional_rerank(items, scores, groups, top_k=10)
        assert len(result) == 10

    def test_empty_candidates(self):
        result = fairness_constrained_rerank(
            np.array([]), np.array([]), {}, {"A": 0.5}, top_k=5
        )
        assert result == []
