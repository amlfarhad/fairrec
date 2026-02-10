"""Tests for evaluation metrics."""

import pytest
import numpy as np

from evaluation.metrics import (
    precision_at_k, recall_at_k, ndcg_at_k, hit_rate_at_k, mrr,
    catalog_coverage, gini_index, intra_list_similarity, novelty,
)


class TestStandardMetrics:

    def test_precision_perfect(self):
        assert precision_at_k([1, 2, 3], [1, 2, 3], k=3) == 1.0

    def test_precision_none_relevant(self):
        assert precision_at_k([1, 2, 3], [4, 5, 6], k=3) == 0.0

    def test_precision_partial(self):
        assert precision_at_k([1, 2, 3, 4], [1, 3], k=4) == 0.5

    def test_precision_respects_k(self):
        # Only first 2 items considered; item 3 is relevant but beyond k
        assert precision_at_k([1, 2, 3], [3], k=2) == 0.0

    def test_recall_perfect(self):
        assert recall_at_k([1, 2, 3], [1, 2], k=3) == 1.0

    def test_recall_partial(self):
        assert recall_at_k([1, 2], [1, 2, 3, 4], k=2) == 0.5

    def test_recall_no_relevant(self):
        assert recall_at_k([1, 2], [], k=2) == 0.0

    def test_ndcg_perfect_ordering(self):
        # All relevant items at the top
        assert ndcg_at_k([1, 2, 3], [1, 2, 3], k=3) == 1.0

    def test_ndcg_no_relevant(self):
        assert ndcg_at_k([1, 2, 3], [4, 5], k=3) == 0.0

    def test_ndcg_position_matters(self):
        # Relevant item at position 1 > position 3
        ndcg_first = ndcg_at_k([1, 2, 3], [1], k=3)
        ndcg_last = ndcg_at_k([2, 3, 1], [1], k=3)
        assert ndcg_first > ndcg_last

    def test_hit_rate_hit(self):
        assert hit_rate_at_k([1, 2, 3], [3], k=3) == 1.0

    def test_hit_rate_miss(self):
        assert hit_rate_at_k([1, 2, 3], [4], k=3) == 0.0

    def test_mrr_first_position(self):
        assert mrr([1, 2, 3], [1]) == 1.0

    def test_mrr_third_position(self):
        assert mrr([1, 2, 3], [3]) == pytest.approx(1 / 3)

    def test_mrr_no_hit(self):
        assert mrr([1, 2, 3], [4]) == 0.0


class TestBeyondAccuracyMetrics:

    def test_coverage_full(self):
        recs = [[1, 2], [3, 4], [5]]
        assert catalog_coverage(recs, 5) == 1.0

    def test_coverage_partial(self):
        recs = [[1, 2], [1, 2]]
        assert catalog_coverage(recs, 10) == 0.2

    def test_coverage_empty(self):
        assert catalog_coverage([], 10) == 0.0

    def test_gini_perfect_equality(self):
        # Every item recommended exactly once
        recs = [[0], [1], [2], [3]]
        gini = gini_index(recs)
        assert gini == pytest.approx(0.0, abs=0.01)

    def test_gini_inequality(self):
        # One item dominates
        recs = [[0, 0, 0], [0, 0, 0], [0, 1]]
        gini = gini_index(recs)
        assert gini > 0.3

    def test_ils_identical_items(self):
        features = {0: [1, 0, 0], 1: [1, 0, 0], 2: [1, 0, 0]}
        ils = intra_list_similarity([0, 1, 2], features)
        assert ils == pytest.approx(1.0, abs=0.01)

    def test_ils_diverse_items(self):
        features = {0: [1, 0, 0], 1: [0, 1, 0], 2: [0, 0, 1]}
        ils = intra_list_similarity([0, 1, 2], features)
        assert ils == pytest.approx(0.0, abs=0.01)

    def test_novelty_popular_items(self):
        # Popular items have low novelty
        popularity = {0: 1000, 1: 1000}
        n = novelty([0, 1], popularity, 1000)
        assert n == pytest.approx(0.0, abs=0.01)

    def test_novelty_rare_items(self):
        popularity = {0: 1, 1: 1}
        n = novelty([0, 1], popularity, 1000)
        assert n > 5  # High self-information
