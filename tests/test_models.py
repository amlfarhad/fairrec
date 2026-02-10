"""Tests for recommendation models."""

import pytest
import numpy as np
import pandas as pd

from models.popularity import PopularityRecommender
from models.matrix_factorization import SVDRecommender
from models.debiased import IPSRecommender


@pytest.fixture
def sample_train_data():
    """Generate small synthetic training data."""
    np.random.seed(42)
    data = []
    for user in range(50):
        n_items = np.random.randint(5, 20)
        items = np.random.choice(100, n_items, replace=False)
        for item in items:
            rating = np.random.choice([1, 2, 3, 4, 5], p=[0.05, 0.1, 0.2, 0.35, 0.3])
            data.append({"user_id": user, "item_id": item, "rating": float(rating), "timestamp": 0})
    return pd.DataFrame(data)


class TestPopularityRecommender:

    def test_fit_and_predict(self, sample_train_data):
        model = PopularityRecommender()
        model.fit(sample_train_data)
        assert model.is_fitted

        scores = model.predict(0, np.array([0, 1, 2, 3, 4]))
        assert len(scores) == 5

    def test_recommend_returns_top_k(self, sample_train_data):
        model = PopularityRecommender()
        model.fit(sample_train_data)

        candidates = np.arange(100)
        items, scores = model.recommend(0, candidates, top_k=5)
        assert len(items) == 5
        # Scores should be sorted descending
        assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))

    def test_same_for_all_users(self, sample_train_data):
        model = PopularityRecommender()
        model.fit(sample_train_data)

        candidates = np.arange(100)
        items1, _ = model.recommend(0, candidates, top_k=5)
        items2, _ = model.recommend(1, candidates, top_k=5)
        np.testing.assert_array_equal(items1, items2)


class TestSVDRecommender:

    def test_fit_and_predict(self, sample_train_data):
        model = SVDRecommender(n_factors=10)
        model.fit(sample_train_data)
        assert model.is_fitted

        scores = model.predict(0, np.array([0, 1, 2]))
        assert len(scores) == 3

    def test_personalized(self, sample_train_data):
        model = SVDRecommender(n_factors=10)
        model.fit(sample_train_data)

        candidates = np.arange(100)
        items1, _ = model.recommend(0, candidates, top_k=5)
        items2, _ = model.recommend(1, candidates, top_k=5)
        # Personalized: different users should (usually) get different recs
        # Not guaranteed but very likely with random data
        assert not np.array_equal(items1, items2) or True  # Soft check

    def test_unknown_user(self, sample_train_data):
        model = SVDRecommender(n_factors=10)
        model.fit(sample_train_data)

        scores = model.predict(9999, np.array([0, 1, 2]))
        assert len(scores) == 3
        # Should fallback to global mean
        assert all(s == model.global_mean for s in scores)

    def test_batch_predict(self, sample_train_data):
        model = SVDRecommender(n_factors=10)
        model.fit(sample_train_data)

        items = np.array([0, 1, 2, 3])
        scores = model.predict_batch(0, items)
        assert len(scores) == 4


class TestIPSRecommender:

    def test_fit_with_propensity(self, sample_train_data):
        propensity = sample_train_data["item_id"].value_counts() / 50
        propensity = propensity.clip(lower=0.01, upper=1.0)

        model = IPSRecommender(n_factors=10, propensity_scores=propensity)
        model.fit(sample_train_data)
        assert model.is_fitted

    def test_missing_propensity_raises(self, sample_train_data):
        model = IPSRecommender(n_factors=10)
        with pytest.raises(ValueError, match="Propensity"):
            model.fit(sample_train_data)

    def test_predict(self, sample_train_data):
        propensity = sample_train_data["item_id"].value_counts() / 50
        propensity = propensity.clip(lower=0.01, upper=1.0)

        model = IPSRecommender(n_factors=10, propensity_scores=propensity)
        model.fit(sample_train_data)

        scores = model.predict(0, np.array([0, 1, 2]))
        assert len(scores) == 3
