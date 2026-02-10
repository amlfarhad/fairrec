"""Inverse Propensity Scoring (IPS) debiased recommender.

Standard recommenders trained on logged interaction data inherit the biases
of how that data was collected. Popular items get more exposure → more clicks
→ model learns to recommend popular items → popularity bias amplifies.

IPS corrects this by weighting each observation inversely to its probability
of being observed (propensity). An interaction with a rare item counts more
than one with a popular item, because the rare item was less likely to be seen.

References:
    - Schnabel et al., 2016 — "Recommendations as Treatments"
    - Saito, 2020 — "Unbiased Recommender Learning from Missing-Not-At-Random Data"
"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

from .base import BaseRecommender
import config


class IPSRecommender(BaseRecommender):
    """SVD-based recommender with Inverse Propensity Scoring debiasing."""

    def __init__(self, n_factors=None, propensity_scores=None):
        super().__init__(name="IPS-SVD")
        self.n_factors = n_factors or config.NUM_FACTORS
        self.propensity_scores = propensity_scores
        self.user_factors = None
        self.item_factors = None
        self.global_mean = 0.0
        self.n_users = 0
        self.n_items = 0

    def fit(self, train_df):
        self.n_users = train_df["user_id"].max() + 1
        self.n_items = train_df["item_id"].max() + 1
        self.global_mean = train_df["rating"].mean()

        if self.propensity_scores is None:
            raise ValueError(
                "Propensity scores must be provided. "
                "Use data.loader.compute_propensity_scores()."
            )

        # Compute IPS weights: w_i = 1 / P(item_i observed)
        weights = train_df["item_id"].map(
            lambda x: 1.0 / self.propensity_scores.get(x, config.PROPENSITY_CLIP_MIN)
        ).values

        # Self-normalized IPS to reduce variance
        weights = weights / weights.sum() * len(weights)

        # Build IPS-weighted interaction matrix
        weighted_ratings = train_df["rating"].values * weights

        matrix = csr_matrix(
            (weighted_ratings,
             (train_df["user_id"].values, train_df["item_id"].values)),
            shape=(self.n_users, self.n_items),
            dtype=np.float32,
        )

        # SVD on the IPS-weighted matrix
        k = min(self.n_factors, min(self.n_users, self.n_items) - 1)
        U, sigma, Vt = svds(matrix, k=k)

        self.user_factors = U * np.sqrt(sigma)
        self.item_factors = Vt.T * np.sqrt(sigma)

        self.is_fitted = True

    def predict(self, user_id, item_ids):
        if user_id >= self.n_users:
            return np.full(len(item_ids), self.global_mean)

        user_vec = self.user_factors[user_id]
        scores = []

        for item_id in item_ids:
            if item_id >= self.n_items:
                scores.append(self.global_mean)
            else:
                scores.append(np.dot(user_vec, self.item_factors[item_id]))

        return np.array(scores)
