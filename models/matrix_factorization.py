"""Matrix Factorization recommender using truncated SVD."""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

from .base import BaseRecommender
import config


class SVDRecommender(BaseRecommender):
    """Recommender using truncated SVD on the user-item interaction matrix.

    Decomposes R ≈ U * Σ * V^T, then predicts as dot product in latent space.
    """

    def __init__(self, n_factors=None, regularization=None):
        super().__init__(name="SVD")
        self.n_factors = n_factors or config.NUM_FACTORS
        self.reg = regularization or config.REGULARIZATION
        self.user_factors = None
        self.item_factors = None
        self.global_mean = 0.0
        self.user_bias = None
        self.item_bias = None
        self.n_users = 0
        self.n_items = 0

    def fit(self, train_df):
        self.n_users = train_df["user_id"].max() + 1
        self.n_items = train_df["item_id"].max() + 1

        # Build sparse matrix
        matrix = csr_matrix(
            (train_df["rating"].values,
             (train_df["user_id"].values, train_df["item_id"].values)),
            shape=(self.n_users, self.n_items),
            dtype=np.float32,
        )

        self.global_mean = train_df["rating"].mean()

        # Compute biases
        user_sums = np.array(matrix.sum(axis=1)).flatten()
        user_counts = np.array((matrix > 0).sum(axis=1)).flatten()
        user_counts[user_counts == 0] = 1
        self.user_bias = user_sums / user_counts - self.global_mean

        item_sums = np.array(matrix.sum(axis=0)).flatten()
        item_counts = np.array((matrix > 0).sum(axis=0)).flatten()
        item_counts[item_counts == 0] = 1
        self.item_bias = item_sums / item_counts - self.global_mean

        # SVD on mean-centered matrix
        k = min(self.n_factors, min(self.n_users, self.n_items) - 1)
        U, sigma, Vt = svds(matrix.asfptype(), k=k)

        self.user_factors = U * np.sqrt(sigma)
        self.item_factors = Vt.T * np.sqrt(sigma)

        self.is_fitted = True

    def predict(self, user_id, item_ids):
        if user_id >= self.n_users:
            return np.full(len(item_ids), self.global_mean)

        user_vec = self.user_factors[user_id]
        u_bias = self.user_bias[user_id]

        scores = []
        for item_id in item_ids:
            if item_id >= self.n_items:
                scores.append(self.global_mean)
                continue

            item_vec = self.item_factors[item_id]
            i_bias = self.item_bias[item_id]
            pred = self.global_mean + u_bias + i_bias + np.dot(user_vec, item_vec)
            scores.append(pred)

        return np.array(scores)

    def predict_batch(self, user_id, item_ids):
        """Vectorized prediction for a batch of items."""
        if user_id >= self.n_users:
            return np.full(len(item_ids), self.global_mean)

        valid_mask = item_ids < self.n_items
        scores = np.full(len(item_ids), self.global_mean)

        valid_items = item_ids[valid_mask]
        user_vec = self.user_factors[user_id]
        item_vecs = self.item_factors[valid_items]

        scores[valid_mask] = (
            self.global_mean
            + self.user_bias[user_id]
            + self.item_bias[valid_items]
            + item_vecs @ user_vec
        )
        return scores
