"""Collaborative filtering recommenders (user-based and item-based)."""

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

from .base import BaseRecommender
import config


class UserBasedCF(BaseRecommender):
    """User-based collaborative filtering with cosine similarity."""

    def __init__(self, k_neighbors=50):
        super().__init__(name="UserCF")
        self.k = k_neighbors
        self.user_item_matrix = None
        self.user_similarity = None
        self.global_mean = 0.0
        self.user_means = None
        self.n_users = 0
        self.n_items = 0

    def fit(self, train_df):
        self.n_users = train_df["user_id"].max() + 1
        self.n_items = train_df["item_id"].max() + 1

        self.user_item_matrix = csr_matrix(
            (train_df["rating"].values,
             (train_df["user_id"].values, train_df["item_id"].values)),
            shape=(self.n_users, self.n_items),
        )

        self.global_mean = train_df["rating"].mean()

        # Mean-center ratings per user
        user_sums = np.array(self.user_item_matrix.sum(axis=1)).flatten()
        user_counts = np.array((self.user_item_matrix > 0).sum(axis=1)).flatten()
        user_counts[user_counts == 0] = 1
        self.user_means = user_sums / user_counts

        self.is_fitted = True

    def _get_neighbors(self, user_id):
        """Find k most similar users using cosine similarity (computed lazily)."""
        user_vec = self.user_item_matrix[user_id]
        similarities = cosine_similarity(user_vec, self.user_item_matrix).flatten()
        similarities[user_id] = -1  # Exclude self
        top_k = np.argsort(similarities)[::-1][: self.k]
        return top_k, similarities[top_k]

    def predict(self, user_id, item_ids):
        if user_id >= self.n_users:
            return np.full(len(item_ids), self.global_mean)

        neighbors, sim_scores = self._get_neighbors(user_id)
        scores = []

        for item_id in item_ids:
            if item_id >= self.n_items:
                scores.append(self.global_mean)
                continue

            # Weighted average of neighbor ratings (mean-centered)
            neighbor_ratings = np.array(
                self.user_item_matrix[neighbors, item_id].todense()
            ).flatten()
            mask = neighbor_ratings > 0

            if mask.sum() == 0:
                scores.append(self.user_means[user_id])
                continue

            neighbor_means = self.user_means[neighbors[mask]]
            centered = neighbor_ratings[mask] - neighbor_means
            weights = sim_scores[mask]

            if np.abs(weights).sum() == 0:
                scores.append(self.user_means[user_id])
            else:
                pred = self.user_means[user_id] + np.dot(weights, centered) / np.abs(weights).sum()
                scores.append(pred)

        return np.array(scores)


class ItemBasedCF(BaseRecommender):
    """Item-based collaborative filtering with cosine similarity."""

    def __init__(self, k_neighbors=30):
        super().__init__(name="ItemCF")
        self.k = k_neighbors
        self.user_item_matrix = None
        self.item_similarity = None
        self.global_mean = 0.0
        self.n_users = 0
        self.n_items = 0

    def fit(self, train_df):
        self.n_users = train_df["user_id"].max() + 1
        self.n_items = train_df["item_id"].max() + 1

        self.user_item_matrix = csr_matrix(
            (train_df["rating"].values,
             (train_df["user_id"].values, train_df["item_id"].values)),
            shape=(self.n_users, self.n_items),
        )

        self.global_mean = train_df["rating"].mean()

        # Precompute item-item similarity (on transposed matrix)
        # For large datasets, compute lazily per-item instead
        item_matrix = self.user_item_matrix.T.tocsr()
        self.item_similarity = cosine_similarity(item_matrix)
        np.fill_diagonal(self.item_similarity, 0)

        self.is_fitted = True

    def predict(self, user_id, item_ids):
        if user_id >= self.n_users:
            return np.full(len(item_ids), self.global_mean)

        user_ratings = np.array(
            self.user_item_matrix[user_id].todense()
        ).flatten()
        rated_items = np.where(user_ratings > 0)[0]

        if len(rated_items) == 0:
            return np.full(len(item_ids), self.global_mean)

        scores = []
        for item_id in item_ids:
            if item_id >= self.n_items:
                scores.append(self.global_mean)
                continue

            sims = self.item_similarity[item_id, rated_items]
            top_k_idx = np.argsort(sims)[::-1][: self.k]

            top_sims = sims[top_k_idx]
            top_ratings = user_ratings[rated_items[top_k_idx]]

            if np.abs(top_sims).sum() == 0:
                scores.append(self.global_mean)
            else:
                scores.append(np.dot(top_sims, top_ratings) / np.abs(top_sims).sum())

        return np.array(scores)
