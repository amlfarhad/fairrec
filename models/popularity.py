"""Popularity-based baseline recommender."""

import numpy as np
import pandas as pd

from .base import BaseRecommender


class PopularityRecommender(BaseRecommender):
    """Recommends items based on global popularity.

    This is the simplest baseline — it recommends the same items to everyone.
    A good recommendation system must significantly outperform this.
    """

    def __init__(self):
        super().__init__(name="Popularity")
        self.item_scores = None

    def fit(self, train_df):
        # Score = average rating weighted by log(count + 1) to balance
        # quality and popularity (avoids recommending items with 1 perfect rating)
        stats = train_df.groupby("item_id").agg(
            mean_rating=("rating", "mean"),
            count=("rating", "count"),
        )
        stats["score"] = stats["mean_rating"] * np.log1p(stats["count"])
        self.item_scores = stats["score"]
        self.is_fitted = True

    def predict(self, user_id, item_ids):
        return np.array([
            self.item_scores.get(iid, 0.0) for iid in item_ids
        ])
