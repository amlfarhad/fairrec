"""Base class for all recommendation models."""

from abc import ABC, abstractmethod
import numpy as np


class BaseRecommender(ABC):
    """Abstract base for recommendation models."""

    def __init__(self, name="BaseRecommender"):
        self.name = name
        self.is_fitted = False

    @abstractmethod
    def fit(self, train_df):
        """Train the model on interaction data.

        Args:
            train_df: DataFrame with user_id, item_id, rating columns.
        """
        pass

    @abstractmethod
    def predict(self, user_id, item_ids):
        """Predict scores for a user on given items.

        Args:
            user_id: Target user.
            item_ids: Array of candidate item IDs.

        Returns:
            Array of predicted scores.
        """
        pass

    def recommend(self, user_id, candidate_items, top_k=10):
        """Generate top-K recommendations for a user.

        Args:
            user_id: Target user.
            candidate_items: Array of item IDs to rank.
            top_k: Number of recommendations.

        Returns:
            Array of recommended item IDs, sorted by predicted score.
        """
        if not self.is_fitted:
            raise RuntimeError(f"{self.name} must be fitted before recommending.")

        scores = self.predict(user_id, candidate_items)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return candidate_items[top_indices], scores[top_indices]
