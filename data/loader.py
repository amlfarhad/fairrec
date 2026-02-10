"""Data loading and preprocessing for MovieLens dataset."""

import os
import zipfile
import requests
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import config


MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"


def download_movielens():
    """Download the MovieLens 25M dataset if not already present."""
    if os.path.exists(config.MOVIELENS_DIR):
        print("MovieLens 25M already downloaded.")
        return

    os.makedirs(config.DATA_DIR, exist_ok=True)
    zip_path = os.path.join(config.DATA_DIR, "ml-25m.zip")

    if not os.path.exists(zip_path):
        print("Downloading MovieLens 25M (~250MB)...")
        response = requests.get(MOVIELENS_URL, stream=True)
        response.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download complete.")

    print("Extracting...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(config.DATA_DIR)
    os.remove(zip_path)
    print("Done.")


def load_ratings(sample_frac=None):
    """Load and preprocess the ratings data.

    Args:
        sample_frac: Optional fraction of data to sample (for faster experimentation).

    Returns:
        DataFrame with columns: user_id, item_id, rating, timestamp.
    """
    path = os.path.join(config.MOVIELENS_DIR, "ratings.csv")
    if not os.path.exists(path):
        download_movielens()

    df = pd.read_csv(path)
    df.columns = ["user_id", "item_id", "rating", "timestamp"]

    if sample_frac and sample_frac < 1.0:
        users = df["user_id"].unique()
        sampled_users = np.random.RandomState(config.RANDOM_SEED).choice(
            users, size=int(len(users) * sample_frac), replace=False
        )
        df = df[df["user_id"].isin(sampled_users)]

    # Filter users with enough interactions
    user_counts = df["user_id"].value_counts()
    active_users = user_counts[user_counts >= config.MIN_INTERACTIONS].index
    df = df[df["user_id"].isin(active_users)]

    # Re-index users and items to contiguous IDs
    df["user_id"] = pd.Categorical(df["user_id"]).codes
    df["item_id"] = pd.Categorical(df["item_id"]).codes

    return df.reset_index(drop=True)


def load_movies():
    """Load movie metadata.

    Returns:
        DataFrame with movie_id, title, genres.
    """
    path = os.path.join(config.MOVIELENS_DIR, "movies.csv")
    if not os.path.exists(path):
        download_movielens()

    df = pd.read_csv(path)
    df.columns = ["movie_id", "title", "genres"]
    return df


def train_test_split_temporal(df, test_ratio=None):
    """Split data by leaving out each user's most recent interactions.

    This simulates a realistic evaluation where we predict future behavior.

    Args:
        df: Ratings DataFrame.
        test_ratio: Fraction of each user's ratings to hold out.

    Returns:
        Tuple of (train_df, test_df).
    """
    if test_ratio is None:
        test_ratio = config.TEST_RATIO

    df = df.sort_values("timestamp")

    train_list = []
    test_list = []

    for user_id, group in df.groupby("user_id"):
        n_test = max(1, int(len(group) * test_ratio))
        train_list.append(group.iloc[:-n_test])
        test_list.append(group.iloc[-n_test:])

    train_df = pd.concat(train_list).reset_index(drop=True)
    test_df = pd.concat(test_list).reset_index(drop=True)

    return train_df, test_df


def compute_item_popularity(train_df):
    """Compute item popularity statistics from training data.

    Returns:
        Series mapping item_id -> interaction count.
    """
    return train_df["item_id"].value_counts()


def compute_propensity_scores(train_df):
    """Estimate propensity scores for IPS debiasing.

    Uses item popularity as a proxy for observation probability.
    Items that are more popular are more likely to be observed (rated),
    regardless of actual user preference.

    Returns:
        Series mapping item_id -> propensity score (clipped).
    """
    item_counts = train_df["item_id"].value_counts()
    total_users = train_df["user_id"].nunique()

    # Propensity = P(item is observed) ≈ count / total_users
    propensity = item_counts / total_users
    propensity = propensity.clip(
        lower=config.PROPENSITY_CLIP_MIN,
        upper=config.PROPENSITY_CLIP_MAX,
    )
    return propensity
