"""End-to-end pipeline: load data, train models, evaluate, rerank, compare."""

import argparse
import time
import numpy as np
import pandas as pd

from data.loader import (
    load_ratings, train_test_split_temporal,
    compute_item_popularity, compute_propensity_scores,
)
from models.popularity import PopularityRecommender
from models.collaborative import UserBasedCF, ItemBasedCF
from models.matrix_factorization import SVDRecommender
from models.neural_cf import NeuralCF
from models.debiased import IPSRecommender
from evaluation.metrics import evaluate_recommendations
from reranking.mmr import mmr_rerank
from reranking.fairness_reranker import fairness_constrained_rerank
import config


def build_genre_features(train_df, movies_df=None):
    """Build item genre feature vectors for diversity metrics.

    Returns:
        Dict mapping item_id -> genre binary vector.
    """
    # If no movie metadata, return empty (metrics that need it will be skipped)
    if movies_df is None:
        try:
            from data.loader import load_movies
            movies_df = load_movies()
        except Exception:
            return {}

    all_genres = set()
    for genres_str in movies_df["genres"].dropna():
        all_genres.update(genres_str.split("|"))
    all_genres = sorted(all_genres)
    genre_to_idx = {g: i for i, g in enumerate(all_genres)}

    item_features = {}
    for _, row in movies_df.iterrows():
        vec = np.zeros(len(all_genres))
        for g in str(row["genres"]).split("|"):
            if g in genre_to_idx:
                vec[genre_to_idx[g]] = 1.0
        item_features[row["movie_id"]] = vec

    return item_features


def generate_recommendations(model, test_users, train_df, all_items, top_k):
    """Generate recommendations for test users."""
    train_items_per_user = train_df.groupby("user_id")["item_id"].apply(set).to_dict()
    user_recs = {}

    for user_id in test_users:
        seen = train_items_per_user.get(user_id, set())
        candidates = np.array([i for i in all_items if i not in seen])

        if len(candidates) == 0:
            user_recs[user_id] = []
            continue

        rec_items, rec_scores = model.recommend(user_id, candidates, top_k=top_k)
        user_recs[user_id] = list(rec_items)

    return user_recs


def run_pipeline(sample_frac=0.05, top_k=10, run_neural=False):
    """Run the full evaluation pipeline.

    Args:
        sample_frac: Fraction of users to sample (for speed).
        top_k: Number of recommendations per user.
        run_neural: Whether to train the neural model (slow).
    """
    print("=" * 70)
    print("  fairrec — Multi-Objective Recommendation Benchmark")
    print("=" * 70)

    # ── Data Loading ────────────────────────────────────────────────
    print("\n[1] Loading data...")
    ratings = load_ratings(sample_frac=sample_frac)
    train_df, test_df = train_test_split_temporal(ratings)
    item_popularity = compute_item_popularity(train_df)
    propensity = compute_propensity_scores(train_df)

    n_users = ratings["user_id"].nunique()
    n_items = ratings["item_id"].nunique()
    all_items = ratings["item_id"].unique()

    print(f"  Users: {n_users:,} | Items: {n_items:,}")
    print(f"  Train: {len(train_df):,} | Test: {len(test_df):,}")

    # Build item features for diversity metrics
    item_features = build_genre_features(train_df)

    # Relevant items per user (from test set, threshold rating >= 4)
    test_relevant = (
        test_df[test_df["rating"] >= 4.0]
        .groupby("user_id")["item_id"]
        .apply(list)
        .to_dict()
    )
    test_users = list(test_relevant.keys())[:500]  # Limit for speed

    # ── Model Training & Evaluation ─────────────────────────────────
    models = [
        PopularityRecommender(),
        SVDRecommender(),
        IPSRecommender(propensity_scores=propensity),
    ]

    if run_neural:
        models.append(NeuralCF(epochs=10))

    results = {}

    for model in models:
        print(f"\n[{model.name}] Training...")
        start = time.time()
        model.fit(train_df)
        train_time = time.time() - start
        print(f"  Trained in {train_time:.1f}s")

        print(f"  Generating recommendations...")
        user_recs = generate_recommendations(model, test_users, train_df, all_items, top_k)
        all_rec_lists = [recs for recs in user_recs.values() if recs]

        print(f"  Evaluating...")
        metrics = evaluate_recommendations(
            user_recommendations=user_recs,
            user_relevant_items=test_relevant,
            all_recommendations=all_rec_lists,
            total_items=n_items,
            item_popularity=item_popularity.to_dict(),
            total_users=n_users,
            item_features=item_features if item_features else None,
            k=top_k,
        )
        metrics["Train Time (s)"] = round(train_time, 1)
        results[model.name] = metrics

    # ── MMR Diversity Reranking ────────────────────────────────────
    if item_features:
        print(f"\n[SVD + MMR] Applying diversity reranking (λ={config.DIVERSITY_LAMBDA})...")
        svd_model = models[1]  # SVD
        mmr_recs = {}

        for user_id in test_users:
            seen = train_df[train_df["user_id"] == user_id]["item_id"].values
            candidates = np.array([i for i in all_items if i not in seen])
            if len(candidates) == 0:
                mmr_recs[user_id] = []
                continue

            scores = svd_model.predict(user_id, candidates)
            # Get top 100 candidates, then rerank top 10 with MMR
            top100_idx = np.argsort(scores)[::-1][:100]
            top100_items = candidates[top100_idx]
            top100_scores = scores[top100_idx]

            reranked = mmr_rerank(
                top100_items, top100_scores, item_features,
                top_k=top_k, lambda_param=config.DIVERSITY_LAMBDA,
            )
            mmr_recs[user_id] = reranked

        all_mmr_lists = [recs for recs in mmr_recs.values() if recs]
        mmr_metrics = evaluate_recommendations(
            user_recommendations=mmr_recs,
            user_relevant_items=test_relevant,
            all_recommendations=all_mmr_lists,
            total_items=n_items,
            item_popularity=item_popularity.to_dict(),
            total_users=n_users,
            item_features=item_features,
            k=top_k,
        )
        results["SVD+MMR"] = mmr_metrics

    # ── Results Table ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  RESULTS COMPARISON")
    print("=" * 70)

    results_df = pd.DataFrame(results).T
    for col in results_df.columns:
        if results_df[col].dtype == float:
            results_df[col] = results_df[col].round(4)

    print(results_df.to_string())
    print("\n" + "=" * 70)

    # ── Key Takeaways ──────────────────────────────────────────────
    print("\n  KEY INSIGHTS:")
    if "SVD+MMR" in results and "SVD" in results:
        svd_ndcg = results["SVD"].get(f"NDCG@{top_k}", 0)
        mmr_ndcg = results["SVD+MMR"].get(f"NDCG@{top_k}", 0)
        svd_cov = results["SVD"].get("Coverage", 0)
        mmr_cov = results["SVD+MMR"].get("Coverage", 0)
        print(f"  → MMR reranking: NDCG {svd_ndcg:.4f} → {mmr_ndcg:.4f} "
              f"| Coverage {svd_cov:.4f} → {mmr_cov:.4f}")

    if "IPS-SVD" in results and "SVD" in results:
        svd_gini = results["SVD"].get("Gini", 0)
        ips_gini = results["IPS-SVD"].get("Gini", 0)
        print(f"  → IPS debiasing: Gini {svd_gini:.4f} → {ips_gini:.4f} "
              f"(lower = more equitable)")

    print()
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="fairrec — Multi-Objective Recommendation Benchmark")
    parser.add_argument("--sample", type=float, default=0.05, help="Fraction of users to sample")
    parser.add_argument("--top-k", type=int, default=10, help="Number of recommendations")
    parser.add_argument("--neural", action="store_true", help="Include Neural CF (slower)")

    args = parser.parse_args()
    run_pipeline(sample_frac=args.sample, top_k=args.top_k, run_neural=args.neural)
