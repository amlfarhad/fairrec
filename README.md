# fairrec — Multi-Objective Recommendation Engine

Most recommendation tutorials optimize for one thing: prediction accuracy. Real production recommender systems at companies like Netflix, Spotify, and Amazon must simultaneously balance **relevance**, **diversity**, **fairness**, and **business value**.

This project benchmarks 5 progressively complex recommendation models and then demonstrates how post-hoc reranking techniques (MMR, IPS debiasing, fairness-constrained reranking) improve the *quality* of recommendations beyond accuracy.

## Why This Matters

| Problem | Why It's Hard | How This Project Addresses It |
|---------|---------------|-------------------------------|
| **Popularity bias** | Models trained on logged data amplify existing biases — popular items get more exposure → more clicks → more training signal | Inverse Propensity Scoring (IPS) reweights training data to correct for observation bias |
| **Filter bubbles** | Optimizing for accuracy leads to homogeneous recommendations (10 similar action movies) | MMR reranking balances relevance with intra-list diversity |
| **Supplier fairness** | In marketplaces, recommendations determine which sellers/creators get revenue | Fairness-constrained reranking ensures equitable exposure across item groups |
| **Misleading evaluation** | RMSE tells you nothing about whether recommendations are useful | Beyond-accuracy metrics: Coverage, Gini, Novelty, ILS, Calibration |

## Models Implemented

| Model | Type | Description |
|-------|------|-------------|
| `PopularityRecommender` | Baseline | Global popularity weighted by rating quality |
| `UserBasedCF` | Collaborative | k-NN with cosine similarity on user-item matrix |
| `ItemBasedCF` | Collaborative | k-NN with precomputed item-item similarity |
| `SVDRecommender` | Matrix Factorization | Truncated SVD with user/item biases |
| `NeuralCF` | Deep Learning | GMF + MLP hybrid (He et al., 2017) using PyTorch |
| `IPSRecommender` | Causal/Debiased | SVD with Inverse Propensity Scoring to correct selection bias |

## Reranking Strategies

- **MMR (Maximal Marginal Relevance)**: Iteratively selects items that are relevant AND different from already-selected items. Tunable λ parameter controls the relevance-diversity tradeoff.
- **Fairness-Constrained Reranking**: Adjusts rankings to match target group exposure distributions while minimizing relevance loss.
- **Proportional Reranking**: Ensures item groups receive representation proportional to their catalog share.

## Evaluation Metrics

### Standard
Precision@K, Recall@K, NDCG@K, Hit Rate@K, MRR

### Beyond-Accuracy
- **Catalog Coverage**: What fraction of items ever gets recommended?
- **Gini Index**: How equally is recommendation exposure distributed?
- **Intra-List Similarity (ILS)**: How diverse is each recommendation list?
- **Novelty**: Are we recommending items users wouldn't find on their own?
- **Popularity Bias**: Distribution of recs across popularity groups

### Fairness
- **Exposure Fairness**: Position-weighted exposure across supplier groups
- **Calibration Error**: KL-divergence between user preferences and recommendation genre distribution
- **Equal Opportunity Gap**: Performance disparity across user groups

## Quick Start

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Run benchmark (samples 5% of users for speed)
python pipeline.py --sample 0.05 --top-k 10

# Include Neural CF (slower)
python pipeline.py --sample 0.05 --neural

# Full dataset
python pipeline.py --sample 1.0 --top-k 10
```

The pipeline will automatically download MovieLens 25M on first run (~250MB).

## Testing

```bash
pytest tests/ -v
```

## Project Structure

```
fairrec/
├── pipeline.py                     # End-to-end benchmark runner
├── config.py                       # Hyperparameters and settings
├── data/
│   └── loader.py                   # MovieLens download, preprocessing, IPS propensity
├── models/
│   ├── base.py                     # Abstract recommender interface
│   ├── popularity.py               # Popularity baseline
│   ├── collaborative.py            # User-CF and Item-CF
│   ├── matrix_factorization.py     # Truncated SVD
│   ├── neural_cf.py                # Neural Collaborative Filtering (PyTorch)
│   └── debiased.py                 # IPS-weighted SVD
├── evaluation/
│   ├── metrics.py                  # Standard + beyond-accuracy metrics
│   └── fairness.py                 # Supplier exposure, calibration, equal opportunity
├── reranking/
│   ├── mmr.py                      # Maximal Marginal Relevance
│   └── fairness_reranker.py        # Fairness-constrained and proportional reranking
└── tests/
    ├── test_metrics.py
    ├── test_models.py
    └── test_reranking.py
```

## References

- He et al., 2017 — *Neural Collaborative Filtering*
- Schnabel et al., 2016 — *Recommendations as Treatments: Debiasing Learning and Evaluation*
- Carbonell & Goldstein, 1998 — *The Use of MMR for Reordering Documents*
- Singh & Joachims, 2018 — *Fairness of Exposure in Rankings*
- Steck, 2018 — *Calibrated Recommendations*
