"""Configuration for the fairrec recommendation engine."""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MOVIELENS_DIR = os.path.join(DATA_DIR, "ml-25m")

# Model defaults
TOP_K = 10
EMBEDDING_DIM = 64
LEARNING_RATE = 1e-3
BATCH_SIZE = 1024
EPOCHS = 20
NUM_FACTORS = 50  # SVD latent factors
REGULARIZATION = 0.02

# Evaluation
TEST_RATIO = 0.2
MIN_INTERACTIONS = 20  # Minimum ratings per user to include

# Fairness
POPULARITY_BINS = 5  # Number of item popularity groups for fairness analysis
DIVERSITY_LAMBDA = 0.3  # MMR trade-off parameter (0=pure relevance, 1=pure diversity)

# IPS Debiasing
PROPENSITY_CLIP_MIN = 0.01  # Clip propensity scores to avoid extreme weights
PROPENSITY_CLIP_MAX = 1.0

RANDOM_SEED = 42
