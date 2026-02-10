"""Neural Collaborative Filtering (NCF) using PyTorch.

Based on He et al., 2017 — "Neural Collaborative Filtering".
Combines Generalized Matrix Factorization (GMF) with a Multi-Layer Perceptron (MLP).
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .base import BaseRecommender
import config


class NCFModel(nn.Module):
    """Neural Collaborative Filtering model combining GMF + MLP."""

    def __init__(self, n_users, n_items, embed_dim=64, mlp_layers=(128, 64, 32)):
        super().__init__()

        # GMF pathway
        self.gmf_user_embed = nn.Embedding(n_users, embed_dim)
        self.gmf_item_embed = nn.Embedding(n_items, embed_dim)

        # MLP pathway
        self.mlp_user_embed = nn.Embedding(n_users, embed_dim)
        self.mlp_item_embed = nn.Embedding(n_items, embed_dim)

        mlp_modules = []
        input_dim = embed_dim * 2
        for hidden_dim in mlp_layers:
            mlp_modules.append(nn.Linear(input_dim, hidden_dim))
            mlp_modules.append(nn.ReLU())
            mlp_modules.append(nn.Dropout(0.2))
            input_dim = hidden_dim
        self.mlp = nn.Sequential(*mlp_modules)

        # Fusion layer
        self.output_layer = nn.Linear(embed_dim + mlp_layers[-1], 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.01)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, user_ids, item_ids):
        # GMF
        gmf_user = self.gmf_user_embed(user_ids)
        gmf_item = self.gmf_item_embed(item_ids)
        gmf_out = gmf_user * gmf_item  # Element-wise product

        # MLP
        mlp_user = self.mlp_user_embed(user_ids)
        mlp_item = self.mlp_item_embed(item_ids)
        mlp_input = torch.cat([mlp_user, mlp_item], dim=-1)
        mlp_out = self.mlp(mlp_input)

        # Combine
        concat = torch.cat([gmf_out, mlp_out], dim=-1)
        output = self.output_layer(concat).squeeze(-1)
        return output


class NeuralCF(BaseRecommender):
    """Neural Collaborative Filtering recommender."""

    def __init__(self, embed_dim=None, epochs=None, batch_size=None, lr=None):
        super().__init__(name="NeuralCF")
        self.embed_dim = embed_dim or config.EMBEDDING_DIM
        self.epochs = epochs or config.EPOCHS
        self.batch_size = batch_size or config.BATCH_SIZE
        self.lr = lr or config.LEARNING_RATE
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_users = 0
        self.n_items = 0
        self.global_mean = 0.0

    def fit(self, train_df):
        self.n_users = train_df["user_id"].max() + 1
        self.n_items = train_df["item_id"].max() + 1
        self.global_mean = train_df["rating"].mean()

        self.model = NCFModel(self.n_users, self.n_items, self.embed_dim).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        loss_fn = nn.MSELoss()

        users = torch.LongTensor(train_df["user_id"].values)
        items = torch.LongTensor(train_df["item_id"].values)
        ratings = torch.FloatTensor(train_df["rating"].values)

        dataset = TensorDataset(users, items, ratings)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            for batch_users, batch_items, batch_ratings in loader:
                batch_users = batch_users.to(self.device)
                batch_items = batch_items.to(self.device)
                batch_ratings = batch_ratings.to(self.device)

                optimizer.zero_grad()
                predictions = self.model(batch_users, batch_items)
                loss = loss_fn(predictions, batch_ratings)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(batch_users)

            avg_loss = total_loss / len(dataset)
            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1}/{self.epochs} — Loss: {avg_loss:.4f}")

        self.model.eval()
        self.is_fitted = True

    def predict(self, user_id, item_ids):
        if user_id >= self.n_users:
            return np.full(len(item_ids), self.global_mean)

        self.model.eval()
        with torch.no_grad():
            users_t = torch.LongTensor([user_id] * len(item_ids)).to(self.device)
            # Clamp item IDs to valid range
            safe_items = np.clip(item_ids, 0, self.n_items - 1)
            items_t = torch.LongTensor(safe_items).to(self.device)

            scores = self.model(users_t, items_t).cpu().numpy()

        # Zero out scores for out-of-range items
        oob_mask = item_ids >= self.n_items
        scores[oob_mask] = self.global_mean

        return scores
