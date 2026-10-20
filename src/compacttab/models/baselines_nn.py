"""Standard tabular neural network baselines: TabularMLP and TabularResNet."""

from typing import Dict, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from compacttab.data.schema import DatasetSchema, TaskType
from compacttab.models.embeddings import CategoricalEmbedding


class TabularMLP(nn.Module):
    """Standard Multilayer Perceptron baseline for tabular data."""

    def __init__(
        self,
        schema: DatasetSchema,
        cat_cardinalities: Dict[str, int],
        hidden_dims: Optional[list] = None,
        embed_dim: int = 16,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.schema = schema
        hidden_dims = hidden_dims or [128, 128]

        # Categorical embedding
        self.has_cat = len(self.schema.categorical_cols) > 0
        if self.has_cat:
            self.cat_embed = CategoricalEmbedding(
                cardinalities=cat_cardinalities,
                cat_cols=self.schema.categorical_cols,
                embed_dim=embed_dim,
            )
        else:
            self.cat_embed = None

        in_dim = len(self.schema.numerical_cols) + len(self.schema.categorical_cols) * embed_dim
        if in_dim == 0:
            in_dim = 1

        # Determine out_dim
        if self.schema.task_type == TaskType.BINARY:
            out_dim = 1
        elif self.schema.task_type == TaskType.MULTICLASS:
            out_dim = self.schema.num_classes if self.schema.num_classes else 2
        else:
            out_dim = 1

        layers = []
        curr_dim = in_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            curr_dim = h_dim

        layers.append(nn.Linear(curr_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        num_vals: torch.Tensor,
        num_masks: torch.Tensor,
        cat_indices: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        tokens = [num_vals]
        if self.has_cat and self.cat_embed is not None:
            c_emb = self.cat_embed(cat_indices).flatten(start_dim=1)
            tokens.append(c_emb)

        x = torch.cat(tokens, dim=1) if len(tokens) > 1 else tokens[0]
        logits = self.net(x)

        if self.training:
            return logits

        if self.schema.task_type == TaskType.BINARY:
            return torch.sigmoid(logits)
        elif self.schema.task_type == TaskType.MULTICLASS:
            return F.softmax(logits, dim=-1)
        else:
            return logits


class TabularResNet(nn.Module):
    """Standard ResNet architecture for tabular data."""

    def __init__(
        self,
        schema: DatasetSchema,
        cat_cardinalities: Dict[str, int],
        hidden_dim: int = 128,
        num_blocks: int = 3,
        embed_dim: int = 16,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.schema = schema
        self.hidden_dim = hidden_dim

        self.has_cat = len(self.schema.categorical_cols) > 0
        if self.has_cat:
            self.cat_embed = CategoricalEmbedding(
                cardinalities=cat_cardinalities,
                cat_cols=self.schema.categorical_cols,
                embed_dim=embed_dim,
            )
        else:
            self.cat_embed = None

        in_dim = len(self.schema.numerical_cols) + len(self.schema.categorical_cols) * embed_dim
        if in_dim == 0:
            in_dim = 1

        self.input_layer = nn.Linear(in_dim, hidden_dim)

        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Dropout(dropout),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Dropout(dropout),
            )
            for _ in range(num_blocks)
        ])

        if self.schema.task_type == TaskType.BINARY:
            out_dim = 1
        elif self.schema.task_type == TaskType.MULTICLASS:
            out_dim = self.schema.num_classes if self.schema.num_classes else 2
        else:
            out_dim = 1

        self.final_norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, out_dim)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        num_vals: torch.Tensor,
        num_masks: torch.Tensor,
        cat_indices: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        tokens = [num_vals]
        if self.has_cat and self.cat_embed is not None:
            c_emb = self.cat_embed(cat_indices).flatten(start_dim=1)
            tokens.append(c_emb)

        x = torch.cat(tokens, dim=1) if len(tokens) > 1 else tokens[0]
        h = self.input_layer(x)

        for block in self.blocks:
            h = h + block(h)

        h = self.final_norm(h)
        logits = self.head(h)

        if self.training:
            return logits

        if self.schema.task_type == TaskType.BINARY:
            return torch.sigmoid(logits)
        elif self.schema.task_type == TaskType.MULTICLASS:
            return F.softmax(logits, dim=-1)
        else:
            return logits
