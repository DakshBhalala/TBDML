"""Embeddings and tokenizers for tabular numerical and categorical features."""

from typing import Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn


class PiecewiseLinearEncoding(nn.Module):
    """Vectorized Piecewise Linear Encoding (PLE) for continuous features.
    
    Transforms scalar continuous values into soft piecewise linear representations
    relative to quantile bin edges fitted on training data.
    """

    def __init__(
        self,
        bin_edges: Dict[str, np.ndarray],
        num_cols: List[str],
        embed_dim: int = 16,
    ):
        super().__init__()
        self.num_cols = num_cols
        self.embed_dim = embed_dim
        self.num_features = len(num_cols)

        # Register bin edges buffers
        edge_tensors = []
        num_bins_list = []
        for col in num_cols:
            edges = torch.from_numpy(bin_edges[col]).float()
            edge_tensors.append(edges)
            num_bins_list.append(len(edges) - 1)

        self.num_bins_list = num_bins_list
        # Pad edge tensors to identical length for vectorized operations
        max_edges = max(len(e) for e in edge_tensors) if edge_tensors else 2
        padded_edges = torch.zeros(self.num_features, max_edges)
        edge_masks = torch.zeros(self.num_features, max_edges, dtype=torch.bool)

        for i, edges in enumerate(edge_tensors):
            padded_edges[i, : len(edges)] = edges
            edge_masks[i, : len(edges)] = True

        self.register_buffer("padded_edges", padded_edges)  # (F, max_edges)
        self.register_buffer("edge_masks", edge_masks)

        # Trainable projection from PLE bins to embedding dimension
        self.projections = nn.ModuleList([
            nn.Linear(num_bins_list[i], embed_dim, bias=False)
            for i in range(self.num_features)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Numerical tensor of shape (batch_size, num_features)
            
        Returns:
            Encoded tensor of shape (batch_size, num_features, embed_dim)
        """
        if self.num_features == 0:
            return torch.empty(x.shape[0], 0, self.embed_dim, device=x.device)

        batch_size = x.shape[0]
        outputs = []

        for i in range(self.num_features):
            feat_val = x[:, i : i + 1]  # (B, 1)
            edges = self.padded_edges[i, self.edge_masks[i]]  # (num_edges,)
            b_left = edges[:-1]  # (num_bins,)
            b_right = edges[1:]  # (num_bins,)
            span = torch.clamp(b_right - b_left, min=1e-6)

            # Piecewise linear formula:
            # e_j = clamp((x - b_left) / (b_right - b_left), 0, 1)
            ple_val = torch.clamp((feat_val - b_left) / span, 0.0, 1.0)  # (B, num_bins)
            proj = self.projections[i](ple_val)  # (B, embed_dim)
            outputs.append(proj)

        return torch.stack(outputs, dim=1)  # (B, F, embed_dim)


class CategoricalEmbedding(nn.Module):
    """Entity embedding tables for categorical features with OOV token handling."""

    def __init__(
        self,
        cardinalities: Dict[str, int],
        cat_cols: List[str],
        embed_dim: int = 16,
    ):
        super().__init__()
        self.cat_cols = cat_cols
        self.embed_dim = embed_dim
        self.num_features = len(cat_cols)

        self.embeddings = nn.ModuleList([
            nn.Embedding(
                num_embeddings=max(2, cardinalities[col]),
                embedding_dim=embed_dim,
                padding_idx=0,
            )
            for col in cat_cols
        ])

    def forward(self, x_cat: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x_cat: Categorical indices tensor of shape (batch_size, num_cat_features)
            
        Returns:
            Embedded tensor of shape (batch_size, num_cat_features, embed_dim)
        """
        if self.num_features == 0:
            return torch.empty(x_cat.shape[0], 0, self.embed_dim, device=x_cat.device)

        outputs = []
        for i in range(self.num_features):
            col_idx = x_cat[:, i]
            emb = self.embeddings[i](col_idx)  # (B, embed_dim)
            outputs.append(emb)

        return torch.stack(outputs, dim=1)  # (B, F_cat, embed_dim)
