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

        if self.num_features > 0:
            edge_tensors = [torch.from_numpy(bin_edges[col]).float() for col in num_cols]
            max_edges = max(len(e) for e in edge_tensors)
            num_bins = max(1, max_edges - 1)
            self.num_bins = num_bins

            b_left_all = []
            b_right_all = []
            for e in edge_tensors:
                if len(e) < max_edges:
                    pad_e = torch.linspace(e[0], e[-1], max_edges)
                else:
                    pad_e = e[:max_edges]
                b_left_all.append(pad_e[:-1])
                b_right_all.append(pad_e[1:])

            b_left = torch.stack(b_left_all, dim=0).unsqueeze(0)  # (1, F, num_bins)
            b_right = torch.stack(b_right_all, dim=0).unsqueeze(0)  # (1, F, num_bins)
            span = torch.clamp(b_right - b_left, min=1e-6)

            self.register_buffer("b_left", b_left)
            self.register_buffer("span", span)

            # Single 3D tensor parameter for all feature projections: (F, num_bins, embed_dim)
            self.weight = nn.Parameter(torch.empty(self.num_features, num_bins, embed_dim))
            nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        else:
            self.num_bins = 0
            self.register_buffer("b_left", torch.empty(0))
            self.register_buffer("span", torch.empty(0))
            self.weight = nn.Parameter(torch.empty(0, 0, embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Numerical tensor of shape (batch_size, num_features)
            
        Returns:
            Encoded tensor of shape (batch_size, num_features, embed_dim)
        """
        if self.num_features == 0:
            return torch.empty(x.shape[0], 0, self.embed_dim, device=x.device)

        # Vectorized soft binning across all features in a single kernel
        x_exp = x.unsqueeze(-1)  # (N, F, 1)
        ple_val = torch.clamp((x_exp - self.b_left) / self.span, 0.0, 1.0)  # (N, F, num_bins)

        # Vectorized einsum projection: (N, F, num_bins) x (F, num_bins, embed_dim) -> (N, F, embed_dim)
        return torch.einsum("nfk,fkd->nfd", ple_val, self.weight)


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
