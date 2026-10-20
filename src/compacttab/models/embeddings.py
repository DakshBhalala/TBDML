"""Embeddings and tokenizers for tabular numerical and categorical features."""

import numpy as np
import torch
import torch.nn as nn


class PiecewiseLinearEncoding(nn.Module):
    """Vectorized Piecewise Linear Encoding (PLE) with Periodic Fourier Modulations.

    Transforms scalar continuous values into soft piecewise linear representations
    relative to quantile bin edges fitted on training data, augmented with learnable
    periodic Fourier basis functions for capturing sharp boundaries and high-frequency patterns.
    """

    def __init__(
        self,
        bin_edges: dict[str, np.ndarray],
        num_cols: list[str],
        embed_dim: int = 16,
        num_freqs: int = 8,
        use_periodic: bool = True,
    ):
        super().__init__()
        self.num_cols = num_cols
        self.embed_dim = embed_dim
        self.num_features = len(num_cols)
        self.num_freqs = num_freqs
        self.use_periodic = use_periodic and (self.num_features > 0)

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

            if self.use_periodic:
                self.frequencies = nn.Parameter(torch.randn(self.num_features, self.num_freqs) * 0.5)
                self.fourier_weight = nn.Parameter(
                    torch.randn(self.num_features, 2 * self.num_freqs, embed_dim) * 0.02
                )
            else:
                self.register_parameter("frequencies", None)
                self.register_parameter("fourier_weight", None)
        else:
            self.num_bins = 0
            self.register_buffer("b_left", torch.empty(0))
            self.register_buffer("span", torch.empty(0))
            self.weight = nn.Parameter(torch.empty(0, 0, embed_dim))
            self.register_parameter("frequencies", None)
            self.register_parameter("fourier_weight", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Numerical tensor of shape (batch_size, num_features)

        Returns:
            Encoded tensor of shape (batch_size, num_features, embed_dim)
        """
        if self.num_features == 0:
            return torch.empty(x.shape[0], 0, self.embed_dim, device=x.device)

        # 1. Vectorized soft binning across all features in a single kernel
        x_exp = x.unsqueeze(-1)  # (N, F, 1)
        ple_val = torch.clamp((x_exp - self.b_left) / self.span, 0.0, 1.0)  # (N, F, num_bins)
        out = torch.einsum("nfk,fkd->nfd", ple_val, self.weight)

        # 2. Learnable Periodic Fourier Modulation
        if self.use_periodic and self.frequencies is not None and self.fourier_weight is not None:
            angles = 2.0 * np.pi * x.unsqueeze(-1) * self.frequencies.unsqueeze(0)
            fourier_features = torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)
            fourier_emb = torch.einsum("nfk,fkd->nfd", fourier_features, self.fourier_weight)
            out = out + fourier_emb

        return out


class CategoricalEmbedding(nn.Module):
    """Entity embedding tables for categorical features with OOV token handling."""

    def __init__(
        self,
        cardinalities: dict[str, int],
        cat_cols: list[str],
        embed_dim: int = 16,
    ):
        super().__init__()
        self.cat_cols = cat_cols
        self.embed_dim = embed_dim
        self.num_features = len(cat_cols)

        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(
                    num_embeddings=max(2, cardinalities[col]),
                    embedding_dim=embed_dim,
                    padding_idx=0,
                )
                for col in cat_cols
            ]
        )

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
