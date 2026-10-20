"""CompactTab: State-of-the-Art Compact Tabular Deep Learning Architecture."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from compacttab.data.schema import DatasetSchema, TaskType
from compacttab.models.embeddings import CategoricalEmbedding, PiecewiseLinearEncoding
from compacttab.models.layers import BatchEnsembleLinear, GatedResidualBlock, RMSNorm


@dataclass
class CompactTabConfig:
    """Configuration dataclass for CompactTab models."""

    preset: str = "small"  # 'tiny', 'small', 'medium', or 'custom'
    hidden_dim: int = 128
    num_layers: int = 4
    num_heads: int = 8
    num_bins: int = 16
    embed_dim: int = 16
    expansion_factor: float = 2.0
    dropout: float = 0.15
    adapter_init_std: float = 0.02

    @classmethod
    def from_preset(cls, preset: str = "small", **overrides: Any) -> "CompactTabConfig":
        """Instantiate configuration from standard research presets."""
        preset = preset.lower()
        if preset == "tiny":
            # ~100k parameters
            cfg = cls(
                preset="tiny",
                hidden_dim=64,
                num_layers=2,
                num_heads=4,
                num_bins=8,
                embed_dim=12,
                expansion_factor=2.0,
                dropout=0.10,
            )
        elif preset == "small":
            # ~1M parameters
            cfg = cls(
                preset="small",
                hidden_dim=128,
                num_layers=4,
                num_heads=8,
                num_bins=16,
                embed_dim=24,
                expansion_factor=2.0,
                dropout=0.15,
            )
        elif preset == "medium":
            # ~10M parameters
            cfg = cls(
                preset="medium",
                hidden_dim=384,
                num_layers=6,
                num_heads=8,
                num_bins=32,
                embed_dim=32,
                expansion_factor=2.0,
                dropout=0.20,
            )
        else:
            cfg = cls(preset="custom")

        for k, v in overrides.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg


class CompactTab(nn.Module):
    """CompactTab neural architecture for tabular data."""

    def __init__(
        self,
        schema: DatasetSchema,
        bin_edges: dict[str, np.ndarray],
        cat_cardinalities: dict[str, int],
        config: CompactTabConfig | None = None,
    ):
        super().__init__()
        self.schema = schema
        self.config = config or CompactTabConfig.from_preset("small")
        self.num_heads = self.config.num_heads

        # Determine output dimension
        if self.schema.task_type == TaskType.BINARY:
            self.out_dim = 1
        elif self.schema.task_type == TaskType.MULTICLASS:
            self.out_dim = self.schema.num_classes if self.schema.num_classes else 2
        else:
            self.out_dim = 1

        # 1. Continuous feature tokenization with Piecewise Linear Embeddings
        self.has_numerical = len(self.schema.numerical_cols) > 0
        if self.has_numerical:
            self.ple = PiecewiseLinearEncoding(
                bin_edges=bin_edges,
                num_cols=self.schema.numerical_cols,
                embed_dim=self.config.embed_dim,
            )
            # Missingness mask projection
            self.mask_proj = nn.Linear(len(self.schema.numerical_cols), self.config.embed_dim, bias=False)
        else:
            self.ple = None
            self.mask_proj = None

        # 2. Categorical feature embeddings
        self.has_categorical = len(self.schema.categorical_cols) > 0
        if self.has_categorical:
            self.cat_embed = CategoricalEmbedding(
                cardinalities=cat_cardinalities,
                cat_cols=self.schema.categorical_cols,
                embed_dim=self.config.embed_dim,
            )
        else:
            self.cat_embed = None

        # Calculate input dimension to backbone projection
        num_feat_count = len(self.schema.numerical_cols)
        cat_feat_count = len(self.schema.categorical_cols)

        # Total flattened feature dimension
        total_tokens_dim = (num_feat_count + cat_feat_count) * self.config.embed_dim
        if self.has_numerical:
            total_tokens_dim += self.config.embed_dim  # For missingness mask projection

        if total_tokens_dim == 0:
            total_tokens_dim = 1

        # Feature projection to backbone hidden dimension
        self.input_proj = nn.Linear(total_tokens_dim, self.config.hidden_dim)

        # 3. Gated Residual Backbone
        self.blocks = nn.ModuleList(
            [
                GatedResidualBlock(
                    hidden_dim=self.config.hidden_dim,
                    expansion_factor=self.config.expansion_factor,
                    num_heads=self.config.num_heads,
                    dropout=self.config.dropout,
                )
                for _ in range(self.config.num_layers)
            ]
        )
        self.final_norm = RMSNorm(self.config.hidden_dim)

        # 4. Multi-head BatchEnsemble Prediction Head
        self.head = BatchEnsembleLinear(
            in_features=self.config.hidden_dim,
            out_features=self.out_dim,
            num_heads=self.config.num_heads,
            bias=True,
            adapter_init_std=self.config.adapter_init_std,
        )

    def count_parameters(self) -> int:
        """Return the exact number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward_heads(
        self,
        num_vals: torch.Tensor,
        num_masks: torch.Tensor,
        cat_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass returning raw logits/outputs for all k ensemble heads.

        Returns:
            Tensor of shape (k, batch_size, out_dim)
        """
        token_list = []

        if self.has_numerical and self.ple is not None and self.mask_proj is not None:
            num_tokens = self.ple(num_vals)  # (B, F_num, embed_dim)
            token_list.append(num_tokens.flatten(start_dim=1))
            mask_token = self.mask_proj(num_masks)  # (B, embed_dim)
            token_list.append(mask_token)

        if self.has_categorical and self.cat_embed is not None:
            cat_tokens = self.cat_embed(cat_indices)  # (B, F_cat, embed_dim)
            token_list.append(cat_tokens.flatten(start_dim=1))

        if token_list:
            x_in = torch.cat(token_list, dim=1)  # (B, total_tokens_dim)
        else:
            x_in = torch.zeros(num_vals.shape[0], 1, device=num_vals.device)

        # Project to hidden_dim
        h = self.input_proj(x_in)  # (B, hidden_dim)

        # Broadcast across k heads
        h = h.unsqueeze(0).expand(self.num_heads, -1, -1)  # (k, B, hidden_dim)

        # Backbone blocks
        for block in self.blocks:
            h = block(h)

        h = self.final_norm(h)

        # Prediction head: (k, B, out_dim)
        logits = self.head(h)
        return logits

    def forward(
        self,
        num_vals: torch.Tensor,
        num_masks: torch.Tensor,
        cat_indices: torch.Tensor,
        return_ensemble: bool = False,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            num_vals: Continuous features tensor (B, F_num)
            num_masks: Missingness indicators tensor (B, F_num)
            cat_indices: Categorical indices tensor (B, F_cat)
            return_ensemble: If True, returns (k, B, out_dim). If False, returns averaged prediction.

        Returns:
            Averaged prediction tensor (B, out_dim) or ensemble heads tensor (k, B, out_dim).
        """
        heads_out = self.forward_heads(num_vals, num_masks, cat_indices)  # (k, B, out_dim)

        if return_ensemble or self.training:
            return heads_out

        # Inference aggregation across k ensemble members
        if self.schema.task_type == TaskType.BINARY:
            probs = torch.sigmoid(heads_out)  # (k, B, 1)
            return probs.mean(dim=0)  # (B, 1)
        elif self.schema.task_type == TaskType.MULTICLASS:
            probs = F.softmax(heads_out, dim=-1)  # (k, B, C)
            return probs.mean(dim=0)  # (B, C)
        else:
            # Regression: scalar mean
            return heads_out.mean(dim=0)  # (B, 1)
