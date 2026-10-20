"""Models subpackage for CompactTab."""

from compacttab.models.baselines_nn import TabularMLP, TabularResNet
from compacttab.models.compacttab import (
    CompactTab,
    CompactTabClassifier,
    CompactTabConfig,
    CompactTabRegressor,
)
from compacttab.models.embeddings import CategoricalEmbedding, PiecewiseLinearEncoding
from compacttab.models.layers import BatchEnsembleLinear, GatedResidualBlock, RMSNorm

__all__ = [
    "PiecewiseLinearEncoding",
    "CategoricalEmbedding",
    "RMSNorm",
    "BatchEnsembleLinear",
    "GatedResidualBlock",
    "CompactTabConfig",
    "CompactTab",
    "CompactTabClassifier",
    "CompactTabRegressor",
    "TabularMLP",
    "TabularResNet",
]
