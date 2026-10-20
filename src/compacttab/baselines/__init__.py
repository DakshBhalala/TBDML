"""Baselines subpackage for CompactTab."""

from compacttab.baselines.tree_models import (
    BaseBaseline,
    CatBoostBaseline,
    DummyBaseline,
    ExtraTreesBaseline,
    HistGradientBoostingBaseline,
    LightGBMBaseline,
    RandomForestBaseline,
    XGBoostBaseline,
)

__all__ = [
    "BaseBaseline",
    "DummyBaseline",
    "RandomForestBaseline",
    "ExtraTreesBaseline",
    "HistGradientBoostingBaseline",
    "LightGBMBaseline",
    "XGBoostBaseline",
    "CatBoostBaseline",
]
