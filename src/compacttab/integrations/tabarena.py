"""AutoGluon AbstractModel plugin for TabArena benchmark integration.

Complies with official TabArena benchmark model requirements:
- Subclasses autogluon.core.models.AbstractModel
- Implements _fit, _predict, _predict_proba
- Supports binary, multiclass, and regression problem types
- Powered natively by CompactTab (Quantile PLE + BatchEnsemble Gated Residuals)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

try:
    from autogluon.core.constants import BINARY, MULTICLASS, REGRESSION
    from autogluon.core.models import AbstractModel

    AUTOGLUON_AVAILABLE = True
except ImportError:
    AbstractModel = object  # type: ignore
    BINARY = "binary"
    MULTICLASS = "multiclass"
    REGRESSION = "regression"
    AUTOGLUON_AVAILABLE = False

from compacttab.models.compacttab import CompactTabClassifier, CompactTabRegressor

logger = logging.getLogger(__name__)


class TabArenaCompactTabModel(AbstractModel):
    """Production AutoGluon model wrapper for CompactTab.

    Designed for direct integration into AutoGluon Tabular and the TabArena benchmark.
    Uses Quantile Piecewise-Linear Embeddings with BatchEnsemble multi-head ensembling.
    """

    ag_key = "COMPACTTAB"
    ag_name = "CompactTab"
    ag_priority = 50

    def __init__(
        self,
        name: str | None = "CompactTab",
        problem_type: str | None = None,
        preset: str = "small",
        max_epochs: int = 50,
        batch_size: int = 128,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        patience: int = 10,
        device: str | None = None,
        **kwargs: Any,
    ):
        if not AUTOGLUON_AVAILABLE:
            raise ImportError(
                "autogluon.core is required to instantiate TabArenaCompactTabModel. "
                "Install via 'pip install autogluon.core'."
            )
        super().__init__(name=name, problem_type=problem_type, **kwargs)
        self.preset = preset
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.patience = patience
        self.device = device
        self._estimator: CompactTabClassifier | CompactTabRegressor | None = None

    def _get_default_auxiliary_params(self) -> dict:
        default_aux = super()._get_default_auxiliary_params()
        default_aux.update(
            {
                "valid_raw_types": ["int", "float", "category", "object"],
                "ignored_type_group_raw": [],
            }
        )
        return default_aux

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
        time_limit: float | None = None,
        num_cpus: int | None = None,
        num_gpus: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Fit CompactTab neural network using training split and optional validation split."""
        params = self._get_model_params()
        preset = params.get("preset", self.preset)
        max_epochs = params.get("max_epochs", self.max_epochs)
        batch_size = params.get("batch_size", self.batch_size)
        learning_rate = params.get("learning_rate", self.learning_rate)
        weight_decay = params.get("weight_decay", self.weight_decay)
        patience = params.get("patience", self.patience)

        device = self.device
        if device is None and num_gpus is not None:
            device = "cuda" if num_gpus > 0 else "cpu"

        if self.problem_type in (BINARY, MULTICLASS):
            self._estimator = CompactTabClassifier(
                preset=preset,
                max_epochs=max_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                patience=patience,
                device=device,
            )
        elif self.problem_type == REGRESSION:
            self._estimator = CompactTabRegressor(
                preset=preset,
                max_epochs=max_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                patience=patience,
                device=device,
            )
        else:
            raise ValueError(f"Unsupported problem_type: {self.problem_type}")

        self._estimator.fit(X=X, y=y, X_val=X_val, y_val=y_val)

    def _predict_proba(self, X: pd.DataFrame, **kwargs: Any) -> np.ndarray:
        """Generate prediction probabilities for classification."""
        if self._estimator is None:
            raise RuntimeError("Model has not been fitted.")
        if self.problem_type == REGRESSION:
            return self._predict(X, **kwargs)
        return self._estimator.predict_proba(X)

    def _predict(self, X: pd.DataFrame, **kwargs: Any) -> np.ndarray:
        """Generate point predictions for classification or regression tasks."""
        if self._estimator is None:
            raise RuntimeError("Model has not been fitted.")
        return self._estimator.predict(X)

    def _set_default_params(self):
        default_params = {
            "preset": "small",
            "max_epochs": 50,
            "batch_size": 128,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "patience": 10,
        }
        for param, val in default_params.items():
            self._set_default_param_value(param, val)
