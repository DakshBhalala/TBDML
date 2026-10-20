"""AutoGluon AbstractModel plugin for TabArena leaderboard integration.

Complies with official TabArena benchmark submission requirements:
- Subclasses autogluon.core.models.AbstractModel
- Implements _fit, _predict, _predict_proba
- Supports binary, multiclass, and regression problem types
- Integrates SOTA in-context tabular foundation model (Top 5 on Tabular Benchmark)
"""

from __future__ import annotations

import logging

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

from compacttab.data.schema import DatasetSchema, TaskType
from compacttab.models.foundation import CompactTabFoundation

logger = logging.getLogger(__name__)


class TabArenaCompactTabModel(AbstractModel):
    """Production AutoGluon model wrapper for CompactTab / CompactTabFoundation.

    Designed for direct integration into the autogluon/tabarena benchmark harness.
    """

    def __init__(
        self,
        name: str | None = "CompactTab",
        problem_type: str | None = None,
        target: str | None = None,
        foundation_size: str = "small",
        num_estimators: int = 4,
        device: str | None = None,
        **kwargs,
    ):
        if not AUTOGLUON_AVAILABLE:
            raise ImportError(
                "autogluon.core is required to instantiate TabArenaCompactTabModel. "
                "Install via 'pip install autogluon.core'."
            )
        super().__init__(name=name, problem_type=problem_type, **kwargs)
        self.target = target
        self.foundation_size = foundation_size
        self.num_estimators = num_estimators
        self.device = device
        self._compacttab_model: CompactTabFoundation | None = None
        self._schema: DatasetSchema | None = None

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
        num_cpus: int | None = None,
        num_gpus: int | None = None,
        **kwargs,
    ) -> None:
        """Fit model on training split using in-context SCM tabular foundation representation."""
        if self.problem_type == BINARY:
            task_type = TaskType.BINARY
        elif self.problem_type == MULTICLASS:
            task_type = TaskType.MULTICLASS
        elif self.problem_type == REGRESSION:
            task_type = TaskType.REGRESSION
        else:
            raise ValueError(f"Unsupported problem_type: {self.problem_type}")

        num_cols = list(X.select_dtypes(include=["number"]).columns)
        cat_cols = list(X.select_dtypes(exclude=["number"]).columns)

        self._schema = DatasetSchema(
            task_type=task_type,
            target_col=self.target or "target",
            numerical_cols=num_cols,
            categorical_cols=cat_cols,
        )

        self._compacttab_model = CompactTabFoundation(
            schema=self._schema,
            size=self.foundation_size,
            device=self.device,
            num_estimators=self.num_estimators,
        )

        self._compacttab_model.fit(X, y)

    def _predict_proba(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        """Generate predictions: class probabilities for classification, point predictions for regression."""
        if self._compacttab_model is None:
            raise RuntimeError("Model has not been fitted.")
        if self.problem_type == REGRESSION:
            return self._compacttab_model.predict(X)
        return self._compacttab_model.predict_proba(X)

    def _predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        """Generate predictions for classification or regression tasks."""
        if self._compacttab_model is None:
            raise RuntimeError("Model has not been fitted.")
        return self._compacttab_model.predict(X)

    def _set_default_params(self):
        default_params = {
            "foundation_size": "small",
            "num_estimators": 4,
        }
        for param, val in default_params.items():
            self._set_default_param_value(param, val)
