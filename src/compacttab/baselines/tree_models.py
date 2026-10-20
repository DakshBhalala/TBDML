"""Baseline tree and non-neural models conforming to unified interface."""

from typing import Any

import numpy as np
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)

from compacttab.data.schema import DatasetSchema

try:
    import lightgbm as lgb

    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    import xgboost as xgb

    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import catboost as cb

    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False


class BaseBaseline:
    """Abstract baseline wrapper."""

    def __init__(self, schema: DatasetSchema, name: str):
        self.schema = schema
        self.name = name
        self.model: Any = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseBaseline":
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def count_parameters(self) -> int:
        """Return parameter count or approximate tree nodes."""
        return 0


class DummyBaseline(BaseBaseline):
    """Sanity baseline predicting majority class or mean target."""

    def __init__(self, schema: DatasetSchema, random_state: int = 42):
        super().__init__(schema, name="Dummy")
        self.random_state = random_state
        if self.schema.is_classification:
            self.model = DummyClassifier(strategy="prior")
        else:
            self.model = DummyRegressor(strategy="mean")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DummyBaseline":
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.schema.is_classification:
            raise ValueError("predict_proba is only supported for classification")
        return self.model.predict_proba(X)


class RandomForestBaseline(BaseBaseline):
    """Scikit-learn Random Forest baseline."""

    def __init__(
        self,
        schema: DatasetSchema,
        n_estimators: int = 100,
        max_depth: int | None = 12,
        random_state: int = 42,
    ):
        super().__init__(schema, name="RandomForest")
        if self.schema.is_classification:
            self.model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state,
                n_jobs=-1,
            )
        else:
            self.model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state,
                n_jobs=-1,
            )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestBaseline":
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.schema.is_classification:
            raise ValueError("predict_proba is only supported for classification")
        return self.model.predict_proba(X)

    def count_parameters(self) -> int:
        return sum(tree.tree_.node_count for tree in self.model.estimators_)


class ExtraTreesBaseline(BaseBaseline):
    """Scikit-learn Extra Trees baseline."""

    def __init__(
        self,
        schema: DatasetSchema,
        n_estimators: int = 100,
        max_depth: int | None = 12,
        random_state: int = 42,
    ):
        super().__init__(schema, name="ExtraTrees")
        if self.schema.is_classification:
            self.model = ExtraTreesClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state,
                n_jobs=-1,
            )
        else:
            self.model = ExtraTreesRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state,
                n_jobs=-1,
            )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ExtraTreesBaseline":
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def count_parameters(self) -> int:
        return sum(tree.tree_.node_count for tree in self.model.estimators_)


class HistGradientBoostingBaseline(BaseBaseline):
    """Scikit-learn native Histogram-based Gradient Boosting."""

    def __init__(
        self,
        schema: DatasetSchema,
        max_iter: int = 100,
        random_state: int = 42,
    ):
        super().__init__(schema, name="HistGradientBoosting")
        if self.schema.is_classification:
            self.model = HistGradientBoostingClassifier(
                max_iter=max_iter,
                random_state=random_state,
                early_stopping=True,
            )
        else:
            self.model = HistGradientBoostingRegressor(
                max_iter=max_iter,
                random_state=random_state,
                early_stopping=True,
            )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HistGradientBoostingBaseline":
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)


class LightGBMBaseline(BaseBaseline):
    """LightGBM GBDT baseline."""

    def __init__(
        self,
        schema: DatasetSchema,
        n_estimators: int = 100,
        learning_rate: float = 0.05,
        random_state: int = 42,
    ):
        super().__init__(schema, name="LightGBM")
        if not HAS_LIGHTGBM:
            raise RuntimeError("lightgbm package is not installed.")

        params = {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "random_state": random_state,
            "verbose": -1,
            "n_jobs": -1,
        }
        if self.schema.is_classification:
            self.model = lgb.LGBMClassifier(**params)
        else:
            self.model = lgb.LGBMRegressor(**params)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LightGBMBaseline":
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)


class XGBoostBaseline(BaseBaseline):
    """XGBoost GBDT baseline."""

    def __init__(
        self,
        schema: DatasetSchema,
        n_estimators: int = 100,
        learning_rate: float = 0.05,
        random_state: int = 42,
    ):
        super().__init__(schema, name="XGBoost")
        if not HAS_XGBOOST:
            raise RuntimeError("xgboost package is not installed.")

        params = {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "random_state": random_state,
            "n_jobs": -1,
            "verbosity": 0,
        }
        if self.schema.is_classification:
            self.model = xgb.XGBClassifier(**params)
        else:
            self.model = xgb.XGBRegressor(**params)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "XGBoostBaseline":
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)


class CatBoostBaseline(BaseBaseline):
    """CatBoost GBDT baseline."""

    def __init__(
        self,
        schema: DatasetSchema,
        iterations: int = 100,
        learning_rate: float = 0.05,
        random_state: int = 42,
    ):
        super().__init__(schema, name="CatBoost")
        if not HAS_CATBOOST:
            raise RuntimeError("catboost package is not installed.")

        params = {
            "iterations": iterations,
            "learning_rate": learning_rate,
            "random_seed": random_state,
            "verbose": False,
            "thread_count": -1,
        }
        if self.schema.is_classification:
            self.model = cb.CatBoostClassifier(**params)
        else:
            self.model = cb.CatBoostRegressor(**params)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CatBoostBaseline":
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)
