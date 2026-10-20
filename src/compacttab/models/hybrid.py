"""CompactTab-Hybrid: Ensembling Quantile PLE Neural Networks with Gradient Boosted Trees."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.model_selection import KFold, StratifiedKFold

from compacttab.models.compacttab import CompactTabClassifier, CompactTabRegressor

logger = logging.getLogger(__name__)


def _get_tree_classifier(random_state: int | None = 42) -> Any:
    """Instantiate high-performance gradient boosted tree classifier."""
    try:
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            n_estimators=150,
            learning_rate=0.05,
            num_leaves=31,
            random_state=random_state,
            verbose=-1,
        )
    except ImportError:
        try:
            from catboost import CatBoostClassifier

            return CatBoostClassifier(
                iterations=150,
                learning_rate=0.05,
                depth=6,
                random_seed=random_state,
                verbose=False,
            )
        except ImportError:
            from sklearn.ensemble import HistGradientBoostingClassifier

            return HistGradientBoostingClassifier(
                max_iter=150,
                learning_rate=0.05,
                random_state=random_state,
            )


def _get_tree_regressor(random_state: int | None = 42) -> Any:
    """Instantiate high-performance gradient boosted tree regressor."""
    try:
        from lightgbm import LGBMRegressor

        return LGBMRegressor(
            n_estimators=150,
            learning_rate=0.05,
            num_leaves=31,
            random_state=random_state,
            verbose=-1,
        )
    except ImportError:
        try:
            from catboost import CatBoostRegressor

            return CatBoostRegressor(
                iterations=150,
                learning_rate=0.05,
                depth=6,
                random_seed=random_state,
                verbose=False,
            )
        except ImportError:
            from sklearn.ensemble import HistGradientBoostingRegressor

            return HistGradientBoostingRegressor(
                max_iter=150,
                learning_rate=0.05,
                random_state=random_state,
            )


class CompactTabHybridClassifier(BaseEstimator, ClassifierMixin):
    """Hybrid Classifier combining CompactTab BatchEnsemble with Gradient Boosted Trees.

    Combines smooth non-linear continuous manifold representations from CompactTab's
    Quantile Piecewise-Linear Embeddings with axis-aligned tree splits via cross-validated
    adaptive probability stacking.
    """

    def __init__(
        self,
        preset: str = "small",
        cv_folds: int = 5,
        mixup_alpha: float = 0.1,
        max_epochs: int = 40,
        batch_size: int = 128,
        device: str | None = None,
        random_state: int | None = 42,
        verbose: bool = False,
    ):
        self.preset = preset
        self.cv_folds = cv_folds
        self.mixup_alpha = mixup_alpha
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.device = device
        self.random_state = random_state
        self.verbose = verbose

        self.nn_model_: CompactTabClassifier | None = None
        self.tree_model_: Any = None
        self.classes_: np.ndarray | None = None
        self.blend_weight_: float = 0.5  # Weight given to neural network

    def _prepare_data(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray | None = None,
    ) -> tuple[pd.DataFrame, pd.Series | None]:
        if isinstance(X, pd.DataFrame):
            df_x = X.copy().reset_index(drop=True)
        else:
            arr_x = np.asarray(X)
            df_x = pd.DataFrame(arr_x, columns=[f"feat_{i}" for i in range(arr_x.shape[1])])

        s_y = None
        if y is not None:
            if isinstance(y, pd.Series):
                s_y = y.copy().reset_index(drop=True)
            else:
                s_y = pd.Series(np.asarray(y))
        return df_x, s_y

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
    ) -> CompactTabHybridClassifier:
        df_x, s_y = self._prepare_data(X, y)
        self.classes_ = np.sort(np.unique(np.asarray(s_y)))
        n_classes = len(self.classes_)
        n_samples = len(df_x)

        # 1. Out-of-fold cross-validation to estimate optimal blend weight
        if self.cv_folds > 1 and n_samples >= 50:
            skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
            oof_nn = np.zeros((n_samples, n_classes), dtype=np.float32)
            oof_tree = np.zeros((n_samples, n_classes), dtype=np.float32)

            for train_idx, val_idx in skf.split(df_x, s_y):
                X_tr, y_tr = df_x.iloc[train_idx], s_y.iloc[train_idx]
                X_val, _y_val = df_x.iloc[val_idx], s_y.iloc[val_idx]

                # Fold NN
                fold_nn = CompactTabClassifier(
                    preset=self.preset,
                    max_epochs=max(10, self.max_epochs // 2),
                    batch_size=self.batch_size,
                    mixup_alpha=self.mixup_alpha,
                    device=self.device,
                    random_state=self.random_state,
                    verbose=False,
                )
                fold_nn.fit(X_tr, y_tr)
                oof_nn[val_idx] = fold_nn.predict_proba(X_val)

                # Fold Tree
                fold_tree = _get_tree_classifier(self.random_state)
                # Tree models need numeric or label encoded categoricals
                df_tr_tree = X_tr.select_dtypes(include=["number"])
                df_val_tree = X_val.select_dtypes(include=["number"])
                if df_tr_tree.shape[1] > 0:
                    fold_tree.fit(df_tr_tree, y_tr)
                    oof_tree[val_idx] = fold_tree.predict_proba(df_val_tree)
                else:
                    oof_tree[val_idx] = oof_nn[val_idx]

            # Optimize blend weight alpha in [0, 1] to minimize cross-entropy loss
            eps = 1e-12
            y_one_hot = np.zeros((n_samples, n_classes), dtype=np.float32)
            for idx, c in enumerate(self.classes_):
                y_one_hot[s_y.to_numpy() == c, idx] = 1.0

            def loss_fn(alpha: float) -> float:
                p_blend = alpha * oof_nn + (1.0 - alpha) * oof_tree
                p_blend = np.clip(p_blend, eps, 1.0 - eps)
                p_blend = p_blend / p_blend.sum(axis=1, keepdims=True)
                return float(-np.mean(np.sum(y_one_hot * np.log(p_blend), axis=1)))

            res = minimize_scalar(loss_fn, bounds=(0.05, 0.95), method="bounded")
            self.blend_weight_ = float(res.x) if res.success else 0.5
        else:
            self.blend_weight_ = 0.5

        # 2. Fit final estimators on full training data
        self.nn_model_ = CompactTabClassifier(
            preset=self.preset,
            max_epochs=self.max_epochs,
            batch_size=self.batch_size,
            mixup_alpha=self.mixup_alpha,
            device=self.device,
            random_state=self.random_state,
            verbose=self.verbose,
        )
        self.nn_model_.fit(df_x, s_y)

        self.tree_model_ = _get_tree_classifier(self.random_state)
        df_x_num = df_x.select_dtypes(include=["number"])
        if df_x_num.shape[1] > 0:
            self.tree_model_.fit(df_x_num, s_y)
        else:
            self.tree_model_ = None

        return self

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if self.nn_model_ is None:
            raise RuntimeError("CompactTabHybridClassifier is not fitted.")
        df_x, _ = self._prepare_data(X)
        p_nn = self.nn_model_.predict_proba(df_x)

        if self.tree_model_ is not None:
            df_x_num = df_x.select_dtypes(include=["number"])
            if df_x_num.shape[1] > 0:
                p_tree = self.tree_model_.predict_proba(df_x_num)
            else:
                p_tree = p_nn
        else:
            p_tree = p_nn

        # Align shapes if binary tree outputs 1D or different classes
        if p_tree.shape != p_nn.shape:
            p_tree = p_nn

        alpha = self.blend_weight_
        p_blend = alpha * p_nn + (1.0 - alpha) * p_tree
        row_sums = p_blend.sum(axis=1, keepdims=True)
        return p_blend / np.maximum(row_sums, 1e-12)

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        idx = np.argmax(probs, axis=1)
        if self.classes_ is not None and len(self.classes_) == probs.shape[1]:
            return self.classes_[idx]
        return idx


class CompactTabHybridRegressor(BaseEstimator, RegressorMixin):
    """Hybrid Regressor combining CompactTab BatchEnsemble with Gradient Boosted Trees."""

    def __init__(
        self,
        preset: str = "small",
        cv_folds: int = 5,
        mixup_alpha: float = 0.1,
        max_epochs: int = 40,
        batch_size: int = 128,
        device: str | None = None,
        random_state: int | None = 42,
        verbose: bool = False,
    ):
        self.preset = preset
        self.cv_folds = cv_folds
        self.mixup_alpha = mixup_alpha
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.device = device
        self.random_state = random_state
        self.verbose = verbose

        self.nn_model_: CompactTabRegressor | None = None
        self.tree_model_: Any = None
        self.blend_weight_: float = 0.5

    def _prepare_data(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray | None = None,
    ) -> tuple[pd.DataFrame, pd.Series | None]:
        if isinstance(X, pd.DataFrame):
            df_x = X.copy().reset_index(drop=True)
        else:
            arr_x = np.asarray(X)
            df_x = pd.DataFrame(arr_x, columns=[f"feat_{i}" for i in range(arr_x.shape[1])])

        s_y = None
        if y is not None:
            if isinstance(y, pd.Series):
                s_y = y.copy().reset_index(drop=True)
            else:
                s_y = pd.Series(np.asarray(y))
        return df_x, s_y

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
    ) -> CompactTabHybridRegressor:
        df_x, s_y = self._prepare_data(X, y)
        n_samples = len(df_x)

        if self.cv_folds > 1 and n_samples >= 50:
            kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
            oof_nn = np.zeros(n_samples, dtype=np.float32)
            oof_tree = np.zeros(n_samples, dtype=np.float32)

            for train_idx, val_idx in kf.split(df_x):
                X_tr, y_tr = df_x.iloc[train_idx], s_y.iloc[train_idx]
                X_val = df_x.iloc[val_idx]

                fold_nn = CompactTabRegressor(
                    preset=self.preset,
                    max_epochs=max(10, self.max_epochs // 2),
                    batch_size=self.batch_size,
                    mixup_alpha=self.mixup_alpha,
                    device=self.device,
                    random_state=self.random_state,
                    verbose=False,
                )
                fold_nn.fit(X_tr, y_tr)
                oof_nn[val_idx] = fold_nn.predict(X_val)

                fold_tree = _get_tree_regressor(self.random_state)
                df_tr_tree = X_tr.select_dtypes(include=["number"])
                df_val_tree = X_val.select_dtypes(include=["number"])
                if df_tr_tree.shape[1] > 0:
                    fold_tree.fit(df_tr_tree, y_tr)
                    oof_tree[val_idx] = fold_tree.predict(df_val_tree)
                else:
                    oof_tree[val_idx] = oof_nn[val_idx]

            y_arr = s_y.to_numpy(dtype=np.float32)

            def loss_fn(alpha: float) -> float:
                pred_blend = alpha * oof_nn + (1.0 - alpha) * oof_tree
                return float(np.mean((pred_blend - y_arr) ** 2))

            res = minimize_scalar(loss_fn, bounds=(0.05, 0.95), method="bounded")
            self.blend_weight_ = float(res.x) if res.success else 0.5
        else:
            self.blend_weight_ = 0.5

        # Fit final estimators
        self.nn_model_ = CompactTabRegressor(
            preset=self.preset,
            max_epochs=self.max_epochs,
            batch_size=self.batch_size,
            mixup_alpha=self.mixup_alpha,
            device=self.device,
            random_state=self.random_state,
            verbose=self.verbose,
        )
        self.nn_model_.fit(df_x, s_y)

        self.tree_model_ = _get_tree_regressor(self.random_state)
        df_x_num = df_x.select_dtypes(include=["number"])
        if df_x_num.shape[1] > 0:
            self.tree_model_.fit(df_x_num, s_y)
        else:
            self.tree_model_ = None

        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if self.nn_model_ is None:
            raise RuntimeError("CompactTabHybridRegressor is not fitted.")
        df_x, _ = self._prepare_data(X)
        pred_nn = self.nn_model_.predict(df_x)

        if self.tree_model_ is not None:
            df_x_num = df_x.select_dtypes(include=["number"])
            if df_x_num.shape[1] > 0:
                pred_tree = self.tree_model_.predict(df_x_num)
            else:
                pred_tree = pred_nn
        else:
            pred_tree = pred_nn

        alpha = self.blend_weight_
        return alpha * pred_nn + (1.0 - alpha) * pred_tree
