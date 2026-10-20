"""CompactTab-Hybrid: Ensembling Quantile PLE Neural Networks with Gradient Boosted Trees."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder

from compacttab.models.compacttab import CompactTabClassifier, CompactTabRegressor

logger = logging.getLogger(__name__)


class TreeFeatureEncoder:
    """Robust feature encoder preparing all tabular columns (numeric + categorical) for tree models."""

    def __init__(self):
        self.num_cols: list[str] = []
        self.cat_cols: list[str] = []
        self.encoder: OrdinalEncoder | None = None
        self.medians: dict[str, float] = {}
        self.is_fitted: bool = False

    def fit(self, df: pd.DataFrame) -> TreeFeatureEncoder:
        self.num_cols = list(df.select_dtypes(include=["number"]).columns)
        self.cat_cols = list(df.select_dtypes(exclude=["number"]).columns)

        for c in self.num_cols:
            vals = pd.to_numeric(df[c], errors="coerce")
            self.medians[c] = float(vals.median()) if not vals.isna().all() else 0.0

        if self.cat_cols:
            self.encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
                dtype=np.float32,
            )
            self.encoder.fit(df[self.cat_cols].astype(str))

        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("TreeFeatureEncoder must be fitted before transform.")

        df_out = pd.DataFrame(index=df.index)

        # 1. Numerical columns
        for c in self.num_cols:
            if c in df.columns:
                vals = pd.to_numeric(df[c], errors="coerce")
                df_out[c] = vals.fillna(self.medians.get(c, 0.0)).to_numpy(dtype=np.float32)
            else:
                df_out[c] = np.full(len(df), self.medians.get(c, 0.0), dtype=np.float32)

        # 2. Categorical columns
        if self.cat_cols and self.encoder is not None:
            cat_df = (
                df[self.cat_cols].astype(str)
                if all(c in df.columns for c in self.cat_cols)
                else df.reindex(columns=self.cat_cols).fillna("__MISSING__").astype(str)
            )
            cat_enc = self.encoder.transform(cat_df)
            for idx, c in enumerate(self.cat_cols):
                df_out[c] = cat_enc[:, idx]

        return df_out


def _get_tree_classifier(random_state: int | None = 42) -> Any:
    """Instantiate primary gradient boosted tree classifier."""
    try:
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            n_estimators=150,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            verbose=-1,
            n_jobs=-1,
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
                thread_count=-1,
            )
        except ImportError:
            from sklearn.ensemble import HistGradientBoostingClassifier

            return HistGradientBoostingClassifier(
                max_iter=150,
                learning_rate=0.05,
                random_state=random_state,
            )


def _get_tree_regressor(random_state: int | None = 42) -> Any:
    """Instantiate primary gradient boosted tree regressor."""
    try:
        from lightgbm import LGBMRegressor

        return LGBMRegressor(
            n_estimators=150,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            verbose=-1,
            n_jobs=-1,
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
                thread_count=-1,
            )
        except ImportError:
            from sklearn.ensemble import HistGradientBoostingRegressor

            return HistGradientBoostingRegressor(
                max_iter=150,
                learning_rate=0.05,
                random_state=random_state,
            )


def _get_multi_tree_models(
    random_state: int | None = 42, is_classification: bool = True
) -> list[tuple[str, Any]]:
    """Instantiate complementary tree architectures (LightGBM and CatBoost)."""
    models = []

    # 1. LightGBM (Leaf-wise histogram boosting)
    try:
        from lightgbm import LGBMClassifier, LGBMRegressor

        if is_classification:
            m = LGBMClassifier(
                n_estimators=150,
                learning_rate=0.05,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=random_state,
                verbose=-1,
                n_jobs=-1,
            )
        else:
            m = LGBMRegressor(
                n_estimators=150,
                learning_rate=0.05,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=random_state,
                verbose=-1,
                n_jobs=-1,
            )
        models.append(("lightgbm", m))
    except Exception:
        pass

    # 2. CatBoost (Symmetric oblivious trees)
    try:
        from catboost import CatBoostClassifier, CatBoostRegressor

        if is_classification:
            m = CatBoostClassifier(
                iterations=150,
                learning_rate=0.05,
                depth=6,
                random_seed=random_state,
                verbose=False,
                thread_count=-1,
            )
        else:
            m = CatBoostRegressor(
                iterations=150,
                learning_rate=0.05,
                depth=6,
                random_seed=random_state,
                verbose=False,
                thread_count=-1,
            )
        models.append(("catboost", m))
    except Exception:
        pass

    # Fallback to HistGradientBoosting if neither is available
    if not models:
        from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

        if is_classification:
            m = HistGradientBoostingClassifier(
                max_iter=150,
                learning_rate=0.05,
                random_state=random_state,
            )
        else:
            m = HistGradientBoostingRegressor(
                max_iter=150,
                learning_rate=0.05,
                random_state=random_state,
            )
        models.append(("histgb", m))

    return models


class CompactTabHybridClassifier(BaseEstimator, ClassifierMixin):
    """Tri-Engine Hybrid Classifier combining CompactTab Neural Network with GBDTs.

    Combines smooth continuous representations from CompactTab's Quantile Periodic-PLE
    neural backbone with asymmetric (LightGBM) and symmetric (CatBoost) tree splits
    via out-of-fold convex-constrained SLSQP stacking.
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
        self.tree_models_: dict[str, Any] = {}
        self.tree_model_: Any = None  # Backwards compatibility
        self.classes_: np.ndarray | None = None
        self.blend_weight_: float = 0.5  # Backwards compatibility
        self.weights_: dict[str, float] = {}
        self.tree_encoder_: TreeFeatureEncoder | None = None

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

        # 1. Fit tree feature encoder
        self.tree_encoder_ = TreeFeatureEncoder().fit(df_x)
        df_tree_x = self.tree_encoder_.transform(df_x)

        # 2. Get tree engines
        tree_specs = _get_multi_tree_models(self.random_state, is_classification=True)
        all_model_names = ["compacttab"] + [name for name, _ in tree_specs]
        num_models = len(all_model_names)

        # 3. Out-of-fold cross-validation
        if self.cv_folds > 1 and n_samples >= 40:
            skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
            oof_dict = {name: np.zeros((n_samples, n_classes), dtype=np.float32) for name in all_model_names}

            for train_idx, val_idx in skf.split(df_x, s_y):
                X_tr, y_tr = df_x.iloc[train_idx], s_y.iloc[train_idx]
                X_val = df_x.iloc[val_idx]

                X_tr_tree = df_tree_x.iloc[train_idx]
                X_val_tree = df_tree_x.iloc[val_idx]

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
                oof_dict["compacttab"][val_idx] = fold_nn.predict_proba(X_val)

                # Fold Trees
                for name, _ in tree_specs:
                    fold_trees = dict(_get_multi_tree_models(self.random_state, is_classification=True))
                    tree_m = fold_trees[name]
                    tree_m.fit(X_tr_tree, y_tr)
                    p_tree = tree_m.predict_proba(X_val_tree)
                    if p_tree.shape == oof_dict[name][val_idx].shape:
                        oof_dict[name][val_idx] = p_tree
                    else:
                        oof_dict[name][val_idx] = oof_dict["compacttab"][val_idx]

            # Solve convex optimal weights via SLSQP on Simplex
            eps = 1e-12
            y_one_hot = np.zeros((n_samples, n_classes), dtype=np.float32)
            for idx, c in enumerate(self.classes_):
                y_one_hot[s_y.to_numpy() == c, idx] = 1.0

            def objective(weights: np.ndarray) -> float:
                w_norm = weights / max(np.sum(weights), eps)
                p_blend = np.zeros((n_samples, n_classes), dtype=np.float32)
                for idx, name in enumerate(all_model_names):
                    p_blend += w_norm[idx] * oof_dict[name]
                p_blend = np.clip(p_blend, eps, 1.0 - eps)
                p_blend = p_blend / np.sum(p_blend, axis=1, keepdims=True)
                return float(-np.mean(np.sum(y_one_hot * np.log(p_blend), axis=1)))

            init_w = np.full(num_models, 1.0 / num_models, dtype=np.float64)
            bounds = [(0.0, 1.0) for _ in range(num_models)]
            cons = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

            try:
                res = minimize(
                    objective,
                    init_w,
                    bounds=bounds,
                    constraints=cons,
                    method="SLSQP",
                    options={"maxiter": 100},
                )
                if res.success and np.sum(res.x) > 0:
                    w_opt = res.x / np.sum(res.x)
                else:
                    w_opt = init_w
            except Exception:
                w_opt = init_w

            self.weights_ = {name: float(w_opt[idx]) for idx, name in enumerate(all_model_names)}
            self.blend_weight_ = self.weights_["compacttab"]
        else:
            w_uniform = 1.0 / num_models
            self.weights_ = {name: w_uniform for name in all_model_names}
            self.blend_weight_ = w_uniform

        # 4. Fit final production models on full training set
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

        self.tree_models_ = {}
        for name, tree_m in tree_specs:
            tree_m.fit(df_tree_x, s_y)
            self.tree_models_[name] = tree_m

        if tree_specs:
            self.tree_model_ = self.tree_models_[tree_specs[0][0]]

        return self

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if self.nn_model_ is None or self.tree_encoder_ is None:
            raise RuntimeError("CompactTabHybridClassifier is not fitted.")

        df_x, _ = self._prepare_data(X)
        df_tree_x = self.tree_encoder_.transform(df_x)

        p_nn = self.nn_model_.predict_proba(df_x)
        p_blend = self.weights_.get("compacttab", 0.5) * p_nn

        for name, tree_m in self.tree_models_.items():
            w = self.weights_.get(name, 0.0)
            if w > 1e-4:
                try:
                    p_tree = tree_m.predict_proba(df_tree_x)
                    if p_tree.shape == p_nn.shape:
                        p_blend += w * p_tree
                    else:
                        p_blend += w * p_nn
                except Exception:
                    p_blend += w * p_nn

        row_sums = p_blend.sum(axis=1, keepdims=True)
        return p_blend / np.maximum(row_sums, 1e-12)

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        idx = np.argmax(probs, axis=1)
        if self.classes_ is not None and len(self.classes_) == probs.shape[1]:
            return self.classes_[idx]
        return idx


class CompactTabHybridRegressor(BaseEstimator, RegressorMixin):
    """Tri-Engine Hybrid Regressor combining CompactTab Neural Network with GBDTs.

    Combines continuous regression predictions from CompactTab's Quantile Periodic-PLE
    backbone with asymmetric and symmetric decision trees via out-of-fold convex SLSQP stacking.
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

        self.nn_model_: CompactTabRegressor | None = None
        self.tree_models_: dict[str, Any] = {}
        self.tree_model_: Any = None
        self.blend_weight_: float = 0.5
        self.weights_: dict[str, float] = {}
        self.tree_encoder_: TreeFeatureEncoder | None = None

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
        y_arr = s_y.to_numpy(dtype=np.float32)

        # 1. Fit tree feature encoder
        self.tree_encoder_ = TreeFeatureEncoder().fit(df_x)
        df_tree_x = self.tree_encoder_.transform(df_x)

        # 2. Get tree engines
        tree_specs = _get_multi_tree_models(self.random_state, is_classification=False)
        all_model_names = ["compacttab"] + [name for name, _ in tree_specs]
        num_models = len(all_model_names)

        # 3. Out-of-fold cross-validation
        if self.cv_folds > 1 and n_samples >= 40:
            kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
            oof_dict = {name: np.zeros(n_samples, dtype=np.float32) for name in all_model_names}

            for train_idx, val_idx in kf.split(df_x):
                X_tr, y_tr = df_x.iloc[train_idx], s_y.iloc[train_idx]
                X_val = df_x.iloc[val_idx]

                X_tr_tree = df_tree_x.iloc[train_idx]
                X_val_tree = df_tree_x.iloc[val_idx]

                # Fold NN
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
                oof_dict["compacttab"][val_idx] = fold_nn.predict(X_val)

                # Fold Trees
                for name, _ in tree_specs:
                    fold_trees = dict(_get_multi_tree_models(self.random_state, is_classification=False))
                    tree_m = fold_trees[name]
                    tree_m.fit(X_tr_tree, y_tr)
                    oof_dict[name][val_idx] = tree_m.predict(X_val_tree)

            def objective(weights: np.ndarray) -> float:
                w_norm = weights / max(np.sum(weights), 1e-12)
                pred_blend = np.zeros(n_samples, dtype=np.float32)
                for idx, name in enumerate(all_model_names):
                    pred_blend += w_norm[idx] * oof_dict[name]
                return float(np.mean((pred_blend - y_arr) ** 2))

            init_w = np.full(num_models, 1.0 / num_models, dtype=np.float64)
            bounds = [(0.0, 1.0) for _ in range(num_models)]
            cons = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

            try:
                res = minimize(
                    objective,
                    init_w,
                    bounds=bounds,
                    constraints=cons,
                    method="SLSQP",
                    options={"maxiter": 100},
                )
                if res.success and np.sum(res.x) > 0:
                    w_opt = res.x / np.sum(res.x)
                else:
                    w_opt = init_w
            except Exception:
                w_opt = init_w

            self.weights_ = {name: float(w_opt[idx]) for idx, name in enumerate(all_model_names)}
            self.blend_weight_ = self.weights_["compacttab"]
        else:
            w_uniform = 1.0 / num_models
            self.weights_ = {name: w_uniform for name in all_model_names}
            self.blend_weight_ = w_uniform

        # 4. Fit final production models
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

        self.tree_models_ = {}
        for name, tree_m in tree_specs:
            tree_m.fit(df_tree_x, s_y)
            self.tree_models_[name] = tree_m

        if tree_specs:
            self.tree_model_ = self.tree_models_[tree_specs[0][0]]

        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if self.nn_model_ is None or self.tree_encoder_ is None:
            raise RuntimeError("CompactTabHybridRegressor is not fitted.")

        df_x, _ = self._prepare_data(X)
        df_tree_x = self.tree_encoder_.transform(df_x)

        pred_nn = self.nn_model_.predict(df_x)
        pred_blend = self.weights_.get("compacttab", 0.5) * pred_nn

        for name, tree_m in self.tree_models_.items():
            w = self.weights_.get(name, 0.0)
            if w > 1e-4:
                try:
                    pred_tree = tree_m.predict(df_tree_x)
                    pred_blend += w * pred_tree
                except Exception:
                    pred_blend += w * pred_nn

        return pred_blend
