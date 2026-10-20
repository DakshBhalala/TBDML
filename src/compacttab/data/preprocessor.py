"""Tabular data preprocessor with strict training isolation and quantile binning."""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from compacttab.data.schema import DatasetSchema, FeatureType, TaskType


class TabularPreprocessor:
    """Robust tabular preprocessor fitted exclusively on training data."""

    def __init__(
        self,
        schema: DatasetSchema,
        num_bins: int = 16,
        handle_unknown: str = "impute",
    ):
        self.schema = schema
        self.num_bins = num_bins
        self.handle_unknown = handle_unknown

        self.num_medians_: Dict[str, float] = {}
        self.num_bin_edges_: Dict[str, np.ndarray] = {}
        self.cat_vocabs_: Dict[str, Dict[Any, int]] = {}
        self.cat_cardinalities_: Dict[str, int] = {}
        self.label_encoder_: Optional[LabelEncoder] = None
        self.target_mean_: float = 0.0
        self.target_std_: float = 1.0
        self.is_fitted_: bool = False

    def fit(
        self,
        df: pd.DataFrame,
        y: Optional[Union[pd.Series, np.ndarray]] = None,
    ) -> "TabularPreprocessor":
        """Fit preprocessor strictly on training features and targets."""
        # 1. Process Numerical Features
        for col in self.schema.numerical_cols:
            vals = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float32)
            valid_mask = ~np.isnan(vals)
            if np.any(valid_mask):
                median_val = float(np.median(vals[valid_mask]))
            else:
                median_val = 0.0
            self.num_medians_[col] = median_val

            clean_vals = np.where(valid_mask, vals, median_val)
            # Calculate quantile bin edges
            quantiles = np.linspace(0.0, 1.0, self.num_bins + 1)
            edges = np.quantile(clean_vals, quantiles)
            edges = np.unique(edges)
            if len(edges) < self.num_bins + 1:
                min_v = float(np.min(clean_vals))
                max_v = float(np.max(clean_vals))
                if np.isclose(min_v, max_v):
                    min_v -= 1.0
                    max_v += 1.0
                edges = np.linspace(min_v, max_v, self.num_bins + 1, dtype=np.float32)
            elif len(edges) > self.num_bins + 1:
                edges = edges[: self.num_bins + 1]
            self.num_bin_edges_[col] = edges.astype(np.float32)

        # 2. Process Categorical Features
        for col in self.schema.categorical_cols:
            series = df[col].astype(str).fillna("__MISSING__")
            unique_cats = sorted(series.unique().tolist())
            # Reserve 0 for Unknown/OOV/Missing
            vocab = {"__UNKNOWN__": 0}
            for idx, cat in enumerate(unique_cats, start=1):
                vocab[cat] = idx
            self.cat_vocabs_[col] = vocab
            # Total categories including OOV
            self.cat_cardinalities_[col] = len(vocab)

        # 3. Process Target
        if y is not None:
            y_arr = np.asarray(y)
            if self.schema.is_classification:
                self.label_encoder_ = LabelEncoder()
                self.label_encoder_.fit(y_arr)
                self.schema.num_classes = len(self.label_encoder_.classes_)
            else:
                self.target_mean_ = float(np.mean(y_arr))
                self.target_std_ = float(np.std(y_arr))
                if self.target_std_ < 1e-6:
                    self.target_std_ = 1.0
                self.schema.target_mean = self.target_mean_
                self.schema.target_std = self.target_std_

        self.is_fitted_ = True
        return self

    def transform(
        self,
        df: pd.DataFrame,
        y: Optional[Union[pd.Series, np.ndarray]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Transform dataframe into:
        - num_values: float array (N, num_cols)
        - num_masks: binary missingness mask (N, num_cols)
        - cat_indices: int64 array (N, cat_cols)
        - y_transformed: processed target array or None
        """
        if not self.is_fitted_:
            raise RuntimeError("TabularPreprocessor must be fitted before transform.")

        n_samples = len(df)

        # 1. Numerical values and missingness masks
        if self.schema.numerical_cols:
            num_vals = np.zeros((n_samples, len(self.schema.numerical_cols)), dtype=np.float32)
            num_masks = np.zeros((n_samples, len(self.schema.numerical_cols)), dtype=np.float32)

            for i, col in enumerate(self.schema.numerical_cols):
                raw = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float32)
                missing = np.isnan(raw)
                num_masks[:, i] = missing.astype(np.float32)
                clean = np.where(missing, self.num_medians_[col], raw)
                num_vals[:, i] = clean
        else:
            num_vals = np.empty((n_samples, 0), dtype=np.float32)
            num_masks = np.empty((n_samples, 0), dtype=np.float32)

        # 2. Categorical indices
        if self.schema.categorical_cols:
            cat_indices = np.zeros((n_samples, len(self.schema.categorical_cols)), dtype=np.int64)
            for i, col in enumerate(self.schema.categorical_cols):
                series = df[col].astype(str).fillna("__MISSING__")
                vocab = self.cat_vocabs_[col]
                # Map unseen or missing categories to 0
                cat_indices[:, i] = series.map(lambda c: vocab.get(c, 0)).to_numpy(dtype=np.int64)
        else:
            cat_indices = np.empty((n_samples, 0), dtype=np.int64)

        # 3. Target transform
        y_trans = None
        if y is not None:
            y_arr = np.asarray(y)
            if self.schema.is_classification:
                if self.label_encoder_ is not None:
                    y_trans = self.label_encoder_.transform(y_arr).astype(np.int64)
                else:
                    y_trans = y_arr.astype(np.int64)
            else:
                y_trans = ((y_arr - self.target_mean_) / self.target_std_).astype(np.float32)

        return num_vals, num_masks, cat_indices, y_trans

    def inverse_transform_target(self, y_pred: np.ndarray) -> np.ndarray:
        """Inverse transform model target predictions to original scale/labels."""
        if not self.is_fitted_:
            raise RuntimeError("TabularPreprocessor must be fitted first.")

        if self.schema.is_classification:
            if self.label_encoder_ is not None:
                # If probabilities are provided (N, C), take argmax first
                if y_pred.ndim == 2:
                    y_idx = np.argmax(y_pred, axis=1)
                else:
                    y_idx = y_pred.astype(np.int64)
                return self.label_encoder_.inverse_transform(y_idx)
            return y_pred
        else:
            # Regression inverse standard scale
            return (y_pred * self.target_std_) + self.target_mean_
