"""CompactTab: State-of-the-Art Compact Tabular Deep Learning Architecture."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.model_selection import train_test_split

from compacttab.data.dataset import TabularDataset, create_dataloader
from compacttab.data.preprocessor import TabularPreprocessor
from compacttab.data.schema import DatasetSchema, TaskType
from compacttab.models.embeddings import CategoricalEmbedding, PiecewiseLinearEncoding
from compacttab.models.layers import BatchEnsembleLinear, GatedResidualBlock, RMSNorm
from compacttab.training.trainer import TabularTrainer


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


class BaseCompactTabEstimator(BaseEstimator):
    """Base class for high-level scikit-learn compatible CompactTab estimators."""

    def __init__(
        self,
        preset: str = "small",
        hidden_dim: int | None = None,
        num_layers: int | None = None,
        num_heads: int | None = None,
        num_bins: int | None = None,
        embed_dim: int | None = None,
        dropout: float | None = None,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        max_epochs: int = 50,
        patience: int = 10,
        batch_size: int = 128,
        val_fraction: float = 0.15,
        device: str | None = None,
        random_state: int | None = 42,
        verbose: bool = False,
    ):
        self.preset = preset
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_bins = num_bins
        self.embed_dim = embed_dim
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.val_fraction = val_fraction
        self.device = device
        self.random_state = random_state
        self.verbose = verbose

        self.model_: CompactTab | None = None
        self.preprocessor_: TabularPreprocessor | None = None
        self.trainer_: TabularTrainer | None = None
        self.schema_: DatasetSchema | None = None
        self.classes_: np.ndarray | None = None

    def _get_config(self) -> CompactTabConfig:
        overrides: dict[str, Any] = {}
        for k in ("hidden_dim", "num_layers", "num_heads", "num_bins", "embed_dim", "dropout"):
            val = getattr(self, k)
            if val is not None:
                overrides[k] = val
        return CompactTabConfig.from_preset(self.preset, **overrides)

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


class CompactTabClassifier(BaseCompactTabEstimator, ClassifierMixin):
    """Scikit-Learn compatible CompactTab Classifier."""

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        X_val: pd.DataFrame | np.ndarray | None = None,
        y_val: pd.Series | np.ndarray | None = None,
    ) -> "CompactTabClassifier":
        df_x, s_y = self._prepare_data(X, y)
        self.classes_ = np.sort(np.unique(np.asarray(s_y)))
        n_classes = len(self.classes_)
        task_type = TaskType.BINARY if n_classes == 2 else TaskType.MULTICLASS

        num_cols = list(df_x.select_dtypes(include=["number"]).columns)
        cat_cols = list(df_x.select_dtypes(exclude=["number"]).columns)

        self.schema_ = DatasetSchema(
            task_type=task_type,
            target_col="__target__",
            numerical_cols=num_cols,
            categorical_cols=cat_cols,
            num_classes=n_classes,
        )

        cfg = self._get_config()
        self.preprocessor_ = TabularPreprocessor(schema=self.schema_, num_bins=cfg.num_bins)

        if X_val is not None and y_val is not None:
            self.preprocessor_.fit(df_x, s_y)
            num_tr, mask_tr, cat_tr, y_tr = self.preprocessor_.transform(df_x, s_y)
            train_dataset = TabularDataset(num_tr, mask_tr, cat_tr, y_tr)

            df_val_x, s_val_y = self._prepare_data(X_val, y_val)
            num_val, mask_val, cat_val, y_v = self.preprocessor_.transform(df_val_x, s_val_y)
            val_dataset = TabularDataset(num_val, mask_val, cat_val, y_v)
        elif self.val_fraction > 0 and len(df_x) >= 40:
            df_train_x, df_val_x, s_train_y, s_val_y = train_test_split(
                df_x, s_y, test_size=self.val_fraction, random_state=self.random_state, stratify=s_y
            )
            self.preprocessor_.fit(df_train_x, s_train_y)
            num_tr, mask_tr, cat_tr, y_tr = self.preprocessor_.transform(df_train_x, s_train_y)
            train_dataset = TabularDataset(num_tr, mask_tr, cat_tr, y_tr)

            num_val, mask_val, cat_val, y_v = self.preprocessor_.transform(df_val_x, s_val_y)
            val_dataset = TabularDataset(num_val, mask_val, cat_val, y_v)
        else:
            self.preprocessor_.fit(df_x, s_y)
            num_tr, mask_tr, cat_tr, y_tr = self.preprocessor_.transform(df_x, s_y)
            train_dataset = TabularDataset(num_tr, mask_tr, cat_tr, y_tr)
            val_dataset = None

        self.model_ = CompactTab(
            schema=self.schema_,
            bin_edges=self.preprocessor_.num_bin_edges_,
            cat_cardinalities=self.preprocessor_.cat_cardinalities_,
            config=cfg,
        )

        self.trainer_ = TabularTrainer(
            model=self.model_,
            schema=self.schema_,
            device=self.device,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            max_epochs=self.max_epochs,
            patience=self.patience,
            batch_size=self.batch_size,
            verbose=self.verbose,
        )

        self.trainer_.fit(train_dataset, val_dataset)
        return self

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if self.trainer_ is None or self.preprocessor_ is None:
            raise RuntimeError("CompactTabClassifier is not fitted.")
        df_x, _ = self._prepare_data(X)
        num_vals, num_masks, cat_indices, _ = self.preprocessor_.transform(df_x)
        dataset = TabularDataset(num_vals, num_masks, cat_indices)
        loader = create_dataloader(dataset, batch_size=self.batch_size, shuffle=False)
        return self.trainer_.predict_proba(loader)

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        idx = np.argmax(probs, axis=1)
        if self.classes_ is not None and len(self.classes_) == probs.shape[1]:
            return self.classes_[idx]
        return idx


class CompactTabRegressor(BaseCompactTabEstimator, RegressorMixin):
    """Scikit-Learn compatible CompactTab Regressor."""

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        X_val: pd.DataFrame | np.ndarray | None = None,
        y_val: pd.Series | np.ndarray | None = None,
    ) -> "CompactTabRegressor":
        df_x, s_y = self._prepare_data(X, y)
        self.classes_ = None

        num_cols = list(df_x.select_dtypes(include=["number"]).columns)
        cat_cols = list(df_x.select_dtypes(exclude=["number"]).columns)

        self.schema_ = DatasetSchema(
            task_type=TaskType.REGRESSION,
            target_col="__target__",
            numerical_cols=num_cols,
            categorical_cols=cat_cols,
            num_classes=None,
        )

        cfg = self._get_config()
        self.preprocessor_ = TabularPreprocessor(schema=self.schema_, num_bins=cfg.num_bins)

        if X_val is not None and y_val is not None:
            self.preprocessor_.fit(df_x, s_y)
            num_tr, mask_tr, cat_tr, y_tr = self.preprocessor_.transform(df_x, s_y)
            train_dataset = TabularDataset(num_tr, mask_tr, cat_tr, y_tr)

            df_val_x, s_val_y = self._prepare_data(X_val, y_val)
            num_val, mask_val, cat_val, y_v = self.preprocessor_.transform(df_val_x, s_val_y)
            val_dataset = TabularDataset(num_val, mask_val, cat_val, y_v)
        elif self.val_fraction > 0 and len(df_x) >= 40:
            df_train_x, df_val_x, s_train_y, s_val_y = train_test_split(
                df_x, s_y, test_size=self.val_fraction, random_state=self.random_state
            )
            self.preprocessor_.fit(df_train_x, s_train_y)
            num_tr, mask_tr, cat_tr, y_tr = self.preprocessor_.transform(df_train_x, s_train_y)
            train_dataset = TabularDataset(num_tr, mask_tr, cat_tr, y_tr)

            num_val, mask_val, cat_val, y_v = self.preprocessor_.transform(df_val_x, s_val_y)
            val_dataset = TabularDataset(num_val, mask_val, cat_val, y_v)
        else:
            self.preprocessor_.fit(df_x, s_y)
            num_tr, mask_tr, cat_tr, y_tr = self.preprocessor_.transform(df_x, s_y)
            train_dataset = TabularDataset(num_tr, mask_tr, cat_tr, y_tr)
            val_dataset = None

        self.model_ = CompactTab(
            schema=self.schema_,
            bin_edges=self.preprocessor_.num_bin_edges_,
            cat_cardinalities=self.preprocessor_.cat_cardinalities_,
            config=cfg,
        )

        self.trainer_ = TabularTrainer(
            model=self.model_,
            schema=self.schema_,
            device=self.device,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            max_epochs=self.max_epochs,
            patience=self.patience,
            batch_size=self.batch_size,
            verbose=self.verbose,
        )

        self.trainer_.fit(train_dataset, val_dataset)
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if self.trainer_ is None or self.preprocessor_ is None:
            raise RuntimeError("CompactTabRegressor is not fitted.")
        df_x, _ = self._prepare_data(X)
        num_vals, num_masks, cat_indices, _ = self.preprocessor_.transform(df_x)
        dataset = TabularDataset(num_vals, num_masks, cat_indices)
        loader = create_dataloader(dataset, batch_size=self.batch_size, shuffle=False)
        raw_preds = self.trainer_.predict(loader)
        return self.preprocessor_.inverse_transform_target(raw_preds)
