# CompactTab API Reference

This document provides detailed API specifications for the core estimators, neural architecture, and utilities in the `compacttab` library.

---

## 1. High-Level Estimators (`compacttab.models`)

### `CompactTabClassifier`

Scikit-Learn compatible classifier using Quantile Piecewise-Linear Embeddings and BatchEnsemble gated residual blocks.

```python
from compacttab import CompactTabClassifier

clf = CompactTabClassifier(
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
)
```

#### Parameters:
- `preset` (*str*, default=`"small"`): Predefined architectural scale:
  - `"tiny"`: ~100k parameters (`hidden_dim=64`, `num_layers=2`, `num_heads=4`, `embed_dim=12`).
  - `"small"`: ~1M parameters (`hidden_dim=128`, `num_layers=4`, `num_heads=8`, `embed_dim=24`).
  - `"medium"`: ~10M parameters (`hidden_dim=384`, `num_layers=6`, `num_heads=8`, `embed_dim=32`).
- `learning_rate` (*float*, default=`1e-3`): Initial AdamW learning rate with cosine annealing schedule.
- `weight_decay` (*float*, default=`1e-4`): L2 regularization factor.
- `max_epochs` (*int*, default=`50`): Maximum training iterations.
- `patience` (*int*, default=`10`): Number of validation check epochs without loss improvement before early stopping.
- `batch_size` (*int*, default=`128`): Mini-batch training size.
- `val_fraction` (*float*, default=`0.15`): Fraction of training data held out for early stopping if no explicit validation split is passed.
- `device` (*str* or *None*, default=`None`): Target device (`"cuda"` or `"cpu"`).

#### Methods:
- `fit(X, y, X_val=None, y_val=None)`: Fits the preprocessor, initializes weights, and trains the network.
- `predict_proba(X)`: Computes normalized class probabilities averaged across the $K$ ensemble heads.
- `predict(X)`: Predicts discrete class labels.
- `score(X, y)`: Computes mean classification accuracy.

---

### `CompactTabRegressor`

Scikit-Learn compatible regressor for continuous target prediction.

```python
from compacttab import CompactTabRegressor

reg = CompactTabRegressor(
    preset: str = "small",
    max_epochs: int = 50,
    patience: int = 10,
    batch_size: int = 128,
)
```

#### Parameters & Methods:
Identical signature to `CompactTabClassifier`, with target standardization and MSE loss optimization. `predict(X)` returns continuous 1D predictions on the original scale.

---

## 2. AutoGluon & TabArena Integration (`compacttab.integrations.tabarena`)

### `TabArenaCompactTabModel`

AutoGluon `AbstractModel` drop-in wrapper designed for TabArena benchmark evaluation, multi-layer stacking, and bagging.

```python
from compacttab.integrations.tabarena import TabArenaCompactTabModel

model = TabArenaCompactTabModel(
    name: str = "CompactTab",
    problem_type: str = "binary",  # "binary", "multiclass", or "regression"
    preset: str = "small",
    max_epochs: int = 50,
    batch_size: int = 128,
)
```

#### AutoGluon Compliance:
- Fully implements `_fit(X, y, X_val, y_val, time_limit, num_cpus, num_gpus, **kwargs)`.
- Implements `_predict(X, **kwargs)` and `_predict_proba(X, **kwargs)`.
- Accepts continuous, categorical, and object dtypes via `_get_default_auxiliary_params()`.

---

## 3. PyTorch Architecture (`compacttab.models.compacttab`)

### `CompactTab`

Low-level PyTorch neural network module.

```python
from compacttab.models.compacttab import CompactTab, CompactTabConfig

model = CompactTab(
    schema: DatasetSchema,
    bin_edges: dict[str, np.ndarray],
    cat_cardinalities: dict[str, int],
    config: CompactTabConfig | None = None,
)
```

#### Core Components:
- **`PiecewiseLinearEncoding`:** Continuous feature numerical tokenizer.
- **`CategoricalEmbedding`:** Entity embedding table with OOV handling.
- **`GatedResidualBlock`:** SwiGLU-inspired gated feed-forward block with RMSNorm.
- **`BatchEnsembleLinear`:** Multi-head rank-1 modulated linear layer.
