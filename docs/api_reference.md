# CompactTab API Reference

This document provides detailed API specifications for the core classes and functions in the `compacttab` library.

---

## 1. Foundation Models (`compacttab.models`)

### `CompactTabFoundation`

In-context Tabular Foundation Model (TFM) utilizing pretrained Structural Causal Model (SCM) transformer architectures.

```python
from compacttab.models import CompactTabFoundation

model = CompactTabFoundation(
    size: str = "small",
    device: str | None = None,
    num_estimators: int = 4,
    random_state: int | None = 42,
)
```

#### Parameters:
- `size` (*str*, default=`"small"`): Model scale checkpoint:
  - `"small"`: ~27.9M parameters, 109 MB weights. Optimal speed-accuracy trade-off targeting ~1803 Elo.
  - `"medium"`: ~62.3M parameters, 246 MB weights. High accuracy targeting ~1902 Elo.
  - `"large"`: ~215M parameters, 860 MB weights. Maximum representation capacity targeting ~1962 Elo.
- `device` (*str* or *None*, default=`None`): Target device (`"cuda"` or `"cpu"`). If `None`, automatically resolves to `"cuda"` if available.
- `num_estimators` (*int*, default=`4`): Number of ensemble forward passes over context subsamples.
- `random_state` (*int* or *None*, default=`42`): Seed for reproducible context sampling.

#### Methods:

##### `fit(X, y)`
Conditions the model on the labeled context dataset $\mathcal{D}_{\text{train}} = (X, y)$.
- **`X`** (*pd.DataFrame* or *np.ndarray*): Feature matrix of shape `(n_samples, n_features)`.
- **`y`** (*pd.Series* or *np.ndarray*): Target vector of shape `(n_samples,)`.
- **Returns:** `self`

##### `predict(X)`
Predicts discrete class labels for classification or continuous values for regression.
- **`X`** (*pd.DataFrame* or *np.ndarray*): Query feature matrix of shape `(n_query, n_features)`.
- **Returns:** `np.ndarray` of shape `(n_query,)`.

##### `predict_proba(X)`
Estimates class probability distributions for classification tasks.
- **`X`** (*pd.DataFrame* or *np.ndarray*): Query feature matrix of shape `(n_query, n_features)`.
- **Returns:** `np.ndarray` of shape `(n_query, n_classes)` normalized such that row sums equal 1.0.

---

## 2. AutoGluon & TabArena Integration (`compacttab.integrations.tabarena`)

### `TabArenaCompactTabModel`

AutoGluon `AbstractModel` drop-in wrapper designed for TabArena benchmark evaluation, multi-layer stacking, and bagging.

```python
from compacttab.integrations.tabarena import TabArenaCompactTabModel

model = TabArenaCompactTabModel(
    problem_type: str,
    foundation_size: str = "small",
    eval_metric: str | None = None,
    hyperparameters: dict | None = None,
    **kwargs,
)
```

#### Parameters:
- `problem_type` (*str*): One of `"binary"`, `"multiclass"`, or `"regression"`.
- `foundation_size` (*str*, default=`"small"`): Foundation model size variant (`"small"`, `"medium"`, `"large"`).
- `eval_metric` (*str* or *None*, default=`None`): Optimization metric string recognized by AutoGluon (e.g. `"roc_auc"`, `"log_loss"`, `"root_mean_squared_error"`).
- `hyperparameters` (*dict* or *None*, default=`None`): Hyperparameter dictionary passed to model (e.g. `{"num_estimators": 4}`).

#### AutoGluon Compliance:
- Fully implements `_fit(X, y, X_val, y_val, time_limit, num_cpus, num_gpus, **kwargs)`.
- Implements `_predict(X, **kwargs)` and `_predict_proba(X, **kwargs)`.
- Properly declares `_get_default_auxiliary_params()` to accept continuous, categorical, and object dtypes without destructive truncation.

---

## 3. Fitted Deep Learning Architecture (`compacttab.models.compacttab`)

### `CompactTab`

Parameter-efficient PyTorch neural network combining Quantile PLE embeddings and BatchEnsemble gated residual blocks.

```python
from compacttab.models.compacttab import CompactTab

model = CompactTab(
    d_in: int,
    d_out: int,
    d_hidden: int = 128,
    n_blocks: int = 4,
    ensemble_size: int = 4,
    dropout: float = 0.1,
)
```

#### Parameters:
- `d_in` (*int*): Total dimensionality of the concatenated input feature embeddings.
- `d_out` (*int*): Output dimension (1 for binary classification / regression; `C` for multiclass).
- `d_hidden` (*int*, default=`128`): Width of the intermediate gated residual representations.
- `n_blocks` (*int*, default=`4`): Number of stacked Gated Residual Blocks.
- `ensemble_size` (*int*, default=`4`): Number of rank-1 ensemble members in BatchEnsemble layers.
- `dropout` (*float*, default=`0.1`): Dropout probability applied across intermediate linear transformations.

---

## 4. Preprocessing & Leakage Isolation (`compacttab.data.preprocessor`)

### `LeakageFreePreprocessor`

Strictly partitioned tabular preprocessor that calculates all summary statistics, quantile bin boundaries, and imputation values solely on the training fold.

```python
from compacttab.data.preprocessor import LeakageFreePreprocessor

preprocessor = LeakageFreePreprocessor(
    n_bins: int = 32,
    cat_min_freq: int = 5,
    handle_unknown: str = "ignore",
)
```

#### Parameters:
- `n_bins` (*int*, default=`32`): Number of quantile bins for Piecewise Linear Embedding.
- `cat_min_freq` (*int*, default=`5`): Minimum occurrences for a categorical level before being collapsed into an Out-of-Vocabulary token.
- `handle_unknown` (*str*, default=`"ignore"`): Strategy for handling unseen categories at test time.

#### Methods:
- `fit(X, y=None)`: Derives quantile thresholds and categorical vocabularies exclusively from `X`.
- `transform(X)`: Applies derived binning and encoding to test/validation splits without updating statistics.
- `fit_transform(X, y=None)`: Convenience method executing `fit` followed by `transform`.
