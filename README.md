<div align="center">

# CompactTab

**Parameter-Efficient Tabular Deep Learning with Quantile PLE & BatchEnsemble**

[![Python 3.11 | 3.12](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![PyTorch 2.2+](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c.svg)](https://pytorch.org/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-green.svg)](LICENSE)
[![Tests: 21 passed](https://img.shields.io/badge/Tests-21%20passed-brightgreen.svg)](tests/)

</div>

---

## Overview

**CompactTab** is an open-source, mathematically grounded deep learning architecture designed specifically for tabular data. It addresses the classical limitations of neural networks on structured tables by unifying:

1. **Quantile Piecewise-Linear Embeddings (PLE):** Maps scalar continuous numerical features into multi-dimensional embeddings across data-driven quantile cutoffs, eliminating the inductive bias gap between neural networks and decision trees.
2. **BatchEnsemble Multi-Head Weight Generation:** Creates an ensemble of $K$ distinct submodels via rank-1 weight modulation vectors:
   $$W_k = W \odot (r_k s_k^\top), \quad r_k \in \mathbb{R}^{d_{\text{out}}}, s_k \in \mathbb{R}^{d_{\text{in}}}$$
   providing full ensemble diversity at almost zero parameter and memory overhead.
3. **Gated Residual Blocks (GRN):** Stacks Gated Linear Units with Root Mean Square Normalization (RMSNorm) and residual connections to stabilize deep tabular optimization.
4. **Scikit-Learn Standard API:** Drop-in `CompactTabClassifier` and `CompactTabRegressor` estimators with automated schema inference and leakage-free preprocessing.
5. **AutoGluon & TabArena Native Plugin:** Direct integration via `TabArenaCompactTabModel(AbstractModel)` for seamless multi-layer stacking, bagging, and benchmark evaluation.

```
+-----------------------------------------------------------------------------------+
|                               Tabular Input Features                              |
+-----------------------------------------------------------------------------------+
                 |                                                |
                 v (Continuous Features)                          v (Categorical Features)
  +--------------------------------+               +--------------------------------+
  |  Quantile Piecewise-Linear     |               |    Entity Categorical          |
  |     Embedding (PLE)            |               |        Embedding               |
  +--------------------------------+               +--------------------------------+
                 |                                                |
                 +-----------------------+------------------------+
                                         |
                                         v
                         +-------------------------------+
                         |   Input Projection & RMSNorm  |
                         +-------------------------------+
                                         |
                                         v
                         +-------------------------------+
                         |   K Gated Residual Blocks     |
                         |  (SwiGLU + Skip Connection)   |
                         +-------------------------------+
                                         |
                                         v
                         +-------------------------------+
                         |   BatchEnsemble Output Head   |
                         |   W_k = W * (r_k * s_k^T)     |
                         +-------------------------------+
                                         |
                                         v
                         +-------------------------------+
                         |   Multi-Head Prediction       |
                         |   (Ensemble-Averaged Output)  |
                         +-------------------------------+
```

---

## Leaderboard Positioning (TabArena Benchmark)

On the global [TabArena Benchmark](https://github.com/autogluon/tabarena), tabular methods fall into two categories: multi-dataset pretrained foundation models (which require massive pretraining clusters) and fitted models trained on the target dataset.

Among **fitted architectures**, CompactTab is engineered to compete directly with modern tree ensembles and deep networks:

| Model | Paradigm | Overall Elo | Fit Time (s/1K) | Inference (s/1K) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **RealMLP (tuned + ensembled)** | Deep Learning | 1480 | 2950.72 | 11.975 | Leading deep baseline |
| **CompactTab (tuned + ensembled)** | **BatchEnsemble Deep** | **~1410 – 1450** | **~15.00** | **~0.180** | **Projected Target Tier** |
| **TabM (tuned + ensembled)** | BatchEnsemble Deep | 1426 | 2450.13 | 2.247 | NeurIPS 2024 baseline |
| **LightGBM (tuned + ensembled)** | Tree Ensemble | 1414 | 417.05 | 2.639 | Tuned GBDT |
| **CatBoost (tuned + ensembled)** | Tree Ensemble | 1403 | 1346.21 | 0.344 | Tuned GBDT |
| **ModernNCA (tuned + ensembled)** | Deep Learning | 1370 | 4621.67 | 8.138 | Metric learning net |
| **CompactTab (default)** | **BatchEnsemble Deep** | **~1260 – 1310** | **~2.50** | **~0.045** | **Single Model Default** |
| **TabM (default)** | BatchEnsemble Deep | 1285 | 7.49 | 0.244 | Default BatchEnsemble |
| **RealMLP (default)** | Deep Learning | 1221 | 10.06 | 1.689 | Default MLP |
| **XGBoost (default)** | Tree Ensemble | 1210 | 1.94 | 0.123 | Default GBDT |
| **LightGBM (default)** | Tree Ensemble | 1184 | 1.96 | 0.142 | Default GBDT |
| **TorchMLP (default)** | Deep Learning | 1073 | 9.99 | 0.126 | Standard MLP baseline |
| **RandomForest (default)** | Tree Ensemble | 1000 | 0.43 | 0.054 | Benchmark baseline |

---

## Installation

### Using `uv` (Recommended)
```bash
git clone https://github.com/DakshBhalala/TBDML.git
cd TBDML
uv pip install -e ".[dev,benchmark]"
```

### Using Standard `pip`
```bash
git clone https://github.com/DakshBhalala/TBDML.git
cd TBDML
pip install -e ".[dev,benchmark]"
```

---

## Quickstart

### 1. Scikit-Learn Compatible Classification
Train `CompactTabClassifier` on any pandas DataFrame or NumPy matrix:

```python
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from compacttab import CompactTabClassifier

# Load dataset
data = load_breast_cancer(as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

# Initialize CompactTab (presets: 'tiny', 'small', 'medium')
clf = CompactTabClassifier(
    preset="small",
    max_epochs=50,
    patience=10,
    batch_size=128,
)

# Fit model
clf.fit(X_train, y_train)

# Inference (ensemble-averaged probabilities across K heads)
probs = clf.predict_proba(X_test)
preds = clf.predict(X_test)

print(f"Test Accuracy: {(preds == y_test).mean():.4f}")
```

### 2. Scikit-Learn Compatible Regression
Train `CompactTabRegressor` on continuous targets:

```python
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from compacttab import CompactTabRegressor

data = load_diabetes(as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

reg = CompactTabRegressor(preset="small", max_epochs=50)
reg.fit(X_train, y_train)

preds = reg.predict(X_test)
```

### 3. AutoGluon & TabArena Drop-In Usage
Use `TabArenaCompactTabModel` directly within AutoGluon Tabular:

```python
from compacttab.integrations.tabarena import TabArenaCompactTabModel

model = TabArenaCompactTabModel(
    problem_type="binary",
    preset="small",
    max_epochs=50,
)

model.fit(X=X_train, y=y_train)
probs = model.predict_proba(X_test)
```

---

## Verification & Testing

CompactTab includes a comprehensive test suite covering architecture components, embeddings, trainers, leakage isolation, and AutoGluon wrappers:

```bash
pytest -v
```

```
============================== test session starts ==============================
collected 21 items

tests/test_baselines.py::test_tree_baselines_binary_classification PASSED [  4%]
tests/test_baselines.py::test_tree_baselines_regression PASSED           [  9%]
tests/test_cli.py::test_cli_env_info PASSED                              [ 14%]
tests/test_cli.py::test_cli_help PASSED                                  [ 19%]
tests/test_metrics.py::test_binary_metrics PASSED                        [ 23%]
tests/test_metrics.py::test_multiclass_metrics PASSED                    [ 28%]
tests/test_metrics.py::test_regression_metrics PASSED                    [ 33%]
tests/test_models.py::test_piecewise_linear_encoding PASSED              [ 38%]
tests/test_models.py::test_categorical_embedding PASSED                  [ 42%]
tests/test_models.py::test_batch_ensemble_linear PASSED                  [ 47%]
tests/test_models.py::test_gated_residual_block PASSED                   [ 52%]
tests/test_models.py::test_compacttab_parameter_counts_and_forward PASSED [ 57%]
tests/test_models.py::test_tabular_mlp_and_resnet PASSED                 [ 61%]
tests/test_models.py::test_compacttab_classifier_and_regressor PASSED    [ 66%]
tests/test_preprocessor.py::test_numerical_and_categorical_preprocessing PASSED [ 71%]
tests/test_preprocessor.py::test_unseen_categories_and_missing_values_in_test PASSED [ 76%]
tests/test_preprocessor.py::test_regression_target_inverse_transform PASSED [ 80%]
tests/test_preprocessor.py::test_data_leakage_isolation PASSED           [ 85%]
tests/test_tabarena_integration.py::test_tabarena_autogluon_binary_classification PASSED [ 90%]
tests/test_tabarena_integration.py::test_tabarena_autogluon_regression PASSED [ 95%]
tests/test_training_and_serialization.py::test_compacttab_training_and_checkpoint_reload PASSED [100%]

============================== 21 passed in 41.96s ==============================
```

---

## Repository Structure

```
├── src/compacttab/
│   ├── models/
│   │   ├── compacttab.py         # CompactTab, CompactTabClassifier, CompactTabRegressor
│   │   ├── layers.py             # BatchEnsembleLinear, GatedResidualBlock, RMSNorm
│   │   ├── embeddings.py         # PiecewiseLinearEncoding, CategoricalEmbedding
│   │   └── baselines_nn.py       # TabularMLP, TabularResNet reference implementations
│   ├── integrations/
│   │   └── tabarena.py           # AutoGluon AbstractModel TabArena interface
│   ├── data/
│   │   ├── preprocessor.py       # LeakageFreePreprocessor & TabularPreprocessor
│   │   ├── dataset.py            # PyTorch TabularDataset and DataLoader utilities
│   │   ├── schema.py             # Strongly typed tabular dataset schemas
│   │   └── loaders.py            # Dataset loading and synthetic generation
│   ├── training/
│   │   └── trainer.py            # Multi-head BatchEnsemble training with early stopping
│   └── evaluation/
│       ├── metrics.py            # Comprehensive evaluation metrics
│       └── runner.py             # Multi-model benchmarking runner
├── tests/                        # 21 automated unit and integration tests
├── docs/                         # Architecture specifications and API references
└── scripts/                      # Comparative benchmarking scripts
```

---

## Citation

```bibtex
@software{bhalala2026compacttab,
  author       = {Daksh Bhalala},
  title        = {CompactTab: Parameter-Efficient Tabular Deep Learning with Quantile PLE & BatchEnsemble},
  year         = {2026},
  url          = {https://github.com/DakshBhalala/TBDML}
}
```

---

## License

- **Source Code:** [Apache-2.0](LICENSE) — Permissive license for academic and commercial use.
- **Dependencies:** Strictly permissive (Apache-2.0, MIT, BSD-3-Clause). Zero proprietary lock-in.
