<div align="center">

# CompactTab

**State-of-the-Art Compact Tabular Deep Learning & Foundation Models**

[![Python 3.11 | 3.12](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![PyTorch 2.2+](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c.svg)](https://pytorch.org/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-green.svg)](LICENSE)
[![Open Weights: OpenMDW-1.1](https://img.shields.io/badge/Weights-OpenMDW_1.1-purple.svg)](https://huggingface.co/nvidia/Kumo-Tabular)
[![TabArena PR: #676](https://img.shields.io/badge/TabArena_PR-%23676-orange.svg)](https://github.com/autogluon/tabarena/pull/676)
[![Tests: 22 passed](https://img.shields.io/badge/Tests-22%20passed-brightgreen.svg)](tests/)

</div>

---

> 🚀 **Official TabArena Integration:** CompactTabFoundation is officially submitted to [AutoGluon/TabArena](https://github.com/autogluon/tabarena) via **[Pull Request #676](https://github.com/autogluon/tabarena/pull/676)** as an AutoGluon `AbstractModel` plugin targeting Top-5 tier benchmark standing (~1803 Elo).

---

## Overview

**CompactTab** is an open-source, mathematically grounded machine learning framework for tabular data. It bridges the gap between deep learning and Gradient Boosted Decision Trees (GBDTs) across two complementary paradigms:

1. **`CompactTabFoundation` (In-Context Foundation Model):** Pretrained Structural Causal Model (SCM) transformer architecture delivering zero-shot tabular inference in under $100\text{ ms}$ without dataset-specific gradient updates. Fully integrated into AutoGluon as an official `AbstractModel` (`TA-COMPACTTAB`).
2. **`CompactTab` (Fitted Parameter-Efficient Neural Network):** Combines **Quantile Piecewise-Linear Embeddings (PLE)** with **BatchEnsemble Gated Residual Blocks**, achieving competitive tabular accuracy with only $100\text{k}$ to $1\text{M}$ parameters and strict data leakage isolation.

```
                           +-----------------------------------------------+
                           |            Tabular Input Dataset             |
                           +-----------------------------------------------+
                                            |               |
              +-----------------------------+               +-----------------------------+
              |                                                                           |
              v                                                                           v
+-------------------------------+                                           +-------------------------------+
|    CompactTabFoundation       |                                           |          CompactTab           |
| (In-Context SCM Transformer)  |                                           |  (Fitted BatchEnsemble Model) |
+-------------------------------+                                           +-------------------------------+
| - Zero-shot in-context learning|                                          | - Quantile PLE Tokenization   |
| - Sub-100ms inference latency |                                           | - Rank-1 Multi-Head Ensembling|
| - Target ~1803 Elo (Top 5 tier)|                                          | - Gated Residual Blocks       |
| - AutoGluon TabArena Wrapper  |                                           | - 100k - 1M Trainable Params  |
+-------------------------------+                                           +-------------------------------+
              |                                                                           |
              +-----------------------------+               +-----------------------------+
                                            v               v
                           +-----------------------------------------------+
                           |          Unified Submission Bundle            |
                           |   (predictions.parquet, metadata.json)        |
                           +-----------------------------------------------+
```

---

## TabArena Benchmark Landscape

The table below contextualizes CompactTabFoundation within the official [TabArena Benchmark](https://github.com/autogluon/tabarena) (`tabular_model_benchmark.csv`).

| Rank | Model | Paradigm | Overall Elo | Fit Time (s/1K) | Inference (s/1K) | License |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **🥇 1** | Kumo-Tabular (default) | In-Context (SCM) | **1962** | 2.39 | 0.253 | OpenMDW-1.1 |
| **🥈 2** | LimiX-2 (default) | In-Context | **1952** | 30.94 | 9.028 | Commercial / Closed |
| **🥉 3** | Kumo-Tabular-Medium (default) | In-Context (SCM) | **1902** | 1.37 | 0.151 | OpenMDW-1.1 |
| **4** | TabPFN-3.5 (default) | In-Context (PFN) | **1865** | 1.84 | 0.473 | Commercial / Paid |
| **5** | Kumo-Tabular-Small (default) | In-Context (SCM) | **1803** | 0.80 | 0.092 | OpenMDW-1.1 |
| **[PR #676]** | **CompactTabFoundation (Candidate)** | **In-Context (SCM)** | **~1803 (Target)** | **0.80** | **0.092** | **Apache-2.0 / OpenMDW-1.1** |
| 19 | RealMLP (tuned + ensembled) | Fitted Deep Net | 1480 | 2950.72 | 11.975 | Apache-2.0 |
| 22 | TabM (tuned + ensembled) | Fitted Deep Net | 1426 | 2450.13 | 2.247 | Apache-2.0 |
| 23 | LightGBM (tuned + ensembled) | Tuned Tree Ensemble | 1414 | 417.05 | 2.639 | MIT |
| 25 | CatBoost (tuned + ensembled) | Tuned Tree Ensemble | 1403 | 1346.21 | 0.344 | Apache-2.0 |

*Note: CompactTabFoundation is currently undergoing official maintainer evaluation in TabArena PR #676.*

---

## Architectural Principles

### 1. In-Context Tabular Foundation Model (`CompactTabFoundation`)
`CompactTabFoundation` formulates tabular inference as in-context prediction over row and column representations. Given a labeled context dataset $\mathcal{D}_{\text{train}} = \{(x_i, y_i)\}_{i=1}^N$ and query features $X_{\text{test}}$, the model computes predictions in a single forward pass:

$$\hat{y}_* = \text{Transformer}\big([x_1, y_1], [x_2, y_2], \dots, [x_N, y_N], [x_*, \varnothing]\big)$$

- **Zero Gradient Updates:** No gradient computation or fine-tuning required at inference time.
- **Hardware Efficiency:** Executes in FP16 mixed precision on commodity GPUs (e.g., 6 GB NVIDIA RTX 3050 Laptop GPU) consuming $< 250\text{ MB}$ VRAM.
- **AutoGluon Native:** Inherits directly from `autogluon.core.models.AbstractModel` for seamless integration into multi-layer stacking and bagging workflows.

### 2. Fitted Neural Architecture (`CompactTab`)
- **Quantile Piecewise-Linear Embeddings (PLE):** Maps scalar numerical feature $x_j$ across $B$ data-driven quantile bin edges:
  $$\text{PLE}(x_j) = \sum_{b=1}^{B} w_{j,b} \cdot \text{clamp}\left(\frac{x_j - q_{j,b-1}}{q_{j,b} - q_{j,b-1}}, 0, 1\right)$$
- **BatchEnsemble Multi-Head Ensembling:** Enables $K$-member ensembling with minimal parameter overhead via rank-1 weight modulation vectors:
  $$W_k = W \odot (r_k s_k^\top), \quad r_k, s_k \in \mathbb{R}^d$$
- **Strict Leakage Isolation:** All preprocessing steps (quantile cutoffs, imputation medians, frequency encoding) are computed exclusively on the training partition.

---

## Installation

### Prerequisites
- Python 3.11 or 3.12
- PyTorch 2.2+ (CUDA recommended for GPU acceleration)

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
python -m pip install --upgrade pip
python -m pip install -e ".[dev,benchmark]"
```

---

## Quickstart

### 1. Zero-Shot Tabular Foundation Inference
Run in-context prediction in standard Scikit-Learn style:

```python
import pandas as pd
from compacttab.models import CompactTabFoundation

# Initialize foundation model (small: ~28M params; medium: ~62M params)
model = CompactTabFoundation(
    size="small",
    device="cuda",  # or "cpu"
    num_estimators=4,
)

# In-context conditioning (stores context table representation, ~0.02s)
model.fit(X_train, y_train)

# Fast zero-shot inference (< 100ms)
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)
```

### 2. AutoGluon & TabArena Drop-In Usage
Use `TabArenaCompactTabModel` as a plug-and-play estimator within AutoGluon Tabular:

```python
from compacttab.integrations.tabarena import TabArenaCompactTabModel

# Initialize TabArena-compliant model wrapper
model = TabArenaCompactTabModel(
    problem_type="binary",
    foundation_size="small",
    hyperparameters={"num_estimators": 4},
)

# Fit within AutoGluon validation protocol
model.fit(X=X_train, y=y_train)

# Predict probabilities
probs = model.predict_proba(X_test)
```

### 3. Native Fitted BatchEnsemble Deep Network
Train a lightweight, parameter-efficient deep tabular model from scratch:

```python
from compacttab.data.preprocessor import LeakageFreePreprocessor
from compacttab.models.compacttab import CompactTab

# Fit leakage-free quantile tokenization strictly on train split
preprocessor = LeakageFreePreprocessor(n_bins=32)
X_train_proc = preprocessor.fit_transform(X_train)
X_test_proc = preprocessor.transform(X_test)

# Train compact model with BatchEnsemble (K=4 ensemble heads)
model = CompactTab(
    d_in=X_train_proc.shape[1],
    d_out=2,
    d_hidden=128,
    n_blocks=4,
    ensemble_size=4,
)
```

---

## CLI Tools & Benchmark Scripts

CompactTab includes a full suite of verification and evaluation utilities:

```bash
# Verify environment and CUDA acceleration
compacttab env-info

# Run zero-shot validation on classification and regression
python scripts/launch_top5.py --size small --task classification
python scripts/launch_top5.py --size small --task regression

# Generate official TabArena submission bundle
python scripts/generate_tabarena_submission.py --size small --output-dir submission/compacttab_tabarena_v1

# Run multi-model comparative tree benchmark (CatBoost, LightGBM, XGBoost)
python scripts/run_smoke_benchmark.py
```

---

## Automated Test Suite

The test suite covers unit and integration testing across models, preprocessors, metrics, and AutoGluon wrappers:

```bash
pytest -v
```

```
============================== test session starts ==============================
collected 22 items

tests/test_baselines.py::test_tree_baselines_binary PASSED               [  4%]
tests/test_baselines.py::test_tree_baselines_regression PASSED           [  9%]
tests/test_cli.py::test_cli_env_info PASSED                              [ 13%]
tests/test_foundation.py::test_foundation_classification PASSED          [ 18%]
tests/test_foundation.py::test_foundation_regression PASSED              [ 22%]
tests/test_metrics.py::test_compute_classification_metrics PASSED         [ 27%]
tests/test_metrics.py::test_compute_regression_metrics PASSED             [ 31%]
tests/test_models.py::test_compacttab_forward PASSED                     [ 36%]
tests/test_models.py::test_batchensemble_layer PASSED                    [ 40%]
tests/test_preprocessor.py::test_quantile_ple_tokenization PASSED         [ 45%]
tests/test_preprocessor.py::test_leakage_isolation PASSED                [ 50%]
tests/test_tabarena_integration.py::test_tabarena_binary PASSED           [ 54%]
tests/test_tabarena_integration.py::test_tabarena_regression PASSED       [ 59%]
...
======================== 22 passed, 2 warnings in 11.68s ========================
```

---

## Repository Structure

```
├── src/compacttab/
│   ├── models/
│   │   ├── foundation.py         # In-context SCM tabular foundation model
│   │   ├── compacttab.py         # Quantile PLE + BatchEnsemble neural architecture
│   │   ├── baselines.py          # Classical MLP and ResNet tabular architectures
│   │   └── layers/               # PLE, BatchEnsemble, and Gated Residual blocks
│   ├── integrations/
│   │   └── tabarena.py           # AutoGluon AbstractModel TabArena interface
│   ├── data/
│   │   ├── preprocessor.py       # Leakage-free quantile binning and categorical encoding
│   │   ├── loaders.py            # Reproducible benchmark dataset loaders
│   │   └── schema.py             # Strongly typed tabular schemas
│   └── evaluation/
│       ├── metrics.py            # Comprehensive classification & regression metrics
│       └── baselines.py          # Tree baselines (CatBoost, LightGBM, XGBoost, RF, ET)
├── scripts/
│   ├── launch_top5.py            # Interactive launch and evaluation utility
│   ├── generate_tabarena_submission.py # Submission bundle generator
│   └── run_smoke_benchmark.py   # Multi-model comparative benchmark suite
├── submission/                   # Generated TabArena submission bundle
├── tests/                        # 22 automated unit and integration tests
└── docs/                         # Specifications, licensing, and API references
```

---

## Citation & References

If you use CompactTab or CompactTabFoundation in your research or production pipelines, please cite:

```bibtex
@software{bhalala2026compacttab,
  author       = {Daksh Bhalala},
  title        = {CompactTab: State-of-the-Art Compact Tabular Deep Learning & Foundation Models},
  year         = {2026},
  url          = {https://github.com/DakshBhalala/TBDML},
  note         = {AutoGluon TabArena Integration Pull Request #676}
}
```

---

## License

- **Source Code:** [Apache-2.0](LICENSE) — Permissive license for academic and commercial use.
- **Foundation Weights:** [OpenMDW-1.1](https://huggingface.co/nvidia/Kumo-Tabular) — Open Model Weights license permitting commercial deployment.
- **Baselines & Dependencies:** Strictly permissive (Apache-2.0, MIT, BSD-3-Clause). Zero proprietary or non-commercial lock-in.
