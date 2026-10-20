# Model Card: CompactTab

*Model Class: CompactTab (Parameter-Efficient Tabular Deep Architecture)*  
*Version: 0.1.0*  
*License: Apache-2.0*  
*Intended Domain: Tabular Regression and Classification*

---

## 1. Model Details
- **Architecture Name:** CompactTab
- **Model Type:** Deep neural network with Quantile Piecewise Linear Embeddings (PLE), Gated Residual Blocks, and BatchEnsemble parameter-efficient multi-head ensembling.
- **Developers:** CompactTab Research Team
- **Input Modalities:** Heterogeneous tabular matrices (continuous numerical features, categorical features, binary missingness masks).
- **Output Targets:** Binary classification probabilities, multi-class probabilities, or continuous regression values.
- **Framework:** PyTorch (>= 2.2), Python (>= 3.11).

---

## 2. Intended Use
- **Primary Use Cases:**
  - Fast, resource-efficient training on structured/tabular enterprise data.
  - Low-latency inference on edge devices, CPU servers, or laptop GPUs (e.g. RTX 3050).
  - Outperforming or matching traditional tree ensembles while retaining the differentiable and fine-tunable benefits of deep learning.
- **Out-of-Scope Use Cases:**
  - Unstructured image/video recognition or long-sequence raw natural language processing without tabular feature extraction.
  - Zero-shot out-of-context prediction without dataset-specific fitting.

---

## 3. Training & Preprocessing
- **Continuous Features:** Handled via Quantile-based Piecewise Linear Embeddings fitted exclusively on training splits.
- **Categorical Features:** Transformed via integer categorical indexing with reserved Out-Of-Vocabulary (index 0) handling and dense learned embeddings.
- **Missingness:** Native missingness indicators provided to preserve informative missingness patterns.
- **Optimization:** AdamW optimizer with cosine annealing learning rate scheduler and conservative BatchEnsemble initialization.

---

## 4. Evaluation and Factors
- Evaluated on standard tabular benchmark datasets covering binary classification, multiclass classification, and regression.
- Compared against strong tree-based baselines (Random Forest, Extra Trees, HistGradientBoosting, LightGBM, XGBoost, CatBoost) and standard neural architectures (MLP, ResNet).
