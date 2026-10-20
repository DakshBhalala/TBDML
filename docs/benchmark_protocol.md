# Benchmark Evaluation Protocol: CompactTab

*Last Reviewed: October 2026*  
*Project: CompactTab*  
*Alignment: TabArena (arXiv:2506.16791) & BeyondArena (arXiv:2606.30410) Standards*

---

## 1. Principles of Rigorous Tabular Evaluation

Benchmarking machine learning models on tabular data requires strict experimental discipline to eliminate common pitfalls such as data leakage, metric scale distortion, and unfair computational advantages.

CompactTab follows a three-stage evaluation methodology:
1. **Stage A (Local Reproducible Smoke Benchmark):** Verification across binary classification, multiclass classification, and regression on standard, verified benchmark datasets. Validates end-to-end functionality, memory stability, and baseline comparability.
2. **Stage B (TabArena Living Benchmark):** Standardized cross-validation protocol using TabArena's official evaluation pipeline, comparing against tuned GBDTs and neural architectures on curated IID datasets.
3. **Stage C (BeyondArena Real-World Stress Test):** Evaluation on non-IID datasets featuring temporal splits, clustered/grouped domains, and extreme feature dimensionalities.

---

## 2. Experimental Controls and Leakage Prevention

To ensure scientific integrity:
1. **Strict Data Split Isolation:**
   - Preprocessing transformations (quantile binning, standard scaling, categorical encoding, imputers) are fitted **strictly on the training split**.
   - Test and validation splits are transformed using fitted parameters without re-estimation.
   - Out-of-vocabulary (OOV) categorical values and unseen missingness in test sets map to dedicated reserved indices/tokens.
2. **Deterministic Random Seeds:**
   - All dataset splits, weight initializations, mini-batch shuffling, and baseline model random states are governed by explicit integer seeds (default: `42`).
3. **No Test Set Tuning:**
   - Hyperparameter selection and early stopping checkpoints are derived solely from validation splits or inner cross-validation folds. The test split is evaluated exactly once per configuration.

---

## 3. Metrics and Aggregation Methodology

### Task-Specific Metrics:
- **Binary Classification:**
  - Primary Metric: **ROC-AUC** (Area Under the Receiver Operating Characteristic Curve)
  - Secondary Metrics: **Accuracy**, **Log-Loss** (Cross-Entropy), **Balanced Accuracy**
- **Multiclass Classification:**
  - Primary Metric: **Multi-class ROC-AUC (OVR with Macro-average)** or **Macro-F1 / Accuracy**
  - Secondary Metrics: **Multiclass Log-Loss**
- **Regression:**
  - Primary Metric: **Root Mean Squared Error (RMSE)**
  - Secondary Metrics: **Mean Absolute Error (MAE)**, **Coefficient of Determination ($R^2$)**

### Aggregation Across Diverse Datasets:
Directly averaging raw loss values across datasets with different scales is mathematically invalid. CompactTab adopts standard benchmark aggregation:
1. **Normalized Performance Score:** Rescaling metrics relative to a baseline (e.g., dummy predictor vs. best tuned baseline) on a $[0, 1]$ scale.
2. **Average Rank:** Computing the ordinal rank of each algorithm per dataset and averaging across all tasks.
3. **Win/Tie/Loss Matrix:** Pairwise statistical sign tests across all evaluation tasks.

---

## 4. Resource and Efficiency Profiling

Every benchmark execution must measure and report:
- **Trainable Parameters:** Exact count of non-frozen model parameters ($\sum \text{numel}$).
- **Serialized Model Size:** Size on disk (in kilobytes or megabytes) of the saved model artifact.
- **Training Time:** Wall-clock time (seconds) required for end-to-end fitting including early stopping.
- **Inference Latency:** Average wall-clock milliseconds per sample ($ms / \text{sample}$) during batch inference.
- **Throughput:** Samples evaluated per second ($\text{samples} / \text{sec}$).
- **Peak Hardware Consumption:**
  - Peak system RAM (MB) monitored via `psutil`.
  - Peak GPU VRAM (MB) monitored via `torch.cuda.max_memory_allocated()`.

---

## 5. Baselines for Comparison

A neural model must be judged against representative, well-tuned non-neural and neural alternatives:
1. **Sanity Baselines:** `DummyClassifier` (majority class / stratified prior), `DummyRegressor` (mean target).
2. **Standard Tree Ensembles:**
   - `RandomForestClassifier` / `RandomForestRegressor`
   - `ExtraTreesClassifier` / `ExtraTreesRegressor`
   - `HistGradientBoostingClassifier` / `HistGradientBoostingRegressor` (scikit-learn's native histogram GBDT)
3. **State-of-the-Art GBDT Implementations:**
   - `LightGBM` (Ke et al., 2017)
   - `XGBoost` (Chen & Guestrin, 2016)
   - `CatBoost` (Prokhorenkova et al., 2018)
4. **Standard Neural Baselines:**
   - `TabularMLP` (Multi-layer perceptron with LayerNorm, Dropout, and ReLU/GELU activations)
   - `TabularResNet` (Pre-activation residual network for tabular inputs)
