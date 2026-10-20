# CompactTab: State-of-the-Art Compact Tabular Deep Learning

CompactTab is a research-grade, highly parameter-efficient deep neural network architecture designed specifically for structured/tabular data. It combines **Quantile Piecewise-Linear Embeddings (PLE)** with **Parameter-Efficient BatchEnsemble Gated Residual Blocks**, closing the historical performance gap between deep learning and Gradient Boosted Decision Trees (GBDTs) while using a fraction of the compute and memory of transformer-based tabular foundation models.

---

## Key Features

- **Extreme Parameter Efficiency:** Achieves competitive tabular performance at ~100k parameters (`CompactTab-Tiny`) to ~1M parameters (`CompactTab-Small`), compared to 10M+ in transformer alternatives.
- **Quantile Piecewise-Linear Embeddings (PLE):** Overcomes the neural network limitation in fitting localized step decisions by learning soft piecewise linear activations over quantile bin edges.
- **BatchEnsemble Multi-Head Ensembling:** Trains and infers an ensemble of $k$ submodels simultaneously in a single forward pass via rank-1 adapters ($r_k \odot W \odot s_k$) with only $\approx 10\%$ extra parameters.
- **Strict Leakage-Free Pipeline:** Preprocessing (imputation, quantile binning, entity encoding, target scaling) is fitted strictly on training data, preventing data leakage.
- **Hardware-Friendly:** Optimized for consumer GPUs (e.g. NVIDIA RTX 3050 Laptop GPU, 6GB VRAM) and CPU-only environments.
- **Unified Benchmark Suite:** Out-of-the-box comparative benchmarking against Random Forest, ExtraTrees, HistGradientBoosting, LightGBM, XGBoost, CatBoost, and standard Tabular MLP / ResNet baselines.
- **Permissive Open-Source Licensing:** 100% Apache-2.0 compliant for commercial deployment without proprietary checkpoint lock-in.

---

## Installation

### Prerequisites
- Python 3.11 or 3.12
- PyTorch 2.2+ (with CUDA support if GPU is available)

### Quick Install (Editable Mode)
```bash
git clone https://github.com/your-org/compacttab.git
cd compacttab
python -m pip install -e .
```

To install development and benchmarking dependencies:
```bash
python -m pip install -e ".[dev,benchmark]"
```

---

## Quickstart

### 1. Check Hardware & Compute Environment
```bash
compacttab env-info
```

### 2. Run Reproducible Smoke Benchmark
Execute a complete comparative benchmark across binary classification and regression:
```bash
compacttab smoke-benchmark --max-epochs 30
```

### 3. Python API Example
```python
from compacttab.data.loaders import load_breast_cancer_dataset
from compacttab.data.preprocessor import TabularPreprocessor
from compacttab.data.dataset import TabularDataset, create_dataloader
from compacttab.models.compacttab import CompactTab, CompactTabConfig
from compacttab.training.trainer import TabularTrainer
from compacttab.evaluation.metrics import compute_metrics

# 1. Load dataset with deterministic split
df_tr, df_val, df_te, y_tr, y_val, y_te, schema = load_breast_cancer_dataset(random_state=42)

# 2. Fit preprocessor strictly on training data
preprocessor = TabularPreprocessor(schema=schema, num_bins=16)
preprocessor.fit(df_tr, y_tr)

# 3. Transform splits
num_tr, mask_tr, cat_tr, y_tr_trans = preprocessor.transform(df_tr, y_tr)
num_val, mask_val, cat_val, y_val_trans = preprocessor.transform(df_val, y_val)
num_te, mask_te, cat_te, y_te_trans = preprocessor.transform(df_te, y_te)

train_ds = TabularDataset(num_tr, mask_tr, cat_tr, y_tr_trans)
val_ds = TabularDataset(num_val, mask_val, cat_val, y_val_trans)
test_ds = TabularDataset(num_te, mask_te, cat_te, y_te_trans)

# 4. Initialize CompactTab model
config = CompactTabConfig.from_preset("small")
model = CompactTab(
    schema=schema,
    bin_edges=preprocessor.num_bin_edges_,
    cat_cardinalities=preprocessor.cat_cardinalities_,
    config=config,
)
print(f"Trainable Parameters: {model.count_parameters():,}")

# 5. Train with early stopping
trainer = TabularTrainer(model=model, schema=schema, max_epochs=30, batch_size=64)
trainer.fit(train_ds, val_ds)

# 6. Evaluate
test_loader = create_dataloader(test_ds, batch_size=64, shuffle=False)
probs = trainer.predict_proba(test_loader)
preds = trainer.predict(test_loader)

metrics = compute_metrics(y_true=y_te.to_numpy(), y_pred=preds, y_prob=probs, task_type=schema.task_type)
print("Test Metrics:", metrics)
```

---

## Model Presets

| Preset | Target Params | Hidden Dim | Layers | Ensemble Heads ($k$) | Bins ($B$) | Expansion | Use Case |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`tiny`** | $\approx 100\text{k}$ | 64 | 2 | 4 | 8 | 2.0 | Ultra-fast training, edge / CPU inference |
| **`small`** | $\approx 1\text{M}$ | 128 | 4 | 8 | 16 | 2.0 | Recommended default for competitive benchmarks |
| **`medium`** | $\approx 10\text{M}$ | 384 | 6 | 8 | 32 | 2.0 | High-capacity modeling for large tabular datasets |

---

## Benchmarking & Comparative Evaluation

Run the master benchmark suite evaluating CompactTab against 8 competitive baselines across binary classification, multiclass classification, and regression:
```bash
python scripts/run_smoke_benchmark.py
```
This automatically logs machine-readable JSON and CSV outputs to `experiments/` and compiles a comprehensive Markdown report in `reports/benchmark_report.md`.

---

## Documentation

- [Research Review](docs/research_review.md): Literature analysis covering TabArena, BeyondArena, TabPFN, TabM, FT-Transformer, GBDTs, parameter comparisons, and research gaps.
- [Architecture Specification](docs/architecture.md): Full mathematical and structural definition of CompactTab.
- [Benchmark Protocol](docs/benchmark_protocol.md): Experimental controls, metric aggregation, and TabArena/BeyondArena alignment.
- [Licensing & Commercial Use Audit](docs/licensing.md): Comprehensive intellectual property audit of code, dependencies, and datasets.
- [Model Card](docs/model_card.md): Model scope, intended domains, and limitations.

---

## Limitations & Scope

1. **Dataset-Specific Fitting:** CompactTab is designed as an ultra-compact, high-accuracy predictor fitted to a specific dataset schema. It is not currently a zero-shot foundational model (like TabPFN), avoiding the quadratic compute and license restrictions of TFMs.
2. **Missingness Mechanisms:** CompactTab implements median imputation + missingness indicator tokenization. On tables where missingness is not missing at random (MNAR), domain-specific imputation may be advantageous.
3. **Very High Dimensional Sparse Data:** For tables with $>10,000$ sparse features (e.g. Bag-of-Words), histogram-based trees like LightGBM may be more memory-efficient.

---

## License

CompactTab is released under the **Apache-2.0 License**. See [docs/licensing.md](docs/licensing.md) for full compliance details.
