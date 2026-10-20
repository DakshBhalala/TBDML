# Official TabArena Leaderboard Submission & Lifecycle Guide

*Active Pull Request:* **[autogluon/tabarena#676](https://github.com/autogluon/tabarena/pull/676)**  
*Fork Repository:* [`DakshBhalala/tabarena`](https://github.com/DakshBhalala/tabarena) (branch `feature/add-compacttab-model`)  
*Living Leaderboard:* [TabArena Leaderboard (Hugging Face Space)](https://huggingface.co/spaces/TabArena/leaderboard)  
*Upstream Repository:* [autogluon/tabarena (GitHub)](https://github.com/autogluon/tabarena)  

---

## 1. Submission Overview

TabArena is the living benchmark and leaderboard for tabular machine learning and Tabular Foundation Models (TFMs). Methods submitted to TabArena are evaluated using a strict, standardized 5-stage protocol on identical benchmark hardware (SkyPilot cluster) across 51 OpenML datasets.

CompactTabFoundation has completed **Stage 1** with the opening of **[Pull Request #676](https://github.com/autogluon/tabarena/pull/676)**.

---

## 2. Integrated Files in TabArena

The integration is implemented under `packages/tabarena/src/tabarena/models/compacttab/`:

```
packages/tabarena/src/tabarena/models/compacttab/
├── __init__.py           # Re-exports CompactTabFoundationModel & Medium
├── model.py              # AbstractModel implementation with FP16 autocast
├── hpo.py                # ConfigGenerator and HPO search spaces
├── info.py               # Method metadata, license, and pip_extra
└── tests/
    ├── __init__.py
    └── test_model.py     # Isolated model unit tests (3/3 passing)
```

### Shared Configuration Updates:
- **`packages/tabarena/pyproject.toml`**: Added `compacttab` extra with pinned `structured-data-models` Git commit and added `"tabarena[compacttab]"` to `extended`.
- **`packages/tabarena/src/tabarena/models/__init__.py`**: Added lazy loader mappings for `CompactTabFoundationModel` and `CompactTabFoundationMediumModel`.
- **`packages/tabarena/src/tabarena/website/website_format.py`**: Added `COMPACTTAB` and `TA-COMPACTTAB` to `Constants.foundational`.

---

## 3. The 5-Stage Evaluation Lifecycle

```
Stage 1: Pull Request (#676)
  └── Author submits model wrapper, unit tests, and hardware verification. [COMPLETE]

Stage 2: Benchmark Cluster Execution
  └── TabArena maintainers queue the method on the SkyPilot GPU cluster across all 51 datasets and folds.

Stage 3: Verification & Sign-off
  └── Maintainers post raw and summarized metrics in PR #676; @DakshBhalala confirms results.

Stage 4: Artifact Hosting & Merge
  └── Predictions hosted in TabArena Cloudflare R2 bucket; PR #676 merged into main.

Stage 5: Leaderboard Release
  └── Official leaderboard website updates with verified Elo score and ranking.
```

---

## 4. Verification Commands

To locally reproduce the model tests and verify registry discovery:

```bash
# Run compacttab isolated unit tests
pytest packages/tabarena/src/tabarena/models/compacttab/tests/test_model.py

# Verify code formatting and linting
ruff check packages/tabarena/src/tabarena/models/compacttab/
ruff format --check packages/tabarena/src/tabarena/models/compacttab/

# Verify TabArena model discovery
python -c "from tabarena.models._registry import discover_models; print('Discovered:', 'CompactTabFoundation' in discover_models())"
```
