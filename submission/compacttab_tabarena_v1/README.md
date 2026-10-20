# TabArena Official Submission Bundle: CompactTabFoundation (Small)

**Project:** CompactTab
**Submission ID:** `compacttab_foundation_small_v1`
**Target Leaderboard:** [TabArena (Living Tabular Benchmark)](https://huggingface.co/spaces/TabArena/leaderboard)
**Leaderboard Target:** Candidate for Top 5 Placement (1803 Elo target, pending official maintainer evaluation)
**Commit:** `244f181c1ff246e79b38e3d2f55b80c01d74fe9f`

---

## Files in this Bundle

| File | Description |
| :--- | :--- |
| `metadata.json` | Model parameters, architecture metadata, licensing, hardware verification |
| `predictions.parquet` | Test predictions across benchmark tasks (492 rows) |
| `results_summary.json` | Detailed per-task evaluation metrics and compute timings |
| `model_submission.py` | Standalone AutoGluon `AbstractModel` integration wrapper |

---

## How to Submit to TabArena / Hugging Face

### Option A: Direct Pull Request to `autogluon/tabarena`
1. Fork `https://github.com/autogluon/tabarena`.
2. Copy `model_submission.py` to `tabarena/models/compacttab.py`.
3. Register `TabArenaCompactTabModel` in `tabarena/models/__init__.py`.
4. Copy `submission/compacttab_tabarena_v1/` into `benchmark_runs/compacttab/`.
5. Submit a Pull Request:
   - **Title:** `[Model] Add CompactTabFoundation (Small)`
   - **Description:**
     - Parameter Count: 27,984,384
     - License: Apache-2.0 / OpenMDW-1.1
     - Benchmark Candidate: 1803 Elo target (pending official re-run)
     - Hardware: NVIDIA GeForce RTX 3050 6GB Laptop GPU

### Option B: Hugging Face Leaderboard Discussion / Issue
1. Navigate to [TabArena Leaderboard Discussions](https://huggingface.co/spaces/TabArena/leaderboard/discussions).
2. Create New Discussion: `[Model Submission] CompactTabFoundation`.
3. Attach `metadata.json` and link your GitHub repository.
