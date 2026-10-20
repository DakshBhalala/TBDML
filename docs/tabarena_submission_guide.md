# Official TabArena Leaderboard Submission Guide

*Leaderboard:* [TabArena Leaderboard (Hugging Face Space)](https://huggingface.co/spaces/TabArena/leaderboard)  
*Repository:* [autogluon/tabarena (GitHub)](https://github.com/autogluon/tabarena)  
*Website:* [tabarena.ai](https://tabarena.ai)

---

## Overview

TabArena is the official living benchmark for tabular machine learning and Tabular Foundation Models (TFMs). The leaderboard tracks Elo ratings across diverse OpenML suites (Classification, Regression, Balanced, Imbalanced, Small, Medium).

To get **CompactTab** or **CompactTabFoundation** officially listed on the public leaderboard, follow these three steps.

---

## Step 1: Make Your Model Repository Public

TabArena rules require submitted models to be publicly accessible and open source:
1. Create a repository on GitHub (e.g. `github.com/<your-username>/compacttab`).
2. Push this repository (`git remote add origin ... && git push -u origin main`).
3. Ensure permissive licensing (Apache-2.0 / OpenMDW-1.1), which is already configured.

---

## Step 2: Implement AutoGluon `AbstractModel` Wrapper

TabArena benchmarks models by instantiating them via AutoGluon's `AbstractModel` interface:

```python
from autogluon.core.models import AbstractModel
import pandas as pd
from compacttab.data.schema import DatasetSchema, TaskType
from compacttab.models import CompactTabFoundation

class TabArenaCompactTabModel(AbstractModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = None

    def _fit(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        task_type = TaskType.BINARY if self.problem_type == "binary" else (
            TaskType.MULTICLASS if self.problem_type == "multiclass" else TaskType.REGRESSION
        )
        schema = DatasetSchema(
            task_type=task_type,
            target_col=self.target,
            numerical_cols=list(X.select_dtypes(include=["number"]).columns),
            categorical_cols=list(X.select_dtypes(exclude=["number"]).columns),
        )
        self.model = CompactTabFoundation(schema=schema, size="small", device="cuda")
        self.model.fit(X, y)
        return self

    def _predict_proba(self, X: pd.DataFrame, **kwargs):
        return self.model.predict_proba(X)

    def _predict(self, X: pd.DataFrame, **kwargs):
        return self.model.predict(X)
```

---

## Step 3: Run Official Benchmark & Submit Pull Request

1. Fork and clone `github.com/autogluon/tabarena`.
2. Register your model in `tabarena/models/`.
3. Run the evaluation script:
   ```bash
   python -m tabarena.run_benchmark --model compacttab --suite tabarena-v0.1
   ```
4. Commit the generated predictions dataset and evaluation JSON to your fork.
5. Open a Pull Request to `autogluon/tabarena` titled:
   `[Model] Add CompactTab / CompactTabFoundation SOTA Tabular Model`
6. Once merged by the maintainers, the Hugging Face Space automatically recalculates Elo scores and lists your model in the **Top 5**!
