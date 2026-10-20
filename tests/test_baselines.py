"""Unit tests for baseline tree models and wrappers."""

import numpy as np

from compacttab.baselines.tree_models import (
    CatBoostBaseline,
    DummyBaseline,
    HistGradientBoostingBaseline,
    LightGBMBaseline,
    RandomForestBaseline,
    XGBoostBaseline,
)
from compacttab.data.schema import DatasetSchema, TaskType


def test_tree_baselines_binary_classification():
    X = np.random.randn(50, 4)
    y = np.random.randint(0, 2, size=50)

    schema = DatasetSchema(
        target_col="target",
        task_type=TaskType.BINARY,
        numerical_cols=[f"num_{i}" for i in range(4)],
        num_classes=2,
    )

    models = [
        DummyBaseline(schema),
        RandomForestBaseline(schema, n_estimators=5),
        HistGradientBoostingBaseline(schema, max_iter=5),
        LightGBMBaseline(schema, n_estimators=5),
        XGBoostBaseline(schema, n_estimators=5),
        CatBoostBaseline(schema, iterations=5),
    ]

    for model in models:
        model.fit(X, y)
        preds = model.predict(X[:5])
        probs = model.predict_proba(X[:5])
        assert len(preds) == 5
        assert probs.shape[0] == 5


def test_tree_baselines_regression():
    X = np.random.randn(50, 4)
    y = np.random.randn(50)

    schema = DatasetSchema(
        target_col="target",
        task_type=TaskType.REGRESSION,
        numerical_cols=[f"num_{i}" for i in range(4)],
    )

    models = [
        DummyBaseline(schema),
        RandomForestBaseline(schema, n_estimators=5),
        HistGradientBoostingBaseline(schema, max_iter=5),
        LightGBMBaseline(schema, n_estimators=5),
        XGBoostBaseline(schema, n_estimators=5),
        CatBoostBaseline(schema, iterations=5),
    ]

    for model in models:
        model.fit(X, y)
        preds = model.predict(X[:5])
        assert len(preds) == 5
