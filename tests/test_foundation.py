"""Unit tests for CompactTabFoundation and CompactTabOmni."""

import numpy as np
import pandas as pd
import pytest
import torch

from compacttab.data.schema import DatasetSchema, TaskType
from compacttab.models.foundation import SDM_AVAILABLE, CompactTabFoundation


@pytest.mark.skipif(not SDM_AVAILABLE, reason="SDM library not installed")
def test_compacttab_foundation_binary_classification():
    """Test CompactTabFoundation on synthetic binary classification data."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    np.random.seed(42)
    n_train, n_test = 100, 25
    X_train = pd.DataFrame(
        {
            "num_1": np.random.randn(n_train),
            "num_2": np.random.randn(n_train),
            "cat_1": np.random.choice(["A", "B", "C"], size=n_train),
        }
    )
    y_train = (X_train["num_1"] + (X_train["cat_1"] == "A").astype(float) > 0.5).astype(int)

    X_test = pd.DataFrame(
        {
            "num_1": np.random.randn(n_test),
            "num_2": np.random.randn(n_test),
            "cat_1": np.random.choice(["A", "B", "C"], size=n_test),
        }
    )

    schema = DatasetSchema(
        task_type=TaskType.BINARY,
        target_col="target",
        numerical_cols=["num_1", "num_2"],
        categorical_cols=["cat_1"],
    )

    model = CompactTabFoundation(schema=schema, size="small", device=device, num_estimators=2)
    assert model.count_parameters() > 10_000_000

    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)
    preds = model.predict(X_test)

    assert probs.shape == (n_test, 2)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-3)
    assert len(preds) == n_test
    assert set(preds).issubset({0, 1})


@pytest.mark.skipif(not SDM_AVAILABLE, reason="SDM library not installed")
def test_compacttab_foundation_regression():
    """Test CompactTabFoundation on synthetic regression data."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    np.random.seed(42)
    n_train, n_test = 80, 20
    X_train = pd.DataFrame(
        {
            "x1": np.random.randn(n_train),
            "x2": np.random.randn(n_train),
        }
    )
    y_train = 2.5 * X_train["x1"] - 1.2 * X_train["x2"] + 0.1 * np.random.randn(n_train)

    X_test = pd.DataFrame(
        {
            "x1": np.random.randn(n_test),
            "x2": np.random.randn(n_test),
        }
    )

    schema = DatasetSchema(
        task_type=TaskType.REGRESSION,
        target_col="target",
        numerical_cols=["x1", "x2"],
        categorical_cols=[],
    )

    model = CompactTabFoundation(schema=schema, size="small", device=device, num_estimators=2)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    assert preds.shape == (n_test,)
    assert not np.isnan(preds).any()
