"""Unit tests for TabArena AutoGluon integration."""

import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split

from compacttab.integrations.tabarena import AUTOGLUON_AVAILABLE, TabArenaCompactTabModel


@pytest.mark.skipif(not AUTOGLUON_AVAILABLE, reason="AutoGluon not installed")
def test_tabarena_autogluon_binary_classification():
    """Verify TabArenaCompactTabModel behaves as valid AutoGluon model on binary classification."""
    data = load_breast_cancer(as_frame=True)
    df = data.frame
    target_col = "target"
    feature_cols = [c for c in df.columns if c != target_col]

    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42, stratify=df[target_col])
    X_train, y_train = df_train[feature_cols], df_train[target_col]
    X_test, y_test = df_test[feature_cols], df_test[target_col]

    model = TabArenaCompactTabModel(
        problem_type="binary",
        preset="tiny",
        max_epochs=5,
        batch_size=64,
    )

    model.fit(X=X_train, y=y_train)
    probs = model.predict_proba(X_test)
    preds = model.predict(X_test)

    assert probs.shape == (len(X_test), 2)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-3)
    assert len(preds) == len(X_test)
    accuracy = (preds == y_test.to_numpy()).mean()
    assert accuracy > 0.85, f"Expected accuracy > 0.85, got {accuracy:.4f}"


@pytest.mark.skipif(not AUTOGLUON_AVAILABLE, reason="AutoGluon not installed")
def test_tabarena_autogluon_regression():
    """Verify TabArenaCompactTabModel behaves as valid AutoGluon model on regression."""
    data = load_diabetes(as_frame=True)
    df = data.frame
    target_col = "target"
    feature_cols = [c for c in df.columns if c != target_col]

    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
    X_train, y_train = df_train[feature_cols], df_train[target_col]
    X_test, _y_test = df_test[feature_cols], df_test[target_col]

    model = TabArenaCompactTabModel(
        problem_type="regression",
        preset="tiny",
        max_epochs=5,
        batch_size=64,
    )

    model.fit(X=X_train, y=y_train)
    preds = model.predict(X_test)

    assert len(preds) == len(X_test)
    assert not np.isnan(preds).any()
