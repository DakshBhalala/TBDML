"""Unit tests for metric calculation across task types."""

import numpy as np
import pytest

from compacttab.data.schema import TaskType
from compacttab.evaluation.metrics import compute_metrics


def test_binary_metrics():
    y_true = np.array([0, 1, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 0, 1])
    y_prob = np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3], [0.6, 0.4], [0.1, 0.9]])

    metrics = compute_metrics(y_true, y_pred, y_prob, task_type=TaskType.BINARY)
    assert "accuracy" in metrics
    assert "roc_auc" in metrics
    assert "balanced_accuracy" in metrics
    assert "log_loss" in metrics
    assert metrics["accuracy"] == 0.8
    assert metrics["roc_auc"] > 0.5


def test_multiclass_metrics():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 2, 2])
    y_prob = np.array([
        [0.8, 0.1, 0.1],
        [0.1, 0.8, 0.1],
        [0.1, 0.1, 0.8],
        [0.7, 0.2, 0.1],
        [0.2, 0.3, 0.5],
        [0.1, 0.1, 0.8],
    ])

    metrics = compute_metrics(y_true, y_pred, y_prob, task_type=TaskType.MULTICLASS)
    assert "accuracy" in metrics
    assert "macro_f1" in metrics
    assert "roc_auc" in metrics
    assert metrics["accuracy"] == 5.0 / 6.0


def test_regression_metrics():
    y_true = np.array([10.0, 20.0, 30.0, 40.0])
    y_pred = np.array([11.0, 19.0, 31.0, 39.0])

    metrics = compute_metrics(y_true, y_pred, task_type=TaskType.REGRESSION)
    assert "rmse" in metrics
    assert "mae" in metrics
    assert "r2" in metrics
    assert np.isclose(metrics["mae"], 1.0)
    assert np.isclose(metrics["rmse"], 1.0)
    assert metrics["r2"] > 0.95
