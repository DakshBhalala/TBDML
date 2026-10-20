"""Evaluation metrics for tabular classification and regression."""

from typing import Dict, Optional, Union
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

from compacttab.data.schema import TaskType


def compute_metrics(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
    y_prob: Optional[Union[np.ndarray, list]] = None,
    task_type: TaskType = TaskType.BINARY,
) -> Dict[str, float]:
    """Compute task-specific evaluation metrics."""
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    metrics: Dict[str, float] = {}

    if task_type == TaskType.BINARY:
        metrics["accuracy"] = float(accuracy_score(y_true_arr, y_pred_arr))
        metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_true_arr, y_pred_arr))
        metrics["f1"] = float(f1_score(y_true_arr, y_pred_arr, zero_division=0))

        if y_prob is not None:
            y_prob_arr = np.asarray(y_prob)
            # Binary probability extraction
            if y_prob_arr.ndim == 2 and y_prob_arr.shape[1] == 2:
                p1 = y_prob_arr[:, 1]
            elif y_prob_arr.ndim == 2 and y_prob_arr.shape[1] == 1:
                p1 = y_prob_arr[:, 0]
            else:
                p1 = y_prob_arr

            try:
                metrics["roc_auc"] = float(roc_auc_score(y_true_arr, p1))
            except Exception:
                metrics["roc_auc"] = 0.5

            try:
                # Clip probabilities for stable log loss
                eps = 1e-12
                p_clipped = np.clip(p1, eps, 1.0 - eps)
                metrics["log_loss"] = float(log_loss(y_true_arr, p_clipped))
            except Exception:
                metrics["log_loss"] = float("nan")

    elif task_type == TaskType.MULTICLASS:
        metrics["accuracy"] = float(accuracy_score(y_true_arr, y_pred_arr))
        metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_true_arr, y_pred_arr))
        metrics["macro_f1"] = float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0))

        if y_prob is not None:
            y_prob_arr = np.asarray(y_prob)
            try:
                metrics["roc_auc"] = float(
                    roc_auc_score(y_true_arr, y_prob_arr, multi_class="ovr", average="macro")
                )
            except Exception:
                metrics["roc_auc"] = 0.5

            try:
                metrics["log_loss"] = float(log_loss(y_true_arr, y_prob_arr))
            except Exception:
                metrics["log_loss"] = float("nan")

    else:
        # Regression
        mse = mean_squared_error(y_true_arr, y_pred_arr)
        metrics["mse"] = float(mse)
        metrics["rmse"] = float(np.sqrt(mse))
        metrics["mae"] = float(mean_absolute_error(y_true_arr, y_pred_arr))
        metrics["r2"] = float(r2_score(y_true_arr, y_pred_arr))

    return metrics
