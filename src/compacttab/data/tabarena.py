"""Official TabArena OpenML benchmark integration and dataset loader."""

import os
from typing import List, Optional, Tuple
import numpy as np
import openml
import pandas as pd
from sklearn.model_selection import train_test_split

from compacttab.data.schema import DatasetSchema, TaskType

TABARENA_SUITE_ID = 457


def get_tabarena_task_ids() -> List[int]:
    """Retrieve all official task IDs in the TabArena-v0.1 benchmark suite."""
    suite = openml.study.get_suite(TABARENA_SUITE_ID)
    return list(suite.tasks)


def load_tabarena_task(
    task_id: int,
    cache_dir: str = "data_cache/tabarena",
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, DatasetSchema]:
    """Load a specific TabArena task and return train/val/test splits with schema."""
    os.makedirs(cache_dir, exist_ok=True)

    task = openml.tasks.get_task(task_id, download_data=True)
    dataset = task.get_dataset()
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        target=task.target_name,
        dataset_format="dataframe",
    )

    # Determine TaskType
    if task.task_type_id == 1:
        # Supervised Classification
        n_classes = len(np.unique(y.dropna()))
        task_type = TaskType.BINARY if n_classes == 2 else TaskType.MULTICLASS
    elif task.task_type_id == 2:
        # Supervised Regression
        task_type = TaskType.REGRESSION
        n_classes = None
    else:
        # Fallback check on target
        if pd.api.types.is_numeric_dtype(y) and len(np.unique(y.dropna())) > 10:
            task_type = TaskType.REGRESSION
            n_classes = None
        else:
            n_classes = len(np.unique(y.dropna()))
            task_type = TaskType.BINARY if n_classes == 2 else TaskType.MULTICLASS

    # Identify numerical and categorical columns
    num_cols = []
    cat_cols = []
    for col, is_cat in zip(attribute_names, categorical_indicator):
        if is_cat or not pd.api.types.is_numeric_dtype(X[col]):
            cat_cols.append(col)
        else:
            num_cols.append(col)

    schema = DatasetSchema(
        target_col=task.target_name,
        task_type=task_type,
        numerical_cols=num_cols,
        categorical_cols=cat_cols,
        num_classes=n_classes,
    )

    # Deterministic 70% Train, 15% Val, 15% Test split
    stratify = y if schema.is_classification else None

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, stratify=stratify, random_state=random_state
    )

    val_ratio = 0.15 / 0.85
    stratify_temp = y_temp if stratify is not None else None

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, stratify=stratify_temp, random_state=random_state
    )

    return X_train, X_val, X_test, y_train, y_val, y_test, schema
