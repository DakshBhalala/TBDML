"""Standard reproducible benchmark dataset loaders with deterministic train/val/test splits."""

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing, load_breast_cancer, load_diabetes, load_wine
from sklearn.model_selection import train_test_split

from compacttab.data.schema import DatasetSchema, TaskType


def split_data(
    df: pd.DataFrame,
    y: pd.Series,
    task_type: TaskType,
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Deterministically split data into train, val, and test partitions."""
    stratify = y if task_type in (TaskType.BINARY, TaskType.MULTICLASS) else None

    # First split off test set
    df_temp, df_test, y_temp, y_test = train_test_split(
        df,
        y,
        test_size=test_size,
        stratify=stratify,
        random_state=random_state,
    )

    # Next split temp into train and val
    val_ratio_of_temp = val_size / (train_size + val_size)
    stratify_temp = y_temp if stratify is not None else None

    df_train, df_val, y_train, y_val = train_test_split(
        df_temp,
        y_temp,
        test_size=val_ratio_of_temp,
        stratify=stratify_temp,
        random_state=random_state,
    )

    return df_train, df_val, df_test, y_train, y_val, y_test


def load_california_housing_dataset(
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, DatasetSchema]:
    """Load California Housing regression benchmark."""
    data = fetch_california_housing(as_frame=True)
    df = data.data
    y = data.target
    schema = DatasetSchema(
        target_col=data.target_names[0] if data.target_names else "MedHouseVal",
        task_type=TaskType.REGRESSION,
        numerical_cols=list(df.columns),
        categorical_cols=[],
    )
    df_train, df_val, df_test, y_train, y_val, y_test = split_data(
        df, y, schema.task_type, random_state=random_state
    )
    return df_train, df_val, df_test, y_train, y_val, y_test, schema


def load_breast_cancer_dataset(
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, DatasetSchema]:
    """Load Breast Cancer Wisconsin binary classification benchmark."""
    data = load_breast_cancer(as_frame=True)
    df = data.data
    y = data.target
    schema = DatasetSchema(
        target_col="target",
        task_type=TaskType.BINARY,
        numerical_cols=list(df.columns),
        categorical_cols=[],
        num_classes=2,
    )
    df_train, df_val, df_test, y_train, y_val, y_test = split_data(
        df, y, schema.task_type, random_state=random_state
    )
    return df_train, df_val, df_test, y_train, y_val, y_test, schema


def load_wine_dataset(
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, DatasetSchema]:
    """Load Wine multiclass classification benchmark."""
    data = load_wine(as_frame=True)
    df = data.data
    y = data.target
    schema = DatasetSchema(
        target_col="target",
        task_type=TaskType.MULTICLASS,
        numerical_cols=list(df.columns),
        categorical_cols=[],
        num_classes=3,
    )
    df_train, df_val, df_test, y_train, y_val, y_test = split_data(
        df, y, schema.task_type, random_state=random_state
    )
    return df_train, df_val, df_test, y_train, y_val, y_test, schema


def load_diabetes_dataset(
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, DatasetSchema]:
    """Load Diabetes regression benchmark."""
    data = load_diabetes(as_frame=True)
    df = data.data
    y = data.target
    schema = DatasetSchema(
        target_col="target",
        task_type=TaskType.REGRESSION,
        numerical_cols=list(df.columns),
        categorical_cols=[],
    )
    df_train, df_val, df_test, y_train, y_val, y_test = split_data(
        df, y, schema.task_type, random_state=random_state
    )
    return df_train, df_val, df_test, y_train, y_val, y_test, schema


def generate_synthetic_tabular_dataset(
    n_samples: int = 1000,
    n_num: int = 6,
    n_cat: int = 2,
    task_type: TaskType = TaskType.BINARY,
    missing_rate: float = 0.05,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, DatasetSchema]:
    """Generate synthetic tabular dataset with mixed features and missingness."""
    rng = np.random.default_rng(random_state)

    # Numerical features
    num_data = rng.standard_normal((n_samples, n_num))
    if missing_rate > 0:
        mask = rng.uniform(0, 1, size=num_data.shape) < missing_rate
        num_data[mask] = np.nan
    num_cols = [f"num_{i}" for i in range(n_num)]
    df = pd.DataFrame(num_data, columns=num_cols)

    # Categorical features
    cat_cols = []
    for j in range(n_cat):
        col_name = f"cat_{j}"
        cat_cols.append(col_name)
        choices = [f"grp_{k}" for k in range(4)]
        cats = rng.choice(choices, size=n_samples)
        df[col_name] = cats

    # Target calculation
    latent = np.sum(np.nan_to_num(num_data, nan=0.0)[:, :3], axis=1)
    if task_type == TaskType.BINARY:
        prob = 1.0 / (1.0 + np.exp(-latent))
        y_arr = (rng.uniform(0, 1, size=n_samples) < prob).astype(int)
        y = pd.Series(y_arr, name="target")
        num_classes = 2
    elif task_type == TaskType.MULTICLASS:
        y_arr = np.digitize(latent, bins=[-1.0, 1.0])
        y = pd.Series(y_arr, name="target")
        num_classes = 3
    else:
        y_arr = latent + rng.standard_normal(n_samples) * 0.2
        y = pd.Series(y_arr, name="target")
        num_classes = None

    schema = DatasetSchema(
        target_col="target",
        task_type=task_type,
        numerical_cols=num_cols,
        categorical_cols=cat_cols,
        num_classes=num_classes,
    )

    df_train, df_val, df_test, y_train, y_val, y_test = split_data(
        df, y, task_type, random_state=random_state
    )
    return df_train, df_val, df_test, y_train, y_val, y_test, schema
