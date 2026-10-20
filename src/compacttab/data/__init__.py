"""Data handling and preprocessing subpackage for CompactTab."""

from compacttab.data.dataset import TabularDataset, create_dataloader
from compacttab.data.loaders import (
    generate_synthetic_tabular_dataset,
    load_breast_cancer_dataset,
    load_california_housing_dataset,
    load_diabetes_dataset,
    load_wine_dataset,
    split_data,
)
from compacttab.data.preprocessor import TabularPreprocessor
from compacttab.data.schema import DatasetSchema, FeatureType, TaskType

__all__ = [
    "DatasetSchema",
    "FeatureType",
    "TaskType",
    "TabularPreprocessor",
    "TabularDataset",
    "create_dataloader",
    "split_data",
    "load_california_housing_dataset",
    "load_breast_cancer_dataset",
    "load_wine_dataset",
    "load_diabetes_dataset",
    "generate_synthetic_tabular_dataset",
]
