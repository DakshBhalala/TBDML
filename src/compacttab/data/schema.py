"""Data schema definitions for CompactTab."""

from dataclasses import dataclass, field
from enum import StrEnum


class TaskType(StrEnum):
    BINARY = "binary"
    MULTICLASS = "multiclass"
    REGRESSION = "regression"


class FeatureType(StrEnum):
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"


@dataclass
class DatasetSchema:
    """Schema metadata describing tabular features and target."""

    target_col: str
    task_type: TaskType
    numerical_cols: list[str] = field(default_factory=list)
    categorical_cols: list[str] = field(default_factory=list)
    num_classes: int | None = None
    target_mean: float = 0.0
    target_std: float = 1.0

    @property
    def num_features(self) -> int:
        return len(self.numerical_cols) + len(self.categorical_cols)

    @property
    def is_classification(self) -> bool:
        return self.task_type in (TaskType.BINARY, TaskType.MULTICLASS)
