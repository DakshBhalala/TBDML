"""Data schema definitions for CompactTab."""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class TaskType(str, Enum):
    BINARY = "binary"
    MULTICLASS = "multiclass"
    REGRESSION = "regression"


class FeatureType(str, Enum):
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"


@dataclass
class DatasetSchema:
    """Schema metadata describing tabular features and target."""

    target_col: str
    task_type: TaskType
    numerical_cols: List[str] = field(default_factory=list)
    categorical_cols: List[str] = field(default_factory=list)
    num_classes: Optional[int] = None
    target_mean: float = 0.0
    target_std: float = 1.0

    @property
    def num_features(self) -> int:
        return len(self.numerical_cols) + len(self.categorical_cols)

    @property
    def is_classification(self) -> bool:
        return self.task_type in (TaskType.BINARY, TaskType.MULTICLASS)
