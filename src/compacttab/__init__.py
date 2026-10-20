"""CompactTab: State-of-the-Art Compact Tabular Deep Learning Architecture."""

from compacttab.data.preprocessor import TabularPreprocessor
from compacttab.data.schema import DatasetSchema, TaskType
from compacttab.evaluation.runner import BenchmarkRunner
from compacttab.models.compacttab import CompactTab, CompactTabConfig
from compacttab.training.trainer import TabularTrainer

__version__ = "0.1.0"

__all__ = [
    "CompactTab",
    "CompactTabConfig",
    "DatasetSchema",
    "TaskType",
    "TabularPreprocessor",
    "TabularTrainer",
    "BenchmarkRunner",
]
