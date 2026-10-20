"""PyTorch Dataset and DataLoader abstractions for Tabular data."""

from typing import Optional, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class TabularDataset(Dataset):
    """PyTorch Dataset wrapping preprocessed tabular inputs."""

    def __init__(
        self,
        num_vals: np.ndarray,
        num_masks: np.ndarray,
        cat_indices: np.ndarray,
        targets: Optional[np.ndarray] = None,
    ):
        self.num_vals = torch.from_numpy(num_vals).float()
        self.num_masks = torch.from_numpy(num_masks).float()
        self.cat_indices = torch.from_numpy(cat_indices).long()
        self.targets = torch.from_numpy(targets) if targets is not None else None

    def __len__(self) -> int:
        return len(self.num_vals)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        target = self.targets[idx] if self.targets is not None else torch.empty(0)
        return self.num_vals[idx], self.num_masks[idx], self.cat_indices[idx], target


def create_dataloader(
    dataset: TabularDataset,
    batch_size: int = 128,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    """Create a high-performance DataLoader."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
