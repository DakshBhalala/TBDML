"""PyTorch training loop with early stopping, multi-head loss, and device management."""

import copy
import os
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from compacttab.data.dataset import TabularDataset, create_dataloader
from compacttab.data.schema import DatasetSchema, TaskType


class TabularTrainer:
    """Production-grade trainer for CompactTab and tabular neural networks."""

    def __init__(
        self,
        model: nn.Module,
        schema: DatasetSchema,
        device: str | None = None,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        max_epochs: int = 50,
        patience: int = 10,
        batch_size: int = 128,
        mixup_alpha: float = 0.0,
        label_smoothing: float = 0.01,
        verbose: bool = False,
    ):
        self.model = model
        self.schema = schema
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.mixup_alpha = mixup_alpha
        self.label_smoothing = label_smoothing if schema.is_classification else 0.0
        self.verbose = verbose

        # Automatic device assignment
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model.to(self.device)

        # Loss function
        if self.schema.task_type == TaskType.BINARY:
            self.criterion = nn.BCEWithLogitsLoss()
        elif self.schema.task_type == TaskType.MULTICLASS:
            self.criterion = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        else:
            self.criterion = nn.MSELoss()

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        warmup_epochs = min(3, max(1, max_epochs // 10))

        def lr_lambda(epoch: int) -> float:
            if epoch < warmup_epochs:
                return float(epoch + 1) / float(max(1, warmup_epochs))
            progress = float(epoch - warmup_epochs) / float(max(1, max_epochs - warmup_epochs))
            return 0.5 * (1.0 + float(np.cos(np.pi * progress)))

        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)

        self.best_val_loss = float("inf")
        self.best_state_dict: dict[str, Any] | None = None
        self.history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

    def _compute_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute loss supporting both multi-head ensemble tensors and single-head tensors."""
        if self.schema.task_type == TaskType.BINARY and self.label_smoothing > 0.0:
            targets_eff = targets.float() * (1.0 - 2.0 * self.label_smoothing) + self.label_smoothing
        else:
            targets_eff = targets.float() if self.schema.task_type != TaskType.MULTICLASS else targets.long()

        if outputs.dim() == 3:
            # Multi-head ensemble output: (k, B, out_dim)
            num_heads = outputs.size(0)
            total_loss = torch.tensor(0.0, device=self.device)
            for m in range(num_heads):
                head_out = outputs[m]  # (B, out_dim)
                if self.schema.task_type == TaskType.BINARY:
                    loss_m = self.criterion(head_out.squeeze(-1), targets_eff)
                elif self.schema.task_type == TaskType.MULTICLASS:
                    loss_m = self.criterion(head_out, targets_eff)
                else:
                    loss_m = self.criterion(head_out.squeeze(-1), targets_eff)
                total_loss = total_loss + loss_m
            return total_loss / num_heads
        else:
            # Single-head output: (B, out_dim)
            if self.schema.task_type == TaskType.BINARY:
                return self.criterion(outputs.squeeze(-1), targets_eff)
            elif self.schema.task_type == TaskType.MULTICLASS:
                return self.criterion(outputs, targets_eff)
            else:
                return self.criterion(outputs.squeeze(-1), targets_eff)

    def fit(
        self,
        train_dataset: TabularDataset,
        val_dataset: TabularDataset | None = None,
    ) -> "TabularTrainer":
        """Execute full training loop with early stopping."""
        train_loader = create_dataloader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = (
            create_dataloader(val_dataset, batch_size=self.batch_size, shuffle=False)
            if val_dataset is not None
            else None
        )

        patience_counter = 0

        for epoch in range(1, self.max_epochs + 1):
            self.model.train()
            train_loss_acc = 0.0
            n_train_batches = 0

            for num_vals, num_masks, cat_indices, targets in train_loader:
                num_vals = num_vals.to(self.device)
                num_masks = num_masks.to(self.device)
                cat_indices = cat_indices.to(self.device)
                targets = targets.to(self.device)

                self.optimizer.zero_grad()
                if self.mixup_alpha > 0.0 and num_vals.size(0) > 1:
                    lam = float(np.random.beta(self.mixup_alpha, self.mixup_alpha))
                    perm = torch.randperm(num_vals.size(0), device=self.device)
                    num_vals_in = lam * num_vals + (1.0 - lam) * num_vals[perm]
                    num_masks_in = torch.maximum(num_masks, num_masks[perm])
                    if cat_indices.numel() > 0:
                        cat_select = (torch.rand(cat_indices.shape, device=self.device) < lam).long()
                        cat_indices_in = cat_select * cat_indices + (1 - cat_select) * cat_indices[perm]
                    else:
                        cat_indices_in = cat_indices

                    outputs = self.model(num_vals_in, num_masks_in, cat_indices_in)
                    loss = lam * self._compute_loss(outputs, targets) + (1.0 - lam) * self._compute_loss(
                        outputs, targets[perm]
                    )
                else:
                    outputs = self.model(num_vals, num_masks, cat_indices)
                    loss = self._compute_loss(outputs, targets)

                loss.backward()

                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                train_loss_acc += loss.item()
                n_train_batches += 1

            self.scheduler.step()
            avg_train_loss = train_loss_acc / max(1, n_train_batches)
            self.history["train_loss"].append(avg_train_loss)

            # Validation phase
            if val_loader is not None:
                val_loss = self.evaluate_loss(val_loader)
                self.history["val_loss"].append(val_loss)

                if self.verbose and (epoch % 5 == 0 or epoch == self.max_epochs):
                    print(f"Epoch {epoch:03d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}")

                if val_loss < self.best_val_loss - 1e-4:
                    self.best_val_loss = val_loss
                    self.best_state_dict = copy.deepcopy(self.model.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        if self.verbose:
                            print(f"Early stopping triggered at epoch {epoch}.")
                        break
            else:
                self.best_state_dict = copy.deepcopy(self.model.state_dict())

        # Load best weights
        if self.best_state_dict is not None:
            self.model.load_state_dict(self.best_state_dict)

        return self

    def evaluate_loss(self, dataloader: DataLoader) -> float:
        """Evaluate loss over a dataloader."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        with torch.no_grad():
            for num_vals, num_masks, cat_indices, targets in dataloader:
                num_vals = num_vals.to(self.device)
                num_masks = num_masks.to(self.device)
                cat_indices = cat_indices.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(num_vals, num_masks, cat_indices)
                loss = self._compute_loss(outputs, targets)
                total_loss += loss.item()
                n_batches += 1
        return total_loss / max(1, n_batches)

    def predict_proba(self, dataloader: DataLoader) -> np.ndarray:
        """Generate prediction probabilities (classification)."""
        self.model.eval()
        preds = []
        with torch.no_grad():
            for num_vals, num_masks, cat_indices, _ in dataloader:
                num_vals = num_vals.to(self.device)
                num_masks = num_masks.to(self.device)
                cat_indices = cat_indices.to(self.device)

                # Inference forward returns ensemble averaged probabilities
                out = self.model(num_vals, num_masks, cat_indices, return_ensemble=False)
                preds.append(out.cpu().numpy())

        res = np.concatenate(preds, axis=0)
        if self.schema.task_type == TaskType.BINARY and res.ndim == 2 and res.shape[1] == 1:
            # Format as (N, 2) probability array [P(y=0), P(y=1)]
            p1 = res[:, 0]
            p0 = 1.0 - p1
            return np.stack([p0, p1], axis=1)
        return res

    def predict(self, dataloader: DataLoader) -> np.ndarray:
        """Generate point predictions."""
        if self.schema.is_classification:
            probs = self.predict_proba(dataloader)
            return np.argmax(probs, axis=1)
        else:
            self.model.eval()
            preds = []
            with torch.no_grad():
                for num_vals, num_masks, cat_indices, _ in dataloader:
                    num_vals = num_vals.to(self.device)
                    num_masks = num_masks.to(self.device)
                    cat_indices = cat_indices.to(self.device)

                    out = self.model(num_vals, num_masks, cat_indices, return_ensemble=False)
                    preds.append(out.cpu().numpy())
            return np.concatenate(preds, axis=0).squeeze(-1)

    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint and training metadata."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "best_val_loss": self.best_val_loss,
                "history": self.history,
                "schema": self.schema,
            },
            path,
        )

    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        self.history = checkpoint.get("history", {"train_loss": [], "val_loss": []})
