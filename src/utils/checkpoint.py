"""
Checkpoint management for saving and loading model states.

Example:
    >>> checkpoint_manager = CheckpointManager(device="cuda", save_dir="checkpoints")
    >>> # During training
    >>> checkpoint_manager.save_if_improved(
    ...     epoch, model, optimizer, scaler, scheduler, val_loss
    ... )
    >>> # For resuming training
    >>> start_epoch = checkpoint_manager.load(
    ...     model, optimizer, scaler, scheduler, resume_training=True
    ... )
"""

from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler


class CheckpointManager:
    """
    Manages model checkpointing with patience-based saving to avoid over-saving.

    Features:
    - Saves only when validation loss improves and patience interval has passed
    - Loads model, optimizer, scaler, and scheduler states
    - Supports training resumption from saved epoch
    """

    def __init__(
        self,
        device: str,
        save_dir: str = "models/checkpoints",
        filename: str = "best_model.pt",
        patience: int = 1,
        verbose: bool = True,
    ) -> None:
        """
        Args:
            device: Device for loading checkpoints ('cuda' or 'cpu').
            save_dir: Directory to save checkpoints.
            filename: Checkpoint filename.
            patience: Minimum epochs between saves when loss improves.
            verbose: Print status messages if True.
        """
        self.device = torch.device(device)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.filename = filename
        self.patience = patience
        self.verbose = verbose

        self.best_loss = float("inf")
        self.counter = 0
        self.start_epoch = 0

    def save(
        self,
        epoch: int,
        model: nn.Module,
        optimizer: Optional[optim.Optimizer] = None,
        scaler: Optional[GradScaler] = None,
        scheduler: Optional[optim.lr_scheduler.LRScheduler] = None,
        val_loss: float = float("inf"),
        **kwargs: Any,
    ) -> None:
        """
        Save checkpoint with model and training state.

        Args:
            epoch: Current epoch.
            model: Model to save.
            optimizer: Optimizer state to save (optional).
            scaler: GradScaler state for mixed precision (optional).
            scheduler: Scheduler state to save (optional).
            val_loss: Validation loss to store in checkpoint.
            **kwargs: Additional data to save.
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
            "scaler_state_dict": scaler.state_dict() if scaler else None,
            "scheduler_state": scheduler.state_dict() if scheduler else None,
            "val_loss": val_loss,
        }

        if kwargs:
            checkpoint.update(**kwargs)

        path = Path(self.save_dir) / self.filename
        torch.save(checkpoint, path)

    def save_if_improved(
        self,
        epoch: int,
        model: nn.Module,
        optimizer: Optional[optim.Optimizer] = None,
        scaler: Optional[GradScaler] = None,
        scheduler: Optional[optim.lr_scheduler.LRScheduler] = None,
        val_loss: float = float("inf"),
    ) -> None:
        """
        Save checkpoint only if validation loss improved and patience interval passed.

        Args:
            epoch: Current epoch.
            model: Model to save.
            optimizer: Optimizer state to save (optional).
            scaler: GradScaler state for mixed precision (optional).
            scheduler: Scheduler state to save (optional).
            val_loss: Current validation loss.
        """
        if val_loss < self.best_loss:
            if epoch - self.counter >= self.patience:
                if self.verbose:
                    print(
                        f"Validation improved on epoch {epoch} "
                        f"({self.best_loss:.4f} → {val_loss:.4f}). Saving checkpoint."
                    )
                self.best_loss = val_loss
                self.save(epoch, model, optimizer, scaler, scheduler, val_loss)
                self.counter = epoch

    def load(
        self,
        model: nn.Module,
        optimizer: Optional[optim.Optimizer] = None,
        scaler: Optional[GradScaler] = None,
        scheduler: Optional[optim.lr_scheduler.LRScheduler] = None,
        resume_training: bool = False,
        **kwargs,
    ) -> int | Dict[str, Any]:
        """
        Load checkpoint into model.

        Args:
            model: Model to load weights into.
            optimizer: Optimizer to load state into (optional).
            scaler: GradScaler to load state into (optional).
            scheduler: Scheduler to load state into (optional).
            resume_training: Return saved epoch + 1 if True, otherwise return 0.

        Returns:
            - Epoch to resume from  (if resume_training=True) or
            - Full checkpoint dictionary (if resume_training=False):
        """
        path = Path(self.save_dir) / self.filename
        if not Path(path).exists():
            print("No checkpoint found.")
            return 0

        checkpoint = torch.load(path, map_location=self.device)

        if "model_state_dict" not in checkpoint:
            raise KeyError("Invalid checkpoint: missing model_state_dict.")

        model.load_state_dict(checkpoint["model_state_dict"])

        if optimizer and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scaler is not None and "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        print(f"Checkpoint loaded from {path}.")

        if kwargs:
            checkpoint.update(**kwargs)

        if resume_training:
            self.start_epoch = checkpoint.get("epoch", 1)
            print(f"Resuming from epoch {self.start_epoch}.")
            return self.start_epoch

        return checkpoint
