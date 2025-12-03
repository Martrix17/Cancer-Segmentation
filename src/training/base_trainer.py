"""
Base trainer for model training and evaluation.

Example:
    >>> trainer = BaseTrainer(
    ...     device="cuda",
    ...     model=model,
    ...     criterion=nn.CrossEntropyLoss(),
    ...     optimizer=optim.Adam(model.parameters())
    ... )
    >>> train_loss = trainer.train(train_loader, epoch=0, total_epochs=10)
    >>> val_results = trainer.evaluate(val_loader, epoch=0, total_epochs=10)
"""

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.metrics.classification_metrics import ClassificationMetrics
from src.metrics.segmentation_metrics import SegmentationMetrics


class BaseTrainer:
    """
    Handles training and evaluation loops with automatic mixed precision (AMP).

    Features:
    - Automatic mixed precision
    - Progress bars for loss tracking
    - Evaluation with loss computation, metrics computation and prediction collection
    """

    def __init__(
        self,
        device: str,
        model: nn.Module,
        criterion: Optional[nn.Module] = None,
        optimizer: Optional[optim.Optimizer] = None,
        seg_metrics: Optional[SegmentationMetrics] = None,
        cls_metrics: Optional[ClassificationMetrics] = None,
        use_compile: bool = False,
        tqdm_refresh_steps: int = 20,
    ) -> None:
        """
        Args:
            device: Device for training ('cuda' or 'cpu').
            model: Neural network to train.
            criterion: Loss function (optional for testing/inference).
            optimizer: Optimizer instance (optional for testing/inference).
            seg_metrics: Segmentation metrics manager (optional).
            cls_metrics: Classification metrics manager (optional).
            use_compile: Whether to use torch.compile for model optimization.
            tqdm_refresh_steps: Frequency of progress bar refresh.
        """
        self.device = torch.device(device)
        self.model = model.to(device)

        if use_compile:
            try:
                self.model = torch.compile(self.model)
                print("Model compiled for optimization.")
            except Exception:
                print("Cannot compile model.")
                pass

        self.criterion = criterion
        self.optimizer = optimizer
        self.scaler = GradScaler()

        self.seg_metrics = seg_metrics
        self.cls_metrics = cls_metrics

        self.tqdm_refresh_steps = max(1, tqdm_refresh_steps)

    @staticmethod
    def _parse_losses(
        loss_obj: Dict[str, torch.Tensor] | torch.Tensor, device: torch.device
    ) -> Tuple[torch.Tensor]:
        """
        Parse loss score dictionary into tensors (total_loss, seg_loss, cls_loss).

        Args:
            loss_obj: Dict containing model output losses.
            device: Torch device to transfer to.

        Returns:
            Tuple of loss tensors.
        """
        if isinstance(loss_obj, dict):
            total = loss_obj.get("loss")
            seg = loss_obj.get("seg_loss", torch.tensor(0.0, device=device))
            cls = loss_obj.get("cls_loss", torch.tensor(0.0, device=device))
            return total, seg, cls
        else:
            return (
                loss_obj,
                torch.tensor(0.0, device=device),
                torch.tensor(0.0, device=device),
            )

    @staticmethod
    def _to_device_batch(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
        """
        Move all tensors in batch to specified device.

        Args:
            batch: Current batch of data.
            device: Torch device to transfer to.

        Returns:
            Dictionary with batch transfered to device.
        """
        out = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.to(device, non_blocking=True)
            else:
                out[k] = v
        return out

    def train(
        self, loader: DataLoader, epoch: int, total_epochs: int
    ) -> Dict[str, float]:
        """
        Run one training epoch with mixed precision.

        Args:
            loader: Training data loader.
            epoch: Current epoch index.
            total_epochs: Total number of epochs.

        Returns:
            Dictionary of averaged losses over the epoch.
        """
        if self.optimizer is None:
            raise RuntimeError("Optimizer is not defined for training.")

        self.model.train()
        n_batches = len(loader)

        acc_total = torch.tensor(0.0, device=self.device)
        acc_seg = torch.tensor(0.0, device=self.device)
        acc_cls = torch.tensor(0.0, device=self.device)

        pbar = tqdm(loader, desc=f"[Train] Epoch {epoch}/{total_epochs}", leave=False)

        for batch_idx, batch in enumerate(pbar):
            batch = self._to_device_batch(batch, self.device)
            imgs = batch["img"]
            masks = batch["mask"]
            labels = batch["label"]

            self.optimizer.zero_grad(set_to_none=True)

            device_type = "cuda" if self.device.type == "cuda" else "cpu"
            with autocast(device_type=device_type):
                seg_pred, cls_pred = self.model(imgs)
                raw_loss = self.criterion(seg_pred, cls_pred, masks, labels)
                total_loss, seg_loss, cls_loss = self._parse_losses(
                    raw_loss, self.device
                )

            self.scaler.scale(total_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            acc_total += total_loss.detach()
            acc_seg += seg_loss.detach()
            acc_cls += cls_loss.detach()

            if (batch_idx % self.tqdm_refresh_steps) == 0 or batch_idx + 1 == n_batches:
                pbar.set_postfix(
                    {
                        "total": f"{(acc_total / (batch_idx + 1)):.4f}",
                        "seg": f"{(acc_seg / (batch_idx + 1)):.4f}",
                        "cls": f"{(acc_cls / (batch_idx + 1)):.4f}",
                    }
                )

        return {
            "train_loss": (acc_total / n_batches).cpu().item(),
            "train_seg_loss": (acc_seg / n_batches).cpu().item(),
            "train_cls_loss": (acc_cls / n_batches).cpu().item(),
        }

    @torch.no_grad()
    def evaluate(
        self,
        loader: DataLoader,
        epoch: int,
        total_epochs: int,
        desc: str = "Validation",
        compute_loss: bool = True,
        compute_metrics: bool = False,
        num_pred_samples: Optional[int] = None,
        class_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Run one evaluation epoch with optional loss and prediction collection.

        Args:
            loader: Val/test/infer data loader.
            epoch: Current epoch index.
            total_epochs: Total number of epochs.
            desc: Progress bar description for current mode.
            compute_loss: Compute loss over set if True.
            compute_metrics: Compute metrics if True.
            num_pred_samples: Number of prediction samples to store (optional).
            class_names: List of class names for multi-label classification (optional).

        Returns:
            Dict containing losses, metrics, and prediction samples.
        """
        self.model.eval()
        n_batches = len(loader)

        if compute_loss:
            acc_total = torch.tensor(0.0, device=self.device)
            acc_seg = torch.tensor(0.0, device=self.device)
            acc_cls = torch.tensor(0.0, device=self.device)

        if compute_metrics:
            if self.seg_metrics and self.cls_metrics:
                self.seg_metrics.reset()
                self.cls_metrics.reset()
            else:
                compute_metrics = False
                print(
                    "Warning: compute_metrics=True but no metrics manager defined. "
                    "Skipping metrics computation."
                )

        pred_samples = [] if num_pred_samples else None
        samples_saved = 0

        pbar = tqdm(loader, desc=f"[{desc}] Epoch {epoch}/{total_epochs}", leave=False)

        for batch_idx, batch in enumerate(pbar):
            batch = self._to_device_batch(batch, self.device)
            imgs = batch["img"]
            masks = batch.get("mask")
            labels = batch.get("label")

            device_type = "cuda" if self.device.type == "cuda" else "cpu"
            with autocast(device_type=device_type):
                seg_pred, cls_pred = self.model(imgs)

                if compute_loss:
                    if masks is None or labels is None:
                        raise ValueError("compute_loss=True but masks/labels missing.")

                    raw_loss = self.criterion(seg_pred, cls_pred, masks, labels)
                    total_loss, seg_loss, cls_loss = self._parse_losses(
                        raw_loss, self.device
                    )
                    acc_total += total_loss.detach()
                    acc_seg += seg_loss.detach()
                    acc_cls += cls_loss.detach()

                if compute_metrics:
                    if masks is None or labels is None:
                        print(
                            "Warning: compute_metrics=True but masks/labels missing. "
                            "Skipping metrics computation."
                        )
                    else:
                        self.seg_metrics.update(seg_pred, masks)
                        self.cls_metrics.update(cls_pred, labels)

            if compute_loss and (
                (batch_idx % self.tqdm_refresh_steps) == 0 or batch_idx + 1 == n_batches
            ):
                pbar.set_postfix(
                    {
                        "total": f"{(acc_total / (batch_idx + 1)):.4f}",
                        "seg": f"{(acc_seg / (batch_idx + 1)):.4f}",
                        "cls": f"{(acc_cls / (batch_idx + 1)):.4f}",
                    }
                )

            if pred_samples is not None and samples_saved < num_pred_samples:
                batch_size = imgs.shape[0]
                to_take = min(batch_size, num_pred_samples - samples_saved)

                imgs_cpu = imgs.detach().cpu()
                seg_cpu = seg_pred.detach().cpu()
                cls_cpu = cls_pred.detach().cpu()
                masks_cpu = masks.detach().cpu() if masks is not None else None
                labels_cpu = labels.detach().cpu() if labels is not None else None
                img_ids = batch.get("img_id", [None] * batch_size)

                for i in range(to_take):
                    seg_prob = torch.sigmoid(seg_cpu[i])
                    seg_bin_mask = (seg_prob > 0.5).squeeze(0)

                    pred_samples.append(
                        {
                            "image_id": img_ids[i],
                            "image": imgs_cpu[i],
                            "seg_pred": seg_bin_mask,
                            "seg_mask": masks_cpu[i] if masks_cpu is not None else None,
                            "cls_pred": int(cls_cpu[i].argmax().item()),
                            "cls_label": (
                                int(labels_cpu[i].item())
                                if labels_cpu is not None
                                else None
                            ),
                        }
                    )
                    samples_saved += 1

                del seg_cpu, cls_cpu, imgs_cpu, masks_cpu, labels_cpu

        output = {}
        if compute_loss:
            output["losses"] = {
                "eval_loss": (acc_total / n_batches).cpu().item(),
                "eval_seg_loss": (acc_seg / n_batches).cpu().item(),
                "eval_cls_loss": (acc_cls / n_batches).cpu().item(),
            }

        if compute_metrics:
            output["metrics"] = self.seg_metrics.compute()
            output["metrics"].update(self.cls_metrics.compute(class_names=class_names))

        if pred_samples:
            output["pred_samples"] = pred_samples

        return output
