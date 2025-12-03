"""
Metrics management for segmentation metrics.

Example:
    >>> metrics = SegmentationMetrics(device="cuda")
    >>> results = metrics.compute(preds, targets)
    >>> print(results["val_seg/iou"])

    >>> metrics.set_mode(test_mode=True)
    >>> test_results = metrics.compute()
    >>> print(results["test_seg/iou"])
"""

from typing import Dict

import torch
from torchmetrics.segmentation import GeneralizedDiceScore, MeanIoU


class SegmentationMetrics:
    """
    Manages segmentation metrics.
    """

    def __init__(
        self,
        num_classes: int = 1,
        device: str = "cpu",
        test_mode: bool = False,
    ) -> None:
        """
        Args:
            num_classes: Number of segmentation classes.
            device: Device for metric computations ('cuda' or 'cpu').
        """
        self.num_classes = num_classes
        self.device = torch.device(device)
        self.test_mode = test_mode

        self.metrics = {
            "iou": MeanIoU(num_classes=num_classes).to(device),
            "dice": GeneralizedDiceScore(num_classes=num_classes).to(device),
        }

    def set_mode(self, test_mode: bool) -> None:
        """
        Switches between training/validation and test mode.

        Args:
            test_mode: Switches mode if True.
        """
        if self.test_mode != test_mode:
            self.test_mode = test_mode

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """
        Update metrics with a batch of predictions.

        Args:
            preds: Model prediction mask [B, H, W].
            targets: Ground truth mask [B, H, W].
        """
        preds = torch.sigmoid(preds)
        preds = (preds > 0.5).long()

        if preds.ndim == 4 and preds.shape[1] == 1:
            preds = preds.squeeze(1)

        if targets.ndim == 4 and targets.shape[1] == 1:
            targets = targets.squeeze(1)

        preds = preds.to(self.device)
        targets = targets.long().to(self.device)

        for metric in self.metrics.values():
            metric.update(preds, targets)

    def compute(self) -> Dict[str, float]:
        """
        Compute segmentation metrics.

        Returns:
            Dict with metric names as keys and computed values.
        """
        results = {}
        for name, metric in self.metrics.items():
            value = metric.compute()

            key = f"val_seg/{name}" if not self.test_mode else f"test_seg/{name}"
            if value.ndim == 0:
                results[key] = value.item()

        return results

    def reset(self) -> None:
        """Reset all metrics for new epoch."""
        for m in self.metrics.values():
            m.reset()
