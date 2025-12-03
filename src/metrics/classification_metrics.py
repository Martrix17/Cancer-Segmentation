"""
Metrics management for multi-class classification metrics.

Example:
    >>> metrics = ClassificationMetrics(num_classes=4, device="cuda")
    >>> results = metrics.compute(preds, targets)
    >>> print(results["val_cls/accuracy"])

    >>> metrics.set_mode(test_mode=True)
    >>> test_results = metrics.compute(
    ...     preds, targets, class_names=["class_0", "class_1", "class_2", "class_3"]
    ... )
    >>> print(results["test_cls/accuracy"])
"""

from typing import Dict, List, Optional

import torch
from sklearn.metrics import classification_report
from torchmetrics import Accuracy, F1Score, Precision, Recall
from torchmetrics.classification import (
    MulticlassAUROC,
    MulticlassAveragePrecision,
    MulticlassConfusionMatrix,
    MulticlassROC,
)


class ClassificationMetrics:
    """
    Manages classification metrics for multi-class classification tasks.

    Supports two modes:
    - Validation: Standard metrics (accuracy, precision, recall, F1, AUROC).
    - Test: Confusion matrix, ROC curves, and classification report.
    """

    def __init__(
        self,
        num_classes: int = 7,
        device: str = "cpu",
        test_mode: bool = False,
        average: str = "macro",
    ) -> None:
        """
        Args:
            num_classes: Number of classification classes.
            device: Device for metric computations ('cuda' or 'cpu').
            test_mode: Initialize in test mode if True.
            average: Averaging method for multi-class metrics
                    ('macro', 'micro', 'weighted').
        """
        self.num_classes = num_classes
        self.device = torch.device(device)
        self.test_mode = test_mode
        self.average = average

        self._all_preds = []
        self._all_targets = []

        self._init_metrics()

    def set_mode(self, test_mode: bool) -> None:
        """
        Switches between training/validation and test mode metrics sets.

        Args:
            test_mode: Switches mode if True.
        """
        if self.test_mode != test_mode:
            self.test_mode = test_mode
            self._init_metrics()

    def _init_metrics(self) -> None:
        """Initialize metrics based on current mode."""
        if self.test_mode:
            self.metrics = {
                "auroc": MulticlassAUROC(num_classes=self.num_classes).to(self.device),
                "roc_curve": MulticlassROC(num_classes=self.num_classes).to(
                    self.device
                ),
                "confmat": MulticlassConfusionMatrix(num_classes=self.num_classes).to(
                    self.device
                ),
            }
        else:
            self.metrics = {
                "accuracy": Accuracy(
                    task="multiclass", num_classes=self.num_classes
                ).to(self.device),
                "precision": Precision(
                    task="multiclass", num_classes=self.num_classes, average="macro"
                ).to(self.device),
                "recall": Recall(
                    task="multiclass", num_classes=self.num_classes, average="macro"
                ).to(self.device),
                "f1": F1Score(
                    task="multiclass", num_classes=self.num_classes, average="macro"
                ).to(self.device),
                "auroc": MulticlassAUROC(num_classes=self.num_classes).to(self.device),
                "avg_prec": MulticlassAveragePrecision(
                    num_classes=self.num_classes, average=None
                ).to(self.device),
            }

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """
        Update metrics with a batch.

        Args:
            preds: Model prediction probabilities [B, num_classes].
            targets: Ground truth label [B].
        """
        preds = torch.softmax(preds, dim=1)
        preds = preds.to(self.device)
        preds_class = preds.argmax(dim=1)
        targets = targets.to(self.device)

        for name, metric in self.metrics.items():
            if name in ("auroc", "roc_curve", "avg_prec"):
                metric.update(preds, targets)
            else:
                metric.update(preds_class, targets)

        if self.test_mode:
            self._all_preds.append(preds_class.cpu())
            self._all_targets.append(targets.cpu())

    def compute(
        self,
        class_names: Optional[List[str]] = None,
    ) -> Dict[str, str | float | torch.Tensor]:
        """
        Compute metrics for current mode.

        Args:
            class_names: Class names for classification report (test mode only).

        Returns:
            Dict with metric names as keys and computed values.
        """
        results = {}

        for name, metric in self.metrics.items():
            value = metric.compute()

            key = f"val_cls/{name}" if not self.test_mode else f"test_cls/{name}"
            if isinstance(value, torch.Tensor):
                if value.ndim == 0:
                    results[key] = value.item()
                elif value.ndim == 1 and name not in ("roc_curve", "confmat"):
                    results[f"{key}_{self.average}"] = value.mean().item()
                else:
                    results[key] = value
            else:
                results[key] = value

        if self.test_mode:
            preds_class = torch.cat(self._all_preds).numpy()
            targets_np = torch.cat(self._all_targets).numpy()

            results["report"] = classification_report(
                y_true=targets_np,
                y_pred=preds_class,
                target_names=class_names,
                zero_division=0,
            )

        return results

    def reset(self) -> None:
        """Reset all metrics for new epoch."""
        for metric in self.metrics.values():
            metric.reset()

        if self.test_mode:
            self._all_preds.clear()
            self._all_targets.clear()
