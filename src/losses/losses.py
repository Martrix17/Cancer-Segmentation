"""
Loss functions for segmentation and classification tasks.

Segementation losses: DiceLoss, BCEDiceLoss, TverskyLoss
Classification losses: FocalLoss
Multitask losses: WeightedMultiTaskLoss, UncertaintyWeightedMultiTaskLoss
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Dice loss for binary segmentation."""

    def __init__(self, eps: float = 1e-7, apply_sigmoid: bool = True) -> None:
        """
        Args:
            eps: Small constant to avoid division by zero.
            apply_sigmoid: Applies sigmoid activation to inputs if True.
        """
        super().__init__()
        self.eps = eps
        self.apply_sigmoid = apply_sigmoid

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Model prediction tensor [B, C, H, W].
            targets: Ground truth tensor [B, H, W].

        Returns:
            Tensor containing dice loss score.
        """
        if self.apply_sigmoid:
            inputs = torch.sigmoid(inputs)

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        union = inputs.sum() + targets.sum()
        dice = (2 * intersection + self.eps) / (union + self.eps)
        return 1 - dice


class BCEDiceLoss(nn.Module):
    """Combined BCE and Dice loss for better training stability."""

    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
    ) -> None:
        """
        Args:
            weight: Weight for BCEWithLogitsLoss (optional).
            bce_weight: Weight for the BCE loss.
            dice_weight: Weight for the Dice loss.
        """
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(weight=weight)
        self.dice = DiceLoss(apply_sigmoid=False)
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Model prediction tensor [B, C, H, W].
            targets: Ground truth tensor [B, H, W].

        Returns:
            Tensor containing BCE dice loss score.
        """
        bce_loss = self.bce(inputs, targets)
        dice_loss = self.dice(torch.sigmoid(inputs), targets)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


class TverskyLoss(nn.Module):
    """Tversky loss - generalization of Dice loss. Defaults to Dice loss."""

    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.5,
        eps: float = 1e-7,
    ) -> None:
        """
        Args:
            alpha: Weight for false positives.
            beta: Weight for false negatives.
            eps: Small constant to avoid division by zero.
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Model prediction tensor [B, C, H, W].
            targets: Ground truth tensor [B, H, W].

        Returns:
            Tensor containing Tversky loss score.
        """
        inputs = torch.sigmoid(inputs)
        inputs = inputs.flatten()
        targets = targets.flatten()

        tp = (inputs * targets).sum()
        fp = (inputs * (1 - targets)).sum()
        fn = ((1 - inputs) * targets).sum()

        tversky = (tp + self.eps) / (tp + self.alpha * fp + self.beta * fn + self.eps)
        return 1 - tversky


class FocalLoss(nn.Module):
    """Focal Loss with label smoothing."""

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ) -> None:
        """
        Args:
            alpha: Class weights for handling class imbalance (optional).
            gamma: Focusing parameter to reduce relative loss for well-classified examples.
            reduction: Reduction method to apply to output ('none', 'mean', 'sum').
            label_smoothing: Smoothing factor in [0, 1). Default = 0.0 (no smoothing).
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Model prediction tensor [B, num_classes].
            targets: Ground truth tensor [B].

        Returns:
            Tensor containing Focal loss score.
        """
        num_classes = inputs.size(1)
        log_probs = F.log_softmax(inputs, dim=1)
        probs = torch.exp(log_probs)

        with torch.no_grad():
            targets_one_hot = F.one_hot(targets, num_classes=num_classes).float()
            if self.label_smoothing > 0.0:
                smooth = self.label_smoothing / num_classes
                targets_one_hot = (
                    1.0 - self.label_smoothing
                ) * targets_one_hot + smooth

        pt = torch.sum(probs * targets_one_hot, dim=1)
        log_pt = torch.sum(log_probs * targets_one_hot, dim=1)

        if self.alpha is not None:
            alpha_factor = self.alpha.to(targets.device)[targets]
        else:
            alpha_factor = torch.ones_like(
                targets, dtype=inputs.dtype, device=targets.device
            )

        focal_term = (1 - pt) ** self.gamma
        loss = -alpha_factor * focal_term * log_pt

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class WeightedMultiTaskLoss(nn.Module):
    """Weighted multi-task loss for simultaneous classification and segmentation."""

    def __init__(
        self,
        seg_loss_fn: nn.Module,
        cls_loss_fn: nn.Module,
        seg_weight: float = 1.0,
        cls_weight: float = 1.0,
    ) -> None:
        """
        Args:
            seg_loss_fn: Segmentation loss function.
            cls_loss_fn: Classification loss function.
            seg_weight: Segmentation loss weighting.
            cls_weight: Classification loss weighting.
        """
        super().__init__()
        self.seg_loss_fn = seg_loss_fn
        self.cls_loss_fn = cls_loss_fn
        self.seg_weight = seg_weight
        self.cls_weight = cls_weight

    def forward(
        self,
        seg_pred: torch.Tensor,
        cls_pred: torch.Tensor,
        masks: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            seg_pred: Model segmentation prediction tensor [B, C, H, W].
            cls_pred: Model classification prediction tensor [B, num_classes].
            masks: Ground truth masks tensor [B, H, W].
            labels: Ground truth labels tensor [B].

        Returns:
            Tuple of Tensor containing total, segmentation and classification loss scores.
        """
        seg_loss = self.seg_loss_fn(seg_pred, masks)
        cls_loss = self.cls_loss_fn(cls_pred, labels)

        loss = self.seg_weight * seg_loss + self.cls_weight * cls_loss
        return {"loss": loss, "seg_loss": seg_loss, "cls_loss": cls_loss}


class UncertaintyWeightedMultiTaskLoss(nn.Module):
    """Multi-task loss with learnable weights for simultaneous classification and segmentation."""

    def __init__(
        self,
        seg_loss_fn: nn.Module,
        cls_loss_fn: nn.Module,
        init_log_var: float = 0.0,
    ) -> None:
        """
        Args:
            seg_loss_fn: Segmentation loss function.
            cls_loss_fn: Classification loss function.
            init_log_var: Log-variance to initialize uncertainty weights.
        """
        super().__init__()
        self.seg_loss_fn = seg_loss_fn
        self.cls_loss_fn = cls_loss_fn

        self.log_var_seg = nn.Parameter(torch.tensor(init_log_var, dtype=torch.float32))
        self.log_var_cls = nn.Parameter(torch.tensor(init_log_var, dtype=torch.float32))

    def forward(
        self,
        seg_pred: torch.Tensor,
        cls_pred: torch.Tensor,
        masks: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            seg_pred: Model segmentation prediction tensor [B, C, H, W].
            cls_pred: Model classification prediction tensor [B, num_classes].
            masks: Ground truth masks tensor [B, H, W].
            labels: Ground truth labels tensor [B].

        Returns:
            Tuple of Tensor containing total, segmentation and classification loss scores.
        """
        seg_loss = self.seg_loss_fn(seg_pred, masks)
        cls_loss = self.cls_loss_fn(cls_pred, labels)

        seg_precision = torch.exp(-self.log_var_seg)
        cls_precision = torch.exp(-self.log_var_cls)

        weighted_seg_loss = 0.5 * seg_precision * seg_loss + 0.5 * self.log_var_seg
        weighted_cls_loss = 0.5 * cls_precision * cls_loss + 0.5 * self.log_var_cls

        loss = weighted_seg_loss + weighted_cls_loss

        return {"loss": loss, "seg_loss": seg_loss, "cls_loss": cls_loss}
