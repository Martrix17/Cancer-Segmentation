"""
Factory functions for creating loss criterions from Hydra configurations.
"""

from typing import Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig

from .losses import (
    BCEDiceLoss,
    DiceLoss,
    FocalLoss,
    TverskyLoss,
    UncertaintyWeightedMultiTaskLoss,
    WeightedMultiTaskLoss,
)

SEGMENTATION_LOSSES = {
    "bce": nn.BCEWithLogitsLoss,
    "dice": DiceLoss,
    "bce_dice": BCEDiceLoss,
    "tversky": TverskyLoss,
}

CLASSIFICATION_LOSSES = {
    "ce": nn.CrossEntropyLoss,
    "focal": FocalLoss,
}

MULTITASK_LOSSES = {
    "weighted": WeightedMultiTaskLoss,
    "uncertainty": UncertaintyWeightedMultiTaskLoss,
}


def create_segmentation_loss(loss_type: str, **kwargs) -> nn.Module:
    """
    Create a segmentation loss function.

    Args:
        loss_type: Type of segmentation loss.
        **kwargs: Loss-specific parameters.

    Returns:
        Instantiated segmentation loss.
    """
    if loss_type not in SEGMENTATION_LOSSES:
        raise ValueError(
            f"Unknown segmentation loss: {loss_type}. "
            f"Choose from: {list(SEGMENTATION_LOSSES.keys())}"
        )

    loss_class = SEGMENTATION_LOSSES[loss_type]

    if loss_type == "bce":
        valid_kwargs = {k: v for k, v in kwargs.items() if k in ["weight", "reduction"]}
    elif loss_type == "dice":
        valid_kwargs = {
            k: v for k, v in kwargs.items() if k in ["eps", "apply_sigmoid"]
        }
    elif loss_type == "bce_dice":
        valid_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k in ["weight", "bce_weight", "dice_weight"]
        }
    elif loss_type == "tversky":
        valid_kwargs = {
            k: v for k, v in kwargs.items() if k in ["alpha", "beta", "eps"]
        }
    else:
        valid_kwargs = kwargs

    return loss_class(**valid_kwargs)


def create_classification_loss(loss_type: str, **kwargs) -> nn.Module:
    """
    Create a classification loss function.

    Args:
        loss_type: Type of classification loss.
        **kwargs: Loss-specific parameters.

    Returns:
        Instantiated classification loss.
    """
    if loss_type not in CLASSIFICATION_LOSSES:
        raise ValueError(
            f"Unknown classification loss: {loss_type}. "
            f"Choose from: {list(CLASSIFICATION_LOSSES.keys())}"
        )

    loss_class = CLASSIFICATION_LOSSES[loss_type]

    if loss_type == "ce":
        valid_kwargs = {}
        for k in ["weight", "reduction", "label_smoothing"]:
            if k in kwargs and kwargs[k] is not None:
                valid_kwargs[k] = kwargs[k]
    elif loss_type == "focal":
        valid_kwargs = {}
        for k in ["alpha", "gamma", "reduction", "label_smoothing"]:
            if k in kwargs and kwargs[k] is not None:
                valid_kwargs[k] = kwargs[k]
    else:
        valid_kwargs = kwargs

    return loss_class(**valid_kwargs)


def create_loss_criterion(
    cfg: DictConfig, class_weights: Optional[torch.Tensor] = None
) -> nn.Module:
    """
    Factory for creating loss criterion from Hydra config.

    Args:
        cfg: Loss config with structure:
            - type: "weighted" | "uncertainty"
            - seg_loss: Segmentation loss config
            - cls_loss: Classification loss config
            - seg_weight: Weight for segmentation (weighted only)
            - cls_weight: Weight for classification (weighted only)
            - init_log_var: Initial log variance (uncertainty only)
        class_weights: Class weights tensor from data module (optional).
                      Will override cfg.cls_loss.weight or cfg.cls_loss.alpha
                      if use_class_weights=True.

    Returns:
        Instantiated multi-task loss.
    """
    loss_type = cfg.type.lower()

    if loss_type not in MULTITASK_LOSSES:
        raise ValueError(
            f"Unknown multi-task loss type: {loss_type}. "
            f"Choose from: {list(MULTITASK_LOSSES.keys())}"
        )

    seg_cfg = cfg.seg_loss
    seg_loss_fn = create_segmentation_loss(
        loss_type=seg_cfg.type, **{k: v for k, v in seg_cfg.items() if k != "type"}
    )

    cls_cfg = cfg.cls_loss
    cls_kwargs = {k: v for k, v in cls_cfg.items() if k != "type"}

    use_class_weights = cfg.get("use_class_weights", False)
    if class_weights is not None and use_class_weights:
        if cls_cfg.type == "ce":
            cls_kwargs["weight"] = class_weights
        elif cls_cfg.type == "focal":
            cls_kwargs["alpha"] = class_weights

    cls_loss_fn = create_classification_loss(loss_type=cls_cfg.type, **cls_kwargs)

    loss_class = MULTITASK_LOSSES[loss_type]

    if loss_type == "weighted":
        return loss_class(
            seg_loss_fn=seg_loss_fn,
            cls_loss_fn=cls_loss_fn,
            seg_weight=cfg.get("seg_weight", 1.0),
            cls_weight=cfg.get("cls_weight", 1.0),
        )
    elif loss_type == "uncertainty":
        return loss_class(
            seg_loss_fn=seg_loss_fn,
            cls_loss_fn=cls_loss_fn,
            init_log_var=cfg.get("init_log_var", 0.0),
        )
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")
