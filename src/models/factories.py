"""
Factory functions for creating models from Hydra configurations.
"""

import torch.nn as nn
from omegaconf import DictConfig

from .smp_models import SMPSegmentationModel


def create_model(cfg: DictConfig) -> nn.Module:
    """
    Factory for creating segmentation models.

    Args:
        cfg: Model config with keys:
            - model_name: Architecture name
            - num_seg_classes: Number of segmentation classes
            - num_cls_classes: Number of classification classes
            - encoder_levels: List of encoder levels to use
            - ... (type-specific parameters)

    Returns:
        Instantiated model.

    Example config:
        model:
          model_name: unet
          encoder_name: resnet50
          num_seg_classes: 1
          num_cls_classes: 7
          encoder_levels: [3, 4]
          encoder_weights: imagenet
    """
    model_kwargs = dict(cfg.get("model_kwargs", {}))

    kwargs = {
        "model_name": cfg.model_name,
        "encoder_name": cfg.encoder_name,
        "num_seg_classes": cfg.num_seg_classes,
        "num_cls_classes": cfg.num_cls_classes,
        "encoder_levels": cfg.get("encoder_levels", [4]),
        "projection_dim": cfg.get("projection_dim", 1),
        "hidden_dims": cfg.get("hidden_dims", [512, 256]),
        "dropout": cfg.get("dropout", 0.5),
        "gem_p": cfg.get("gem_p", 3.0),
        "freeze_encoder": cfg.get("freeze_encoder", False),
        "encoder_weights": cfg.get("encoder_weights", "imagenet"),
    }
    kwargs.update(model_kwargs)
    return SMPSegmentationModel(**kwargs)
