"""
Factory functions for creating metrics components from Hydra configurations.
"""

from typing import Tuple

from omegaconf import DictConfig

from .classification_metrics import ClassificationMetrics
from .segmentation_metrics import SegmentationMetrics


def create_metrics_manager(
    cfg: DictConfig, device: str
) -> Tuple[SegmentationMetrics, ClassificationMetrics]:
    """Factory for creating metric managers."""
    seg_metrics = None
    cls_metrics = None

    if "segmentation" in cfg:
        seg_metrics = SegmentationMetrics(
            num_classes=cfg.segmentation.num_classes,
            device=device,
        )

    if "classification" in cfg:
        cls_metrics = ClassificationMetrics(
            num_classes=cfg.classification.num_classes,
            device=device,
            test_mode=False,
            average=cfg.classification.get("average", "macro"),
        )

    return seg_metrics, cls_metrics
