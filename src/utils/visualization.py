"""
Visualization utilities for segmentation tasks.
"""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw


def compute_iou(
    pred_mask: torch.Tensor, true_mask: Optional[torch.Tensor]
) -> Optional[float]:
    """
    Compute Intersection over Union between predicted and true masks.

    Args:
        pred_mask: Predicted segmentation mask.
        true_mask: Ground truth mask.

    Returns:
        IoU score as float.
    """
    if true_mask is None:
        return None

    pred = pred_mask.cpu().numpy().astype(bool)
    gt = true_mask.cpu().numpy().astype(bool)

    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()

    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    return float(intersection / union)


def colorcode_segmentation_overlay(
    image: Image.Image,
    pred_mask: np.ndarray,
    true_mask: Optional[np.ndarray],
    mode: str = "overlay",
    alpha: float = 0.3,
    outline_thickness: int = 1,
) -> Image.Image:
    """
    Overlay color-coded segmentation with multiple visualization modes.

    Args:
        image: Base PIL Image (RGB).
        pred_mask: Predicted mask (2D numpy array).
        true_mask: Ground truth mask (2D numpy array).
        mode: Visualization mode ('overlay', 'outline').
        alpha: Blending factor for overlay (used in 'overlay' mode).
        outline_thickness: Thickness of outline in pixels (used in 'outline' mode).

    Returns:
        PIL Image with visualization.

    Color coding:
        If GT exists:
            - Red: GT only
            - Green: Pred only
            - Yellow: overlap
        If GT missing:
            - Green: Predicted mask only
    """

    img = np.array(image).astype(np.uint8)
    pred = pred_mask.astype(bool)

    if mode == "overlay":
        if true_mask is not None:
            gt = true_mask.astype(np.uint8)

            gt_only = np.logical_and(gt, np.logical_not(pred)).astype(bool)
            pred_only = np.logical_and(pred, np.logical_not(gt)).astype(bool)
            correct = np.logical_and(pred, gt).astype(bool)

            overlay = np.zeros_like(img)
            overlay[gt_only] = [255, 0, 0]  # red
            overlay[pred_only] = [0, 255, 0]  # green
            overlay[correct] = [255, 255, 0]  # yellow

            blended = (img * (1 - alpha) + overlay * alpha).astype(np.uint8)
            return Image.fromarray(blended)
        else:
            overlay = np.zeros_like(img)
            overlay[pred.astype(bool)] = [0, 255, 0]

            blended = (img * (1 - alpha) + overlay * alpha).astype(np.uint8)
            return Image.fromarray(blended)
    elif mode == "outline":
        result = img.copy()

        if true_mask is not None:
            gt = true_mask.astype(np.uint8)

            gt_only = np.logical_and(gt, np.logical_not(pred)).astype(np.uint8)
            pred_only = np.logical_and(pred, np.logical_not(gt)).astype(np.uint8)
            correct = np.logical_and(pred, gt).astype(np.uint8)

            contours_gt, _ = cv2.findContours(
                gt_only, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            contours_pred, _ = cv2.findContours(
                pred_only, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            contours_correct, _ = cv2.findContours(
                correct, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            cv2.drawContours(result, contours_gt, -1, (255, 0, 0), outline_thickness)
            cv2.drawContours(result, contours_pred, -1, (0, 255, 0), outline_thickness)
            cv2.drawContours(
                result, contours_correct, -1, (255, 255, 0), outline_thickness
            )
        else:
            contours, _ = cv2.findContours(
                pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(result, contours, -1, (0, 255, 0), outline_thickness)

        return Image.fromarray(result)
    else:
        raise ValueError("mode must be 'overlay' or 'outline'")


def draw_sample_panel(
    sample: Dict[str, str | float | torch.Tensor],
    class_names: Optional[List[str]] = None,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> Image.Image:
    """
    Draws a panel for a single sample with image, overlay, and text info.

    Args:
        sample: Dictionary with keys:
            - "image_id": image identifier
            - "image": image tensor [C, H, W]
            - "seg_pred": predicted mask tensor [H, W]
            - "cls_pred": predicted class label (int)
            - "seg_mask": ground truth mask tensor [H, W]
            - "cls_label": ground truth class label (int)
        class_names: Optional list of class names for labeling.

    Returns:
        PIL Image with color-coded overlay, ground-truth mask, pred class, IoU score.
    """
    image = sample["image"].permute(1, 2, 0).numpy()
    image = (image * std) + mean
    image = (image * 255).clip(0, 255).astype(np.uint8)
    image = Image.fromarray(image)

    pred_mask = sample["seg_pred"].numpy().astype(np.uint8)
    gt_mask = (
        sample["seg_mask"].cpu().numpy().astype(np.uint8)
        if sample.get("seg_mask") is not None
        else None
    )

    overlay = colorcode_segmentation_overlay(image, pred_mask, gt_mask, mode="outline")
    iou = compute_iou(sample["seg_pred"], sample.get("seg_mask"))

    w, h = image.size
    panel_height = h + 70
    panel = Image.new("RGB", (w, panel_height), (30, 30, 30))
    panel.paste(overlay, (0, 70))
    draw = ImageDraw.Draw(panel)

    pred_class = sample.get("cls_pred")
    gt_class = sample.get("cls_label")

    pred_name = (
        class_names[pred_class]
        if class_names and pred_class is not None
        else str(pred_class)
    )
    gt_name = (
        class_names[gt_class]
        if class_names and gt_class is not None
        else (str(gt_class) if gt_class is not None else "N/A")
    )

    text = f"ID: {sample.get('image_id', '')} \nPred: {pred_name}"

    if gt_class is not None:
        text += f"\nGT: {gt_name}"
    if iou is not None:
        text += f"\nIoU: {iou:.3f}"

    draw.text((10, 10), text, fill=(255, 255, 255))
    return panel


def make_collage(
    samples: List[Dict[str, str | float | torch.Tensor | None]],
    epoch: int,
    max_samples: int = 16,
    class_names: Optional[List[str]] = None,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> Image.Image:
    """
    Builds a collage with header and multiple sample panels.

    Args:
        samples: List of sample dictionaries.
        epoch: Current epoch.
        max_samples: Maximum number of samples to include.
        class_names: Optional list of class names for labeling.

    Returns:
        PIL Image collage of samples.
    """
    samples_to_plot = samples[:max_samples]
    n_samples = len(samples_to_plot)

    panels = [
        draw_sample_panel(s, class_names=class_names, mean=mean, std=std)
        for s in samples_to_plot
    ]
    pw, ph = panels[0].size

    cols = int(np.ceil(np.sqrt(n_samples)))
    rows = int(np.ceil(n_samples / cols))

    header_height = 80
    collage = Image.new(
        "RGB",
        (cols * pw, rows * ph + header_height),
        (20, 20, 20),
    )
    draw = ImageDraw.Draw(collage)

    has_gt = any(s.get("seg_mask") is not None for s in samples_to_plot)

    title = f"Predictions - Epoch {epoch}"

    if has_gt:
        legend = "[Red: GT Only]" "\n[Green: Pred Only]" "\n[Yellow: Correct Overlap]"
    else:
        legend = "[Green: Prediction]"

    draw.text((10, 10), title, fill=(255, 255, 255))
    draw.text((10, 40), legend, fill=(200, 200, 200))

    idx = 0
    for r in range(rows):
        for c in range(cols):
            if idx >= n_samples:
                return collage
            collage.paste(panels[idx], (c * pw, header_height + r * ph))
            idx += 1

    return collage
