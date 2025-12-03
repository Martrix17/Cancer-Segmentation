"""Utility functions for saving predictions, metrics, and visualizations."""

import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import torch
from omegaconf import OmegaConf
from PIL import Image


def save_metrics(
    metrics: Dict[str, str | float | torch.Tensor], output_dir: str
) -> None:
    """
    Save metrics locally.

    - Extracts the 'report' string to a txt file.
    - Converts torch.Tensors to scalars or lists.
    - Stores all remaining metrics in a JSON file.

    Args:
        metrics: Dict containing metrics.
        output_dir: Directory to save images.
        metrics_filename: Filename for the JSON metrics.
        report_filename: Filename for the classification report.
    """
    output_dir = Path(output_dir)
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    report_text = metrics.get("report", None)

    metrics_for_json: Dict[str, Any] = {}
    for k, v in metrics.items():
        if k == "report":
            continue

        if isinstance(v, torch.Tensor):
            if v.ndim == 0:
                metrics_for_json[k] = v.item()
            else:
                metrics_for_json[k] = v.detach().cpu().tolist()
        else:
            metrics_for_json[k] = v

    metrics_path = metrics_dir / "metrics.json"
    with metrics_path.open("w") as f:
        json.dump(metrics_for_json, f, indent=4)

    if report_text is not None:
        report_path = metrics_dir / "classification_report.txt"
        with report_path.open("w") as f:
            f.write(report_text)


def save_metrics_figures(
    figures: Dict[str, plt.Figure], output_dir: str, dpi: int = 300
) -> None:
    """
    Save matplotlib figures as PNG images.

    Args:
        figures: Dict mapping filenames to Figure objects.
        output_dir: Directory to save images.
        dpi: Image resolution (dots per inch).
    """
    output_dir = Path(output_dir)
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    for name, fig in figures.items():
        fig_path = metrics_dir / f"{name}.png"
        fig.savefig(fig_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)


def save_prediction_figures(figures: Dict[str, Image.Image], output_dir: str) -> None:
    """
    Save PIL prediction figures as PNG images.

    Args:
        figures: Dict mapping filenames to Image objects.
        output_dir: Directory to save images.
        dpi: Image resolution (dots per inch).
    """
    output_dir = Path(output_dir)
    preds_dir = output_dir / "predictions"
    preds_dir.mkdir(parents=True, exist_ok=True)

    for name, fig in figures.items():
        fig_path = preds_dir / f"{name}.png"
        fig.save(fig_path)


def flatten_hparams_dict(d, parent_key="", sep=".") -> Dict[str, Any]:
    """
    Recursively flatten nested dict with dot-separated keys.

    Args:
        d: Nested dictionary to flatten.
        parent_key: Parent key prefix for recursion.
        sep: Separator for nested keys.

    Returns:
        Flattened dict (e.g., {"model.lr": 0.001, "data.batch_size": 32}).
    """
    items: List[Any] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_hparams_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def extract_hparams_from_cfg(cfg) -> Dict[str, Any]:
    """
    Convert Hydra config to flattened hyperparameter dict.

    Args:
        cfg: Hydra DictConfig or OmegaConf config.

    Returns:
        Flattened dict (e.g., {"model.lr": 0.001}).
    """
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    return flatten_hparams_dict(cfg_dict)
