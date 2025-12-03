"""Inference function for multitask segmentation and image classification."""

from pathlib import Path

import hydra
from omegaconf import DictConfig

from src.factories import build_inference_pipeline
from src.utils.helper import save_prediction_figures


@hydra.main(version_base=None, config_path="config", config_name="infer")
def main(cfg: DictConfig):
    """Inference script."""
    trainer = build_inference_pipeline(cfg)

    print("\nStarting inference...")
    results = trainer.infer(
        data_dir=cfg.data.data_dir,
        plot_samples=True,
    )

    output_dir = Path(cfg.local_save_dir) / "inference" / cfg.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving prediction figures to {output_dir}...")

    save_prediction_figures(
        figures=results["pred_figures"], output_dir=output_dir
    )
    print("\nSaving outputs completed!")


if __name__ == "__main__":
    main()
