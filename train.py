"""Main train function for multitask segmentation and image classification."""

from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from src.factories import build_training_pipeline
from src.utils.helper import save_metrics, save_metrics_figures, save_prediction_figures


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    """Main training script."""

    print("=" * 80)
    print("Configuration:")
    print("=" * 80)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)

    trainer = build_training_pipeline(cfg)

    print("\nStarting training:")
    trainer.fit(resume=cfg.get("resume", False))
    print("\nTraining completed!")

    print("\nStarting testing...")
    results = trainer.test(
        plot_metrics=True,
        plot_samples=True,
    )

    print("\nTesting completed!")
    if trainer.logger:
        trainer.logger.end_run()

    output_dir = Path(cfg.local_save_dir) / "test" / cfg.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving outputs to {output_dir}...")

    save_metrics(
        metrics=results["metrics"],
        output_dir=output_dir,
    )
    save_metrics_figures(
        figures=results["metrics_figures"], output_dir=output_dir
    )
    save_prediction_figures(
        figures=results["pred_figures"], output_dir=output_dir
    )
    print("\nSaving outputs completed!")


if __name__ == "__main__":
    main()
