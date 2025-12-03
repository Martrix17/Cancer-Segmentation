"""
Logger wrapper for experiment tracking and artifact management.

Example:
    >>> logger = MLflowLogger(
    ...   run_name="vit_b_16_run1",
    ...   experiment_name="COVID_Classification"
    ... )
    >>> logger.start_run()
    >>> logger.log_params({"lr": 0.001, "batch_size": 32})
    >>> logger.log_metrics({"train_loss": 0.5}, step=1)
    >>> logger.log_figure(fig, "confusion_matrix.png")
    >>> logger.end_run()
"""

from typing import Any, Dict

import mlflow
from matplotlib.figure import Figure
from PIL import Image


class MLflowLogger:
    """Wrapper for MLflow experiment tracking."""

    def __init__(
        self,
        run_name: str,
        uri: str = "mlruns",
        experiment_name: str = "ViT_COVID19_Classification",
        log_system: bool = True,
    ) -> None:
        """
        Args:
            run_name: Name for this MLflow run.
            uri: Tracking URI (local directory or remote server).
            experiment_name: Experiment name for grouping runs.
            log_system: Log system metrics if True.
        """
        self.uri = uri
        self.experiment_name = experiment_name
        self.log_system = log_system
        self.run_name = run_name

    def start_run(self) -> None:
        """Start MLflow run."""
        mlflow.set_tracking_uri(self.uri)
        mlflow.set_experiment(self.experiment_name)
        mlflow.start_run(run_name=self.run_name, log_system_metrics=self.log_system)

    def set_experiment_name(self, experiment_name: str) -> None:
        """Set experiment name."""
        self.experiment_name = experiment_name

    def set_run_name(self, run_name: str) -> None:
        """Set run name."""
        self.run_name = run_name

    def set_tags(self, tags: Dict[str, Any]) -> None:
        """Set tags to MLflow."""
        mlflow.set_tag(tags)

    def log_metrics(self, metrics: Dict[str, Any], step: int) -> None:
        """Log scalar metrics at a specific training step."""
        mlflow.log_metrics(metrics=metrics, step=step)

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log hyperparameters for the run."""
        mlflow.log_params(params=params)

    def log_artifact(self, file_path: str) -> None:
        """Log file or directory as an artifact."""
        mlflow.log_artifact(artifact_path=file_path)

    def log_text(self, text: Any, file_path: str) -> None:
        """Log text content as an artifact file."""
        mlflow.log_text(text=text, artifact_file=file_path)

    def log_figure(self, fig: Figure, file_path: str) -> None:
        """Log matplotlib figure as an image artifact."""
        mlflow.log_figure(figure=fig, artifact_file=file_path)

    def log_image(self, image: Image.Image, file_path: str) -> None:
        """Log PIL image as an image artifact."""
        mlflow.log_image(image=image, artifact_file=file_path)

    def end_run(self) -> None:
        """End MLflow run."""
        mlflow.end_run()
