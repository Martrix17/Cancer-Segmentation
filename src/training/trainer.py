"""
High-level trainer for model training loops, validation, testing and inference.

Example:
    >>> trainer = Trainer(
    ...     base_trainer=base_trainer,
    ...     epochs=50,
    ...     data_module=data_module,
    ...     scheduler=scheduler,
    ...     metrics_manager=metrics,
    ...     logger=logger,
    ...     early_stopping=early_stopping,
    ...     checkpoint_manager=checkpoint
    ... )
    >>> trainer.fit()
    >>> results = trainer.test(plot_metrics=True)
    >>> prediction = trainer.infer(data_dir='data/infer', plot_samples=True)
"""

import gc
from typing import Any, Dict, List, Optional

import torch
from matplotlib.figure import Figure
from torch import optim

from src.data.dataloader import ISICDataModule
from src.utils.checkpoint import CheckpointManager
from src.utils.visualization import make_collage

from .base_trainer import BaseTrainer
from .callbacks import EarlyStopping


class Trainer:
    """
    Handles full training pipeline with optional logging, checkpointing, and callbacks.

    Features:
    - Periodic validation with configurable frequency
    - Automatic checkpointing on validation improvement
    - Early stopping
    - Evaluation with metric visualization
    - Logging integration
    """

    def __init__(
        self,
        base_trainer: BaseTrainer,
        epochs: int,
        data_module: Optional[ISICDataModule],
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        logger: Optional[Any] = None,
        early_stopping: Optional[EarlyStopping] = None,
        checkpoint_manager: Optional[CheckpointManager] = None,
        val_every_n_epochs: int = 1,
        compute_metrics_every_n_val: int = 1,
        save_vis_every_n_val: int = 10,
        n_vis_samples: int = 50,
    ) -> None:
        """
        Args:
            base_trainer: BaseTrainer for train/val/test.
            epochs: Total number of epochs.
            data_module: Provides train/val/test dataloaders.
            scheduler: Learning rate scheduler (optional).
            metrics_manager: Computes and tracks metrics (optional).
            logger: Logger for experiment tracking (optional).
            early_stopping: Stops training on validation plateau (optional).
            checkpoint_manager: Saves/loads model checkpoints (optional).
            val_every_n_epochs: Validation frequency (0 = no validation).
            compute_metrics_every_n_val: Metric computation frequency during validation
                (relative to validation epochs, requires logger).
            save_vis_every_n_val: Save visualization samples every N validation runs.
            n_vis_samples: Number of samples to save for visualization.
        """
        self.base_trainer = base_trainer
        self.epochs = max(2, epochs)
        self.data_module = data_module
        self.scheduler = scheduler
        self.logger = logger
        self.early_stopping = early_stopping
        self.checkpoint_manager = checkpoint_manager

        self.val_every_n_epochs = max(1, val_every_n_epochs)
        self.compute_metrics_every_n_val = max(1, compute_metrics_every_n_val)
        self.save_vis_every_n_val = max(1, save_vis_every_n_val)
        self.n_vis_samples = n_vis_samples

    def fit(self, resume: bool = False) -> None:
        """
        Run full training loop with optional validation, checkpointing,
        and early stopping.

        Args:
            resume: Loads checkpoint and resumes training if True.
        """
        start_epoch = 1
        if self.checkpoint_manager and resume:
            start_epoch = self.checkpoint_manager.load(
                model=self.base_trainer.model,
                optimizer=self.base_trainer.optimizer,
                scaler=self.base_trainer.scaler,
                scheduler=self.scheduler,
                resume_training=True,
            )

        val_count = 0

        for epoch in range(start_epoch, self.epochs + 1):
            val_losses = {}
            val_metrics = {}
            val_preds = None

            train_losses = self.base_trainer.train(
                loader=self.data_module.train_dataloader(),
                epoch=epoch,
                total_epochs=self.epochs,
            )

            if epoch % self.val_every_n_epochs == 0:
                val_count += 1

                compute_metrics = val_count % self.compute_metrics_every_n_val == 0
                num_pred_samples = (
                    self.n_vis_samples
                    if (val_count % self.save_vis_every_n_val == 0)
                    else None
                )

                val_output = self.base_trainer.evaluate(
                    loader=self.data_module.val_dataloader(),
                    epoch=epoch,
                    total_epochs=self.epochs,
                    desc="Validation",
                    compute_loss=True,
                    compute_metrics=compute_metrics,
                    num_pred_samples=num_pred_samples,
                )

                val_losses = val_output.get("losses")
                val_metrics = val_output.get("metrics")
                val_preds = val_output.get("pred_samples")

                del val_output

                if self.checkpoint_manager:
                    self.checkpoint_manager.save_if_improved(
                        epoch=epoch,
                        model=self.base_trainer.model,
                        optimizer=self.base_trainer.optimizer,
                        scaler=self.base_trainer.scaler,
                        scheduler=self.scheduler,
                        val_loss=val_losses["eval_loss"],
                    )

                if self.early_stopping and self.early_stopping(val_losses["eval_loss"]):
                    break

            if self.logger:
                log_dict = {
                    **train_losses,
                    "lr": self.base_trainer.optimizer.param_groups[0]["lr"],
                }
                if val_losses:
                    log_dict.update(val_losses)
                if val_metrics:
                    log_dict.update(val_metrics)

                self.logger.log_metrics(metrics=log_dict, step=epoch)

                if val_preds:
                    collage = make_collage(
                        samples=val_preds,
                        epoch=epoch,
                        max_samples=self.n_vis_samples,
                        class_names=self.data_module.class_names,
                        mean=self.data_module.mean,
                        std=self.data_module.std,
                    )
                    self.logger.log_image(
                        image=collage, file_path="prediction_collage.png"
                    )

                    del collage
                    del val_preds
                    val_preds = None

            if self.scheduler:
                self.scheduler.step(val_losses["eval_loss"])

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def test(
        self,
        plot_metrics: bool = True,
        plot_samples: bool = True,
    ) -> Dict[str, Any]:
        """
        Evaluate model on test set with metric and sample visualization.

        Args:
            plot_metrics: Generate and return metric plots if True.
            plot_samples: Generate and return collage plots of samples if True.

        Returns:
            Dict containing test metrics, predictions, and figures.
        """
        if not self.checkpoint_manager:
            raise ValueError("Valid checkpoint manager is required for testing.")

        self.checkpoint_manager.load(
            model=self.base_trainer.model,
            optimizer=self.base_trainer.optimizer,
            scaler=self.base_trainer.scaler,
            scheduler=self.scheduler,
            resume_training=False,
        )

        self.base_trainer.seg_metrics.set_mode(test_mode=True)
        self.base_trainer.cls_metrics.set_mode(test_mode=True)
        test_output = self.base_trainer.evaluate(
            loader=self.data_module.test_dataloader(),
            epoch=1,
            total_epochs=1,
            desc="Testing",
            compute_loss=False,
            compute_metrics=True,
            num_pred_samples=self.n_vis_samples * 3,
            class_names=self.data_module.class_names,
        )

        test_metrics = test_output.get("metrics")
        test_preds = test_output.get("pred_samples")

        if self.logger:
            if plot_metrics and test_metrics:
                metrics_figures = self._visualize_cls_metrics(
                    class_names=self.data_module.class_names
                )
                test_metrics.pop("test_cls/roc_curve")
                test_metrics.pop("test_cls/confmat")

                test_output["metrics_figures"] = metrics_figures

                for name, fig in metrics_figures.items():
                    self.logger.log_figure(fig=fig, file_path=f"{name}.png")

            if test_metrics:
                classification_report = test_metrics["report"]
                self.logger.log_text(
                    text=classification_report, file_path="classification_report.txt"
                )
                test_metrics.pop("report")
                self.logger.log_metrics(metrics=test_metrics, step=1)
                test_metrics["report"] = classification_report

            if plot_samples and test_preds:
                collages = {}
                total = len(test_preds)
                max_samples_per_collage = 16 if total > 16 else total

                for start in range(0, total, max_samples_per_collage):
                    end = min(start + max_samples_per_collage, total)
                    file_name = f"pred_figures[{start+1}-{end}]"

                    collage = make_collage(
                        samples=test_preds[start:end],
                        epoch=0,
                        max_samples=max_samples_per_collage,
                        class_names=self.data_module.class_names,
                        mean=self.data_module.mean,
                        std=self.data_module.std,
                    )
                    self.logger.log_image(
                        image=collage,
                        file_path=file_name + ".png",
                    )
                    collages.update({file_name: collage})

                test_output["pred_figures"] = collages
                return test_output

        return test_output

    def infer(
        self,
        data_dir: str,
        plot_samples: bool = True,
    ) -> Dict[str, Any]:
        """
        Run model prediction on inference set.

        Args:
            data_dir: Path to inference dataset directory.
            plot_samples: Generate and return collage plots of samples if True.

        Returns:
            Dict containing predictions, and optional figures.
        """
        if not self.checkpoint_manager:
            raise ValueError("Valid checkpoint manager is required for inference.")

        self.checkpoint_manager.load(
            model=self.base_trainer.model,
            optimizer=None,
            scaler=None,
            scheduler=None,
            resume_training=False,
        )

        output = self.base_trainer.evaluate(
            loader=self.data_module.infer_dataloader(data_dir=data_dir),
            epoch=1,
            total_epochs=1,
            desc="Inference",
            compute_loss=False,
            compute_metrics=False,
            num_pred_samples=self.n_vis_samples,
        )
        pred_samples = output.get("pred_samples")

        if plot_samples and pred_samples:
            collages = {}
            total = len(pred_samples)
            max_samples_per_collage = 16 if total > 16 else total

            for start in range(0, total, max_samples_per_collage):
                end = min(start + max_samples_per_collage, total)
                file_name = f"pred_figures[{start+1}-{end}]"

                collage = make_collage(
                    samples=pred_samples[start:end],
                    epoch=0,
                    max_samples=max_samples_per_collage,
                    class_names=self.data_module.class_names,
                    mean=self.data_module.mean,
                    std=self.data_module.std,
                )
                collages.update({file_name: collage})
            output["pred_figures"] = collages

        return output

    def _visualize_cls_metrics(self, class_names: List[str]) -> Dict[str, Figure]:
        """
        Visualize classification metrics ROC and confusion matrix plots.

        Args:
            class_names: List of class names for multi-label classification.

        Returns:
            Dict containing classfication metrics figures.
        """
        figures = {}
        metrics_dict = self.base_trainer.cls_metrics.metrics

        if "roc_curve" in metrics_dict:
            roc_metric = metrics_dict.get("roc_curve")
            fig, ax = roc_metric.plot(score=True, labels=class_names)
            ax.set_title("ROC Curves")
            figures["roc_curve"] = fig

        if "confmat" in metrics_dict:
            cm_metric = metrics_dict.get("confmat")
            fig, ax = cm_metric.plot(labels=class_names)
            ax.set_title("Confusion Matrix")
            figures["confusion_matrix"] = fig

        return figures
