"""
Factory functions for creating trainer pipeline from Hydra configurations.
"""

from omegaconf import DictConfig

from src.data.factories import create_data_module
from src.losses.factories import create_loss_criterion
from src.metrics.factories import create_metrics_manager
from src.models.factories import create_model
from src.training.factories import (
    create_base_trainer,
    create_early_stopping,
    create_optimizer,
    create_scheduler,
    create_trainer,
)
from src.training.trainer import Trainer
from src.utils.checkpoint import CheckpointManager
from src.utils.helper import extract_hparams_from_cfg
from src.utils.logger import MLflowLogger


def build_training_pipeline(cfg: DictConfig) -> Trainer:
    """
    Main factory function that builds entire training pipeline.

    Returns:
        Trainer instance for training/validation/testing.
    """
    device = cfg.trainer.device
    print("=" * 80)
    print(f"Building trainer pipeline on {device}.")
    print("=" * 80)

    print("1. Building model...")
    model = create_model(cfg.model)

    print("2. Building data module...")
    data_module = create_data_module(cfg.data)
    data_module.setup(**cfg.trainer.class_weighting)

    print("3. Building metrics...")
    seg_metrics, cls_metrics = create_metrics_manager(cfg.trainer.metrics, device)

    print("4. Building loss criterion...")
    criterion = create_loss_criterion(cfg.loss, class_weights=data_module.class_weights)

    print("5. Building optimizer...")
    optimizer = create_optimizer(cfg.trainer.optimizer, model.parameters())

    scheduler = None
    if "scheduler" in cfg.trainer:
        print("6. Building scheduler...")
        scheduler = create_scheduler(cfg.trainer.scheduler, optimizer)
    else:
        print("6. No scheduler defined. Skipping.")

    print("7. Building base trainer...")
    base_trainer = create_base_trainer(
        cfg.trainer,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        seg_metrics=seg_metrics,
        cls_metrics=cls_metrics,
    )

    logger = None
    if cfg.get("logging", {}).get("use_mlflow", False):
        print("8. Building logger...")
        logger = MLflowLogger(
            uri=cfg.logging.uri,
            experiment_name=cfg.logging.experiment_name,
            run_name=cfg.logging.run_name + "_train",
        )
        hparams = extract_hparams_from_cfg(cfg)
        logger.start_run()
        logger.log_params(hparams)

        if cfg.loss.get("use_class_weights", False):
            logger.log_params(
                {
                    f"class_weight_{name}": weight
                    for name, weight in zip(
                        data_module.class_names, data_module.class_weights.tolist()
                    )
                }
            )
    else:
        print("8. No logger defined. Skipping.")

    print("9. Building checkpoint manager...")
    checkpoint_manager = CheckpointManager(
        device=device,
        save_dir=cfg.checkpointing.get("save_dir", "checkpoints"),
        filename=cfg.checkpointing.get("filename", "best_model.pt"),
        patience=cfg.checkpointing.get("patience", 1),
        verbose=cfg.checkpointing.get("patience", False),
    )

    early_stopping = None
    if "early_stopping" in cfg.trainer:
        print("10. Building early stopping...")
        early_stopping = create_early_stopping(cfg.trainer.early_stopping)
    else:
        print("10. No early stopping defined. Skipping.")

    print("11. Building trainer...")
    trainer = create_trainer(
        cfg.trainer,
        base_trainer=base_trainer,
        data_module=data_module,
        scheduler=scheduler,
        logger=logger,
        early_stopping=early_stopping,
        checkpoint_manager=checkpoint_manager,
    )

    print("=" * 80)
    print("Training pipeline built.")
    print("=" * 80)
    return trainer


def build_testing_pipeline(cfg: DictConfig) -> Trainer:
    """
    Factory function that builds the testing pipeline.

    Returns:
        Trainer instance for testing.
    """
    print("=" * 80)
    print("Building testing pipeline.")
    print("=" * 80)
    device = cfg.trainer.device

    print("1. Building model...")
    model = create_model(cfg.model)

    print("2. Building data module...")
    data_module = create_data_module(cfg.data)
    data_module.setup(**cfg.trainer.class_weighting)

    print("3. Building metrics...")
    seg_metrics, cls_metrics = create_metrics_manager(cfg.trainer.metrics, device)

    criterion = None
    if "loss" in cfg:
        print("4. Building loss criterion...")
        criterion = create_loss_criterion(
            cfg.loss, class_weights=data_module.class_weights
        )
    else:
        print("4. No loss criterion defined. Skipping.")

    print("5. Building base trainer...")
    base_trainer = create_base_trainer(
        cfg.trainer,
        model=model,
        criterion=criterion,
        seg_metrics=seg_metrics,
        cls_metrics=cls_metrics,
    )

    logger = None
    if cfg.get("logging", {}).get("use_mlflow", False):
        print("6. Building logger...")
        logger = MLflowLogger(
            experiment_name=cfg.logging.experiment_name,
            run_name=cfg.logging.run_name + "_test",
        )
        hparams = extract_hparams_from_cfg(cfg)
        logger.start_run()
        logger.log_params(hparams)

        if cfg.loss.get("use_class_weights", False):
            logger.log_params(
                {
                    f"class_weight_{name}": weight
                    for name, weight in zip(
                        data_module.class_names, data_module.class_weights.tolist()
                    )
                }
            )
    else:
        print("6. No logger defined. Skipping.")

    print("7. Building checkpoint manager...")
    checkpoint_manager = CheckpointManager(
        device=device,
        save_dir=cfg.checkpointing.get("save_dir", "checkpoints"),
        filename=cfg.checkpointing.get("filename", "best_model.pt"),
        patience=cfg.checkpointing.get("patience", 1),
    )

    print("8. Building trainer...")
    trainer = create_trainer(
        cfg.trainer,
        base_trainer=base_trainer,
        data_module=data_module,
        logger=logger,
        checkpoint_manager=checkpoint_manager,
    )

    print("=" * 80)
    print("Testing pipeline built.")
    print("=" * 80)
    return trainer


def build_inference_pipeline(cfg: DictConfig) -> Trainer:
    """
    Factory function that builds the inference pipeline.

    Returns:
        Trainer instance for inference.
    """
    print("=" * 80)
    print("Building inference pipeline")
    print("=" * 80)
    device = cfg.trainer.device

    print("1. Building model...")
    model = create_model(cfg.model)

    print("2. Building data module...")
    data_module = create_data_module(cfg.data)

    print("3. Building base trainer...")
    base_trainer = create_base_trainer(
        cfg.trainer,
        model=model,
    )

    print("4. Building checkpoint manager...")
    checkpoint_manager = CheckpointManager(
        device=device,
        save_dir=cfg.checkpointing.get("save_dir", "checkpoints"),
        filename=cfg.checkpointing.get("filename", "best_model.pt"),
    )

    print("5. Building trainer...")
    trainer = create_trainer(
        cfg.trainer,
        base_trainer=base_trainer,
        data_module=data_module,
        checkpoint_manager=checkpoint_manager,
    )

    print("=" * 80)
    print("Inference pipeline built.")
    print("=" * 80)
    return trainer
