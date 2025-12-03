"""
Factory functions for creating trainer instances from Hydra configurations.
"""

from typing import Optional, Any

from omegaconf import DictConfig
import torch.nn as nn
import torch.optim as optim

from .base_trainer import BaseTrainer
from .trainer import Trainer
from .callbacks import EarlyStopping


def create_optimizer(cfg: DictConfig, model_parameters) -> optim.Optimizer:
    """Create optimizer from config (e.g., Adam, SGD) via reflection."""
    return getattr(optim, cfg.name)(
        params=model_parameters,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )


def create_scheduler(
    cfg: DictConfig, optimizer: optim.Optimizer
) -> optim.lr_scheduler.LRScheduler:
    """Create learning rate scheduler from config via reflection."""
    params = {k: v for k, v in cfg.items() if k != "name"}
    return getattr(optim.lr_scheduler, cfg.name)(optimizer=optimizer, **params)


def create_early_stopping(cfg: DictConfig) -> EarlyStopping:
    """Create early stopping callback from config."""
    return EarlyStopping(**cfg)


def create_base_trainer(
    cfg: DictConfig,
    model: nn.Module,
    criterion: Optional[nn.Module] = None,
    optimizer: Optional[nn.Module] = None,
    seg_metrics: Optional[Any] = None,
    cls_metrics: Optional[Any] = None,
) -> BaseTrainer:
    """Factory for creating base trainer."""
    return BaseTrainer(
        device=cfg.device,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        seg_metrics=seg_metrics,
        cls_metrics=cls_metrics,
        use_compile=cfg.get("use_compile", False),
        tqdm_refresh_steps=cfg.get("tqdm_refresh_steps", 20),
    )


def create_trainer(
    cfg: DictConfig,
    base_trainer: BaseTrainer,
    data_module: Optional[Any],
    scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
    logger: Optional[Any] = None,
    early_stopping: Optional[EarlyStopping] = None,
    checkpoint_manager: Optional[Any] = None,
) -> Trainer:
    """Factory for creating high-level trainer."""
    return Trainer(
        base_trainer=base_trainer,
        epochs=cfg.epochs,
        data_module=data_module,
        scheduler=scheduler,
        logger=logger,
        early_stopping=early_stopping,
        checkpoint_manager=checkpoint_manager,
        val_every_n_epochs=cfg.get("val_every_n_epochs", 1),
        compute_metrics_every_n_val=cfg.get("compute_metrics_every_n_val", 5),
        save_vis_every_n_val=cfg.get("save_vis_every_n_val", 10),
        n_vis_samples=cfg.get("n_vis_samples", 16),
    )
