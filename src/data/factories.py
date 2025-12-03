"""
Factory functions for creating datamodules from Hydra configurations.
"""

from typing import Tuple, cast

from omegaconf import DictConfig

from src.data.dataloader import ISICDataModule


def create_data_module(cfg: DictConfig) -> ISICDataModule:
    """Factory for creating data modules."""
    data_module = ISICDataModule(
        data_dir=cfg.data_dir,
        class_names=cfg.class_names,
        image_size=cast(Tuple[int, int], tuple(cfg.image_size)),
        mean=cast(Tuple[float, float, float], tuple(cfg.mean)),
        std=cast(Tuple[float, float, float], tuple(cfg.std)),
        batch_size=cfg.get('batch_size'),
        num_workers=cfg.get('num_workers'),
        val_split=cfg.get('val_split'),
        test_split=cfg.get('test_split'),
        seed=cfg.get('seed'),
    )
    return data_module
