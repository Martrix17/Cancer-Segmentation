"""
DataModule class for ISIC dataset.

Example:
    >>> dm = ISICDataModule(data_dir='data/', batch_size=32)
    >>> dm.setup()
    >>> train_loader = dm.train_dataloader()

    >>> dm = ISICDataModule(data_dir='data/', batch_size=32)
    >>> infer_loader = dm.infer_dataloader(data_dir='data/inf')
"""

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from torch.utils.data import DataLoader, WeightedRandomSampler

from .dataset import ISICDataset


class ISICDataModule:
    """
    DataModule wrapper for ISIC dataset.

    Features:
    - Stratified train/val/test splits
    - Class weight computation for imbalanced datasets
    - WeightedRandomSampler for balanced batch sampling

    Workflow:
    Train/Test: Initialize -> setup() -> access train/val/test_dataloader()
    Inference: Initialize -> create/access infer_dataloader(data_dir)
    """

    def __init__(
        self,
        data_dir: str,
        class_names: List[str],
        image_size: Tuple[int, int] = (224, 224),
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        batch_size: int = 32,
        num_workers: int = 4,
        val_split: float = 0.15,
        test_split: float = 0.15,
        seed: int = 42,
    ) -> None:
        """
        Args:
            data_dir: Path to dataset root.
            class_names: List of class names for multi-label classification.
            image_size: Target size for resizing.
            mean: Channel means for normalization (ImageNet defaults).
            std: Channel stds for normalization (ImageNet defaults).
            batch_size: Batch size for DataLoaders.
            num_workers: Number of workers for parallel loading.
            val_split: Fraction for validation set (0.0-0.5).
            test_split: Fraction for testing set (0.0-0.5).
            seed: Random seed for reproducible splits.
        """
        self.data_dir = Path(data_dir)
        self.class_names = class_names
        self.image_size = image_size
        self.mean = mean
        self.std = std
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.test_split = test_split
        self.seed = seed

        metadata_path = Path(data_dir) / "metadata.csv"
        self.metadata_path = metadata_path if metadata_path.exists() else None

        self.train_dataset: Optional[ISICDataset] = None
        self.val_dataset: Optional[ISICDataset] = None
        self.test_dataset: Optional[ISICDataset] = None
        self.train_df: Optional[pd.DataFrame] = None
        self.val_df: Optional[pd.DataFrame] = None
        self.test_df: Optional[pd.DataFrame] = None
        self.train_labels: Optional[pd.Series] = None
        self.class_weights: Optional[torch.Tensor] = None

    def setup(
        self, weighting_method: str = "balanced", beta: Optional[float] = 0.999
    ) -> None:
        """
        Create stratified train/val/test splits and compute class weights.

        Args:
            weighting_method: Method to compute class weights for loss criterion
            beta: Factor for weighting method 'effective' (optional).
        """
        if self.metadata_path is None:
            raise ValueError(f"No valid metadata file (.csv) in {self.data_dir}.")

        df = pd.read_csv(self.metadata_path)
        labels = df[self.class_names].idxmax(axis=1)

        train_df, temp_df, train_labels, temp_labels = train_test_split(
            df,
            labels,
            test_size=self.val_split + self.test_split,
            stratify=labels,
            random_state=self.seed,
        )

        val_rel_size = self.val_split / (self.val_split + self.test_split)
        val_df, test_df = train_test_split(
            temp_df,
            test_size=1 - val_rel_size,
            stratify=temp_labels,
            random_state=self.seed,
        )

        self.train_df = train_df.reset_index(drop=True)
        self.val_df = val_df.reset_index(drop=True)
        self.test_df = test_df.reset_index(drop=True)

        self.train_dataset = ISICDataset(
            data_dir=self.data_dir,
            df=self.train_df,
            class_names=self.class_names,
            mode="train",
            image_size=self.image_size,
            mean=self.mean,
            std=self.std,
        )

        self.val_dataset = ISICDataset(
            data_dir=self.data_dir,
            df=self.val_df,
            class_names=self.class_names,
            mode="eval",
            image_size=self.image_size,
            mean=self.mean,
            std=self.std,
        )

        self.test_dataset = ISICDataset(
            data_dir=self.data_dir,
            df=self.test_df,
            class_names=self.class_names,
            mode="eval",
            image_size=self.image_size,
            mean=self.mean,
            std=self.std,
        )

        self.train_labels = self.train_df[self.class_names].values.argmax(axis=1)
        self.class_weights = self._compute_class_weights(
            weighting_method=weighting_method, beta=beta
        )
        self._print_split_info()

    def _compute_class_weights(
        self, weighting_method: str = "balanced", beta: Optional[float] = 0.999
    ) -> torch.Tensor:
        """
        Compute class weights for handling class imbalance.
        Uses sklearn's 'balanced' strategy.

        Args:
            weighting_method: Method to compute class weights for loss criterion
            beta: Factor for weighting method 'effective' (optional).

        Returns:
            Tensor of class weights with shape (num_classes, 1).
        """
        classes = np.unique(self.train_labels)

        if weighting_method == "balanced":
            class_weights = compute_class_weight(
                class_weight="balanced",
                classes=classes,
                y=self.train_labels,
            )
        elif weighting_method == "effective":
            # Effective number of samples (Cui et al., CVPR 2019)
            samples_per_class = np.bincount(self.train_labels)
            effective_num = 1.0 - np.power(beta, samples_per_class)
            weights = (1.0 - beta) / effective_num
            class_weights = weights / weights.sum() * len(classes)
        elif weighting_method == "sqrt":
            samples_per_class = np.bincount(self.train_labels)
            class_weights = np.sqrt(
                samples_per_class.sum() / (len(classes) * samples_per_class)
            )
        else:
            raise ValueError(f"Unknown method: {weighting_method}")

        return torch.tensor(class_weights, dtype=torch.float32)

    def _make_weighted_sampler(self) -> WeightedRandomSampler:
        """
        Create a weighted sampler for balanced batch sampling.
        Each sample is weighted by its class weight.

        Returns:
            WeightedRandomSampler instance.
        """
        sample_weights = compute_sample_weight(
            class_weight="balanced", y=self.train_labels
        )
        return WeightedRandomSampler(
            weights=torch.Tensor(sample_weights),
            num_samples=len(sample_weights),
            replacement=True,
        )

    def _print_split_info(self) -> None:
        """Print dataset split information and class distributions."""
        print("=" * 80)
        print("Data Information")
        print("=" * 80)
        print("\nSplit sizes:")
        print(
            f"  Train: {len(self.train_df)} samples "
            f"({(1 - (self.val_split + self.test_split)) * 100:.1f}%)"
        )
        print(f"  Val:   {len(self.val_df)} samples ({100 * self.val_split:.1f}%)")
        print(f"  Test:  {len(self.test_df)} samples ({100 * self.test_split:.1f}%)")
        self.train_dataset._print_class_distribution()
        print("\nClass weights:")
        for name, weight in zip(self.class_names, self.class_weights.tolist()):
            print(f" class_weight_{name}: {weight:.4f}")
        print("=" * 80)

    def train_dataloader(self, use_weighted_sampler: bool = True) -> DataLoader:
        """
        Return DataLoader with weighted sampling for training.

        Args:
            use_weighted_sampler: Uses sampler for balanced batches if True.
                                  Uses standard shuffling if False.

        Returns:
            DataLoader intance for training set.
        """
        if use_weighted_sampler:
            sampler = self._make_weighted_sampler()
            return DataLoader(
                dataset=self.train_dataset,
                batch_size=self.batch_size,
                sampler=sampler,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=True,
            )
        else:
            return DataLoader(
                dataset=self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
                drop_last=True,
            )

    def val_dataloader(self) -> DataLoader:
        """Return DataLoader for validation."""
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )

    def test_dataloader(self) -> DataLoader:
        """Return DataLoader for testing."""
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )

    def infer_dataloader(self, data_dir: str) -> DataLoader:
        """
        Return DataLoader for inference.

        Args:
            data_dir: Path to inference data directory.
            df: DataFrame containing image id (optional).

        Returns:
            DataLoader intance for inference set.
        """
        infer_set = ISICDataset(
            data_dir=data_dir,
            df=None,
            class_names=self.class_names,
            mode="infer",
            image_size=self.image_size,
            mean=self.mean,
            std=self.std,
        )
        return DataLoader(
            dataset=infer_set,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )
