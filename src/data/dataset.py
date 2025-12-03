"""
Dataset class for ISIC dataset images and masks.

Example:
    >>> dataset = ISICDataset(
    ...     data_dir='data/',
    ...     df=data_frame
    ... )
    >>> sample = dataset[0]
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import v2


class ISICDataset(Dataset):
    """
    ISIC Dataset class for segmentation and classification.
    """

    def __init__(
        self,
        data_dir: str,
        class_names: List[str],
        df: Optional[pd.DataFrame],
        mode: str = "train",
        image_size: Tuple[int, int] = (224, 224),
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ) -> None:
        """
        Args:
            data_dir: Path to dataset root directory.
            class_names: List of class names for multi-label classification.
            df: DataFrame containing img_id and labels.
            mode: Mode to prepare data for ("train", "eval", "infer").
            image_size: Target size for resizing.
            mean: Channel means for normalization (ImageNet defaults).
            std: Channel stds for normalization (ImageNet defaults).
        """
        self.data_dir = Path(data_dir)
        self.images_dir = self.data_dir / "images"
        self.masks_dir = self.data_dir / "masks"
        self.class_names = class_names
        self.mode = mode

        valid_modes = ["train", "eval", "infer"]
        if mode not in valid_modes:
            raise ValueError(f"mode must be one of {valid_modes}, got {mode}")

        if df is not None:
            self.df = df.reset_index(drop=True)
            self.has_labels = True
        else:
            self.df = None
            self.has_labels = False

            if mode == "infer":
                self.image_paths = sorted(self.data_dir.glob("*.jpg"))
                if not self.image_paths:
                    raise ValueError(f"No images found in {self.data_dir}.")
            else:
                raise ValueError(f"Dataframe required for mode '{mode}'. ")

        self.load_mask = mode in ("train", "eval")
        self.load_label = mode in ("train", "eval")
        self.augment = mode == "train"

        self.image_size = image_size
        self.mean = mean
        self.std = std

        self._transforms()

    def _transforms(self) -> None:
        """Build transforms for images and masks."""
        base_transform = [
            v2.Resize(self.image_size),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ]

        if self.augment:
            self.geom_transform = v2.Compose(
                [
                    *base_transform,
                    v2.RandomHorizontalFlip(p=0.5),
                    v2.RandomVerticalFlip(p=0.5),
                    v2.RandomRotation(degrees=15),
                    v2.RandomAffine(
                        degrees=0,
                        translate=(0.05, 0.05),
                        scale=(0.95, 1.05),
                        shear=None,
                    ),
                ]
            )

            self.color_transform = v2.Compose(
                [
                    v2.ColorJitter(
                        brightness=0.15,
                        contrast=0.1,
                        saturation=0.1,
                        hue=0.02,
                    ),
                    v2.RandomGrayscale(p=0.15),
                    v2.GaussianBlur(kernel_size=3, sigma=(0.2, 1.5)),
                    v2.RandomAdjustSharpness(sharpness_factor=1.5, p=0.3),
                    v2.Normalize(mean=self.mean, std=self.std),
                ]
            )
        else:
            self.transform = v2.Compose([*base_transform])
            self.normalize = v2.Normalize(mean=self.mean, std=self.std)

    def _apply_transforms(
        self, img: Image.Image, mask: Image.Image
    ) -> Tuple[torch.Tensor]:
        """
        Apply transforms to image and optionally mask.

        Args:
            img: input img.
            mask: corresponding mask img.

        Returns:
            Tuple of (img tensor, mask tensor) after augmentation.
        """
        if self.augment:
            if mask is not None:
                img, mask = self.geom_transform(img, mask)
            else:
                img = self.geom_transform(img)
            img = self.color_transform(img)
        else:
            if mask is not None:
                img, mask = self.transform(img, mask)
            else:
                img = self.transform(img)
            img = self.normalize(img)

        return img, mask

    def _print_class_distribution(self) -> None:
        """Print class distribution for current split."""
        if not self.has_labels:
            print("No labels available (inference mode)")
            return

        class_counts = self.df[self.class_names].sum()
        print(f"Class distribution ({self.mode} set):")
        for class_name, count in class_counts.items():
            percentage = (count / len(self.df)) * 100
            print(f"  {class_name}: {int(count)} ({percentage:.1f}%)")

    def __len__(self) -> int:
        if self.df:
            return len(self.df)
        else:
            return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Args:
            idx: Sample index.

        Returns:
            Dictionary containing:
                - 'img_id': Image identifier
                - 'img': Image tensor [C, H, W]
                - 'mask': Mask tensor [H, W] (if load_mask=True)
                - 'label': Label tensor (scalar) (if load_label=True)
        """
        if self.has_labels:
            row = self.df.iloc[idx]
            img_id = row["image"]
            image_path = self.images_dir / f"{img_id}.jpg"

            img = Image.open(image_path).convert("RGB")

            mask = None
            if self.load_mask:
                mask_path = self.masks_dir / f"{img_id}.png"
                mask = Image.open(mask_path).convert("L")

            label = None
            if self.load_label:
                cls_values = row[self.class_names].astype(float).values
                label = torch.tensor(cls_values, dtype=torch.float32)
                label = torch.argmax(label)

            img, mask = self._apply_transforms(img, mask)
            mask = (mask > 0.5).float()

            sample = {
                "img_id": img_id,
                "img": img,
            }

            if mask is not None:
                sample["mask"] = mask.squeeze(0)

            if label is not None:
                sample["label"] = label

        else:
            image_path = self.image_paths[idx]
            img_id = image_path.stem
            img = Image.open(image_path).convert("RGB")

            img, _ = self._apply_transforms(img, mask=None)

            sample = {
                "img_id": img_id,
                "img": img,
            }

        return sample
