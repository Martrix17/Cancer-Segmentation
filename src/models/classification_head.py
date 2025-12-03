"""
ClassificationHead class for building a classification head.
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationHead(nn.Module):
    """Multi-scale classification head with feature projections from encoder levels."""

    def __init__(
        self,
        in_channels_list: List[int],
        num_classes: int = 7,
        projection_dim: int = 256,
        hidden_dims: List[int] = [512, 256],
        dropout: float = 0.5,
        gem_p: float = 3.0,
    ) -> None:
        """
        Args:
            in_channels_list: List of channel dimensions from each encoder level.
            num_classes: Number of output classes.
            projection_dim: Dimension to project each encoder level to.
            hidden_dims: List of hidden layer dimensions. Empty list = direct projection.
                        Example: [512, 256] -> two hidden layers with these dimensions.
            dropout: Dropout rate applied after each hidden layer.
            gem_p: Initial GeM pooling parameter.
        """
        super().__init__()
        self.projection_dim = projection_dim
        self.num_scales = len(in_channels_list)

        self.projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(channels, projection_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(projection_dim),
                    nn.ReLU(inplace=True),
                    CBAMBlock(projection_dim, reduction=16, kernel_size=7),
                )
                for channels in in_channels_list
            ]
        )

        self.pool = GeMPooling(p=gem_p)

        num_scales = len(in_channels_list)

        self.attention = MultiScaleAttention(
            num_scales=num_scales, embed_dim=projection_dim
        )

        layers = []
        in_features = projection_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(in_features, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                ]
            )
            in_features = hidden_dim

        layers.append(nn.Linear(in_features, num_classes))
        self.fc = nn.Sequential(*layers)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: List of feature maps [B, C, H, W].

        Returns:
            Tensor containing classification logits [B, num_classes].
        """
        pooled = [
            self.pool(proj(feat)) for proj, feat in zip(self.projections, features)
        ]
        attended = self.attention(pooled)
        return self.fc(attended)


class CBAMBlock(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        # Channel attention
        self.channel_mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
        )

        # Spatial attention
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Channel attention
        chn = self.channel_mlp(x)
        chn = torch.sigmoid(chn)
        x = x * chn

        # Spatial attention
        max_map, _ = torch.max(x, dim=1, keepdim=True)
        mean_map = torch.mean(x, dim=1, keepdim=True)
        spatial_input = torch.cat([max_map, mean_map], dim=1)

        sp = self.spatial(spatial_input)
        x = x * sp
        return x


class MultiScaleAttention(nn.Module):
    def __init__(self, num_scales: int, embed_dim: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(embed_dim * num_scales, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, num_scales),
            nn.Softmax(dim=1),
        )

    def forward(self, pooled_feats: List[torch.Tensor]) -> torch.Tensor:
        stacked = torch.stack(pooled_feats, dim=1)
        flattened = stacked.reshape(stacked.size(dim=0), -1)

        weights = self.fc(flattened)
        weights = weights.unsqueeze(-1)

        weighted = (stacked * weights).sum(dim=1)
        return weighted


class GeMPooling(nn.Module):
    """Generalized Mean Pooling - learnable pooling parameter."""

    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(p))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Feature map [B, C, H, W]

        Returns:
            Pooled features [B, C]
        """
        return (
            F.avg_pool2d(
                x.clamp(min=self.eps).pow(self.p), kernel_size=(x.size(-2), x.size(-1))
            )
            .pow(1.0 / self.p)
            .flatten(1)
        )
