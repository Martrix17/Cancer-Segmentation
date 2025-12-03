"""
Segmentation Models PyTorch models.
"""

from typing import List, Optional, Tuple

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn

from .classification_head import ClassificationHead


class SMPSegmentationModel(nn.Module):
    """SMP module with classification head from encoder level outputs."""

    AVAILABLE_MODELS = {
        "unet": smp.Unet,
        "unet++": smp.UnetPlusPlus,
        "deeplab": smp.DeepLabV3,
        "deeplabplus": smp.DeepLabV3Plus,
        "fpn": smp.FPN,
        "linknet": smp.Linknet,
        "manet": smp.MAnet,
        "pan": smp.PAN,
        "pspnet": smp.PSPNet,
        "segformer": smp.Segformer,
    }

    BACKBONE_CHANNELS = {
        "resnet18": [64, 64, 128, 256, 512],
        "resnet34": [64, 64, 128, 256, 512],
        "resnet50": [64, 256, 512, 1024, 2048],
        "efficientnet-b0": [32, 24, 40, 112, 320],
        "efficientnet-b1": [32, 24, 40, 112, 320],
        "efficientnet-b2": [32, 24, 48, 120, 352],
        "efficientnet-b3": [40, 32, 48, 136, 384],
        "efficientnet-b4": [48, 32, 56, 160, 448],
    }

    DUMMY_INPUT_SIZE = (1, 3, 224, 224)

    def __init__(
        self,
        model_name: str,
        num_seg_classes: int,
        num_cls_classes: int,
        encoder_levels: List[int] = [5],
        projection_dim: int = 256,
        hidden_dims: List[int] = [512, 256],
        dropout: float = 0.5,
        gem_p: float = 3.0,
        freeze_encoder: bool = False,
        encoder_name: Optional[str] = None,
        encoder_weights: Optional[str | None] = None,
        **model_kwargs,
    ) -> None:
        """
        Args:
            model_name: Name of SMP model (see list_available_models()).
            num_seg_classes: Number of segmentation classes.
            num_cls_classes: Number of classification classes.
            encoder_levels: List of encoder layer indices to extract (e.g., [3, 4] for
                            layers 3 and 4). Defaults to [5] (bottleneck only).
            num_hidden_layers: Number of hidden layers in classification head.
            dropout: Dropout rate in classification head.
            gem_p: Initial GeM pooling parameter in classification head.
            freeze_encoder: Freeze encoder parameters if True.
            encoder_name: Name of encoder backbone (e.g., 'resnet50').
            encoder_weights: Pretrained weights for backbone.
            **model_kwargs: Keyword arguments for segmentation model constructor.
        """
        super().__init__()
        self.model_name = model_name
        self.num_seg_classes = num_seg_classes
        self.num_cls_classes = num_cls_classes
        self.encoder_levels = encoder_levels
        self.projection_dim = projection_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout
        self.gem_p = gem_p
        self.freeze_encoder = freeze_encoder
        self.encoder_name = encoder_name
        self.encoder_weights = encoder_weights

        self.seg_model = self._build_segmentation_model(**model_kwargs)
        self._freeze_backbone()

        self.cls_head = self._build_classsification_head()

    def _build_segmentation_model(self, **model_kwargs) -> nn.Module:
        """
        Build the base segmentation model.

        Returns:
            Segmentation model instance.
        """
        if self.model_name not in self.AVAILABLE_MODELS:
            raise ValueError(
                f"Input model_name ({self.model_name}) not found."
                f"See available models with 'list_available_models'."
            )

        model_class = self.AVAILABLE_MODELS[self.model_name]
        return model_class(
            encoder_name=self.encoder_name,
            encoder_weights=self.encoder_weights,
            classes=self.num_seg_classes,
            **model_kwargs,
        )

    def _freeze_backbone(self) -> None:
        """Freeze backbone encoder parameters."""
        if hasattr(self.seg_model, "encoder") and self.freeze_encoder:
            for param in self.seg_model.encoder.parameters():
                param.requires_grad = False
            print(f"Encoder '{self.encoder_name}' frozen")

    def _get_encoder_channels(self) -> List[int]:
        """
        Get channel dimensions for encoder levels.
        First tries static lookup, falls back to dynamic detection if needed.

        Returns:
            List of channel dimensions for each encoder level.
        """
        try:
            if self.encoder_name not in self.BACKBONE_CHANNELS:
                raise KeyError(f"Encoder '{self.encoder_name}' not in static lookup")

            in_channels_list = [
                self.BACKBONE_CHANNELS[self.encoder_name][level - 1]
                for level in self.encoder_levels
            ]
            print(f"Using static channel dimensions: {in_channels_list}")
            return in_channels_list
        except (KeyError, ValueError) as e:
            print(f"Static lookup failed ({e}), detecting channels dynamically...")
            return self._detect_channels_dynamically()

    def _detect_channels_dynamically(self) -> List[int]:
        """
        Detect channel dimensions by running a dummy forward pass.

        Returns:
            List of channel dimensions for each encoder level.
        """
        with torch.no_grad():
            dummy_input = torch.randn(*self.DUMMY_INPUT_SIZE)
            encoder_features = self.seg_model.encoder(dummy_input)

        in_channels_list = [
            encoder_features[level].shape[1] for level in self.encoder_levels
        ]

        print(f"Detected channel dimensions: {in_channels_list}")
        return in_channels_list

    def _build_classsification_head(self) -> nn.Module:
        """
        Build classification head from encoder channels.

        Returns:
            Classification head instance.
        """
        in_channels_list = self._get_encoder_channels()
        cls_head = ClassificationHead(
            in_channels_list=in_channels_list,
            num_classes=self.num_cls_classes,
            projection_dim=self.projection_dim,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout_rate,
            gem_p=self.gem_p,
        )

        print(
            f"Classification head initialized with {len(in_channels_list)}"
            " encoder levels"
        )
        return cls_head

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            Tuple with segmentation logits [B, num_seg_classes, H, W] and
            classification logits [B, num_cls_classes].
        """
        encoder_features = self.seg_model.encoder(x)

        decoder_features = self.seg_model.decoder(encoder_features)
        seg_logits = self.seg_model.segmentation_head(decoder_features)

        cls_features = [encoder_features[level] for level in self.encoder_levels]
        cls_logits = self.cls_head(cls_features)

        return seg_logits, cls_logits

    @classmethod
    def list_available_models(cls) -> list[str]:
        """Return list of available model names."""
        return list(cls.AVAILABLE_MODELS.keys())
