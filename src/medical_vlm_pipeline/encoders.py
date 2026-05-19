"""Encoder interfaces and implementations for medical images and clinical text."""

import logging
from typing import Protocol, Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class ImageEncoder(Protocol):
    def encode_image(self, images: Tensor) -> Tensor:
        """Return image embeddings before cross-modal projection."""
        ...


class TextEncoder(Protocol):
    def encode_text(self, token_ids: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        """Return report embeddings before cross-modal projection."""
        ...


class ProjectionHead(nn.Module):
    """Projection head for mapping modality-specific embeddings into shared space."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.LayerNorm(output_dim),
        )

    def forward(self, embeddings: Tensor) -> Tensor:
        return F.normalize(self.layers(embeddings), dim=-1)


class MedicalImageEncoder(nn.Module):
    """Swin Transformer or CNN encoder for 2D and 3D medical images."""

    def __init__(
        self,
        model_name: str = "swin_tiny_patch4_window7_224",
        pretrained: bool = True,
        embedding_dim: int = 768,
        volume_depth: int | None = None,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.pretrained = pretrained
        self.embedding_dim = embedding_dim
        self.volume_depth = volume_depth

        # Handle 3D inputs: we project 3D (C, D, H, W) to 2D (C, H, W) before Swin
        if volume_depth is not None:
            self.volume_projector = nn.Sequential(
                nn.Conv3d(1, 3, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
                nn.AdaptiveAvgPool3d((3, 224, 224)),  # Project to 3 channels of 224x224
            )
        else:
            self.volume_projector = None

        # Build core 2D encoder
        try:
            import timm
            logger.info(f"Loading 2D Image Encoder: {model_name} from timm (pretrained={pretrained})")
            self.backbone = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=0,  # return pooled features
            )
            # Find the feature dimension
            if hasattr(self.backbone, "num_features"):
                self.feature_dim = self.backbone.num_features
            elif hasattr(self.backbone, "head") and hasattr(self.backbone.head, "in_features"):
                self.feature_dim = self.backbone.head.in_features
            else:
                self.feature_dim = 768
        except Exception as e:
            logger.warning(f"Could not load {model_name} via timm: {e}. Falling back to custom CNN baseline.")
            self.backbone = None
            self.feature_dim = 512

        # Final linear mapping to desired embedding dimension
        if self.feature_dim != embedding_dim or self.backbone is None:
            self.fc = nn.Linear(self.feature_dim, embedding_dim)
        else:
            self.fc = nn.Identity()

        # Custom CNN baseline fallback
        if self.backbone is None:
            self.fallback_cnn = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(128, self.feature_dim)
            )

    def encode_image(self, images: Tensor) -> Tensor:
        """
        Args:
            images: Tensor of shape (B, C, H, W) for 2D, or (B, 1, D, H, W) for 3D.
        """
        # If 3D, project depth dimension
        if self.volume_projector is not None and len(images.shape) == 5:
            # images shape: (B, 1, D, H, W) -> Projector -> (B, 3, H, W)
            images = self.volume_projector(images).squeeze(2)
        elif len(images.shape) == 5:
            # Collapse depth by averaging
            images = images.mean(dim=2)
            if images.shape[1] == 1:
                images = images.repeat(1, 3, 1, 1)

        if self.backbone is not None:
            features = self.backbone(images)
        else:
            features = self.fallback_cnn(images)

        return self.fc(features)

    def forward(self, images: Tensor) -> Tensor:
        return self.encode_image(images)


class ClinicalTextEncoder(nn.Module):
    """PubMedBERT or ClinicalBERT transformer encoder for clinical texts."""

    def __init__(
        self,
        model_name: str = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
        embedding_dim: int = 768,
        fine_tune: bool = True,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.embedding_dim = embedding_dim

        try:
            from transformers import AutoModel
            logger.info(f"Loading Text Encoder: {model_name} from HuggingFace")
            self.transformer = AutoModel.from_pretrained(model_name)
            self.feature_dim = self.transformer.config.hidden_size
        except Exception as e:
            logger.warning(f"Could not load {model_name} via transformers: {e}. Falling back to standard BERT-base or custom model.")
            try:
                from transformers import AutoModel
                self.transformer = AutoModel.from_pretrained("bert-base-uncased")
                self.feature_dim = self.transformer.config.hidden_size
            except Exception:
                self.transformer = None
                self.feature_dim = 512

        # Freeze/fine-tune config
        if not fine_tune and self.transformer is not None:
            for param in self.transformer.parameters():
                param.requires_grad = False

        # Projection head if dimensions do not match
        if self.feature_dim != embedding_dim or self.transformer is None:
            self.fc = nn.Linear(self.feature_dim, embedding_dim)
        else:
            self.fc = nn.Identity()

        if self.transformer is None:
            # Pure PyTorch fallback embedding model if offline and HF fails completely
            self.vocab_size = 30522
            self.token_embeddings = nn.Embedding(self.vocab_size, 128)
            self.lstm = nn.LSTM(128, 256, batch_first=True, bidirectional=True)
            self.feature_dim = 512

    def encode_text(self, token_ids: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        """
        Args:
            token_ids: Tensor of shape (B, SeqLen)
            attention_mask: Optional attention mask Tensor of shape (B, SeqLen)
        """
        if self.transformer is not None:
            outputs = self.transformer(input_ids=token_ids, attention_mask=attention_mask)
            # Use mean pooling over token embeddings weighted by attention mask
            token_embeddings = outputs.last_hidden_state
            if attention_mask is not None:
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                features = sum_embeddings / sum_mask
            else:
                features = token_embeddings.mean(dim=1)
        else:
            # Fallback LSTM model
            emb = self.token_embeddings(token_ids)
            lstm_out, _ = self.lstm(emb)
            features = lstm_out.mean(dim=1)  # Mean pooling over sequence

        return self.fc(features)

    def forward(self, token_ids: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        return self.encode_text(token_ids, attention_mask)
