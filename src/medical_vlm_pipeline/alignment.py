"""Cross-modal alignment objectives."""

import torch
from torch import Tensor
from torch.nn import functional as F


def infonce_loss(image_embeddings: Tensor, text_embeddings: Tensor, temperature: float = 0.07) -> Tensor:
    """Symmetric InfoNCE loss for image-text alignment."""
    image_embeddings = F.normalize(image_embeddings, dim=-1)
    text_embeddings = F.normalize(text_embeddings, dim=-1)

    logits = image_embeddings @ text_embeddings.T
    logits = logits / temperature
    targets = torch.arange(logits.shape[0], device=logits.device)

    image_to_text = F.cross_entropy(logits, targets)
    text_to_image = F.cross_entropy(logits.T, targets)
    return (image_to_text + text_to_image) / 2
