"""Prediction heads for diagnosis and auxiliary medical tasks."""

from torch import Tensor, nn


class DiagnosisHead(nn.Module):
    """Disease classification head over the shared medical latent space."""

    def __init__(self, embedding_dim: int, num_classes: int) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, num_classes),
        )

    def forward(self, embeddings: Tensor) -> Tensor:
        return self.classifier(embeddings)
