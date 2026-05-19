import torch
import torch.nn.functional as F
from torch import Tensor, nn


class DiagnosisHead(nn.Module):
    """Disease classification head over the shared medical latent space.
    
    Includes active Monte Carlo (MC) Dropout support for epistemic uncertainty estimation.
    """

    def __init__(self, embedding_dim: int, num_classes: int, dropout_prob: float = 0.2) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_prob)
        self.classifier = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            self.dropout,
            nn.Linear(embedding_dim, num_classes),
        )

    def forward(self, embeddings: Tensor) -> Tensor:
        return self.classifier(embeddings)

    def predict_with_uncertainty(
        self, embeddings: Tensor, num_samples: int = 20
    ) -> dict[str, Tensor]:
        """Perform Monte Carlo Dropout sampling to estimate model uncertainty.

        Returns:
            dict containing:
                - 'mean_probs': Mean probability distribution over classes.
                - 'predicted_class': Index of class with highest mean probability.
                - 'confidence': Mean probability of the predicted class.
                - 'uncertainty': Epistemic uncertainty calculated as entropy of the mean probability.
                - 'variance': Standard deviation of class probabilities across samples.
        """
        # Force dropout layer to remain ACTIVE during inference
        self.dropout.train()

        probs_list = []
        with torch.no_grad():
            for _ in range(num_samples):
                logits = self.forward(embeddings)
                probs = F.softmax(logits, dim=-1)
                probs_list.append(probs.unsqueeze(0))  # Shape: (1, Batch, Classes)

        # Shape: (Samples, Batch, Classes)
        probs_tensor = torch.cat(probs_list, dim=0)

        # Compute stats
        mean_probs = probs_tensor.mean(dim=0)  # Shape: (Batch, Classes)
        variance = probs_tensor.std(dim=0)     # Shape: (Batch, Classes)

        # Entropy of the mean probabilities: H(p) = -sum(p * log(p + epsilon))
        epsilon = 1e-8
        entropy = -torch.sum(mean_probs * torch.log(mean_probs + epsilon), dim=-1)

        # Max class selection
        confidence, predicted_class = torch.max(mean_probs, dim=-1)

        return {
            "mean_probs": mean_probs,
            "predicted_class": predicted_class,
            "confidence": confidence,
            "uncertainty": entropy,
            "variance": variance,
        }

