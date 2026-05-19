"""Latent quantization interfaces and implementations for compressed retrieval and VQ training."""

import logging
from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class QuantizationResult:
    codes: Tensor
    reconstructed: Tensor | None = None
    commitment_loss: Tensor | None = None


class LatentQuantizer(nn.Module):
    """Base interface for PQ/VQ/TurboQuant-style latent compression."""

    def fit(self, embeddings: Tensor) -> None:
        """Fit quantizer parameters (used for non-differentiable like PQ)."""
        raise NotImplementedError

    def encode(self, embeddings: Tensor) -> QuantizationResult:
        """Encode embeddings to discrete codes."""
        raise NotImplementedError

    def decode(self, codes: Tensor) -> Tensor:
        """Decode discrete codes back to continuous embeddings."""
        raise NotImplementedError


class IdentityQuantizer(LatentQuantizer):
    """No-op quantizer used while establishing the retrieval baseline."""

    def fit(self, embeddings: Tensor) -> None:
        pass

    def encode(self, embeddings: Tensor) -> QuantizationResult:
        return QuantizationResult(codes=embeddings, reconstructed=embeddings, commitment_loss=torch.tensor(0.0, device=embeddings.device))

    def decode(self, codes: Tensor) -> Tensor:
        return codes


class ProductQuantizer(LatentQuantizer):
    """Product Quantization (PQ) compressing high-dimensional vectors.

    Splits embedding of dimension D into M sub-spaces of dimension D/M,
    and runs K-Means clustering on each sub-space independently.
    """

    def __init__(self, embedding_dim: int = 256, num_subvectors: int = 8, num_centroids: int = 256) -> None:
        super().__init__()
        assert embedding_dim % num_subvectors == 0, f"Embedding dim {embedding_dim} must be divisible by {num_subvectors}."
        self.embedding_dim = embedding_dim
        self.num_subvectors = num_subvectors
        self.subvector_dim = embedding_dim // num_subvectors
        self.num_centroids = num_centroids

        # Register codebooks buffer for persistence
        self.register_buffer(
            "codebooks",
            torch.zeros(num_subvectors, num_centroids, self.subvector_dim)
        )
        self.is_fitted = False

    def fit(self, embeddings: Tensor) -> None:
        """Perform K-Means on each sub-space independently to build the codebook."""
        logger.info(f"Fitting Product Quantizer (M={self.num_subvectors}, K={self.num_centroids})...")
        from sklearn.cluster import KMeans

        # Convert to numpy CPU for scikit-learn
        embeddings_np = embeddings.detach().cpu().numpy()
        N = embeddings_np.shape[0]

        if N < self.num_centroids:
            logger.warning("Fewer sample embeddings than centroids. Using random centroids.")
            random_centroids = np.random.randn(self.num_subvectors, self.num_centroids, self.subvector_dim)
            self.codebooks.copy_(torch.tensor(random_centroids, dtype=torch.float32))
            self.is_fitted = True
            return

        new_codebooks = torch.zeros(self.num_subvectors, self.num_centroids, self.subvector_dim)

        for m in range(self.num_subvectors):
            sub_start = m * self.subvector_dim
            sub_end = (m + 1) * self.subvector_dim
            sub_embeddings = embeddings_np[:, sub_start:sub_end]

            # Fit K-Means
            kmeans = KMeans(n_clusters=self.num_centroids, n_init=3, random_state=42)
            kmeans.fit(sub_embeddings)
            new_codebooks[m] = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)

        self.codebooks.copy_(new_codebooks)
        self.is_fitted = True
        logger.info("Product Quantizer fitting completed.")

    def encode(self, embeddings: Tensor) -> QuantizationResult:
        """Encode embeddings into code indices (B, M)."""
        if not self.is_fitted:
            # Fit on the fly with a warning
            logger.warning("Product Quantizer is not fitted yet! Running on-the-fly fit.")
            self.fit(embeddings)

        B = embeddings.shape[0]
        codes = torch.zeros(B, self.num_subvectors, dtype=torch.long, device=embeddings.device)

        # Reshape: (B, M, sub_dim)
        reshaped = embeddings.view(B, self.num_subvectors, self.subvector_dim)

        for m in range(self.num_subvectors):
            # Compute Euclidean distance to all centroids in this sub-space
            sub_vectors = reshaped[:, m, :].unsqueeze(1)  # (B, 1, sub_dim)
            centroids = self.codebooks[m].unsqueeze(0)    # (1, K, sub_dim)
            distances = torch.sum((sub_vectors - centroids) ** 2, dim=-1)  # (B, K)
            codes[:, m] = torch.argmin(distances, dim=-1)

        reconstructed = self.decode(codes)
        # Commitment loss as reconstruction MSE
        commitment_loss = F.mse_loss(embeddings, reconstructed)

        return QuantizationResult(codes=codes, reconstructed=reconstructed, commitment_loss=commitment_loss)

    def decode(self, codes: Tensor) -> Tensor:
        """Decode indices (B, M) back into full embeddings (B, D)."""
        B = codes.shape[0]
        reconstructed = torch.zeros(B, self.num_subvectors, self.subvector_dim, device=codes.device, dtype=torch.float32)

        for m in range(self.num_subvectors):
            reconstructed[:, m, :] = self.codebooks[m][codes[:, m]]

        return reconstructed.view(B, self.embedding_dim)


class VectorQuantizer(LatentQuantizer):
    """Vector Quantization (VQ) layer (differentiable codebook).

    Utilizes straight-through estimator (STE) to enable backpropagation through
    discrete quantization indices.
    """

    def __init__(self, embedding_dim: int = 256, num_embeddings: int = 512, commitment_cost: float = 0.25) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        # Codebook weights
        self.codebook = nn.Embedding(num_embeddings, embedding_dim)
        self.codebook.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    def fit(self, embeddings: Tensor) -> None:
        # Learned dynamically during PyTorch training
        pass

    def encode(self, embeddings: Tensor) -> QuantizationResult:
        """Quantizes the incoming embeddings.

        Args:
            embeddings: Tensor of shape (B, D)
        """
        # Calculate distances: ||z - e||^2 = ||z||^2 + ||e||^2 - 2 * z @ e.T
        distances = (
            torch.sum(embeddings**2, dim=1, keepdim=True)
            + torch.sum(self.codebook.weight**2, dim=1)
            - 2 * torch.matmul(embeddings, self.codebook.weight.t())
        )  # (B, K)

        encoding_indices = torch.argmin(distances, dim=1)  # (B,)
        quantized = self.codebook(encoding_indices)  # (B, D)

        # Loss formulations
        # commitment loss: push encoder representations close to codebook
        e_latent_loss = F.mse_loss(quantized.detach(), embeddings)
        # codebook loss: push codebook centroids close to encoder outputs
        q_latent_loss = F.mse_loss(quantized, embeddings.detach())
        commitment_loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Straight-through estimator
        quantized = embeddings + (quantized - embeddings).detach()

        return QuantizationResult(
            codes=encoding_indices.unsqueeze(-1),
            reconstructed=quantized,
            commitment_loss=commitment_loss
        )

    def decode(self, codes: Tensor) -> Tensor:
        """Decode indices (B, 1) or (B,) back into full embeddings (B, D)."""
        flat_codes = codes.squeeze(-1) if len(codes.shape) == 2 else codes
        return self.codebook(flat_codes)
