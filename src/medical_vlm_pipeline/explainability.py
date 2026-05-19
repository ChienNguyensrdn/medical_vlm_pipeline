"""Explainability hooks for diagnostic visualization, GradCAM, and retrieval-augmented explanations."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F
import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExplanationArtifact:
    method: str
    summary: str
    heatmap: Tensor | None = None
    artifact_path: Path | None = None


def retrieval_explanation(query_case_id: str, similar_cases: list[Any]) -> ExplanationArtifact:
    """Generate a readable textual comparison explaining why these cases were retrieved.

    Args:
        query_case_id: ID of the query patient case.
        similar_cases: List of RetrievalResult objects.
    """
    case_summaries = []
    for c in similar_cases:
        score_pct = f"{c.score * 100:.1f}%" if c.score <= 1.0 else f"{c.score:.3f}"
        diag = f" (Diagnosis: {c.label})" if c.label else ""
        case_summaries.append(f"Case {c.case_id} [Similarity: {score_pct}]{diag}")

    joined_cases = "; ".join(case_summaries)
    summary = (
        f"Case {query_case_id} was matched in the shared latent space based on semantic similarity. "
        f"The most structurally similar precedent cases in our medical database are: {joined_cases}."
    )
    return ExplanationArtifact(method="retrieval", summary=summary)


class MedicalGradCAM:
    """Custom Grad-CAM (Gradient-weighted Class Activation Mapping) for medical CNNs and Transformers.

    Enables visual explainability by highlighting target anatomical zones that influenced
    a disease classification decision. Includes automatic tensor shape handling.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        """
        Args:
            model: PyTorch model containing the image encoder and diagnostic head.
            target_layer: The specific layer (e.g. final conv or block norm) to inspect.
        """
        self.model = model
        self.target_layer = target_layer
        self.activations: Tensor | None = None
        self.gradients: Tensor | None = None

        # Register forward and backward hooks
        self.forward_hook = target_layer.register_forward_hook(self._save_activations)
        self.backward_hook = target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module: nn.Module, input: Any, output: Any) -> None:
        # In transformers output is often a tuple or tensor
        if isinstance(output, tuple):
            self.activations = output[0].detach()
        else:
            self.activations = output.detach()

    def _save_gradients(self, module: nn.Module, grad_input: Any, grad_output: Any) -> None:
        if isinstance(grad_output, tuple):
            self.gradients = grad_output[0].detach()
        else:
            self.gradients = grad_output.detach()

    def generate_heatmap(self, input_tensor: Tensor, class_idx: int | None = None) -> Tensor:
        """Forward pass and backward pass to calculate Grad-CAM heatmaps.

        Args:
            input_tensor: Image tensor of shape (1, C, H, W) or (1, C, D, H, W) for 3D.
            class_idx: Index of the disease class to explain. Defaults to the highest scoring class.
        """
        self.model.eval()

        # Forward pass
        predictions = self.model(input_tensor)
        if class_idx is None:
            class_idx = int(predictions.argmax(dim=-1).item())

        # Zero gradients
        self.model.zero_grad()

        # Backward pass for the target class
        target_score = predictions[0, class_idx]
        target_score.backward(retain_graph=True)

        if self.activations is None or self.gradients is None:
            logger.warning("GradCAM could not capture activations or gradients. Returning blank heatmap.")
            # Return blank heatmap matching spatial size of input
            spatial_dims = input_tensor.shape[2:]
            return torch.zeros(spatial_dims)

        # Apply Grad-CAM math: weights = mean(gradients, spatial_dims)
        # Handle 2D vs 3D shapes
        is_3d = len(self.activations.shape) == 5

        if is_3d:
            # Activations: (B, C, D, H, W)
            weights = torch.mean(self.gradients, dim=[2, 3, 4], keepdim=True)  # (B, C, 1, 1, 1)
            cam = torch.sum(weights * self.activations, dim=1).squeeze(0)  # (D, H, W)
        else:
            # If activations have shape (B, SeqLen, Dim) e.g. from Transformer block norm
            # we need to reshape it back to 2D spatial dimensions if possible, or handle it.
            if len(self.activations.shape) == 3:
                # Shape: (B, N_patches, C)
                B, N, C = self.activations.shape
                H = W = int(np.sqrt(N))  # Assume square patch grid
                if H * W == N:
                    act_reshaped = self.activations.transpose(1, 2).view(B, C, H, W)
                    grad_reshaped = self.gradients.transpose(1, 2).view(B, C, H, W)
                    weights = torch.mean(grad_reshaped, dim=[2, 3], keepdim=True)
                    cam = torch.sum(weights * act_reshaped, dim=1).squeeze(0)
                else:
                    # Generic linear token sequence fallback
                    weights = torch.mean(self.gradients, dim=1, keepdim=True)
                    cam = torch.sum(weights * self.activations, dim=-1).squeeze(0)
                    # Reshape cam to square if possible
                    H = W = int(np.sqrt(N))
                    if H * W == N:
                        cam = cam.view(H, W)
                    else:
                        cam = cam.unsqueeze(0)
            else:
                activations = self.activations
                gradients = self.gradients

                # Swin/timm features can be channels-last: (B, H, W, C).
                if (
                    len(activations.shape) == 4
                    and activations.shape[-1] > activations.shape[1]
                    and activations.shape[-1] > activations.shape[2]
                ):
                    activations = activations.permute(0, 3, 1, 2)
                    gradients = gradients.permute(0, 3, 1, 2)

                # Standard Conv2D activations: (B, C, H, W)
                weights = torch.mean(gradients, dim=[2, 3], keepdim=True)  # (B, C, 1, 1)
                cam = torch.sum(weights * activations, dim=1).squeeze(0)  # (H, W)

        # Relu on CAM: only positive influence matters
        cam = F.relu(cam)

        # Normalize heatmap to [0, 1]
        cam_max = cam.max()
        if cam_max > 0:
            cam = cam / cam_max

        # Interpolate/upsample heatmap to match input image size
        spatial_dims = input_tensor.shape[2:]  # (H, W) or (D, H, W)
        cam_expanded = cam.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions

        mode = "trilinear" if is_3d else "bilinear"
        upsampled_cam = F.interpolate(
            cam_expanded,
            size=spatial_dims,
            mode=mode,
            align_corners=False
        ).squeeze(0).squeeze(0)

        return upsampled_cam

    def remove_hooks(self) -> None:
        """Remove registered hooks to prevent memory leaks."""
        self.forward_hook.remove()
        self.backward_hook.remove()
