"""Stage 6 — Adaptive Multimodal Fusion.

Fuses four information streams:
    Z_v  : visual embedding
    Z_t  : text (clinical context) embedding
    Z_r  : retrieval evidence embedding
    Z_g  : knowledge graph embedding

Fusion strategy (from pipeline spec):
    Z_vg = CrossAttention(Q=Z_v, K=Z_g, V=Z_g)
    Z_vr = CrossAttention(Q=Z_v, K=Z_r, V=Z_r)
    Z_gt = CrossAttention(Q=Z_g, K=Z_t, V=Z_t)

    w_v, w_t, w_r, w_g = AdaptiveGate(Z_v, Z_t, Z_r, Z_g)
    Z_f  = w_v·Z_v + w_t·Z_t + w_r·Z_r + w_g·Z_g
"""

import logging
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output container
# ---------------------------------------------------------------------------

@dataclass
class FusionOutput:
    """Output of the adaptive multimodal fusion stage."""
    fused: Tensor                       # (B, D) final fused representation Z_f
    gate_weights: Tensor                # (B, 4) softmax gate weights [w_v, w_t, w_r, w_g]
    cross_vg: Tensor | None = None      # (B, D) visual-graph cross-attention result
    cross_vr: Tensor | None = None      # (B, D) visual-retrieval cross-attention result
    cross_gt: Tensor | None = None      # (B, D) graph-text cross-attention result


# ---------------------------------------------------------------------------
# Pairwise Cross-Attention block
# ---------------------------------------------------------------------------

class CrossAttentionBlock(nn.Module):
    """Single cross-attention block: query from source, key/value from target.

    Args:
        dim: Embedding dimension.
        num_heads: Number of attention heads.
        dropout: Dropout on attention weights.
    """

    def __init__(self, dim: int = 256, num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )
        self.ffn_norm = nn.LayerNorm(dim)

    def forward(self, query: Tensor, key_value: Tensor) -> Tensor:
        """
        Args:
            query:     (B, 1, D) or (B, N, D) — query sequence.
            key_value: (B, M, D) — key & value sequence.

        Returns:
            (B, N, D) cross-attended query representation.
        """
        attn_out, _ = self.attn(query, key_value, key_value)
        x = self.norm(query + attn_out)           # residual + norm
        x = self.ffn_norm(x + self.ffn(x))        # FFN + residual + norm
        return x


# ---------------------------------------------------------------------------
# Adaptive Gate
# ---------------------------------------------------------------------------

class AdaptiveGate(nn.Module):
    """Learns scalar importance weights for each modality stream.

    Implements:
        w = softmax(MLP([Z_v, Z_t, Z_r, Z_g]))

    Args:
        dim:        Per-stream embedding dimension.
        num_streams: Number of modality streams (default 4).
        hidden_dim: Hidden MLP width.
    """

    def __init__(self, dim: int = 256, num_streams: int = 4, hidden_dim: int = 128) -> None:
        super().__init__()
        self.num_streams = num_streams
        self.gate_mlp = nn.Sequential(
            nn.Linear(dim * num_streams, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_streams),
        )

    def forward(self, streams: list[Tensor]) -> Tensor:
        """Compute softmax gate weights.

        Args:
            streams: List of num_streams tensors, each (B, D).

        Returns:
            weights: (B, num_streams) softmax weights summing to 1.
        """
        concat = torch.cat(streams, dim=-1)       # (B, num_streams * D)
        logits = self.gate_mlp(concat)            # (B, num_streams)
        return F.softmax(logits, dim=-1)          # (B, num_streams)


# ---------------------------------------------------------------------------
# Main Fusion Module
# ---------------------------------------------------------------------------

class AdaptiveMultimodalFusion(nn.Module):
    """Stage 6 — Adaptive Multimodal Fusion.

    Combines visual, textual, retrieval, and graph embeddings into a single
    fused representation Z_f using cross-attention and learned adaptive gates.

    The module gracefully handles missing streams (None inputs) by replacing
    them with zero tensors and downweighting them implicitly via the gate.

    Args:
        dim:       Shared embedding dimension for all streams.
        num_heads: Attention heads per cross-attention block.
        dropout:   Dropout rate for attention and FFN layers.
    """

    def __init__(self, dim: int = 256, num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.dim = dim

        # Three cross-attention interactions (as per pipeline spec)
        self.cross_vg = CrossAttentionBlock(dim, num_heads, dropout)   # visual ← graph
        self.cross_vr = CrossAttentionBlock(dim, num_heads, dropout)   # visual ← retrieval
        self.cross_gt = CrossAttentionBlock(dim, num_heads, dropout)   # graph  ← text

        # Adaptive gating over 4 streams
        self.gate = AdaptiveGate(dim, num_streams=4, hidden_dim=dim // 2)

        # Final projection after weighted sum
        self.output_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.LayerNorm(dim),
        )

    def _ensure_3d(self, x: Tensor) -> Tensor:
        """Ensure tensor is (B, 1, D) for use as attention query/key."""
        if x.dim() == 2:   # (B, D)
            return x.unsqueeze(1)
        return x

    def _ensure_2d(self, x: Tensor) -> Tensor:
        """Collapse sequence dimension: (B, 1, D) → (B, D)."""
        if x.dim() == 3:
            return x.squeeze(1)
        return x

    def forward(
        self,
        Z_v: Tensor,
        Z_t: Tensor | None = None,
        Z_r: Tensor | None = None,
        Z_g: Tensor | None = None,
    ) -> FusionOutput:
        """Run adaptive multimodal fusion.

        Args:
            Z_v: Visual embedding (B, D) — required.
            Z_t: Text embedding (B, D) — optional; zero-filled if None.
            Z_r: Retrieval evidence embedding (B, D) — optional.
            Z_g: Knowledge graph embedding (B, D) — optional.

        Returns:
            FusionOutput with fused embedding and diagnostic gate weights.
        """
        B, D = Z_v.shape
        device = Z_v.device

        # Fill missing streams with zeros (gate will downweight them)
        Z_t = Z_t if Z_t is not None else torch.zeros(B, D, device=device)
        Z_r = Z_r if Z_r is not None else torch.zeros(B, D, device=device)
        Z_g = Z_g if Z_g is not None else torch.zeros(B, D, device=device)

        # Reshape to (B, 1, D) for MultiheadAttention (batch_first=True)
        Zv3 = self._ensure_3d(Z_v)  # (B, 1, D)
        Zt3 = self._ensure_3d(Z_t)
        Zr3 = self._ensure_3d(Z_r)
        Zg3 = self._ensure_3d(Z_g)

        # ── Cross-attention interactions ──────────────────────────────────
        # Z_vg: visual enriched with graph context
        Z_vg_3d = self.cross_vg(query=Zv3, key_value=Zg3)   # (B, 1, D)

        # Z_vr: visual enriched with retrieval evidence
        Z_vr_3d = self.cross_vr(query=Zv3, key_value=Zr3)   # (B, 1, D)

        # Z_gt: graph enriched with text context
        Z_gt_3d = self.cross_gt(query=Zg3, key_value=Zt3)   # (B, 1, D)

        # Collapse to (B, D)
        Z_vg = self._ensure_2d(Z_vg_3d)
        Z_vr = self._ensure_2d(Z_vr_3d)
        Z_gt = self._ensure_2d(Z_gt_3d)

        # ── Adaptive gate over original 4 streams ────────────────────────
        # Use cross-attended visual as Z_v proxy to avoid double-counting
        streams = [Z_vg, Z_t, Z_vr, Z_gt]     # enriched [v, t, r, g]
        weights = self.gate(streams)            # (B, 4)

        # Weighted sum
        w_v = weights[:, 0:1]   # (B, 1)
        w_t = weights[:, 1:2]
        w_r = weights[:, 2:3]
        w_g = weights[:, 3:4]

        Z_f_raw = w_v * Z_vg + w_t * Z_t + w_r * Z_vr + w_g * Z_gt   # (B, D)
        Z_f = self.output_proj(Z_f_raw)                                  # (B, D)

        logger.debug(
            f"Fusion gate weights — visual: {w_v.mean():.3f}, text: {w_t.mean():.3f}, "
            f"retrieval: {w_r.mean():.3f}, graph: {w_g.mean():.3f}"
        )

        return FusionOutput(
            fused=Z_f,
            gate_weights=weights,
            cross_vg=Z_vg,
            cross_vr=Z_vr,
            cross_gt=Z_gt,
        )


# ---------------------------------------------------------------------------
# Retrieval evidence aggregator
# ---------------------------------------------------------------------------

class RetrievalEvidenceAggregator(nn.Module):
    """Pools k retrieved case embeddings into a single evidence vector Z_r.

    Uses score-weighted mean pooling so higher-similarity cases contribute more.

    Args:
        embedding_dim: Dimension of per-case embeddings.
    """

    def __init__(self, embedding_dim: int = 256) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
            nn.LayerNorm(embedding_dim),
        )

    def forward(self, case_embeddings: Tensor, scores: Tensor | None = None) -> Tensor:
        """Aggregate k retrieved case embeddings into Z_r.

        Args:
            case_embeddings: (k, D) — embeddings of k retrieved cases.
            scores:          (k,) — similarity scores; uniform if None.

        Returns:
            Z_r: (1, D) aggregated retrieval evidence embedding.
        """
        if case_embeddings.shape[0] == 0:
            return torch.zeros(1, case_embeddings.shape[-1],
                               device=case_embeddings.device)

        if scores is None:
            scores = torch.ones(case_embeddings.shape[0], device=case_embeddings.device)

        # Clamp scores to [0, 1] and normalize
        w = torch.clamp(scores, min=0.0)
        w = w / (w.sum() + 1e-8)             # (k,)
        Z_r = (case_embeddings * w.unsqueeze(-1)).sum(dim=0, keepdim=True)  # (1, D)
        return self.proj(Z_r)
