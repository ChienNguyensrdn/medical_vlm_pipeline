"""Retrieval interfaces and implementations for similar-case diagnosis."""

import logging
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor
import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievalResult:
    case_id: str
    score: float
    report_text: str | None = None
    label: str | None = None
    projection: str | None = None


class VectorRetriever:
    """Vector database contract for FAISS, Qdrant, Milvus, or ScaNN backends."""

    def add(self, case_ids: list[str], embeddings: Tensor) -> None:
        raise NotImplementedError

    def search(self, query_embedding: Tensor, top_k: int = 5) -> list[RetrievalResult]:
        raise NotImplementedError

    def save(self, path: Path) -> None:
        raise NotImplementedError

    def load(self, path: Path) -> None:
        raise NotImplementedError


class FAISSRetriever(VectorRetriever):
    """FAISS-backed vector database for fast nearest-neighbor search on latent embeddings.

    Supports a high-quality pure-PyTorch fallback when FAISS is unavailable, ensuring
    maximum reliability across compute environments.
    """

    def __init__(
        self,
        embedding_dim: int = 256,
        metric: str = "cosine",
        index_path: Path | None = None,
    ) -> None:
        """
        Args:
            embedding_dim: Dimensionality of latent embeddings.
            metric: Distance metric, either "cosine" or "l2".
            index_path: Optional path to load/save index.
        """
        self.embedding_dim = embedding_dim
        self.metric = metric.lower()
        self.index_path = index_path

        self.case_ids: list[str] = []
        self.case_metadata: dict[str, dict[str, str]] = {}
        self.db_embeddings: list[Tensor] = []  # Used for PyTorch fallback

        # Load FAISS index or initialize fallback
        try:
            import faiss
            self.faiss_available = True
            logger.info("Initializing FAISS Retriever backend.")
            if self.metric == "cosine":
                # For cosine similarity, normalize vectors and use inner product (IP) index
                self.index = faiss.IndexFlatIP(embedding_dim)
            else:
                self.index = faiss.IndexFlatL2(embedding_dim)
        except ImportError:
            self.faiss_available = False
            self.index = None
            logger.warning("FAISS is not installed. Falling back to high-performance PyTorch vector search.")

        if index_path and index_path.exists():
            self.load(index_path)

    def add(self, case_ids: list[str], embeddings: Tensor, metadata: list[dict[str, str]] | None = None) -> None:
        """Add case embeddings to the vector database.

        Args:
            case_ids: List of unique case identifiers.
            embeddings: Tensor of shape (N, D).
            metadata: Optional list of dictionaries containing report_text, label, etc.
        """
        assert len(case_ids) == embeddings.shape[0], "Count of case_ids must match embedding batch size."
        embeddings_cpu = embeddings.detach().cpu()

        # Update metadata maps
        for i, cid in enumerate(case_ids):
            if cid not in self.case_ids:
                self.case_ids.append(cid)
            meta = metadata[i] if metadata is not None and i < len(metadata) else {}
            self.case_metadata[cid] = meta

        # Add to index
        if self.faiss_available:
            import faiss
            embeddings_np = embeddings_cpu.numpy().astype("float32")
            if self.metric == "cosine":
                # Normalize for Cosine Similarity (IndexFlatIP computes dot product)
                faiss.normalize_L2(embeddings_np)
            self.index.add(embeddings_np)
        else:
            # Fallback PyTorch representation
            self.db_embeddings.append(embeddings_cpu)

        logger.info(f"Added {len(case_ids)} items to Vector DB. Total size: {len(self.case_ids)}.")

    def _get_fallback_db(self) -> Tensor | None:
        if not self.db_embeddings:
            return None
        return torch.cat(self.db_embeddings, dim=0)

    def search(self, query_embedding: Tensor, top_k: int = 5) -> list[RetrievalResult]:
        """Query similar cases from the database.

        Args:
            query_embedding: Tensor of shape (1, D) or (D,).
            top_k: Number of nearest neighbors to retrieve.
        """
        if not self.case_ids:
            return []

        # Ensure correct query shape (1, D)
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.unsqueeze(0)
        query_cpu = query_embedding.detach().cpu()

        actual_k = min(top_k, len(self.case_ids))
        results: list[RetrievalResult] = []

        if self.faiss_available:
            import faiss
            query_np = query_cpu.numpy().astype("float32")
            if self.metric == "cosine":
                faiss.normalize_L2(query_np)

            # Search FAISS index
            scores, indices = self.index.search(query_np, actual_k)
            scores = scores[0]
            indices = indices[0]

            for score, idx in zip(scores, indices):
                if idx < 0 or idx >= len(self.case_ids):
                    continue
                cid = self.case_ids[idx]
                meta = self.case_metadata.get(cid, {})
                results.append(RetrievalResult(
                    case_id=cid,
                    score=float(score),
                    report_text=meta.get("report_text"),
                    label=meta.get("label"),
                    projection=meta.get("projection"),
                ))
        else:
            # High-performance PyTorch vectorized search fallback
            db_tensor = self._get_fallback_db()
            if db_tensor is None:
                return []

            if self.metric == "cosine":
                # Cosine Similarity: query * db / (||query|| * ||db||)
                q_norm = torch.nn.functional.normalize(query_cpu, dim=-1)
                db_norm = torch.nn.functional.normalize(db_tensor, dim=-1)
                similarities = torch.matmul(q_norm, db_norm.t()).squeeze(0)  # (N,)
            else:
                # Euclidean Distance (L2) converted to negative distance for score consistency
                similarities = -torch.sum((query_cpu - db_tensor) ** 2, dim=-1).squeeze(0)

            top_values, top_indices = torch.topk(similarities, actual_k)

            for score, idx in zip(top_values.tolist(), top_indices.tolist()):
                cid = self.case_ids[idx]
                meta = self.case_metadata.get(cid, {})
                results.append(RetrievalResult(
                    case_id=cid,
                    score=float(score),
                    report_text=meta.get("report_text"),
                    label=meta.get("label"),
                    projection=meta.get("projection"),
                ))

        return results

    def save(self, path: Path) -> None:
        """Serialize current state and indices to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "case_ids": self.case_ids,
            "case_metadata": self.case_metadata,
            "metric": self.metric,
            "embedding_dim": self.embedding_dim,
        }

        if self.faiss_available:
            import faiss
            # Save FAISS index
            faiss.write_index(self.index, str(path.with_suffix(".index")))
        else:
            # Save raw PyTorch embeddings
            state["db_embeddings"] = self._get_fallback_db()

        torch.save(state, str(path.with_suffix(".state")))
        logger.info(f"Retriever index saved to {path}.")

    def load(self, path: Path) -> None:
        """Load serialized index and state from disk."""
        state_path = path.with_suffix(".state")
        if not state_path.exists():
            logger.warning(f"Retriever state file not found at {state_path}.")
            return

        state = torch.load(str(state_path))
        self.case_ids = state["case_ids"]
        self.case_metadata = state["case_metadata"]
        self.metric = state["metric"]
        self.embedding_dim = state["embedding_dim"]

        index_file = path.with_suffix(".index")
        if self.faiss_available and index_file.exists():
            import faiss
            self.index = faiss.read_index(str(index_file))
            logger.info("Successfully loaded FAISS index.")
        elif "db_embeddings" in state:
            db_tensor = state["db_embeddings"]
            if db_tensor is not None:
                self.db_embeddings = [db_tensor]
            logger.info("Successfully loaded PyTorch fallback retriever state.")
        else:
            logger.warning("No index search representations found to load.")
