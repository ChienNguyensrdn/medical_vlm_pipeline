"""End-to-end orchestration pipeline for quantized retrieval-augmented medical vision-language diagnosis."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .config import PipelineConfig
from .retrieval import RetrievalResult, VectorRetriever, FAISSRetriever
from .encoders import MedicalImageEncoder, ClinicalTextEncoder, ProjectionHead
from .quantization import LatentQuantizer, IdentityQuantizer, ProductQuantizer
from .heads import DiagnosisHead
from .generation import LLMReportGenerator, GeneratedReport

logger = logging.getLogger(__name__)


@dataclass
class DiagnosisOutput:
    diagnosis: str
    confidence: float
    retrieved_cases: list[RetrievalResult]
    report: str | None = None
    latent_embedding: Tensor | None = None
    quantized_embedding: Tensor | None = None


class MedicalVLMPipeline(nn.Module):
    """Coordinates image encoding, cross-modal alignment, quantized retrieval,

    disease classification, and text report generation in a unified interface.
    """

    def __init__(
        self,
        config: PipelineConfig,
        num_classes: int = 3,
        class_names: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.num_classes = num_classes
        self.class_names = class_names or ["Healthy", "Meningioma", "Glioma"]

        # 1. Encoders & Projection Heads (InfoNCE Shared Latent Alignment)
        self.image_encoder = MedicalImageEncoder(
            model_name=config.encoders.image_encoder,
            embedding_dim=config.encoders.embedding_dim,
            volume_depth=config.data.volume_depth,
        )

        self.text_encoder = ClinicalTextEncoder(
            model_name=config.encoders.text_encoder,
            embedding_dim=config.encoders.embedding_dim,
        )

        self.image_proj = ProjectionHead(
            input_dim=config.encoders.embedding_dim,
            output_dim=config.encoders.projection_dim,
        )

        self.text_proj = ProjectionHead(
            input_dim=config.encoders.embedding_dim,
            output_dim=config.encoders.projection_dim,
        )

        # 2. Latent Space Quantizer
        if config.retrieval.quantized:
            logger.info("Initializing Product Quantization (PQ) Compression Layer.")
            self.quantizer = ProductQuantizer(
                embedding_dim=config.encoders.projection_dim,
                num_subvectors=8,
                num_centroids=256
            )
        else:
            self.quantizer = IdentityQuantizer()

        # 3. Vector Database Retriever
        self.retriever = FAISSRetriever(
            embedding_dim=config.encoders.projection_dim,
            metric="cosine",
            index_path=config.retrieval.index_path,
        )

        # 4. Diagnostic Classifier Head (defined over shared aligned space)
        self.diagnosis_head = DiagnosisHead(
            embedding_dim=config.encoders.projection_dim,
            num_classes=num_classes,
        )

        # 5. Report Generator (Seq2Seq / Fallback decoder)
        self.report_generator = LLMReportGenerator(
            embedding_dim=config.encoders.projection_dim,
        )

    def encode_image(self, image: Tensor) -> Tensor:
        """Process image tensor to aligned multimodal projection space."""
        # Visual feature representation
        feats = self.image_encoder(image)
        # Shared space alignment
        proj = self.image_proj(feats)
        return proj

    def encode_text(self, token_ids: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        """Process clinical report tokens to aligned multimodal projection space."""
        # Textual representation
        feats = self.text_encoder(token_ids, attention_mask)
        # Shared space alignment
        proj = self.text_proj(feats)
        return proj

    def build_vector_index(self, train_loader: Any) -> None:
        """Fit the quantizer and build the retrieval index using reference cases.

        Args:
            train_loader: Dataloader yielding dicts with "image", "case_id", "report_text", "label"
        """
        self.eval()
        all_embeddings = []
        all_case_ids = []
        all_metadata = []

        logger.info("Extracting embeddings for vector index database...")
        with torch.no_grad():
            for batch in train_loader:
                images = batch["image"]
                case_ids = batch["case_id"]
                reports = batch["report_text"]
                labels = batch["label"]

                # Extract aligned projections
                projs = self.encode_image(images)
                all_embeddings.append(projs.cpu())
                all_case_ids.extend(case_ids)

                for cid, rep, lbl in zip(case_ids, reports, labels):
                    all_metadata.append({
                        "case_id": cid,
                        "report_text": rep,
                        "label": lbl
                    })

        # Concatenate reference features
        reference_features = torch.cat(all_embeddings, dim=0)

        # Fit quantization codebook
        self.quantizer.fit(reference_features)

        # Compress features
        quant_res = self.quantizer.encode(reference_features)
        compressed_embeddings = quant_res.reconstructed if quant_res.reconstructed is not None else reference_features

        # Index compressed vectors in retriever
        self.retriever.add(all_case_ids, compressed_embeddings, all_metadata)
        logger.info("Vector database indexing successfully constructed.")

    def diagnose(self, image: Tensor) -> DiagnosisOutput:
        """Execute the complete end-to-end RAG clinical inference pipeline.

        Args:
            image: Single input scan or batch image tensor. Shape (1, C, H, W) or (1, C, D, H, W).
        """
        self.eval()
        with torch.no_grad():
            # 1. Encode image to aligned space
            latent = self.encode_image(image)

            # 2. Apply quantization compression
            quant_res = self.quantizer.encode(latent)
            query_embedding = quant_res.reconstructed if quant_res.reconstructed is not None else latent

            # 3. Vector Database Retrieval (similar cases)
            retrieved = self.retriever.search(query_embedding, top_k=self.config.retrieval.top_k)

            # 4. Diagnose Head prediction
            logits = self.diagnosis_head(query_embedding)
            probabilities = F.softmax(logits, dim=-1)
            conf, pred_idx = torch.max(probabilities, dim=-1)

            confidence = float(conf.item())
            diagnosis_label = self.class_names[int(pred_idx.item())]

            # 5. Multimodal Report Generation
            report_out = self.report_generator.generate(
                image_embedding=query_embedding,
                retrieved_context=retrieved,
                diagnosis_hint=diagnosis_label
            )

            return DiagnosisOutput(
                diagnosis=diagnosis_label,
                confidence=confidence,
                retrieved_cases=retrieved,
                report=report_out.text,
                latent_embedding=latent,
                quantized_embedding=query_embedding
            )

    def forward(self, images: Tensor) -> Tensor:
        """Direct classification path (useful for downstream trainer interfaces)."""
        latent = self.encode_image(images)
        quant_res = self.quantizer.encode(latent)
        query_emb = quant_res.reconstructed if quant_res.reconstructed is not None else latent
        return self.diagnosis_head(query_emb)
