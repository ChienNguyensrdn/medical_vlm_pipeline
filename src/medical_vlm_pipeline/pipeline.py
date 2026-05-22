"""TrustMed-RAG — End-to-end orchestration for uncertainty-aware retrieval-augmented
medical vision-language diagnosis.

Exposes two pipeline classes:
    MedicalVLMPipeline    — original quantised RAG pipeline (backward compat)
    TrustMedRAGPipeline   — full 10-stage pipeline from medical_vlm_pipeline.md
"""

import logging
from dataclasses import dataclass, field
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

# New pipeline stages
from .grounding import LesionAnatomyGrounder, GroundingResult
from .knowledge_graph import DynamicKGBuilder, MedicalKGEncoder, MedicalKnowledgeGraph
from .fusion import AdaptiveMultimodalFusion, RetrievalEvidenceAggregator, FusionOutput
from .uncertainty import UncertaintyEstimator, UncertaintyOutput
from .reasoning import ClinicalReasoningAgent, ReasoningTrace

logger = logging.getLogger(__name__)


@dataclass
class DiagnosisOutput:
    diagnosis: str
    confidence: float
    retrieved_cases: list[RetrievalResult]
    report: str | None = None
    latent_embedding: Tensor | None = None
    quantized_embedding: Tensor | None = None
    uncertainty: float | None = None
    step_metrics: dict[str, Any] | None = None


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
        model_device = next(self.parameters()).device

        logger.info("Extracting embeddings for vector index database...")
        with torch.no_grad():
            for batch in train_loader:
                images = batch["image"].to(model_device)
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
        """Execute the complete end-to-end RAG clinical inference pipeline with step profiling.

        Args:
            image: Single input scan or batch image tensor. Shape (1, C, H, W) or (1, C, D, H, W).
        """
        import time
        self.eval()
        step_metrics = {}

        with torch.no_grad():
            # 1. Encode image to aligned space
            t_start = time.perf_counter()
            latent = self.encode_image(image)
            t_image = time.perf_counter() - t_start
            
            step_metrics["image_encoder"] = {
                "latency_ms": t_image * 1000,
                "latent_shape": list(latent.shape),
                "l2_norm": float(torch.norm(latent, p=2).item())
            }

            # 2. Apply quantization compression
            t_start = time.perf_counter()
            quant_res = self.quantizer.encode(latent)
            query_embedding = quant_res.reconstructed if quant_res.reconstructed is not None else latent
            t_quant = time.perf_counter() - t_start
            
            # Reconstruction distortion (MSE)
            reconstruction_mse = float(F.mse_loss(latent, query_embedding).item())
            step_metrics["quantizer"] = {
                "latency_ms": t_quant * 1000,
                "reconstruction_mse": reconstruction_mse,
                "quantizer_type": self.quantizer.__class__.__name__
            }

            # 3. Vector Database Retrieval (similar cases)
            t_start = time.perf_counter()
            retrieved = self.retriever.search(query_embedding, top_k=self.config.retrieval.top_k)
            t_retrieve = time.perf_counter() - t_start
            
            scores = [r.score for r in retrieved]
            mean_score = sum(scores) / len(scores) if scores else 0.0
            step_metrics["retriever"] = {
                "latency_ms": t_retrieve * 1000,
                "num_retrieved": len(retrieved),
                "similarity_scores": scores,
                "mean_similarity_score": mean_score
            }

            # 4. Diagnose Head prediction with epistemic uncertainty estimation via MC Dropout
            t_start = time.perf_counter()
            if hasattr(self.diagnosis_head, "predict_with_uncertainty"):
                mc_res = self.diagnosis_head.predict_with_uncertainty(query_embedding, num_samples=20)
                pred_idx = mc_res["predicted_class"][0]
                confidence = float(mc_res["confidence"][0].item())
                uncertainty_score = float(mc_res["uncertainty"][0].item())
            else:
                logits = self.diagnosis_head(query_embedding)
                probabilities = F.softmax(logits, dim=-1)
                conf, pred_idx = torch.max(probabilities, dim=-1)
                confidence = float(conf.item())
                uncertainty_score = 0.0
            t_class = time.perf_counter() - t_start
            
            diagnosis_label = self.class_names[int(pred_idx.item())]
            step_metrics["diagnosis_head"] = {
                "latency_ms": t_class * 1000,
                "confidence": confidence,
                "uncertainty_entropy": uncertainty_score,
                "predicted_label": diagnosis_label
            }

            # 5. Multimodal Report Generation
            t_start = time.perf_counter()
            report_out = self.report_generator.generate(
                image_embedding=query_embedding,
                retrieved_context=retrieved,
                diagnosis_hint=diagnosis_label
            )
            t_report = time.perf_counter() - t_start
            
            step_metrics["report_generator"] = {
                "latency_ms": t_report * 1000,
                "report_length_chars": len(report_out.text) if report_out.text else 0,
                "repetition_fallback_triggered": getattr(report_out, "fallback_triggered", False),
                "tokens_generated": len(report_out.text.split()) if report_out.text else 0
            }

            return DiagnosisOutput(
                diagnosis=diagnosis_label,
                confidence=confidence,
                retrieved_cases=retrieved,
                report=report_out.text,
                latent_embedding=latent,
                quantized_embedding=query_embedding,
                uncertainty=uncertainty_score,
                step_metrics=step_metrics
            )

    def forward(self, images: Tensor) -> Tensor:
        """Direct classification path (useful for downstream trainer interfaces)."""
        latent = self.encode_image(images)
        quant_res = self.quantizer.encode(latent)
        query_emb = quant_res.reconstructed if quant_res.reconstructed is not None else latent
        return self.diagnosis_head(query_emb)


# ---------------------------------------------------------------------------
# TrustMedRAGPipeline — Full 10-stage pipeline
# ---------------------------------------------------------------------------

@dataclass
class TrustMedOutput:
    """Complete structured output from the full TrustMed-RAG 10-stage pipeline."""

    # Stage 2 — Encoded embeddings
    Z_v: Tensor | None = None
    Z_t: Tensor | None = None

    # Stage 3 — Grounding
    grounding: GroundingResult | None = None

    # Stage 4 — Retrieved cases
    retrieved_cases: list[RetrievalResult] = field(default_factory=list)

    # Stage 5 — Knowledge graph
    graph: MedicalKnowledgeGraph | None = None

    # Stage 6 — Fused representation
    fusion: FusionOutput | None = None

    # Stage 7 — Uncertainty
    uncertainty: UncertaintyOutput | None = None

    # Stage 8 — Reasoning trace
    reasoning_trace: ReasoningTrace | None = None

    # Stage 9 — Generated report
    report: str = ""

    # Metadata
    step_latencies_ms: dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        """Produce a human-readable one-page summary of the full pipeline output."""
        lines: list[str] = ["=" * 60, "TrustMed-RAG Pipeline Output", "=" * 60]

        # Grounding
        if self.grounding:
            lines.append(
                f"\n[Stage 3 — Grounding] score={self.grounding.global_grounding_score:.3f}"
            )
            for phrase, region in self.grounding.phrase_to_region.items():
                score = self.grounding.phrase_scores.get(phrase, 0.0)
                lines.append(f"  • {phrase!r} → {region} ({score:.2f})")

        # Retrieval
        if self.retrieved_cases:
            lines.append(f"\n[Stage 4 — Retrieval] {len(self.retrieved_cases)} cases")
            for r in self.retrieved_cases[:3]:
                lines.append(
                    f"  • {r.case_id} | sim={r.score:.3f} | label={r.label}"
                )

        # Knowledge graph
        if self.graph:
            lines.append(
                f"\n[Stage 5 — Knowledge Graph] "
                f"{self.graph.num_nodes} nodes, {self.graph.num_edges} edges"
            )

        # Fusion gate weights
        if self.fusion is not None:
            w = self.fusion.gate_weights[0].tolist()
            labels = ["visual", "text", "retrieval", "graph"]
            gate_str = " | ".join(f"{l}={v:.2f}" for l, v in zip(labels, w))
            lines.append(f"\n[Stage 6 — Fusion] Gate: {gate_str}")

        # Uncertainty
        if self.uncertainty:
            u = self.uncertainty
            lines.append(
                f"\n[Stage 7 — Uncertainty] global={u.U_global:.3f} "
                f"aleatoric={u.aleatoric:.3f} epistemic={u.epistemic:.3f} "
                f"→ {u.decision.upper()}"
            )

        # Reasoning
        if self.reasoning_trace:
            rt = self.reasoning_trace
            hyp = rt.primary_hypothesis
            lines.append(
                f"\n[Stage 8 — Reasoning] "
                f"findings={len(rt.findings)} "
                f"contradiction={'YES' if rt.contradiction.has_contradiction else 'NO'}"
            )
            if hyp:
                lines.append(
                    f"  Primary: {hyp.diagnosis} (conf={hyp.confidence:.2f})"
                )
                if hyp.differential:
                    lines.append(f"  Differentials: {', '.join(hyp.differential)}")

        # Report
        lines.append(f"\n[Stage 9 — Report]\n{self.report}")

        # Latencies
        if self.step_latencies_ms:
            lines.append("\n[Latencies]")
            for stage, ms in self.step_latencies_ms.items():
                lines.append(f"  {stage}: {ms:.1f} ms")

        lines.append("=" * 60)
        return "\n".join(lines)


class TrustMedRAGPipeline(nn.Module):
    """Full TrustMed-RAG 10-stage clinical vision-language pipeline.

    Stages:
        1  Data Preprocessing           (caller-side; expects clean Tensor input)
        2  Vision-Language Encoding     → MedicalImageEncoder + ClinicalTextEncoder
        3  Lesion / Anatomy Grounding   → LesionAnatomyGrounder
        4  Retrieval-Augmented Search   → FAISSRetriever
        5  Dynamic Knowledge Graph      → DynamicKGBuilder + MedicalKGEncoder
        6  Adaptive Multimodal Fusion   → AdaptiveMultimodalFusion
        7  Uncertainty Estimation       → UncertaintyEstimator
        8  Clinical Reasoning Agent     → ClinicalReasoningAgent
        9  Controlled Report Generation → LLMReportGenerator
       10  Evaluation                   → (offline metrics scripts)

    Args:
        config:      PipelineConfig dataclass.
        num_classes: Number of diagnostic classes.
        class_names: Human-readable class names.
        tau1:        Uncertainty threshold: confident boundary (default 0.35).
        tau2:        Uncertainty threshold: defer boundary (default 0.65).
    """

    def __init__(
        self,
        config: PipelineConfig,
        num_classes: int = 3,
        class_names: list[str] | None = None,
        tau1: float = 0.35,
        tau2: float = 0.65,
    ) -> None:
        super().__init__()
        self.config = config
        self.num_classes = num_classes
        self.class_names = class_names or ["Healthy", "Meningioma", "Glioma"]
        dim = config.encoders.projection_dim

        # ── Stage 2: Vision-Language Encoders ────────────────────────────
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
            output_dim=dim,
        )
        self.text_proj = ProjectionHead(
            input_dim=config.encoders.embedding_dim,
            output_dim=dim,
        )

        # ── Stage 3: Grounding ────────────────────────────────────────────
        self.grounder = LesionAnatomyGrounder(feature_dim=dim)

        # ── Stage 4: Retrieval ────────────────────────────────────────────
        self.retriever = FAISSRetriever(
            embedding_dim=dim,
            metric="cosine",
            index_path=config.retrieval.index_path,
        )

        # ── Stage 5: Knowledge Graph ──────────────────────────────────────
        self.kg_builder = DynamicKGBuilder(include_base_ontology=True)
        self.kg_encoder = MedicalKGEncoder(embedding_dim=dim, hidden_dim=dim // 2)

        # ── Stage 6: Fusion ───────────────────────────────────────────────
        self.retrieval_aggregator = RetrievalEvidenceAggregator(embedding_dim=dim)
        self.fusion = AdaptiveMultimodalFusion(dim=dim, num_heads=4)

        # ── Stage 7: Uncertainty ──────────────────────────────────────────
        self.uncertainty_estimator = UncertaintyEstimator(
            embedding_dim=dim,
            num_classes=num_classes,
            tau1=tau1,
            tau2=tau2,
        )

        # ── Stage 8: Reasoning ────────────────────────────────────────────
        self.reasoning_agent = ClinicalReasoningAgent()

        # ── Stage 9: Report Generation ────────────────────────────────────
        self.report_generator = LLMReportGenerator(embedding_dim=dim)

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------

    def encode_image(self, image: Tensor) -> Tensor:
        """Stage 2a: image → (B, D) projected visual embedding."""
        return self.image_proj(self.image_encoder(image))

    def encode_text(
        self,
        token_ids: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        """Stage 2b: clinical text → (B, D) projected text embedding."""
        return self.text_proj(self.text_encoder(token_ids, attention_mask))

    # ------------------------------------------------------------------
    # Index building (Stage 4 pre-step)
    # ------------------------------------------------------------------

    def build_vector_index(self, dataloader: Any) -> None:
        """Fit retrieval index on training/reference cases.

        Args:
            dataloader: Yields dicts with keys: image, case_id, report_text, label.
        """
        self.eval()
        all_embs, all_ids, all_meta = [], [], []
        device = next(self.parameters()).device

        with torch.no_grad():
            for batch in dataloader:
                imgs = batch["image"].to(device)
                Z_v = self.encode_image(imgs)
                all_embs.append(Z_v.cpu())
                all_ids.extend(batch["case_id"])
                for cid, rep, lbl in zip(
                    batch["case_id"], batch["report_text"], batch["label"]
                ):
                    all_meta.append({"case_id": cid, "report_text": rep, "label": lbl})

        ref = torch.cat(all_embs, dim=0)
        self.retriever.add(all_ids, ref, all_meta)
        logger.info(f"TrustMedRAG index built with {len(all_ids)} cases.")

    # ------------------------------------------------------------------
    # Full 10-stage inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def diagnose(
        self,
        image: Tensor,
        clinical_text: str = "",
        token_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
    ) -> TrustMedOutput:
        """Execute all 10 pipeline stages for a single patient case.

        Args:
            image:          (1, C, H, W) or (1, C, D, H, W) medical image tensor.
            clinical_text:  Raw clinical note string.
            token_ids:      Optional pre-tokenised text tensor (1, SeqLen).
            attention_mask: Optional attention mask tensor (1, SeqLen).

        Returns:
            TrustMedOutput with all intermediate and final results.
        """
        import time
        self.eval()
        out = TrustMedOutput()
        device = next(self.parameters()).device

        # ── Stage 2: Encode ───────────────────────────────────────────────
        t0 = time.perf_counter()
        Z_v = self.encode_image(image.to(device))       # (1, D)

        Z_t: Tensor | None = None
        if token_ids is not None:
            Z_t = self.encode_text(
                token_ids.to(device),
                attention_mask.to(device) if attention_mask is not None else None,
            )
        out.Z_v = Z_v
        out.Z_t = Z_t
        out.step_latencies_ms["stage2_encoding"] = (time.perf_counter() - t0) * 1000

        # ── Stage 3: Grounding ────────────────────────────────────────────
        t0 = time.perf_counter()
        # Use visual features as token sequence (1, 1, D) — expand if needed
        visual_tokens = Z_v.unsqueeze(1)   # (1, 1, D) — minimal token dim
        grounding = self.grounder(visual_tokens, clinical_text)
        out.grounding = grounding
        out.step_latencies_ms["stage3_grounding"] = (time.perf_counter() - t0) * 1000

        # ── Stage 4: Retrieval ────────────────────────────────────────────
        t0 = time.perf_counter()
        retrieved = self.retriever.search(Z_v, top_k=self.config.retrieval.top_k)
        out.retrieved_cases = retrieved
        out.step_latencies_ms["stage4_retrieval"] = (time.perf_counter() - t0) * 1000

        # ── Stage 5: Dynamic Knowledge Graph ─────────────────────────────
        t0 = time.perf_counter()
        graph = self.kg_builder.build(
            clinical_text=clinical_text,
            phrase_to_region=grounding.phrase_to_region,
            phrase_scores=grounding.phrase_scores,
            retrieved_cases=retrieved,
        )
        out.graph = graph
        Z_g = self.kg_encoder(graph, device=device)     # (1, D)
        out.step_latencies_ms["stage5_knowledge_graph"] = (time.perf_counter() - t0) * 1000

        # ── Stage 6: Adaptive Multimodal Fusion ───────────────────────────
        t0 = time.perf_counter()
        # Aggregate retrieval evidence into Z_r
        if retrieved:
            case_embs = torch.stack([
                Z_v.squeeze(0) * r.score for r in retrieved
            ], dim=0)                               # rough proxy; replace with actual embeddings
            retrieval_scores = torch.tensor(
                [r.score for r in retrieved], device=device
            )
            Z_r = self.retrieval_aggregator(case_embs, retrieval_scores)  # (1, D)
        else:
            Z_r = None

        fusion_out = self.fusion(
            Z_v=Z_v,
            Z_t=Z_t,
            Z_r=Z_r,
            Z_g=Z_g,
        )
        out.fusion = fusion_out
        out.step_latencies_ms["stage6_fusion"] = (time.perf_counter() - t0) * 1000

        # ── Stage 7: Uncertainty Estimation ──────────────────────────────
        t0 = time.perf_counter()
        retrieval_score_map = {
            r.case_id: r.score for r in retrieved
        } if retrieved else {}

        uncertainty = self.uncertainty_estimator(
            Z_f=fusion_out.fused,
            phrase_scores=grounding.phrase_scores,
            retrieval_scores=retrieval_score_map,
            graph_num_nodes=graph.num_nodes,
        )
        out.uncertainty = uncertainty
        out.step_latencies_ms["stage7_uncertainty"] = (time.perf_counter() - t0) * 1000

        # ── Stage 8: Clinical Reasoning ───────────────────────────────────
        t0 = time.perf_counter()
        reasoning_trace = self.reasoning_agent.reason(
            grounding=grounding,
            retrieved_cases=retrieved,
            graph=graph,
            uncertainty=uncertainty,
            clinical_text=clinical_text,
        )
        out.reasoning_trace = reasoning_trace
        out.step_latencies_ms["stage8_reasoning"] = (time.perf_counter() - t0) * 1000

        # ── Stage 9: Controlled Report Generation ─────────────────────────
        t0 = time.perf_counter()
        diagnosis_hint = (
            reasoning_trace.primary_hypothesis.diagnosis
            if reasoning_trace.primary_hypothesis
            else None
        )
        report_out = self.report_generator.generate(
            image_embedding=fusion_out.fused,
            retrieved_context=retrieved,
            diagnosis_hint=diagnosis_hint,
        )
        # Append uncertainty statement and recommendation
        uncertainty_stmt = uncertainty.uncertainty_statement()
        recommendation = reasoning_trace.recommendation
        full_report = (
            f"{report_out.text}\n\n"
            f"Uncertainty: {uncertainty_stmt}\n\n"
            f"Recommendation: {recommendation}"
        )
        out.report = full_report
        out.step_latencies_ms["stage9_generation"] = (time.perf_counter() - t0) * 1000

        logger.info(
            f"TrustMedRAG inference complete — "
            f"decision={uncertainty.decision}, "
            f"hypothesis={diagnosis_hint}, "
            f"total_ms={sum(out.step_latencies_ms.values()):.1f}"
        )

        return out

    def forward(self, image: Tensor, clinical_text: str = "") -> TrustMedOutput:
        """Alias for diagnose() — supports nn.Module forward conventions."""
        return self.diagnose(image, clinical_text)

