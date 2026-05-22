"""Medical VLM pipeline — TrustMed-RAG.

Stages:
    1  Data Preprocessing          → data.py
    2  Vision-Language Encoding    → encoders.py
    3  Lesion / Anatomy Grounding  → grounding.py
    4  Retrieval-Augmented Search  → retrieval.py
    5  Dynamic Knowledge Graph     → knowledge_graph.py
    6  Adaptive Multimodal Fusion  → fusion.py
    7  Uncertainty Estimation      → uncertainty.py
    8  Clinical Reasoning Agent    → reasoning.py
    9  Controlled Report Generation→ generation.py
   10  Evaluation                  → (external metrics scripts)
"""

from .config import PipelineConfig
from .pipeline import MedicalVLMPipeline, TrustMedRAGPipeline, DiagnosisOutput
from .data import MedicalCase, MedicalCaseDataset, load_iu_chest_xray_cases

# Stage modules
from .grounding import LesionAnatomyGrounder, GroundingResult, extract_pathology_phrases
from .knowledge_graph import (
    DynamicKGBuilder, MedicalKGEncoder,
    MedicalKnowledgeGraph, KGNode, KGEdge,
)
from .fusion import AdaptiveMultimodalFusion, RetrievalEvidenceAggregator, FusionOutput
from .uncertainty import UncertaintyEstimator, UncertaintyOutput
from .reasoning import ClinicalReasoningAgent, ReasoningTrace

__all__ = [
    # Pipeline
    "PipelineConfig",
    "MedicalVLMPipeline",
    "TrustMedRAGPipeline",
    "DiagnosisOutput",
    # Data
    "MedicalCase",
    "MedicalCaseDataset",
    "load_iu_chest_xray_cases",
    # Stage 3
    "LesionAnatomyGrounder",
    "GroundingResult",
    "extract_pathology_phrases",
    # Stage 5
    "DynamicKGBuilder",
    "MedicalKGEncoder",
    "MedicalKnowledgeGraph",
    "KGNode",
    "KGEdge",
    # Stage 6
    "AdaptiveMultimodalFusion",
    "RetrievalEvidenceAggregator",
    "FusionOutput",
    # Stage 7
    "UncertaintyEstimator",
    "UncertaintyOutput",
    # Stage 8
    "ClinicalReasoningAgent",
    "ReasoningTrace",
]
