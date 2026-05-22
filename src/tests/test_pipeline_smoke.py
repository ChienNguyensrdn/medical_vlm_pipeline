"""Smoke test — TrustMedRAGPipeline end-to-end (no GPU, no real data required).

Run from src/ directory:
    python -m pytest tests/test_pipeline_smoke.py -v
    # or directly:
    python tests/test_pipeline_smoke.py

Tests:
    1. Module imports
    2. Grounding module (Stage 3)
    3. Knowledge Graph builder + encoder (Stage 5)
    4. Adaptive Fusion module (Stage 6)
    5. Uncertainty estimator (Stage 7)
    6. Clinical Reasoning Agent (Stage 8)
    7. TrustMedRAGPipeline.diagnose() full 10-stage forward pass
"""

import sys
import logging
from pathlib import Path

# Allow running from any CWD — resolve to absolute path first
_SRC_DIR = str(Path(__file__).resolve().parent.parent)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import torch
import pytest

logging.basicConfig(level=logging.WARNING)   # suppress info during tests

DIM = 64          # small dim for speed
BATCH = 1
NUM_CLASSES = 3


# ───────────────────────────────────────────────────────────────────────────
# Fixtures
# ───────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def dummy_image() -> torch.Tensor:
    """(1, 3, 224, 224) random image tensor."""
    return torch.randn(BATCH, 3, 224, 224)


@pytest.fixture(scope="module")
def dummy_visual_tokens() -> torch.Tensor:
    """(1, 4, DIM) simulated visual token sequence."""
    return torch.randn(BATCH, 4, DIM)


@pytest.fixture(scope="module")
def dummy_embedding() -> torch.Tensor:
    """(1, DIM) simulated fused embedding."""
    return torch.randn(BATCH, DIM)


# ───────────────────────────────────────────────────────────────────────────
# Test 1 — Module imports
# ───────────────────────────────────────────────────────────────────────────

def test_imports():
    from medical_vlm_pipeline import (
        TrustMedRAGPipeline,
        LesionAnatomyGrounder,
        DynamicKGBuilder,
        MedicalKGEncoder,
        AdaptiveMultimodalFusion,
        UncertaintyEstimator,
        ClinicalReasoningAgent,
    )
    assert TrustMedRAGPipeline is not None


# ───────────────────────────────────────────────────────────────────────────
# Test 2 — Stage 3: Grounding
# ───────────────────────────────────────────────────────────────────────────

def test_grounding_module(dummy_visual_tokens):
    from medical_vlm_pipeline.grounding import LesionAnatomyGrounder, extract_pathology_phrases

    grounder = LesionAnatomyGrounder(feature_dim=DIM, num_anatomy_regions=5)
    clinical_text = "Right lower lobe opacity with fever and cough."

    # Test phrase extraction
    phrases = extract_pathology_phrases(clinical_text)
    assert len(phrases) > 0, "Should extract at least one pathology phrase"

    # Test grounding forward
    result = grounder(dummy_visual_tokens, clinical_text)
    assert result.global_grounding_score >= 0.0
    assert result.global_grounding_score <= 1.0
    assert len(result.phrase_to_region) > 0
    assert len(result.region_attention) > 0
    print(f"  Grounding score: {result.global_grounding_score:.3f}")
    print(f"  Phrases: {list(result.phrase_to_region.keys())}")


# ───────────────────────────────────────────────────────────────────────────
# Test 3 — Stage 5: Knowledge Graph
# ───────────────────────────────────────────────────────────────────────────

def test_knowledge_graph(dummy_embedding):
    from medical_vlm_pipeline.knowledge_graph import DynamicKGBuilder, MedicalKGEncoder
    from medical_vlm_pipeline.grounding import extract_pathology_phrases

    builder = DynamicKGBuilder(include_base_ontology=True)
    encoder = MedicalKGEncoder(embedding_dim=DIM, hidden_dim=DIM // 2)

    # Build graph
    graph = builder.build(
        clinical_text="fever cough right lower lobe opacity",
        phrase_to_region={"right lower lobe opacity": "right_lower_lobe"},
        phrase_scores={"right lower lobe opacity": 0.78},
        retrieved_cases=[],
    )

    assert graph.num_nodes > 0, "Graph should have nodes"
    assert graph.num_edges > 0, "Graph should have edges"
    print(f"  KG: {graph.summary()}")

    # Encode
    Z_g = encoder(graph, device=torch.device("cpu"))
    assert Z_g.shape == (1, DIM), f"Expected (1, {DIM}), got {Z_g.shape}"


# ───────────────────────────────────────────────────────────────────────────
# Test 4 — Stage 6: Fusion
# ───────────────────────────────────────────────────────────────────────────

def test_fusion_module(dummy_embedding):
    from medical_vlm_pipeline.fusion import AdaptiveMultimodalFusion

    fusion = AdaptiveMultimodalFusion(dim=DIM, num_heads=4)

    # With all streams
    out = fusion(
        Z_v=dummy_embedding,
        Z_t=torch.randn(BATCH, DIM),
        Z_r=torch.randn(BATCH, DIM),
        Z_g=torch.randn(BATCH, DIM),
    )
    assert out.fused.shape == (BATCH, DIM)
    assert out.gate_weights.shape == (BATCH, 4)
    weights_sum = out.gate_weights.sum(dim=-1)
    assert torch.allclose(weights_sum, torch.ones(BATCH), atol=1e-4), \
        "Gate weights should sum to 1"

    # With missing streams (Z_t, Z_r = None)
    out_partial = fusion(Z_v=dummy_embedding, Z_t=None, Z_r=None, Z_g=None)
    assert out_partial.fused.shape == (BATCH, DIM)

    print(f"  Gate weights: {out.gate_weights[0].tolist()}")


# ───────────────────────────────────────────────────────────────────────────
# Test 5 — Stage 7: Uncertainty
# ───────────────────────────────────────────────────────────────────────────

def test_uncertainty_estimator(dummy_embedding):
    from medical_vlm_pipeline.uncertainty import UncertaintyEstimator

    estimator = UncertaintyEstimator(
        embedding_dim=DIM,
        num_classes=NUM_CLASSES,
        tau1=0.35,
        tau2=0.65,
        mc_samples=5,   # small for speed
    )

    out = estimator(
        Z_f=dummy_embedding,
        phrase_scores={"opacity": 0.8, "effusion": 0.5},
        retrieval_scores={"case_001": 0.9, "case_002": 0.6},
        graph_num_nodes=18,
    )

    assert 0.0 <= out.U_global <= 1.0
    assert out.decision in ("confident", "cautious", "defer")
    assert 0.0 <= out.aleatoric <= 1.0
    assert 0.0 <= out.epistemic <= 1.0
    assert len(out.U_finding) == 2
    print(f"  U_global={out.U_global:.3f} decision={out.decision}")
    print(f"  Statement: {out.uncertainty_statement()[:80]}...")


# ───────────────────────────────────────────────────────────────────────────
# Test 6 — Stage 8: Clinical Reasoning Agent
# ───────────────────────────────────────────────────────────────────────────

def test_reasoning_agent():
    from medical_vlm_pipeline.reasoning import ClinicalReasoningAgent
    from medical_vlm_pipeline.grounding import GroundingResult
    from medical_vlm_pipeline.knowledge_graph import DynamicKGBuilder
    from medical_vlm_pipeline.uncertainty import UncertaintyEstimator
    from medical_vlm_pipeline.retrieval import RetrievalResult

    # Minimal mocks
    grounding = GroundingResult(
        phrase_to_region={"right lower lobe opacity": "right_lower_lobe"},
        phrase_scores={"right lower lobe opacity": 0.78},
        region_attention={},
        global_grounding_score=0.78,
    )

    retrieved = [
        RetrievalResult(
            case_id="case_001",
            score=0.92,
            report_text="Patchy opacity in right lower lobe consistent with pneumonia.",
            label="Pneumonia",
        ),
        RetrievalResult(
            case_id="case_002",
            score=0.75,
            report_text="Right lower lobe infiltrate. Possible atelectasis.",
            label="Atelectasis",
        ),
    ]

    graph = DynamicKGBuilder().build(
        clinical_text="fever cough right lower lobe opacity",
        phrase_to_region=grounding.phrase_to_region,
        phrase_scores=grounding.phrase_scores,
        retrieved_cases=retrieved,
    )

    estimator = UncertaintyEstimator(embedding_dim=DIM, num_classes=NUM_CLASSES, mc_samples=5)
    uncertainty = estimator(torch.randn(1, DIM))

    agent = ClinicalReasoningAgent()
    trace = agent.reason(
        grounding=grounding,
        retrieved_cases=retrieved,
        graph=graph,
        uncertainty=uncertainty,
        clinical_text="fever cough right lower lobe opacity",
    )

    assert len(trace.findings) > 0
    assert trace.primary_hypothesis is not None
    assert trace.recommendation != ""
    assert isinstance(trace.contradiction.has_contradiction, bool)

    print(f"  Primary hypothesis: {trace.primary_hypothesis.diagnosis}")
    print(f"  Contradiction: {trace.contradiction.has_contradiction}")
    print(f"  Recommendation: {trace.recommendation[:80]}")

    # Check prompt context serialisation
    prompt = trace.to_prompt_context()
    assert len(prompt) > 50


# ───────────────────────────────────────────────────────────────────────────
# Test 7 — Full TrustMedRAGPipeline.diagnose()
# ───────────────────────────────────────────────────────────────────────────

def test_full_pipeline_forward(dummy_image):
    """End-to-end smoke test: TrustMedRAGPipeline.diagnose() must complete
    without exceptions and return a non-empty structured TrustMedOutput."""
    from medical_vlm_pipeline.pipeline import TrustMedRAGPipeline
    from medical_vlm_pipeline.config import PipelineConfig, EncoderConfig

    # Small config for CPU smoke test
    cfg = PipelineConfig()
    cfg.encoders = EncoderConfig(
        image_encoder="resnet18",    # lightweight for test
        embedding_dim=512,
        projection_dim=DIM,
    )

    pipeline = TrustMedRAGPipeline(cfg, num_classes=NUM_CLASSES)
    pipeline.eval()

    clinical_text = "Fever, cough, shortness of breath. Right lower lobe opacity suspected."

    with torch.no_grad():
        out = pipeline.diagnose(dummy_image, clinical_text=clinical_text)

    # Verify all stage outputs populated
    assert out.Z_v is not None and out.Z_v.shape[-1] == DIM
    assert out.grounding is not None
    assert out.graph is not None
    assert out.fusion is not None
    assert out.uncertainty is not None
    assert out.reasoning_trace is not None
    assert len(out.report) > 0

    # Latency tracking populated for all stages
    for stage in [
        "stage2_encoding", "stage3_grounding", "stage4_retrieval",
        "stage5_knowledge_graph", "stage6_fusion",
        "stage7_uncertainty", "stage8_reasoning", "stage9_generation",
    ]:
        assert stage in out.step_latencies_ms, f"Missing latency for {stage}"

    print("\n" + out.summary())


# ───────────────────────────────────────────────────────────────────────────
# Run directly
# ───────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("TrustMed-RAG Pipeline Smoke Test")
    print("=" * 60)

    img = torch.randn(1, 3, 224, 224)
    vtok = torch.randn(1, 4, DIM)
    emb = torch.randn(1, DIM)

    tests = [
        ("Test 1 — Imports",        lambda: test_imports()),
        ("Test 2 — Grounding",      lambda: test_grounding_module(vtok)),
        ("Test 3 — Knowledge Graph",lambda: test_knowledge_graph(emb)),
        ("Test 4 — Fusion",         lambda: test_fusion_module(emb)),
        ("Test 5 — Uncertainty",    lambda: test_uncertainty_estimator(emb)),
        ("Test 6 — Reasoning",      lambda: test_reasoning_agent()),
        ("Test 7 — Full Pipeline",  lambda: test_full_pipeline_forward(img)),
    ]

    passed = failed = 0
    for name, fn in tests:
        print(f"\n{'─'*50}")
        print(f"Running: {name}")
        try:
            fn()
            print(f"  ✅ PASSED")
            passed += 1
        except Exception as e:
            print(f"  ❌ FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests.")
    print("=" * 60)
    sys.exit(0 if failed == 0 else 1)
