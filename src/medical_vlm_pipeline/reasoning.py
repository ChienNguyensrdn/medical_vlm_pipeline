"""Stage 8 — Clinical Reasoning Agent.

Executes structured clinical reasoning before report generation:

    Step 1: Detect candidate findings (from grounding output)
    Step 2: Locate findings in anatomy regions
    Step 3: Retrieve similar cases (already done in Stage 4)
    Step 4: Query knowledge graph
    Step 5: Check contradiction
    Step 6: Estimate uncertainty (from Stage 7)
    Step 7: Form diagnostic hypothesis
    Step 8: Prepare report evidence

Output: ReasoningTrace — a structured object consumed by Stage 9 (Report Generator).
"""

import logging
from dataclasses import dataclass, field
from typing import Any

from .grounding import GroundingResult
from .knowledge_graph import MedicalKnowledgeGraph, KGNode
from .uncertainty import UncertaintyOutput

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reasoning data containers
# ---------------------------------------------------------------------------

@dataclass
class CandidateFinding:
    phrase: str
    anatomy_region: str
    grounding_score: float
    supported_by_retrieval: bool = False
    supported_by_graph: bool = False


@dataclass
class DiagnosticHypothesis:
    diagnosis: str
    confidence: float
    supporting_findings: list[str] = field(default_factory=list)
    supporting_cases: list[str] = field(default_factory=list)
    graph_relations: list[str] = field(default_factory=list)
    differential: list[str] = field(default_factory=list)


@dataclass
class ContradictionReport:
    has_contradiction: bool = False
    details: list[str] = field(default_factory=list)


@dataclass
class ReasoningTrace:
    """Full structured reasoning trace produced by the Clinical Reasoning Agent."""

    # Step 1-2: Findings + locations
    findings: list[CandidateFinding] = field(default_factory=list)

    # Step 3: Retrieved case summary
    num_retrieved: int = 0
    retrieved_case_ids: list[str] = field(default_factory=list)
    retrieved_diagnoses: list[str] = field(default_factory=list)

    # Step 4: Graph relations
    graph_relations: list[str] = field(default_factory=list)
    graph_size: int = 0

    # Step 5: Contradiction check
    contradiction: ContradictionReport = field(default_factory=ContradictionReport)

    # Step 6: Uncertainty
    uncertainty: UncertaintyOutput | None = None

    # Step 7: Hypotheses
    primary_hypothesis: DiagnosticHypothesis | None = None
    differential_hypotheses: list[DiagnosticHypothesis] = field(default_factory=list)

    # Step 8: Recommendation
    recommendation: str = ""

    def to_prompt_context(self) -> str:
        """Serialise the trace to a structured text prompt for the report generator."""
        lines = []

        # Findings
        lines.append("=== CLINICAL REASONING TRACE ===")
        lines.append(f"\n[FINDINGS] ({len(self.findings)} detected)")
        for f in self.findings:
            support_tags = []
            if f.supported_by_retrieval:
                support_tags.append("retrieval-supported")
            if f.supported_by_graph:
                support_tags.append("graph-supported")
            tag_str = f" [{', '.join(support_tags)}]" if support_tags else ""
            lines.append(
                f"  • {f.phrase} → {f.anatomy_region} "
                f"(grounding={f.grounding_score:.2f}){tag_str}"
            )

        # Retrieved evidence
        lines.append(f"\n[RETRIEVED EVIDENCE] ({self.num_retrieved} cases)")
        if self.retrieved_diagnoses:
            from collections import Counter
            diag_counts = Counter(self.retrieved_diagnoses)
            for diag, cnt in diag_counts.most_common():
                lines.append(f"  • {diag}: {cnt}/{self.num_retrieved} cases")

        # Graph
        lines.append(f"\n[KNOWLEDGE GRAPH] ({self.graph_size} nodes)")
        for rel in self.graph_relations[:5]:
            lines.append(f"  • {rel}")

        # Contradiction
        lines.append("\n[CONTRADICTION CHECK]")
        if self.contradiction.has_contradiction:
            for detail in self.contradiction.details:
                lines.append(f"  ⚠ {detail}")
        else:
            lines.append("  ✓ No contradictions detected.")

        # Uncertainty
        if self.uncertainty:
            lines.append(f"\n[UNCERTAINTY] global={self.uncertainty.U_global:.2f} "
                         f"decision={self.uncertainty.decision.upper()}")
            lines.append(f"  {self.uncertainty.uncertainty_statement()}")

        # Hypothesis
        if self.primary_hypothesis:
            h = self.primary_hypothesis
            lines.append(f"\n[PRIMARY HYPOTHESIS] {h.diagnosis} "
                         f"(confidence={h.confidence:.2f})")
            if h.differential:
                lines.append(f"  Differentials: {', '.join(h.differential)}")

        # Recommendation
        if self.recommendation:
            lines.append(f"\n[RECOMMENDATION] {self.recommendation}")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Hypothesis generator
# ---------------------------------------------------------------------------

# Rule-based diagnosis scoring table (finding phrase → diagnosis votes)
FINDING_TO_DIAGNOSIS: dict[str, list[tuple[str, float]]] = {
    "opacity":            [("Pneumonia", 0.7), ("Atelectasis", 0.4), ("Lung Cancer", 0.2)],
    "consolidation":      [("Pneumonia", 0.8), ("Atelectasis", 0.3)],
    "pleural effusion":   [("Congestive Heart Failure", 0.6), ("Pneumonia", 0.3)],
    "cardiomegaly":       [("Congestive Heart Failure", 0.8)],
    "nodule":             [("Lung Cancer", 0.5), ("Granuloma", 0.4)],
    "pneumothorax":       [("Pneumothorax", 1.0)],
    "atelectasis":        [("Atelectasis", 0.9)],
    "effusion":           [("Congestive Heart Failure", 0.6), ("Pneumonia", 0.3)],
}

SYMPTOM_TO_DIAGNOSIS: dict[str, list[tuple[str, float]]] = {
    "fever":              [("Pneumonia", 0.6)],
    "cough":              [("Pneumonia", 0.4)],
    "dyspnea":            [("Congestive Heart Failure", 0.4), ("Pneumonia", 0.3)],
    "shortness of breath": [("Congestive Heart Failure", 0.5)],
}


def _score_hypotheses(
    findings: list[CandidateFinding],
    clinical_text: str,
    retrieved_diagnoses: list[str],
) -> list[DiagnosticHypothesis]:
    """Generate and rank diagnostic hypotheses from findings + context."""
    from collections import defaultdict

    scores: dict[str, float] = defaultdict(float)
    supporting: dict[str, list[str]] = defaultdict(list)

    # Score from findings
    for f in findings:
        phrase_lower = f.phrase.lower()
        for kw, diag_votes in FINDING_TO_DIAGNOSIS.items():
            if kw in phrase_lower:
                for diag, vote in diag_votes:
                    w = vote * f.grounding_score
                    if f.supported_by_graph:
                        w *= 1.2
                    if f.supported_by_retrieval:
                        w *= 1.1
                    scores[diag] += w
                    supporting[diag].append(f.phrase)

    # Score from clinical context
    text_lower = clinical_text.lower()
    for symptom, diag_votes in SYMPTOM_TO_DIAGNOSIS.items():
        if symptom in text_lower:
            for diag, vote in diag_votes:
                scores[diag] += vote * 0.5

    # Score from retrieved diagnoses
    from collections import Counter
    diag_counter = Counter(retrieved_diagnoses)
    total_retrieved = len(retrieved_diagnoses) or 1
    for diag, cnt in diag_counter.items():
        retrieval_boost = (cnt / total_retrieved) * 0.8
        # Map retrieved label to canonical diagnosis name
        for canonical in list(scores.keys()) + list(FINDING_TO_DIAGNOSIS.keys()):
            if canonical.lower() in diag.lower() or diag.lower() in canonical.lower():
                scores[canonical] += retrieval_boost
                break
        else:
            scores[diag] += retrieval_boost

    if not scores:
        # Default normal hypothesis
        return [DiagnosticHypothesis(
            diagnosis="Normal / No acute finding",
            confidence=0.5,
        )]

    # Normalise scores to [0, 1]
    max_score = max(scores.values()) or 1.0
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    hypotheses = []
    for i, (diag, raw_score) in enumerate(ranked[:3]):
        conf = min(raw_score / max_score, 0.99)
        differentials = [d for d, _ in ranked if d != diag][:2]
        hypotheses.append(DiagnosticHypothesis(
            diagnosis=diag,
            confidence=conf,
            supporting_findings=supporting.get(diag, []),
            differential=differentials,
        ))

    return hypotheses


# ---------------------------------------------------------------------------
# Contradiction checker
# ---------------------------------------------------------------------------

# Mutually exclusive finding pairs
CONTRADICTORY_PAIRS: list[tuple[str, str]] = [
    ("no pleural effusion", "pleural effusion"),
    ("no pneumothorax", "pneumothorax"),
    ("no cardiomegaly", "cardiomegaly"),
    ("normal", "opacity"),
    ("left lung", "right lung"),
]


def check_contradictions(findings: list[CandidateFinding]) -> ContradictionReport:
    """Check for logical contradictions among detected findings."""
    phrases = [f.phrase.lower() for f in findings]
    details: list[str] = []

    for neg, pos in CONTRADICTORY_PAIRS:
        neg_found = any(neg in p for p in phrases)
        pos_found = any(pos in p for p in phrases)
        if neg_found and pos_found:
            details.append(
                f"Contradiction: '{neg}' and '{pos}' both present in findings."
            )

    # Check grounding-location mismatch (left vs right)
    for f in findings:
        phrase_lower = f.phrase.lower()
        region_lower = f.anatomy_region.lower()
        if "left" in phrase_lower and "right" in region_lower:
            details.append(
                f"Location mismatch: phrase '{f.phrase}' mentions LEFT but "
                f"grounded to RIGHT region '{f.anatomy_region}'."
            )
        if "right" in phrase_lower and "left" in region_lower:
            details.append(
                f"Location mismatch: phrase '{f.phrase}' mentions RIGHT but "
                f"grounded to LEFT region '{f.anatomy_region}'."
            )

    return ContradictionReport(
        has_contradiction=bool(details),
        details=details,
    )


# ---------------------------------------------------------------------------
# Main Clinical Reasoning Agent
# ---------------------------------------------------------------------------

class ClinicalReasoningAgent:
    """Stage 8 — Clinical Reasoning Agent.

    Executes 8-step structured reasoning to produce a ``ReasoningTrace``
    consumed by the report generator (Stage 9).

    Args:
        min_grounding_score: Minimum grounding score to consider a finding valid.
        min_retrieval_support_score: Score threshold for retrieval support flag.
    """

    def __init__(
        self,
        min_grounding_score: float = 0.3,
        min_retrieval_support_score: float = 0.5,
    ) -> None:
        self.min_grounding_score = min_grounding_score
        self.min_retrieval_support_score = min_retrieval_support_score

    def _step1_2_detect_findings(
        self,
        grounding: GroundingResult,
        retrieved_cases: list[Any],
        graph: MedicalKnowledgeGraph,
    ) -> list[CandidateFinding]:
        """Steps 1-2: Detect candidate findings and locate them anatomically."""
        findings: list[CandidateFinding] = []

        # Gather retrieval-mentioned pathologies for cross-support check
        retrieval_pathologies: set[str] = set()
        for case in retrieved_cases:
            report = getattr(case, "report_text", "") or ""
            retrieval_pathologies.update(report.lower().split())

        # Gather graph-known pathology node ids
        graph_pathology_ids: set[str] = {
            n.node_id for n in graph.nodes if n.node_type == "pathology"
        }

        for phrase, region in grounding.phrase_to_region.items():
            score = grounding.phrase_scores.get(phrase, 0.0)

            if score < self.min_grounding_score:
                logger.debug(f"Skipping low-score finding: '{phrase}' ({score:.2f})")
                continue

            # Check retrieval support
            retrieval_support = any(
                kw in phrase.lower() for kw in retrieval_pathologies
            )

            # Check graph support (any pathology node matches phrase)
            graph_support = any(
                pid in phrase.lower() or phrase.lower() in pid
                for pid in graph_pathology_ids
            )

            findings.append(CandidateFinding(
                phrase=phrase,
                anatomy_region=region,
                grounding_score=score,
                supported_by_retrieval=retrieval_support,
                supported_by_graph=graph_support,
            ))

        logger.info(f"Step 1-2: {len(findings)} candidate findings extracted.")
        return findings

    def _step4_query_graph(self, graph: MedicalKnowledgeGraph) -> list[str]:
        """Step 4: Extract human-readable relations from the KG."""
        relations: list[str] = []
        node_labels = {n.node_id: n.label for n in graph.nodes}

        for edge in graph.edges:
            if edge.source in ("detected", "retrieved", "inferred"):
                src_label = node_labels.get(edge.src_id, edge.src_id)
                dst_label = node_labels.get(edge.dst_id, edge.dst_id)
                relations.append(
                    f"{src_label} --[{edge.relation}]--> {dst_label} "
                    f"(w={edge.weight:.2f})"
                )

        return relations[:10]  # top 10 relations

    def _step8_recommendation(
        self,
        primary: DiagnosticHypothesis | None,
        uncertainty: UncertaintyOutput,
        has_contradiction: bool,
    ) -> str:
        """Step 8: Generate clinical recommendation string."""
        parts: list[str] = []

        if has_contradiction:
            parts.append(
                "Findings contain potential contradictions requiring careful review."
            )

        if uncertainty.decision == "defer":
            parts.append(
                "Automated analysis is inconclusive. "
                "Urgent radiologist review is recommended."
            )
        elif uncertainty.decision == "cautious":
            parts.append(
                "Clinical correlation with laboratory results and patient history is advised."
            )
        else:
            parts.append("Findings appear consistent with automated analysis.")

        if primary and primary.diagnosis not in ("Normal / No acute finding", "Normal"):
            parts.append(
                f"Consider follow-up imaging if {primary.diagnosis.lower()} "
                "is clinically suspected."
            )

        return " ".join(parts)

    def reason(
        self,
        grounding: GroundingResult,
        retrieved_cases: list[Any],
        graph: MedicalKnowledgeGraph,
        uncertainty: UncertaintyOutput,
        clinical_text: str = "",
    ) -> ReasoningTrace:
        """Execute full 8-step clinical reasoning pipeline.

        Args:
            grounding:       Phrase-region grounding map from Stage 3.
            retrieved_cases: List of RetrievalResult from Stage 4.
            graph:           Dynamic knowledge graph from Stage 5.
            uncertainty:     Uncertainty estimates from Stage 7.
            clinical_text:   Raw clinical note for symptom extraction.

        Returns:
            ReasoningTrace structured evidence object.
        """
        trace = ReasoningTrace()

        # ── Step 1-2: Detect findings and locate them ─────────────────────
        trace.findings = self._step1_2_detect_findings(
            grounding, retrieved_cases, graph
        )

        # ── Step 3: Summarise retrieved evidence ──────────────────────────
        trace.num_retrieved = len(retrieved_cases)
        for case in retrieved_cases:
            cid = getattr(case, "case_id", None) or "unknown"
            lbl = getattr(case, "label", None) or ""
            trace.retrieved_case_ids.append(str(cid))
            if lbl:
                trace.retrieved_diagnoses.append(str(lbl))

        # ── Step 4: Query knowledge graph ─────────────────────────────────
        trace.graph_relations = self._step4_query_graph(graph)
        trace.graph_size = graph.num_nodes

        # ── Step 5: Check contradictions ──────────────────────────────────
        trace.contradiction = check_contradictions(trace.findings)
        if trace.contradiction.has_contradiction:
            logger.warning(
                f"Contradiction detected: {trace.contradiction.details}"
            )

        # ── Step 6: Attach uncertainty ────────────────────────────────────
        trace.uncertainty = uncertainty

        # ── Step 7: Form diagnostic hypotheses ───────────────────────────
        hypotheses = _score_hypotheses(
            trace.findings,
            clinical_text,
            trace.retrieved_diagnoses,
        )
        if hypotheses:
            trace.primary_hypothesis = hypotheses[0]
            trace.differential_hypotheses = hypotheses[1:]

        # ── Step 8: Generate recommendation ──────────────────────────────
        trace.recommendation = self._step8_recommendation(
            trace.primary_hypothesis,
            uncertainty,
            trace.contradiction.has_contradiction,
        )

        logger.info(
            f"Reasoning complete — primary: "
            f"{trace.primary_hypothesis.diagnosis if trace.primary_hypothesis else 'None'}, "
            f"findings: {len(trace.findings)}, "
            f"decision: {uncertainty.decision}"
        )

        return trace
