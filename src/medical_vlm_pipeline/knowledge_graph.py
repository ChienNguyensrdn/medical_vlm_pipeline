"""Stage 5 — Dynamic Medical Knowledge Graph Construction.

Builds a case-specific knowledge graph combining:
- Base medical ontology (RadGraph / UMLS nodes)
- Detected anatomy + pathology findings
- Retrieved evidence from similar cases

Graph structure:
    V = V_base ∪ V_detected ∪ V_retrieved ∪ V_context
    E = E_base ∪ E_inferred ∪ E_retrieved

Encoders:
    GATConv (Graph Attention Network) — learns edge-type-aware node representations.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Graph data containers
# ---------------------------------------------------------------------------

@dataclass
class KGNode:
    node_id: str
    node_type: str   # anatomy | pathology | symptom | diagnosis | treatment | outcome | uncertainty
    label: str
    confidence: float = 1.0
    source: str = "base"    # base | detected | retrieved | context


@dataclass
class KGEdge:
    src_id: str
    dst_id: str
    relation: str   # located_in | associated_with | indicates | causes | excludes | suggests | progresses_to | treated_by | similar_to
    weight: float = 1.0
    source: str = "base"


@dataclass
class MedicalKnowledgeGraph:
    nodes: list[KGNode] = field(default_factory=list)
    edges: list[KGEdge] = field(default_factory=list)

    def add_node(self, node: KGNode) -> None:
        if not any(n.node_id == node.node_id for n in self.nodes):
            self.nodes.append(node)

    def add_edge(self, edge: KGEdge) -> None:
        self.edges.append(edge)

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def num_edges(self) -> int:
        return len(self.edges)

    def node_index(self) -> dict[str, int]:
        return {n.node_id: i for i, n in enumerate(self.nodes)}

    def summary(self) -> str:
        return (
            f"KG: {self.num_nodes} nodes, {self.num_edges} edges | "
            f"types: {set(n.node_type for n in self.nodes)}"
        )


# ---------------------------------------------------------------------------
# Base medical ontology (minimal seed — expand from RadGraph/UMLS in prod)
# ---------------------------------------------------------------------------

BASE_ONTOLOGY_NODES: list[dict[str, str]] = [
    # Anatomy
    {"id": "right_upper_lobe",    "type": "anatomy",    "label": "Right Upper Lobe"},
    {"id": "right_lower_lobe",    "type": "anatomy",    "label": "Right Lower Lobe"},
    {"id": "left_upper_lobe",     "type": "anatomy",    "label": "Left Upper Lobe"},
    {"id": "left_lower_lobe",     "type": "anatomy",    "label": "Left Lower Lobe"},
    {"id": "pleural_space",       "type": "anatomy",    "label": "Pleural Space"},
    {"id": "cardiac_silhouette",  "type": "anatomy",    "label": "Cardiac Silhouette"},
    {"id": "mediastinum",         "type": "anatomy",    "label": "Mediastinum"},
    # Pathology
    {"id": "opacity",             "type": "pathology",  "label": "Opacity / Consolidation"},
    {"id": "pleural_effusion",    "type": "pathology",  "label": "Pleural Effusion"},
    {"id": "cardiomegaly",        "type": "pathology",  "label": "Cardiomegaly"},
    {"id": "pneumothorax",        "type": "pathology",  "label": "Pneumothorax"},
    {"id": "nodule",              "type": "pathology",  "label": "Pulmonary Nodule"},
    {"id": "atelectasis",         "type": "pathology",  "label": "Atelectasis"},
    # Diagnoses
    {"id": "pneumonia",           "type": "diagnosis",  "label": "Pneumonia"},
    {"id": "heart_failure",       "type": "diagnosis",  "label": "Congestive Heart Failure"},
    {"id": "lung_cancer",         "type": "diagnosis",  "label": "Lung Cancer"},
    {"id": "normal",              "type": "diagnosis",  "label": "Normal"},
    # Symptoms
    {"id": "fever",               "type": "symptom",    "label": "Fever"},
    {"id": "cough",               "type": "symptom",    "label": "Cough"},
    {"id": "dyspnea",             "type": "symptom",    "label": "Dyspnea"},
]

BASE_ONTOLOGY_EDGES: list[dict[str, str]] = [
    {"src": "opacity",          "dst": "right_lower_lobe", "rel": "located_in"},
    {"src": "opacity",          "dst": "pneumonia",        "rel": "associated_with"},
    {"src": "opacity",          "dst": "atelectasis",      "rel": "associated_with"},
    {"src": "pleural_effusion", "dst": "pleural_space",    "rel": "located_in"},
    {"src": "pleural_effusion", "dst": "heart_failure",    "rel": "indicates"},
    {"src": "cardiomegaly",     "dst": "cardiac_silhouette","rel": "located_in"},
    {"src": "cardiomegaly",     "dst": "heart_failure",    "rel": "indicates"},
    {"src": "fever",            "dst": "pneumonia",        "rel": "suggests"},
    {"src": "cough",            "dst": "pneumonia",        "rel": "suggests"},
    {"src": "dyspnea",          "dst": "heart_failure",    "rel": "suggests"},
    {"src": "nodule",           "dst": "lung_cancer",      "rel": "indicates"},
    {"src": "pneumothorax",     "dst": "pleural_space",    "rel": "located_in"},
]

# Keyword → node_id mapping for auto-linking detected phrases
KEYWORD_TO_NODE: dict[str, str] = {
    "opacity": "opacity", "consolidation": "opacity", "infiltrate": "opacity",
    "effusion": "pleural_effusion", "pleural effusion": "pleural_effusion",
    "cardiomegaly": "cardiomegaly", "cardiac": "cardiomegaly",
    "pneumothorax": "pneumothorax",
    "nodule": "nodule", "mass": "nodule",
    "atelectasis": "atelectasis",
    "fever": "fever", "cough": "cough", "dyspnea": "dyspnea", "shortness of breath": "dyspnea",
    "right lower": "right_lower_lobe", "right upper": "right_upper_lobe",
    "left lower": "left_lower_lobe", "left upper": "left_upper_lobe",
}


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

class DynamicKGBuilder:
    """Constructs a case-specific knowledge graph at inference time.

    Combines:
    1. Static base ontology (anatomy, pathology, diagnosis nodes).
    2. Detected nodes from grounding output (phrases + anatomy regions).
    3. Retrieved evidence from similar cases.
    4. Clinical context keywords (symptoms, history).

    Args:
        include_base_ontology: Whether to seed the graph with the base medical KG.
    """

    def __init__(self, include_base_ontology: bool = True) -> None:
        self.include_base_ontology = include_base_ontology

    def _seed_base_ontology(self, graph: MedicalKnowledgeGraph) -> None:
        """Populate graph with base medical ontology nodes and edges."""
        for nd in BASE_ONTOLOGY_NODES:
            graph.add_node(KGNode(
                node_id=nd["id"],
                node_type=nd["type"],
                label=nd["label"],
                source="base",
            ))
        for ed in BASE_ONTOLOGY_EDGES:
            graph.add_edge(KGEdge(
                src_id=ed["src"],
                dst_id=ed["dst"],
                relation=ed["rel"],
                source="base",
            ))

    def _add_detected_nodes(
        self,
        graph: MedicalKnowledgeGraph,
        phrase_to_region: dict[str, str],
        phrase_scores: dict[str, float],
    ) -> None:
        """Add detected pathology phrases and their grounded anatomy regions."""
        for phrase, region_id in phrase_to_region.items():
            score = phrase_scores.get(phrase, 0.5)
            phrase_kw = phrase.lower()

            # Map phrase to ontology node if possible
            matched_node_id = None
            for kw, nid in KEYWORD_TO_NODE.items():
                if kw in phrase_kw:
                    matched_node_id = nid
                    break

            if matched_node_id is None:
                # Create a new custom detected node
                matched_node_id = f"detected_{phrase[:20].replace(' ', '_')}"
                graph.add_node(KGNode(
                    node_id=matched_node_id,
                    node_type="pathology",
                    label=phrase.title(),
                    confidence=score,
                    source="detected",
                ))

            # Link detected finding → anatomy region
            if any(n.node_id == region_id for n in graph.nodes):
                graph.add_edge(KGEdge(
                    src_id=matched_node_id,
                    dst_id=region_id,
                    relation="located_in",
                    weight=score,
                    source="detected",
                ))

    def _add_retrieved_evidence(
        self,
        graph: MedicalKnowledgeGraph,
        retrieved_cases: list[Any],
    ) -> None:
        """Add retrieved similar cases as evidence nodes linked to matching diagnoses."""
        for i, case in enumerate(retrieved_cases[:5]):  # limit to top-5
            case_id = getattr(case, "case_id", None) or case.get("case_id", f"case_{i}")
            label = getattr(case, "label", None) or case.get("label", "unknown")
            score = getattr(case, "score", 0.5)
            report_text = getattr(case, "report_text", "") or ""

            # Add evidence node
            ev_node_id = f"evidence_{case_id}"
            graph.add_node(KGNode(
                node_id=ev_node_id,
                node_type="outcome",
                label=f"Retrieved Case {case_id} (label={label})",
                confidence=float(score),
                source="retrieved",
            ))

            # Link evidence to matching diagnosis node
            label_lower = str(label).lower()
            for diag_id in ["pneumonia", "heart_failure", "lung_cancer", "normal"]:
                if diag_id.replace("_", " ") in label_lower or diag_id in label_lower:
                    graph.add_edge(KGEdge(
                        src_id=ev_node_id,
                        dst_id=diag_id,
                        relation="similar_to",
                        weight=float(score),
                        source="retrieved",
                    ))
                    break

            # Extract phrases from retrieved report and link to matching ontology
            if report_text:
                for kw, nid in KEYWORD_TO_NODE.items():
                    if kw in report_text.lower():
                        graph.add_edge(KGEdge(
                            src_id=ev_node_id,
                            dst_id=nid,
                            relation="associated_with",
                            weight=float(score) * 0.8,
                            source="retrieved",
                        ))

    def _add_clinical_context(
        self,
        graph: MedicalKnowledgeGraph,
        clinical_text: str,
    ) -> None:
        """Link symptom keywords from the clinical note to symptom nodes."""
        text_lower = clinical_text.lower()
        for kw, nid in KEYWORD_TO_NODE.items():
            if kw in text_lower:
                if any(n.node_id == nid for n in graph.nodes):
                    # Add context activation edge (self-loop with increased weight)
                    graph.add_edge(KGEdge(
                        src_id=nid,
                        dst_id=nid,
                        relation="activated_by_context",
                        weight=1.2,
                        source="context",
                    ))

    def build(
        self,
        clinical_text: str = "",
        phrase_to_region: dict[str, str] | None = None,
        phrase_scores: dict[str, float] | None = None,
        retrieved_cases: list[Any] | None = None,
    ) -> MedicalKnowledgeGraph:
        """Construct the dynamic case-specific knowledge graph.

        Args:
            clinical_text:    Raw clinical note for symptom extraction.
            phrase_to_region: Grounding output — phrase → anatomy region map.
            phrase_scores:    Grounding output — phrase → confidence score.
            retrieved_cases:  List of RetrievalResult objects from Stage 4.

        Returns:
            MedicalKnowledgeGraph instance for this case.
        """
        graph = MedicalKnowledgeGraph()

        if self.include_base_ontology:
            self._seed_base_ontology(graph)

        if phrase_to_region and phrase_scores:
            self._add_detected_nodes(graph, phrase_to_region, phrase_scores)

        if retrieved_cases:
            self._add_retrieved_evidence(graph, retrieved_cases)

        if clinical_text:
            self._add_clinical_context(graph, clinical_text)

        logger.info(f"Dynamic KG built: {graph.summary()}")
        return graph


# ---------------------------------------------------------------------------
# Graph Attention Network (GAT) encoder
# ---------------------------------------------------------------------------

class GATLayer(nn.Module):
    """Single Graph Attention Network layer.

    Computes attention-weighted neighbourhood aggregation for all nodes.

    Args:
        in_dim:  Input node feature dimension.
        out_dim: Output node feature dimension.
        num_heads: Number of attention heads.
    """

    def __init__(self, in_dim: int, out_dim: int, num_heads: int = 4) -> None:
        super().__init__()
        assert out_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads

        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.attn = nn.Linear(2 * self.head_dim, 1, bias=False)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x: Tensor, adj: Tensor) -> Tensor:
        """
        Args:
            x:   (N, in_dim) node features.
            adj: (N, N) adjacency matrix (weighted or binary).

        Returns:
            (N, out_dim) updated node embeddings.
        """
        N = x.shape[0]
        H = self.W(x)  # (N, out_dim)
        H_heads = H.view(N, self.num_heads, self.head_dim)  # (N, h, d)

        # Pairwise attention logits
        H_i = H_heads.unsqueeze(1).expand(N, N, self.num_heads, self.head_dim)
        H_j = H_heads.unsqueeze(0).expand(N, N, self.num_heads, self.head_dim)
        e = self.leaky_relu(
            self.attn(torch.cat([H_i, H_j], dim=-1)).squeeze(-1)
        )  # (N, N, h)

        # Mask non-edges
        mask = (adj > 0).float().unsqueeze(-1)  # (N, N, 1)
        e = e * mask + (1 - mask) * (-1e9)

        alpha = F.softmax(e, dim=1)  # (N, N, h)

        # Aggregate neighbours
        out = torch.einsum("ijh,jhd->ihd", alpha, H_heads)  # (N, h, d)
        return out.reshape(N, -1)  # (N, out_dim)


class MedicalKGEncoder(nn.Module):
    """Two-layer GAT encoder for the dynamic medical knowledge graph.

    Converts node embeddings from the graph into a single fused graph
    representation vector ``Z_g`` used by the fusion module (Stage 6).

    Args:
        node_types: Ordered list of all node type strings.
        embedding_dim: Input/output embedding dimension.
        hidden_dim: Internal GAT hidden dimension.
        num_heads: Attention heads in each GAT layer.
    """

    NODE_TYPES = ["anatomy", "pathology", "symptom", "diagnosis",
                  "treatment", "outcome", "uncertainty"]

    def __init__(
        self,
        embedding_dim: int = 256,
        hidden_dim: int = 128,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim

        # Node type embedding (one per node type category)
        self.type_embedding = nn.Embedding(len(self.NODE_TYPES), embedding_dim)

        # Two-layer GAT
        self.gat1 = GATLayer(embedding_dim, hidden_dim, num_heads)
        self.gat2 = GATLayer(hidden_dim, embedding_dim, num_heads)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)

        # Global pooling MLP
        self.pool_mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
            nn.LayerNorm(embedding_dim),
        )

    def _build_feature_matrix(
        self,
        graph: MedicalKnowledgeGraph,
        device: torch.device,
    ) -> tuple[Tensor, Tensor]:
        """Convert graph to (N, D) feature matrix and (N, N) adjacency matrix.

        Returns:
            x:   (N, D) node features.
            adj: (N, N) adjacency matrix.
        """
        N = graph.num_nodes
        node_idx = graph.node_index()

        # Node type → integer index
        type_to_int = {t: i for i, t in enumerate(self.NODE_TYPES)}

        type_ids = torch.zeros(N, dtype=torch.long, device=device)
        confidence_weights = torch.ones(N, device=device)
        for i, node in enumerate(graph.nodes):
            t_idx = type_to_int.get(node.node_type, 0)
            type_ids[i] = t_idx
            confidence_weights[i] = node.confidence

        # Initial node features = type embedding scaled by confidence
        x = self.type_embedding(type_ids) * confidence_weights.unsqueeze(-1)

        # Adjacency matrix (weighted by edge weight)
        adj = torch.zeros(N, N, device=device)
        for edge in graph.edges:
            si = node_idx.get(edge.src_id)
            di = node_idx.get(edge.dst_id)
            if si is not None and di is not None:
                adj[si, di] = edge.weight
                adj[di, si] = edge.weight  # undirected for aggregation

        return x, adj

    def forward(self, graph: MedicalKnowledgeGraph, device: torch.device | None = None) -> Tensor:
        """Encode the knowledge graph to a single embedding vector Z_g.

        Args:
            graph:  MedicalKnowledgeGraph to encode.
            device: Target computation device.

        Returns:
            Z_g: (1, embedding_dim) graph-level embedding.
        """
        if device is None:
            device = next(self.parameters()).device

        if graph.num_nodes == 0:
            logger.warning("Empty KG — returning zero graph embedding.")
            return torch.zeros(1, self.embedding_dim, device=device)

        x, adj = self._build_feature_matrix(graph, device)

        # GAT layers with residual connections
        h = self.norm1(self.gat1(x, adj))       # (N, hidden_dim)
        h = F.gelu(h)
        h_pad = F.pad(h, (0, self.embedding_dim - h.shape[-1]))  # match dim for residual
        z = self.norm2(self.gat2(h, adj))        # (N, embedding_dim)

        # Global mean pooling → graph-level representation
        z_g = z.mean(dim=0, keepdim=True)        # (1, embedding_dim)
        return self.pool_mlp(z_g)
