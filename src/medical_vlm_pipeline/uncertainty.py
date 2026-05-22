"""Stage 7 — Uncertainty Estimation.

Estimates two types of clinical uncertainty:
    Aleatoric  — inherent data noise (blurry image, ambiguous lesion)
    Epistemic  — model ignorance (rare disease, OOD scanner, unseen pattern)

Techniques implemented:
    - Monte Carlo Dropout (epistemic)
    - Evidential Deep Learning via Normal-Inverse-Gamma (aleatoric + epistemic)
    - Temperature Scaling calibration

Decision rule (from pipeline spec):
    U_global < τ1              → generate confident report
    τ1 ≤ U_global < τ2         → generate cautious report with uncertainty statement
    U_global ≥ τ2              → defer to human radiologist
"""

import logging
import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output containers
# ---------------------------------------------------------------------------

@dataclass
class UncertaintyOutput:
    """Multi-granularity uncertainty estimates for a single inference pass."""
    # Global report-level uncertainty [0, 1]
    U_global: float

    # Per-finding uncertainty (finding_phrase → score)
    U_finding: dict[str, float] = field(default_factory=dict)

    # Retrieval evidence uncertainty (case_id → score)
    U_retrieval: dict[str, float] = field(default_factory=dict)

    # Graph reasoning uncertainty [0, 1]
    U_graph: float = 0.0

    # Decomposed components
    aleatoric: float = 0.0
    epistemic: float = 0.0

    # Decision outcome
    decision: str = "confident"          # confident | cautious | defer
    tau1: float = 0.35
    tau2: float = 0.65

    def __post_init__(self) -> None:
        # Apply decision rule
        if self.U_global < self.tau1:
            self.decision = "confident"
        elif self.U_global < self.tau2:
            self.decision = "cautious"
        else:
            self.decision = "defer"

    def uncertainty_statement(self) -> str:
        """Generate a human-readable uncertainty statement for the report."""
        if self.decision == "confident":
            return (
                f"High diagnostic confidence (uncertainty={self.U_global:.2f}). "
                "Findings are well-supported by visual evidence."
            )
        elif self.decision == "cautious":
            return (
                f"Moderate uncertainty (score={self.U_global:.2f}). "
                "Findings should be correlated with clinical history and laboratory results. "
                f"Aleatoric: {self.aleatoric:.2f} | Epistemic: {self.epistemic:.2f}."
            )
        else:
            return (
                f"High uncertainty (score={self.U_global:.2f}). "
                "Automated analysis is inconclusive. "
                "Human radiologist review is strongly recommended."
            )


# ---------------------------------------------------------------------------
# Monte Carlo Dropout Estimator
# ---------------------------------------------------------------------------

class MCDropoutEstimator(nn.Module):
    """Epistemic uncertainty via Monte Carlo Dropout.

    Runs T stochastic forward passes with dropout active and measures
    prediction variance (entropy of the mean prediction distribution).

    Args:
        embedding_dim: Input feature dimension.
        num_classes:   Number of diagnostic classes.
        dropout_rate:  Dropout probability during MC sampling.
        num_samples:   Number of MC forward passes (T).
    """

    def __init__(
        self,
        embedding_dim: int = 256,
        num_classes: int = 3,
        dropout_rate: float = 0.3,
        num_samples: int = 20,
    ) -> None:
        super().__init__()
        self.num_samples = num_samples
        self.num_classes = num_classes

        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(embedding_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes),
        )

    def forward(self, z: Tensor, num_samples: int | None = None) -> dict[str, Tensor]:
        """Run MC Dropout sampling.

        Args:
            z:           (B, D) fused embedding.
            num_samples: Override default T if provided.

        Returns:
            Dict with keys:
                mean_probs  (B, C) — mean class probabilities
                variance    (B, C) — variance across MC samples
                epistemic   (B,)  — predictive entropy (epistemic proxy)
                pred_class  (B,)  — argmax of mean_probs
                confidence  (B,)  — max of mean_probs
        """
        T = num_samples or self.num_samples
        self.train()   # activate dropout

        sample_probs = []
        with torch.no_grad():
            for _ in range(T):
                logits = self.classifier(z)             # (B, C)
                probs = F.softmax(logits, dim=-1)       # (B, C)
                sample_probs.append(probs)

        self.eval()

        # Stack: (T, B, C)
        stacked = torch.stack(sample_probs, dim=0)
        mean_probs = stacked.mean(dim=0)               # (B, C)
        variance = stacked.var(dim=0)                  # (B, C)

        # Predictive entropy H[y|x] = -Σ p log p
        eps = 1e-8
        entropy = -(mean_probs * (mean_probs + eps).log()).sum(dim=-1)  # (B,)
        # Normalise to [0, 1] using max possible entropy log(C)
        max_entropy = math.log(self.num_classes + eps)
        epistemic = (entropy / max_entropy).clamp(0.0, 1.0)

        pred_class = mean_probs.argmax(dim=-1)
        confidence = mean_probs.max(dim=-1).values

        return {
            "mean_probs": mean_probs,
            "variance": variance,
            "epistemic": epistemic,
            "pred_class": pred_class,
            "confidence": confidence,
        }


# ---------------------------------------------------------------------------
# Evidential Deep Learning (Normal-Inverse-Gamma head)
# ---------------------------------------------------------------------------

class EvidentialHead(nn.Module):
    """Aleatoric + epistemic uncertainty via Evidential Deep Learning.

    Outputs parameters (γ, ν, α, β) of a Normal-Inverse-Gamma distribution.
    Aleatoric uncertainty ≈ β / (α − 1).
    Epistemic uncertainty ≈ 1 / ν.

    Args:
        embedding_dim: Input dimension.
        num_classes:   Number of output classes.
    """

    def __init__(self, embedding_dim: int = 256, num_classes: int = 3) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.evidence_head = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.GELU(),
            nn.Linear(128, num_classes * 4),   # [gamma, nu, alpha, beta] × C
        )

    def forward(self, z: Tensor) -> dict[str, Tensor]:
        """
        Args:
            z: (B, D) fused embedding.

        Returns:
            Dict with:
                gamma  (B, C) — predicted mean
                nu     (B, C) — evidence on variance (≥ 0)
                alpha  (B, C) — shape parameter (≥ 1)
                beta   (B, C) — rate parameter (≥ 0)
                aleatoric  (B,) — per-sample aleatoric uncertainty
                epistemic  (B,) — per-sample epistemic uncertainty
        """
        B = z.shape[0]
        raw = self.evidence_head(z).view(B, self.num_classes, 4)  # (B, C, 4)

        gamma = raw[..., 0]                             # unrestricted
        nu    = F.softplus(raw[..., 1]) + 1e-4          # ≥ 0
        alpha = F.softplus(raw[..., 2]) + 1.0 + 1e-4   # ≥ 1
        beta  = F.softplus(raw[..., 3]) + 1e-4          # ≥ 0

        # Aleatoric: expected variance of data likelihood
        aleatoric = (beta / (alpha - 1 + 1e-8)).mean(dim=-1)      # (B,)
        aleatoric = aleatoric.clamp(0.0, 10.0)
        aleatoric_norm = 1.0 - torch.exp(-aleatoric)               # map to [0,1]

        # Epistemic: uncertainty in parameter estimate
        epistemic = (1.0 / (nu + 1e-8)).mean(dim=-1)              # (B,)
        epistemic_norm = 1.0 - torch.exp(-epistemic)

        return {
            "gamma": gamma,
            "nu": nu,
            "alpha": alpha,
            "beta": beta,
            "aleatoric": aleatoric_norm,
            "epistemic": epistemic_norm,
        }

    @staticmethod
    def evidential_loss(
        gamma: Tensor,
        nu: Tensor,
        alpha: Tensor,
        beta: Tensor,
        y: Tensor,
        lam: float = 0.1,
    ) -> Tensor:
        """NIG-NLL loss with evidence regularisation.

        Args:
            gamma, nu, alpha, beta: NIG parameters (B, C).
            y: True labels (B,) — long integer class indices.
            lam: Regularisation coefficient.

        Returns:
            Scalar loss.
        """
        # One-hot encode targets
        C = gamma.shape[-1]
        y_one_hot = F.one_hot(y, C).float()   # (B, C)

        omega = 2 * beta * (1 + nu)
        nll = (
            0.5 * (math.pi / (nu + 1e-8)).log()
            - alpha * omega.log()
            + (alpha + 0.5) * (nu * (gamma - y_one_hot) ** 2 + omega).log()
            + torch.lgamma(alpha)
            - torch.lgamma(alpha + 0.5)
        )

        # Evidence regularisation: penalise high evidence on wrong class
        error = (gamma - y_one_hot).abs()
        reg = error * (2 * nu + alpha)

        loss = (nll + lam * reg).mean()
        return loss


# ---------------------------------------------------------------------------
# Temperature Scaling Calibrator
# ---------------------------------------------------------------------------

class TemperatureScaler(nn.Module):
    """Post-hoc calibration via a single learned temperature parameter.

    After training, fit temperature on a validation set to minimise NLL.

    Args:
        init_temp: Initial temperature value (1.0 = identity).
    """

    def __init__(self, init_temp: float = 1.5) -> None:
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(init_temp))

    def forward(self, logits: Tensor) -> Tensor:
        """Scale logits by learned temperature."""
        return logits / self.temperature.clamp(min=1e-3)

    def calibrate(
        self,
        logits: Tensor,
        labels: Tensor,
        lr: float = 0.01,
        max_iters: int = 50,
    ) -> float:
        """Fit temperature on held-out validation logits / labels.

        Args:
            logits: (N, C) raw model logits on validation set.
            labels: (N,) ground-truth integer class labels.
            lr: Learning rate for temperature optimisation.
            max_iters: Max optimisation steps.

        Returns:
            Final calibrated temperature value.
        """
        optimiser = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iters)

        def closure():
            optimiser.zero_grad()
            scaled = self.forward(logits)
            loss = F.cross_entropy(scaled, labels)
            loss.backward()
            return loss

        optimiser.step(closure)
        return float(self.temperature.item())


# ---------------------------------------------------------------------------
# Main Uncertainty Estimation Module
# ---------------------------------------------------------------------------

class UncertaintyEstimator(nn.Module):
    """Stage 7 — Multi-method Uncertainty Estimator.

    Combines MC Dropout (epistemic) and Evidential DL (aleatoric) to produce
    a comprehensive ``UncertaintyOutput`` for each case.

    Args:
        embedding_dim: Fused embedding dimension from Stage 6.
        num_classes:   Number of diagnostic classes.
        tau1:          Lower uncertainty threshold (confident → cautious boundary).
        tau2:          Upper uncertainty threshold (cautious → defer boundary).
        mc_samples:    Number of MC Dropout forward passes.
    """

    def __init__(
        self,
        embedding_dim: int = 256,
        num_classes: int = 3,
        tau1: float = 0.35,
        tau2: float = 0.65,
        mc_samples: int = 20,
    ) -> None:
        super().__init__()
        self.tau1 = tau1
        self.tau2 = tau2

        self.mc_estimator = MCDropoutEstimator(
            embedding_dim=embedding_dim,
            num_classes=num_classes,
            dropout_rate=0.3,
            num_samples=mc_samples,
        )

        self.evidential_head = EvidentialHead(
            embedding_dim=embedding_dim,
            num_classes=num_classes,
        )

        self.temperature_scaler = TemperatureScaler(init_temp=1.5)

    def forward(
        self,
        Z_f: Tensor,
        phrase_scores: dict[str, float] | None = None,
        retrieval_scores: dict[str, float] | None = None,
        graph_num_nodes: int = 0,
    ) -> UncertaintyOutput:
        """Estimate multi-granularity uncertainty.

        Args:
            Z_f:              (B, D) fused representation from Stage 6.
            phrase_scores:    Grounding confidences — phrase → score.
            retrieval_scores: Retrieval similarities — case_id → score.
            graph_num_nodes:  Number of nodes in the dynamic KG.

        Returns:
            UncertaintyOutput with all uncertainty components.
        """
        # MC Dropout — epistemic
        mc_out = self.mc_estimator(Z_f)
        epistemic = float(mc_out["epistemic"].mean().item())

        # Evidential DL — aleatoric + epistemic
        ev_out = self.evidential_head(Z_f)
        aleatoric = float(ev_out["aleatoric"].mean().item())
        ev_epistemic = float(ev_out["epistemic"].mean().item())

        # Combine epistemics (weighted average)
        combined_epistemic = 0.6 * epistemic + 0.4 * ev_epistemic

        # Global uncertainty = harmonic blend of aleatoric + epistemic
        U_global = 0.5 * aleatoric + 0.5 * combined_epistemic
        U_global = float(min(max(U_global, 0.0), 1.0))

        # Per-finding uncertainty: invert grounding confidence
        U_finding: dict[str, float] = {}
        if phrase_scores:
            for phrase, conf in phrase_scores.items():
                U_finding[phrase] = float(1.0 - min(max(conf, 0.0), 1.0))

        # Retrieval uncertainty: low similarity → high uncertainty
        U_retrieval: dict[str, float] = {}
        if retrieval_scores:
            for case_id, sim in retrieval_scores.items():
                U_retrieval[case_id] = float(1.0 - min(max(sim, 0.0), 1.0))

        # Graph uncertainty: fewer nodes → higher uncertainty
        if graph_num_nodes == 0:
            U_graph = 1.0
        elif graph_num_nodes < 5:
            U_graph = 0.6
        elif graph_num_nodes < 15:
            U_graph = 0.3
        else:
            U_graph = 0.1

        logger.debug(
            f"Uncertainty — global: {U_global:.3f}, "
            f"aleatoric: {aleatoric:.3f}, epistemic: {combined_epistemic:.3f}, "
            f"graph: {U_graph:.3f}"
        )

        return UncertaintyOutput(
            U_global=U_global,
            U_finding=U_finding,
            U_retrieval=U_retrieval,
            U_graph=U_graph,
            aleatoric=aleatoric,
            epistemic=combined_epistemic,
            tau1=self.tau1,
            tau2=self.tau2,
        )
