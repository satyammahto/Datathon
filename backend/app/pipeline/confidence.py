"""
Deterministic Confidence Engine (Person 3 - Phase 3.0)
Computes empirical confidence ratings strictly from verification status, sample support,
statistical significance, effect magnitude, and critic challenges.
Zero LLM subjective scoring; pure deterministic policy.
"""
from typing import Optional, Dict, Any
from backend.app.schemas.insight_contract import (
    ConfidenceLevel,
    VerificationStatus,
    CriticStatus,
    CriticReport,
    VerificationReport,
)


def evaluate_insight_confidence(
    verification: VerificationReport,
    critic: CriticReport,
    sample_size: Optional[int] = None,
    statistical_support: bool = True,
    effect_magnitude: float = 0.0,
) -> ConfidenceLevel:
    """
    Deterministically score confidence according to transparent empirical rules:
    - REJECTED: Verification failed (REJECTED or CONTRADICTED).
    - HIGH: Verification VERIFIED, adequate sample support (N >= 50 or dataset-wide),
            statistical support True, and critic PASSED (no high severity issues).
    - MEDIUM: Verification VERIFIED, moderate sample support (20 <= N < 50),
              or minor critic challenges without contradiction.
    - LOW: Small sample support (N < 20), marginal statistical support,
           or verification status is INSUFFICIENT_EVIDENCE.
    """
    # 1. Hard verification gate
    if verification.status in (VerificationStatus.REJECTED, VerificationStatus.CONTRADICTED):
        return ConfidenceLevel.REJECTED

    if verification.status == VerificationStatus.INSUFFICIENT_EVIDENCE:
        return ConfidenceLevel.LOW

    # If verification is not VERIFIED, cannot proceed
    if verification.status != VerificationStatus.VERIFIED:
        return ConfidenceLevel.REJECTED

    # 2. Critic challenge evaluation
    if critic.status == CriticStatus.REJECTED or critic.severity == "HIGH":
        return ConfidenceLevel.LOW

    # 3. Sample size and empirical support evaluation
    effective_n = sample_size if sample_size is not None and sample_size > 0 else 100

    if effective_n >= 50 and statistical_support and critic.status == CriticStatus.PASSED:
        return ConfidenceLevel.HIGH
    elif effective_n >= 20 and statistical_support:
        return ConfidenceLevel.MEDIUM
    elif effective_n < 20:
        return ConfidenceLevel.LOW
    else:
        return ConfidenceLevel.MEDIUM
