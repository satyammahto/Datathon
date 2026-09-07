"""
Deterministic Insight Ranking Engine (Person 3 - Phase 3.0)
Ranks ONLY verified insights based on empirical importance, confidence weight,
effect size, and sample support.
Zero LLM subjective sorting; purely deterministic.
"""
from typing import List
from backend.app.schemas.insight_contract import (
    VerifiedInsight,
    VerificationStatus,
    ConfidenceLevel,
)


def compute_insight_rank_score(insight: VerifiedInsight) -> float:
    """
    Compute a composite deterministic ranking score for a verified insight:
    Score = (BaseImportance * 0.4) + (ConfidenceWeight * 0.3) + (EffectMagnitude * 0.2) + (SampleWeight * 0.1)
    """
    conf_weights = {
        ConfidenceLevel.HIGH: 1.0,
        ConfidenceLevel.MEDIUM: 0.75,
        ConfidenceLevel.LOW: 0.40,
        ConfidenceLevel.REJECTED: 0.0,
    }
    conf_factor = conf_weights.get(insight.confidence, 0.5)

    # Extract effect magnitude from evidence
    ev = insight.evidence or {}
    delta = abs(float(ev.get("delta") or 0.0))
    disp = abs(float(ev.get("disparity_ratio") or 1.0) - 1.0)
    r_val = abs(float(ev.get("r") or 0.0))
    score_val = abs(float(ev.get("score") or 0.0))
    effect_val = max(delta * 2.0, disp * 0.5, r_val, min(1.0, score_val))

    # Sample weight
    samples = int(ev.get("sample_count") or ev.get("support") or 100)
    sample_weight = min(1.0, samples / 100.0)

    raw_score = (
        (insight.importance * 0.40)
        + (conf_factor * 0.30)
        + (min(1.0, effect_val) * 0.20)
        + (sample_weight * 0.10)
    )
    return round(float(raw_score), 4)


def rank_verified_insights(insights: List[VerifiedInsight]) -> List[VerifiedInsight]:
    """
    Filter and rank VERIFIED insights deterministically.
    Insights that are REJECTED or CONTRADICTED are never ranked in the final list.
    """
    # Only keep genuinely VERIFIED insights
    verified_only = [
        ins for ins in insights
        if ins.verification.status == VerificationStatus.VERIFIED
        and ins.confidence != ConfidenceLevel.REJECTED
    ]

    for ins in verified_only:
        ins.rank_score = compute_insight_rank_score(ins)

    # Sort descending by rank_score, then lexicographically by insight_id for determinism
    sorted_insights = sorted(
        verified_only,
        key=lambda x: (x.rank_score if x.rank_score is not None else 0.0, x.insight_id),
        reverse=True,
    )

    for rank_idx, ins in enumerate(sorted_insights, 1):
        ins.rank = rank_idx
        ins.ranking_rationale = (
            f"Rank {rank_idx} assigned based on composite score {ins.rank_score:.4f} "
            f"(Confidence: {ins.confidence.value}, Importance: {ins.importance:.2f})."
        )

    return sorted_insights
