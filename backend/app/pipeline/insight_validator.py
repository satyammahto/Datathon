"""
Insight Validator & Provenance Engine (Person 3 - Phase 3.0)
Constructs the full evidence-to-claim provenance graph and detects contradictions
between initial findings and subsequent investigation evidence.
"""
from typing import List, Dict, Any, Optional
from backend.app.schemas.insight_contract import (
    VerifiedInsight,
    VerificationStatus,
    ConfidenceLevel,
    InvestigationEvidence,
)


def build_provenance_graph(insights: List[VerifiedInsight]) -> List[Dict[str, Any]]:
    """
    Construct a complete end-to-end provenance graph for every verified insight:
    claim -> evidence -> source_path -> method -> critic -> verification -> rank.
    """
    graph: List[Dict[str, Any]] = []

    for ins in insights:
        node = {
            "insight_id": ins.insight_id,
            "rank": ins.rank,
            "claim": ins.claim,
            "confidence": ins.confidence.value if hasattr(ins.confidence, "value") else str(ins.confidence),
            "verification_status": ins.verification.status.value if hasattr(ins.verification.status, "value") else str(ins.verification.status),
            "critic_status": ins.critic.status.value if hasattr(ins.critic.status, "value") else str(ins.critic.status),
            "source_paths": ins.source_paths,
            "evidence_snapshot": ins.evidence,
            "method": ins.method,
            "provenance_chain": [
                {"step": "analysis_source", "paths": ins.source_paths},
                {"step": "investigation_method", "method": ins.method},
                {"step": "critic_review", "status": ins.critic.status.value, "issues": ins.critic.issues},
                {"step": "numerical_verification", "status": ins.verification.status.value, "checks_passed": ins.verification.checks_passed},
                {"step": "ranking", "rank": ins.rank, "score": ins.rank_score},
            ],
        }
        graph.append(node)

    return graph


def detect_and_handle_contradictions(
    insight: VerifiedInsight,
    followup_evidence: Optional[InvestigationEvidence] = None,
) -> bool:
    """
    Evaluate whether follow-up evidence directly contradicts an initial claim.
    If contradiction detected, marks the insight as CONTRADICTED and downgrades confidence to REJECTED.
    Returns True if contradicted, False otherwise.
    """
    if not followup_evidence:
        return False

    ev = followup_evidence.evidence or {}
    # Example 1: initial claim was "group disparity", follow-up says disparity is not significant or ratio <= 1.05
    if "disparity_ratio" in ev:
        ratio = float(ev["disparity_ratio"])
        if ratio <= 1.05 and "elevated" in insight.claim.lower():
            insight.verification.status = VerificationStatus.CONTRADICTED
            insight.verification.checks_failed.append("Follow-up investigation found disparity ratio <= 1.05; initial disparity contradicted.")
            insight.confidence = ConfidenceLevel.REJECTED
            return True

    # Example 2: statistical support was disproven
    if not followup_evidence.statistical_support and "significant" in insight.claim.lower():
        insight.verification.status = VerificationStatus.CONTRADICTED
        insight.verification.checks_failed.append("Follow-up statistical test disproved significance.")
        insight.confidence = ConfidenceLevel.REJECTED
        return True

    # Example 3: explicit contradiction flag in evidence
    if ev.get("contradiction_detected"):
        insight.verification.status = VerificationStatus.CONTRADICTED
        insight.verification.checks_failed.append(ev.get("contradiction_reason", "Follow-up evidence contradicted claim."))
        insight.confidence = ConfidenceLevel.REJECTED
        return True

    return False
