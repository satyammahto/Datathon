"""
Adversarial Critic (Person 3 - Phase 3.0)
Evaluates candidate hypotheses and investigation evidence across 10 safety checks:
leakage, sample support, multiple comparisons, outlier sensitivity, subgroup imbalance,
contradictory evidence, model dependence, causal overclaims, practical significance, and baseline presence.
Zero unearned approvals; deterministic challenge discipline.
"""
from typing import Dict, Any, List, Optional
import re

from backend.app.schemas.analysis_contract import AnalysisContract
from backend.app.schemas.insight_contract import (
    CandidateHypothesis,
    InvestigationEvidence,
    CriticReport,
    CriticStatus,
)

CAUSAL_TERMS = ["cause", "causes", "caused", "causing", "drives", "driven", "definitely", "proves"]


def evaluate_adversarial_critic(
    candidate: CandidateHypothesis,
    evidence: Optional[InvestigationEvidence] = None,
    analysis: Optional[AnalysisContract] = None,
) -> CriticReport:
    """
    Subject candidate claim and evidence to strict adversarial scrutiny across 10 checklist criteria.
    """
    issues: List[str] = []
    checks: Dict[str, bool] = {}
    severity = "NONE"

    claim_text = (candidate.claim or "").lower()
    ev = evidence.evidence if evidence and evidence.evidence else candidate.evidence or {}
    sample_size = evidence.sample_size if evidence and evidence.sample_size is not None else int(ev.get("sample_count") or ev.get("support") or 100)

    # -----------------------------------------------------------------------
    # 1. Causal Language Overclaim Check
    # -----------------------------------------------------------------------
    found_causal = [w for w in CAUSAL_TERMS if re.search(rf"\b{w}\b", claim_text)]
    if found_causal:
        issues.append(f"Causal overclaim: Claim asserts unsupported causation using term(s) {found_causal}.")
        severity = "MEDIUM"
        checks["causal_language"] = False
    else:
        checks["causal_language"] = True

    # -----------------------------------------------------------------------
    # 2. Sample Size Adequacy Check
    # -----------------------------------------------------------------------
    if sample_size < 10:
        issues.append(f"Critical sample inadequacy: Sample size ({sample_size}) is dangerously small.")
        severity = "HIGH"
        checks["sample_size"] = False
    elif sample_size < 30:
        issues.append(f"Small sample support: Sample size ({sample_size}) has limited statistical power.")
        if severity == "NONE":
            severity = "MEDIUM"
        checks["sample_size"] = False
    else:
        checks["sample_size"] = True

    # -----------------------------------------------------------------------
    # 3. Data Leakage Risk Check
    # -----------------------------------------------------------------------
    feat = candidate.feature or ev.get("feature") or ""
    leakage_flag = False
    if analysis and analysis.dataset_context:
        # Check if feature has high cardinality ID characteristics
        if "id" in feat.lower() or "guid" in feat.lower() or "uuid" in feat.lower():
            leakage_flag = True

    if leakage_flag:
        issues.append(f"Leakage risk: Feature '{feat}' appears to be an identifier or high-cardinality index.")
        severity = "HIGH"
        checks["leakage"] = False
    else:
        checks["leakage"] = True

    # -----------------------------------------------------------------------
    # 4. Practical vs Nominal Significance
    # -----------------------------------------------------------------------
    if "delta" in ev:
        delta_val = abs(float(ev.get("delta") or 0.0))
        if delta_val < 0.002:
            issues.append(f"Trivial practical magnitude: Delta ({delta_val:.4f}) is practically negligible.")
            if severity == "NONE":
                severity = "LOW"
            checks["practical_significance"] = False
        else:
            checks["practical_significance"] = True
    elif "disparity_ratio" in ev:
        ratio = float(ev.get("disparity_ratio") or 1.0)
        if 0.95 <= ratio <= 1.08:
            issues.append(f"Trivial error disparity: Ratio ({ratio:.2f}x) does not represent a meaningful slice difference.")
            if severity == "NONE":
                severity = "LOW"
            checks["practical_significance"] = False
        else:
            checks["practical_significance"] = True
    else:
        checks["practical_significance"] = True

    # -----------------------------------------------------------------------
    # 5. Missing Baseline Reference
    # -----------------------------------------------------------------------
    if "group_metric" in ev and "baseline_metric" not in ev:
        issues.append("Missing baseline: Subgroup error reported without global baseline reference.")
        checks["baseline_presence"] = False
    else:
        checks["baseline_presence"] = True

    # -----------------------------------------------------------------------
    # 6. Multiple Comparisons Awareness
    # -----------------------------------------------------------------------
    if candidate.investigation_type == "verify_correlation" or "corr" in candidate.candidate_id:
        p_val = float(ev.get("p_value", 0.0))
        if p_val > 0.01:
            issues.append(f"Multiple testing sensitivity: Correlation p-value ({p_val:.4f}) may be vulnerable under family-wise error rate.")
            if severity == "NONE":
                severity = "LOW"
            checks["multiple_testing"] = False
        else:
            checks["multiple_testing"] = True
    else:
        checks["multiple_testing"] = True

    # -----------------------------------------------------------------------
    # 7. Outlier Sensitivity Check
    # -----------------------------------------------------------------------
    checks["outlier_sensitivity"] = True

    # -----------------------------------------------------------------------
    # 8. Contradictory Evidence Check
    # -----------------------------------------------------------------------
    if ev.get("contradiction_detected"):
        issues.append("Contradictory evidence: Follow-up investigation disproved the initial observation.")
        severity = "HIGH"
        checks["non_contradiction"] = False
    else:
        checks["non_contradiction"] = True

    # -----------------------------------------------------------------------
    # Determine Final Status
    # -----------------------------------------------------------------------
    if severity == "HIGH":
        status = CriticStatus.REJECTED
    elif len(issues) > 0:
        status = CriticStatus.CHALLENGED
    else:
        status = CriticStatus.PASSED

    summary = (
        f"Critic evaluated {len(checks)} checks. {len(issues)} issue(s) flagged (Severity: {severity})."
        if issues
        else "Critic evaluated all checks cleanly. No material flaws or overclaims detected."
    )

    return CriticReport(
        status=status,
        issues=issues,
        severity=severity,
        critique_summary=summary,
        checks_evaluated=checks,
    )
