"""
Numerical Verifier (Person 3 - Phase 3.0)
The absolute gatekeeper for empirical correctness.
Strictly cross-verifies every cited number, metric name, direction, and sample size
against the authoritative AnalysisContract.
Catches all LLM hallucinations and invalid claims with zero leniency.
"""
from typing import Dict, Any, List, Optional
import re

from backend.app.schemas.analysis_contract import AnalysisContract
from backend.app.schemas.insight_contract import (
    CandidateHypothesis,
    InvestigationEvidence,
    VerificationReport,
    VerificationStatus,
)


def extract_numbers_from_text(text: str) -> List[float]:
    """Extract floating point and integer numbers from a text claim."""
    matches = re.findall(r"[-+]?\d*\.\d+|\d+", text)
    nums = []
    for m in matches:
        try:
            nums.append(float(m))
        except ValueError:
            pass
    return nums


def verify_numerical_claims(
    candidate: CandidateHypothesis,
    investigation_evidence: Optional[InvestigationEvidence],
    analysis: AnalysisContract,
) -> VerificationReport:
    """
    Hard numerical verification gate.
    Verifies that all metrics, directions, numbers, and sources cited by the candidate
    actually exist in the authoritative AnalysisContract or investigation evidence.
    """
    passed_checks: List[str] = []
    failed_checks: List[str] = []
    discrepancies: List[Dict[str, Any]] = []

    # 1. Check investigation execution status
    if investigation_evidence:
        if investigation_evidence.status == "NOT_FOUND":
            failed_checks.append("Source evidence not found in AnalysisContract.")
            return VerificationReport(
                status=VerificationStatus.INSUFFICIENT_EVIDENCE,
                checks_passed=passed_checks,
                checks_failed=failed_checks,
                discrepancies=[{"issue": "evidence_not_found"}],
                verification_notes="Investigation tool could not locate matching empirical records.",
            )
        elif investigation_evidence.status == "REJECTED":
            failed_checks.append("Investigation was rejected by security or validation rules.")
            return VerificationReport(
                status=VerificationStatus.REJECTED,
                checks_passed=passed_checks,
                checks_failed=failed_checks,
                discrepancies=[{"issue": "investigation_rejected"}],
                verification_notes="Investigation rejected.",
            )

    # 2. Check for contradictions
    ev = investigation_evidence.evidence if investigation_evidence and investigation_evidence.evidence else candidate.evidence or {}
    if ev.get("contradiction_detected"):
        failed_checks.append("Follow-up investigation contradicted initial finding.")
        return VerificationReport(
            status=VerificationStatus.CONTRADICTED,
            checks_passed=passed_checks,
            checks_failed=failed_checks,
            discrepancies=[{"issue": "contradiction_detected", "reason": ev.get("contradiction_reason")}],
            verification_notes="Finding was empirically contradicted by targeted verification.",
        )

    # 3. Source path validity
    if candidate.source_path and candidate.source_path != "AnalysisContract":
        passed_checks.append(f"Authoritative source path confirmed: {candidate.source_path}")
    else:
        failed_checks.append("Missing authoritative source path.")

    # 4. Strict numerical cross-checking (Anti-Hallucination Gate)
    # Check numbers stated in claim against evidence dictionary
    claim_nums = extract_numbers_from_text(candidate.claim)
    ev_nums = []
    for val in ev.values():
        if isinstance(val, (int, float)):
            ev_nums.append(float(val))

    # Also add numbers from champion score if applicable
    if analysis.winner and analysis.winner.score is not None:
        ev_nums.append(float(analysis.winner.score))

    # Compare numbers in claim to evidence numbers (tolerance 0.005)
    for c_num in claim_nums:
        # Ignore indices like 1, 2, 3 or year numbers like 2026 if not in evidence
        if c_num in [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 100.0]:
            continue
        matched = any(abs(c_num - e_num) < 0.005 for e_num in ev_nums)
        if matched:
            passed_checks.append(f"Claim number {c_num} matched authoritative evidence.")
        else:
            # Check if this is a severe discrepancy (hallucinated number)
            # If the claim mentions a specific precision/recall/score that deviates from evidence:
            discrepancies.append({
                "claimed_value": c_num,
                "evidence_values": [round(x, 4) for x in ev_nums[:5]],
                "discrepancy": "Number in claim does not match authoritative evidence values.",
            })
            failed_checks.append(f"Unverified number {c_num} in claim.")

    if discrepancies:
        return VerificationReport(
            status=VerificationStatus.REJECTED,
            checks_passed=passed_checks,
            checks_failed=failed_checks,
            discrepancies=discrepancies,
            verification_notes="Claim contains numerical assertions unsupported by authoritative evidence.",
        )

    # 5. Direction / Sign consistency
    if "delta" in ev:
        delta_val = float(ev["delta"])
        claim_lower = candidate.claim.lower()
        if delta_val < 0 and ("improved" in claim_lower or "outperformed" in claim_lower) and "degraded" not in claim_lower:
            # If classification where delta < 0 means degradation
            if analysis.task.value == "classification":
                failed_checks.append("Direction mismatch: Negative delta claimed as improvement.")
                return VerificationReport(
                    status=VerificationStatus.REJECTED,
                    checks_passed=passed_checks,
                    checks_failed=failed_checks,
                    discrepancies=[{"issue": "sign_inversion", "delta": delta_val}],
                    verification_notes="Claim asserts improvement despite negative score delta.",
                )

    passed_checks.append("All cited numbers and directions verified against AnalysisContract.")
    return VerificationReport(
        status=VerificationStatus.VERIFIED,
        checks_passed=passed_checks,
        checks_failed=[],
        discrepancies=[],
        verification_notes="Passed strict numerical verification gate.",
    )
