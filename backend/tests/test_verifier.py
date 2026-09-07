"""
Tests for Numerical Verifier (Person 3 - Phase 3.0)
Verifies hard anti-hallucination gates, number discrepancies, sign mismatches,
contradictions, and valid numerical approvals.
"""
import pytest
from backend.app.schemas.analysis_contract import AnalysisContract, ChampionResult
from backend.app.schemas.insight_contract import (
    CandidateHypothesis,
    InvestigationEvidence,
    VerificationStatus,
)
from backend.app.schemas.enums import PipelineStatus, MLTaskType
from backend.app.agents.verifier import verify_numerical_claims


def test_llm_hallucination_number_rejected():
    """
    CRITICAL SPECIFICATION TEST:
    Authoritative evidence: accuracy = 0.81
    Candidate/LLM claims: accuracy = 0.91
    Verifier MUST return REJECTED.
    """
    analysis = AnalysisContract(
        schema_version="1.0",
        status=PipelineStatus.SUCCESS,
        winner=ChampionResult(
            status=PipelineStatus.SUCCESS,
            model_name="RandomForest",
            score=0.81,
            primary_metric="accuracy",
        ),
    )

    hallucinated_candidate = CandidateHypothesis(
        candidate_id="c_hallucinated",
        claim="Champion model achieved exceptional accuracy of 0.91 on validation folds.",
        source_path="winner",
        evidence={"score": 0.81, "metric": "accuracy"},
    )

    ev = InvestigationEvidence(
        claim="Champion verified",
        method="champion_verification",
        evidence={"score": 0.81},
        sample_size=100,
        statistical_support=True,
    )

    ver_report = verify_numerical_claims(hallucinated_candidate, ev, analysis)

    assert ver_report.status == VerificationStatus.REJECTED
    assert len(ver_report.discrepancies) > 0
    assert ver_report.discrepancies[0]["claimed_value"] == 0.91
    assert any("unverified number" in check.lower() for check in ver_report.checks_failed)


def test_direction_sign_inversion_rejected():
    """Verify that claiming 'improved' when delta is negative (-0.09) is rejected."""
    analysis = AnalysisContract(
        schema_version="1.0",
        status=PipelineStatus.SUCCESS,
        task=MLTaskType.CLASSIFICATION,
    )

    candidate = CandidateHypothesis(
        candidate_id="c_inversion",
        claim="Removing feature A improved classification performance significantly.",
        source_path="experiments[0]",
        evidence={"delta": -0.09, "metric": "f1_macro"},
    )

    ev = InvestigationEvidence(
        claim="Ablation checked",
        method="feature_ablation_verification",
        evidence={"delta": -0.09},
        sample_size=100,
        statistical_support=True,
    )

    ver_report = verify_numerical_claims(candidate, ev, analysis)

    assert ver_report.status == VerificationStatus.REJECTED
    assert any("direction mismatch" in check.lower() or "negative delta" in check.lower() for check in ver_report.checks_failed)


def test_contradicted_evidence_marked_contradicted():
    """Verify that contradiction evidence sets status to CONTRADICTED."""
    analysis = AnalysisContract(schema_version="1.0", status=PipelineStatus.SUCCESS)

    candidate = CandidateHypothesis(
        candidate_id="c_init",
        claim="Subgroup disparity observed in Region South.",
        source_path="error_analysis.subgroup_analysis",
        evidence={"disparity_ratio": 2.1},
    )

    ev = InvestigationEvidence(
        claim="Re-verification completed",
        method="subgroup_verification",
        evidence={
            "contradiction_detected": True,
            "contradiction_reason": "Controlled covariate test removed disparity.",
        },
        sample_size=100,
        statistical_support=False,
    )

    ver_report = verify_numerical_claims(candidate, ev, analysis)

    assert ver_report.status == VerificationStatus.CONTRADICTED
    assert any("contradicted" in check.lower() for check in ver_report.checks_failed)


def test_valid_evidence_passes_verification():
    """Verify that accurate numbers and valid sources receive clean VERIFIED status."""
    analysis = AnalysisContract(
        schema_version="1.0",
        status=PipelineStatus.SUCCESS,
        winner=ChampionResult(
            status=PipelineStatus.SUCCESS,
            model_name="RandomForest",
            score=0.88,
            primary_metric="f1_macro",
        ),
    )

    candidate = CandidateHypothesis(
        candidate_id="c_accurate",
        claim="Champion model RandomForest achieved score 0.88 on primary metric.",
        source_path="winner",
        evidence={"score": 0.88, "model_name": "RandomForest"},
    )

    ev = InvestigationEvidence(
        claim="Champion verified",
        method="champion_verification",
        evidence={"score": 0.88, "model_name": "RandomForest"},
        sample_size=120,
        statistical_support=True,
    )

    ver_report = verify_numerical_claims(candidate, ev, analysis)

    assert ver_report.status == VerificationStatus.VERIFIED
    assert len(ver_report.checks_failed) == 0
    assert len(ver_report.discrepancies) == 0
