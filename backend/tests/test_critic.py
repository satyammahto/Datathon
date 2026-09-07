"""
Tests for Adversarial Critic (Person 3 - Phase 3.0)
Verifies 10-check safety scrutiny: sample support, causal overclaims, leakage,
practical significance, and contradictions.
"""
import pytest
from backend.app.schemas.insight_contract import (
    CandidateHypothesis,
    InvestigationEvidence,
    CriticStatus,
)
from backend.app.agents.critic import evaluate_adversarial_critic


def test_critic_flags_causal_overclaim():
    """Verify that critic challenges unsupported causal vocabulary."""
    candidate = CandidateHypothesis(
        candidate_id="c_causal",
        claim="Lower income causes high customer churn directly.",
        source_path="correlations",
        evidence={"r": 0.65},
    )
    report = evaluate_adversarial_critic(candidate)

    assert report.status == CriticStatus.CHALLENGED
    assert any("causal" in issue.lower() for issue in report.issues)
    assert report.checks_evaluated["causal_language"] is False


def test_critic_flags_small_sample_size():
    """Verify that critic challenges small samples and rejects tiny samples."""
    # N = 8 -> High severity / Rejected
    c_tiny = CandidateHypothesis(
        candidate_id="c_tiny",
        claim="Rare group disparity noted.",
        source_path="subgroups",
        evidence={"sample_count": 8},
    )
    rep_tiny = evaluate_adversarial_critic(c_tiny)
    assert rep_tiny.status == CriticStatus.REJECTED
    assert rep_tiny.severity == "HIGH"
    assert rep_tiny.checks_evaluated["sample_size"] is False

    # N = 22 -> Medium severity / Challenged
    c_mod = CandidateHypothesis(
        candidate_id="c_mod",
        claim="Moderate slice disparity.",
        source_path="subgroups",
        evidence={"sample_count": 22},
    )
    rep_mod = evaluate_adversarial_critic(c_mod)
    assert rep_mod.status == CriticStatus.CHALLENGED
    assert rep_mod.severity == "MEDIUM"


def test_critic_flags_trivial_magnitude():
    """Verify that critic flags trivial practical differences."""
    candidate = CandidateHypothesis(
        candidate_id="c_trivial",
        claim="Feature delta detected.",
        source_path="experiments",
        evidence={"delta": 0.0008},
    )
    report = evaluate_adversarial_critic(candidate)

    assert report.status == CriticStatus.CHALLENGED
    assert any("trivial" in issue.lower() for issue in report.issues)
    assert report.checks_evaluated["practical_significance"] is False


def test_critic_catches_contradiction():
    """Verify that critic rejects finding if follow-up evidence reports contradiction."""
    candidate = CandidateHypothesis(
        candidate_id="c_contra",
        claim="Slice disparity in Region X.",
        source_path="subgroups",
    )
    ev = InvestigationEvidence(
        claim="Contradiction verified",
        method="subgroup_verification",
        evidence={"contradiction_detected": True, "contradiction_reason": "Equalized under normalization"},
        sample_size=100,
        statistical_support=False,
    )
    report = evaluate_adversarial_critic(candidate, evidence=ev)

    assert report.status == CriticStatus.REJECTED
    assert report.severity == "HIGH"
    assert any("contradict" in issue.lower() for issue in report.issues)


def test_critic_clean_pass():
    """Verify that a well-supported, non-causal finding passes cleanly."""
    candidate = CandidateHypothesis(
        candidate_id="c_clean",
        claim="Class 1 exhibits degraded recall of 0.45 under stratified cross-validation.",
        source_path="class_metrics",
        evidence={"recall": 0.45, "support": 85},
    )
    ev = InvestigationEvidence(
        claim="Class 1 metrics verified",
        method="class_metrics_verification",
        evidence={"recall": 0.45, "support": 85},
        sample_size=85,
        statistical_support=True,
    )
    report = evaluate_adversarial_critic(candidate, evidence=ev)

    assert report.status == CriticStatus.PASSED
    assert len(report.issues) == 0
    assert report.severity == "NONE"
