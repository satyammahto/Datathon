"""Unit and integration tests for AIDA Trust Layer and Deterministic Verifier Node."""

import pytest
from backend.aida.contracts.discovery_contract import DiscoveryContract
from backend.aida.contracts.analysis_contract import AnalysisContract
from backend.aida.contracts.insight_contract import (
    CandidateInsight,
    MetricClaim,
    InsightContract,
    VerifiedInsight,
    RejectedInsight,
)
from backend.aida.state import AidaState, CritiqueEvaluation
from backend.aida.verifier.metrics import verify_metric_exactness
from backend.aida.verifier.correlations import verify_correlation_claim
from backend.aida.verifier.feature_importance import (
    verify_feature_importance_value,
    verify_top_features_ranking,
)
from backend.aida.verifier.confidence import calculate_deterministic_confidence
from backend.aida.verifier.verifier_node import verifier_node
from backend.aida.mocks.mock_contracts import (
    get_mock_discovery_contract,
    get_mock_analysis_contract,
    get_mock_candidate_insights,
)


@pytest.fixture
def discovery_fixture() -> DiscoveryContract:
    return get_mock_discovery_contract()


@pytest.fixture
def analysis_fixture() -> AnalysisContract:
    return get_mock_analysis_contract()


# --- Metric Exactness Tests ---

def test_metric_exactness_success(discovery_fixture, analysis_fixture):
    """Accurate values matching the contracts must pass within tolerance."""
    # Test discovery metric: session_duration mean = 42.50
    claim_discovery = MetricClaim(
        metric_name="mean",
        source_contract="discovery",
        entity_path="columns.session_duration.distribution.mean",
        claimed_value=42.50,
        tolerance=0.01,
    )
    res = verify_metric_exactness(claim_discovery, discovery_contract=discovery_fixture)
    assert res.passed is True
    assert res.verified_value == 42.50
    assert res.error_detail is None

    # Test analysis metric: RandomForest accuracy = 0.884
    claim_analysis = MetricClaim(
        metric_name="accuracy",
        source_contract="analysis",
        entity_path="models.RandomForestClassifier.evaluation_metrics.accuracy",
        claimed_value=0.884,
        tolerance=0.001,
    )
    res_model = verify_metric_exactness(claim_analysis, analysis_contract=analysis_fixture)
    assert res_model.passed is True
    assert res_model.verified_value == 0.884


def test_metric_exactness_hallucination_rejection(analysis_fixture):
    """Hallucinated metrics MUST be caught deterministically and fail."""
    claim = MetricClaim(
        metric_name="accuracy",
        source_contract="analysis",
        entity_path="models.RandomForestClassifier.evaluation_metrics.accuracy",
        claimed_value=0.960,  # Actual is 0.884
        tolerance=0.005,
    )
    res = verify_metric_exactness(claim, analysis_contract=analysis_fixture)
    assert res.passed is False
    assert res.verified_value == 0.884
    assert res.error_detail is not None
    assert "exceeds tolerance" in res.error_detail


def test_metric_exactness_unresolved_path(discovery_fixture):
    """Non-existent entities must fail gracefully without crashing."""
    claim = MetricClaim(
        metric_name="mean",
        source_contract="discovery",
        entity_path="columns.non_existent_column.distribution.mean",
        claimed_value=10.0,
    )
    res = verify_metric_exactness(claim, discovery_contract=discovery_fixture)
    assert res.passed is False
    assert "could not be resolved" in res.message


# --- Correlation Verification Tests ---

def test_correlation_verification_success(discovery_fixture):
    """Valid correlation coefficients and directions must pass."""
    res = verify_correlation_claim(
        feature_a="session_duration",
        feature_b="monthly_charges",
        discovery_contract=discovery_fixture,
        claimed_r=0.724,
        claimed_direction="positive",
        claimed_strength="strong",
        tolerance=0.01,
    )
    assert res.passed is True
    assert res.verified_value == 0.724


def test_correlation_contradiction_rejection(discovery_fixture):
    """Contradictory direction or strength claims must fail."""
    res = verify_correlation_claim(
        feature_a="session_duration",
        feature_b="support_tickets",
        discovery_contract=discovery_fixture,
        claimed_direction="positive",  # True correlation is -0.482
    )
    assert res.passed is False
    assert "contradicts actual direction 'negative'" in res.error_detail


# --- Feature Importance & Ranking Tests ---

def test_feature_importance_value_success(analysis_fixture):
    """Accurate feature importance scores must pass."""
    res = verify_feature_importance_value(
        model_name="RandomForestClassifier",
        feature_name="monthly_charges",
        analysis_contract=analysis_fixture,
        claimed_importance=0.420,
        tolerance=0.01,
    )
    assert res.passed is True
    assert res.verified_value == 0.420


def test_feature_importance_ranking_success(analysis_fixture):
    """Valid descending top feature order must pass."""
    top_3 = ["monthly_charges", "session_duration", "support_tickets"]
    res = verify_top_features_ranking(
        model_name="RandomForestClassifier",
        claimed_top_features=top_3,
        analysis_contract=analysis_fixture,
    )
    assert res.passed is True
    assert res.verified_value == top_3


def test_feature_importance_ranking_inverted_failure(analysis_fixture):
    """Inverted feature order must fail ranking verification."""
    inverted_top = ["support_tickets", "monthly_charges"]
    res = verify_top_features_ranking(
        model_name="RandomForestClassifier",
        claimed_top_features=inverted_top,
        analysis_contract=analysis_fixture,
    )
    assert res.passed is False
    assert "Order mismatch" in res.error_detail


# --- Deterministic Confidence Calculation Tests ---

def test_confidence_zero_on_failed_checks():
    """Confidence MUST be strictly 0.0 if any check fails."""
    from aida.contracts.insight_contract import VerificationCheckResult
    checks = [
        VerificationCheckResult(
            check_type="metric_exactness",
            passed=False,
            claimed_value=0.9,
            verified_value=0.5,
            evidence_source="test",
            message="failed",
        )
    ]
    score = calculate_deterministic_confidence(checks, sample_size=1000)
    assert score == 0.0


def test_confidence_sample_size_and_p_value_sensitivity():
    """High sample size and statistical significance boost deterministic confidence."""
    from aida.contracts.insight_contract import VerificationCheckResult
    good_checks = [
        VerificationCheckResult(
            check_type="metric_exactness",
            passed=True,
            claimed_value=10.0,
            verified_value=10.0,
            evidence_source="test",
            message="passed",
        )
    ]

    # Large dataset with p < 0.001
    high_score = calculate_deterministic_confidence(
        good_checks, sample_size=15000, missing_percentage=0.0, p_value=0.0001
    )
    assert 0.90 <= high_score <= 0.99

    # Tiny dataset with non-significant p-value
    low_score = calculate_deterministic_confidence(
        good_checks, sample_size=20, missing_percentage=10.0, p_value=0.25
    )
    assert low_score < high_score


# --- End-to-End Verifier Node Test ---

def test_verifier_node_end_to_end(discovery_fixture, analysis_fixture):
    """Full LangGraph Verifier Node execution with candidate segregation."""
    candidates = get_mock_candidate_insights()

    state: AidaState = {
        "discovery_contract": discovery_fixture,
        "analysis_contract": analysis_fixture,
        "candidate_insights": candidates,
        "critique_evaluations": [
            CritiqueEvaluation(
                insight_id=c.insight_id,
                soundness_score=0.9,
                is_approved_for_verification=True,
                critique_notes="Passed sanity check",
            )
            for c in candidates
        ],
        "errors": [],
    }

    result = verifier_node(state)

    verified = result["verified_insights"]
    rejected = result["rejected_insights"]
    contract: InsightContract = result["insight_contract"]

    # Exactly 2 should pass and 2 should fail
    assert len(verified) == 2
    assert len(rejected) == 2

    # Assert verified insights have positive confidence and verified evidence
    for vi in verified:
        assert isinstance(vi, VerifiedInsight)
        assert vi.status == "VERIFIED"
        assert vi.confidence_score > 0.5
        assert len(vi.evidence) > 0
        assert all(ev.passed for ev in vi.evidence)

    # Assert rejected insights have complete audit trails
    for ri in rejected:
        assert isinstance(ri, RejectedInsight)
        assert ri.status == "REJECTED"
        assert len(ri.failed_checks) > 0
        assert len(ri.rejection_reason) > 0

    # Assert summary contract for Person 4
    assert contract.dataset_id == discovery_fixture.dataset_id
    assert contract.summary["total_candidates"] == 4
    assert contract.summary["verified_count"] == 2
    assert contract.summary["rejected_count"] == 2
    assert contract.summary["verification_pass_rate"] == 0.5
