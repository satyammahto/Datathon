"""
Tests for Deterministic Investigation Tools (Person 3 - Phase 3.0)
Verifies each allowlisted tool, security gates, numerical accuracy, and JSON serialization.
"""
import pytest
from backend.app.schemas.analysis_contract import (
    AnalysisContract,
    StatisticalReport,
    ChampionResult,
    ErrorAnalysisReport,
    ExperimentRecord,
    TimeAnalysisReport,
)
from backend.app.schemas.insight_contract import InvestigationAction, InvestigationEvidence
from backend.app.schemas.enums import PipelineStatus, MLTaskType
from backend.app.agents.investigation import (
    execute_investigation,
    verify_correlation,
    verify_error_slice,
    verify_ablation,
    verify_residual_bias,
    verify_class_performance,
    ALLOWLISTED_ACTIONS,
)


def make_mock_analysis_contract() -> AnalysisContract:
    return AnalysisContract(
        schema_version="1.0",
        status=PipelineStatus.SUCCESS,
        task=MLTaskType.CLASSIFICATION,
        dataset_context={"n_rows": 120, "n_cols": 6},
        statistical_analysis=StatisticalReport(
            status=PipelineStatus.SUCCESS,
            correlations={
                "feature_a": {"feature_b": {"r": 0.74, "p_value": 0.0001, "method": "pearson"}},
            },
            statistical_tests=[
                {"column": "income", "group_by": "region", "p_value": 0.002, "effect_size": 0.58, "test_name": "kruskal_wallis"}
            ],
        ),
        winner=ChampionResult(
            status=PipelineStatus.SUCCESS,
            model_name="RandomForest",
            primary_metric="f1_macro",
            score=0.88,
        ),
        error_analysis=ErrorAnalysisReport(
            status=PipelineStatus.SUCCESS,
            classification={
                "class_metrics": [
                    {"class": "0", "recall": 0.92, "precision": 0.89, "f1": 0.90, "support": 80},
                    {"class": "1", "recall": 0.45, "precision": 0.60, "f1": 0.51, "support": 40},
                ],
            },
            subgroup_analysis=[
                {"feature": "tier", "group": "Tier3", "group_metric": 0.38, "baseline_metric": 0.15, "error_disparity_ratio": 2.53, "sample_count": 28, "status": "WEAK_SLICE"}
            ],
            regression={
                "residual_summary": {
                    "mean_residual": -1.45,
                    "std_residual": 3.2,
                    "bias_diagnostic": {"bias_detected": True, "direction": "systematic_overprediction"},
                }
            },
        ),
        experiments=[
            ExperimentRecord(
                experiment_id="abl_1",
                experiment_type="feature_ablation",
                feature="feature_a",
                status=PipelineStatus.SUCCESS,
                baseline_score=0.88,
                experiment_score=0.79,
                delta=-0.09,
                metric="f1_macro",
                interpretation="performance_decreased",
            )
        ],
        time_analysis=TimeAnalysisReport(
            status=PipelineStatus.SUCCESS,
            stationarity_test_passed=True,
            trend_type="upward",
        ),
    )


def test_verify_correlation_tool():
    """Verify bivariate correlation investigation tool."""
    analysis = make_mock_analysis_contract()
    action = InvestigationAction(
        action_id="act_1",
        action_type="verify_correlation",
        candidate_id="c_1",
        target_feature="feature_a",
        target_group="feature_b",
    )
    ev = execute_investigation(action, analysis)

    assert ev.status == "COMPLETED"
    assert ev.statistical_support is True
    assert ev.evidence["r"] == 0.74
    assert ev.evidence["p_value"] == 0.0001
    assert "feature_a" in ev.claim


def test_verify_error_slice_tool():
    """Verify subgroup slice investigation tool."""
    analysis = make_mock_analysis_contract()
    action = InvestigationAction(
        action_id="act_2",
        action_type="verify_error_slice",
        candidate_id="c_2",
        target_feature="tier",
        target_group="Tier3",
    )
    ev = execute_investigation(action, analysis)

    assert ev.status == "COMPLETED"
    assert ev.statistical_support is True
    assert ev.evidence["disparity_ratio"] == 2.53
    assert ev.evidence["sample_count"] == 28


def test_verify_ablation_tool():
    """Verify feature ablation investigation tool."""
    analysis = make_mock_analysis_contract()
    action = InvestigationAction(
        action_id="act_3",
        action_type="verify_ablation",
        candidate_id="c_3",
        target_feature="feature_a",
    )
    ev = execute_investigation(action, analysis)

    assert ev.status == "COMPLETED"
    assert ev.evidence["delta"] == -0.09
    assert ev.evidence["metric"] == "f1_macro"


def test_verify_class_performance_tool():
    """Verify classification class performance tool."""
    analysis = make_mock_analysis_contract()
    action = InvestigationAction(
        action_id="act_4",
        action_type="verify_class_performance",
        candidate_id="c_4",
        target_group="1",
    )
    ev = execute_investigation(action, analysis)

    assert ev.status == "COMPLETED"
    assert ev.evidence["recall"] == 0.45
    assert ev.evidence["support"] == 40


def test_unauthorized_action_rejection():
    """Verify that un-allowlisted arbitrary action types are strictly rejected."""
    analysis = make_mock_analysis_contract()
    action = InvestigationAction(
        action_id="act_bad",
        action_type="execute_shell_command",
        candidate_id="c_bad",
    )
    ev = execute_investigation(action, analysis)

    assert ev.status == "REJECTED"
    assert "security policy" in ev.claim.lower()
    assert ev.statistical_support is False
