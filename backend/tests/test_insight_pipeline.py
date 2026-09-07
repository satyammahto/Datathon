"""
End-to-End Insight Pipeline Tests (Person 3 - Phase 3.0)
Tests complete multi-step autonomous loop:
AnalysisContract -> Discovery -> Planning -> Investigation -> Critic -> Verifier -> Ranking -> Summary -> InsightContract.
Includes determinism check, bounded iteration test, and JSON serialization check.
"""
import json
import pytest

from backend.app.schemas.analysis_contract import (
    AnalysisContract,
    StatisticalReport,
    ChampionResult,
    ErrorAnalysisReport,
    ExperimentRecord,
    EnsembleReport,
)
from backend.app.schemas.insight_contract import (
    InsightContract,
    VerificationStatus,
    ConfidenceLevel,
)
from backend.app.schemas.enums import PipelineStatus, MLTaskType
from backend.app.agents.graph import run_aida_reasoning


def make_comprehensive_analysis_contract() -> AnalysisContract:
    return AnalysisContract(
        schema_version="1.0",
        status=PipelineStatus.SUCCESS,
        task=MLTaskType.CLASSIFICATION,
        dataset_context={"n_rows": 150, "n_cols": 8, "target_column": "churn", "task_type": "classification"},
        statistical_analysis=StatisticalReport(
            status=PipelineStatus.SUCCESS,
            correlations={
                "monthly_charges": {"total_charges": {"r": 0.82, "p_value": 0.0001, "method": "pearson"}},
            },
            statistical_tests=[
                {"column": "monthly_charges", "group_by": "contract_type", "p_value": 0.0002, "effect_size": 0.65, "test_name": "anova"}
            ],
        ),
        winner=ChampionResult(
            status=PipelineStatus.SUCCESS,
            model_name="HistGradientBoosting",
            primary_metric="f1_macro",
            score=0.86,
            selection_rule="empirical_validation_score",
        ),
        error_analysis=ErrorAnalysisReport(
            status=PipelineStatus.SUCCESS,
            classification={
                "class_metrics": [
                    {"class": "0", "recall": 0.94, "precision": 0.91, "f1": 0.92, "support": 110},
                    {"class": "1", "recall": 0.52, "precision": 0.65, "f1": 0.58, "support": 40},
                ],
                "weak_classes": [
                    {"class": "1", "recall": 0.52, "f1": 0.58, "support": 40, "severity": "MEDIUM"}
                ],
            },
            subgroup_analysis=[
                {
                    "feature": "contract_type",
                    "group": "month-to-month",
                    "group_metric": 0.35,
                    "baseline_metric": 0.14,
                    "error_disparity_ratio": 2.50,
                    "sample_count": 45,
                    "status": "WEAK_SLICE",
                }
            ],
        ),
        experiments=[
            ExperimentRecord(
                experiment_id="exp_ablation_tenure",
                experiment_type="feature_ablation",
                feature="tenure",
                status=PipelineStatus.SUCCESS,
                baseline_score=0.86,
                experiment_score=0.74,
                delta=-0.12,
                metric="f1_macro",
                interpretation="performance_decreased",
            )
        ],
        ensemble=EnsembleReport(
            status=PipelineStatus.SUCCESS,
            champion_score=0.86,
            ensemble_score=0.875,
            delta=0.015,
            improves_champion=True,
            models=["HistGradientBoosting", "RandomForest"],
        ),
    )


def test_end_to_end_reasoning_loop():
    """Verify complete reasoning pass produces an authoritative InsightContract."""
    analysis = make_comprehensive_analysis_contract()
    contract = run_aida_reasoning(analysis, max_iterations=4)

    assert isinstance(contract, InsightContract)
    assert contract.status == PipelineStatus.SUCCESS
    assert contract.schema_version == "1.0"
    assert contract.candidate_count > 0
    assert contract.investigations_count > 0
    assert len(contract.insights) > 0

    # Verify ranking
    assert contract.insights[0].rank == 1
    assert contract.insights[0].rank_score is not None
    assert contract.insights[0].ranking_rationale is not None

    # Verify confidence
    for ins in contract.insights:
        assert ins.confidence in (ConfidenceLevel.HIGH, ConfidenceLevel.MEDIUM, ConfidenceLevel.LOW)
        assert ins.verification.status == VerificationStatus.VERIFIED
        assert ins.critic.status in ("PASSED", "CHALLENGED")

    # Verify executive summary
    summary = contract.executive_summary
    assert "title" in summary
    assert "overview" in summary
    assert len(summary["key_takeaways"]) > 0

    # Verify provenance graph
    assert len(contract.provenance_graph) == len(contract.insights)
    first_node = contract.provenance_graph[0]
    assert "provenance_chain" in first_node
    assert first_node["insight_id"] == contract.insights[0].insight_id


def test_bounded_iterations_and_early_stopping():
    """Verify that loop terminates within max_iterations and does not loop infinitely."""
    analysis = make_comprehensive_analysis_contract()

    # Bounded to 2 iterations
    contract_b2 = run_aida_reasoning(analysis, max_iterations=2)
    assert contract_b2.metadata["iterations_executed"] <= 2

    # Bounded to 5 iterations
    contract_b5 = run_aida_reasoning(analysis, max_iterations=5)
    assert contract_b5.metadata["iterations_executed"] <= 5


def test_determinism_reproducibility():
    """
    SPECIFICATION TEST (Section 19):
    Run the same AnalysisContract through the deterministic pipeline twice.
    Verify equivalent candidate counts, rankings, scores, and summary.
    """
    analysis = make_comprehensive_analysis_contract()

    contract1 = run_aida_reasoning(analysis, max_iterations=3)
    contract2 = run_aida_reasoning(analysis, max_iterations=3)

    assert contract1.candidate_count == contract2.candidate_count
    assert contract1.investigations_count == contract2.investigations_count
    assert len(contract1.insights) == len(contract2.insights)

    for ins1, ins2 in zip(contract1.insights, contract2.insights):
        assert ins1.insight_id == ins2.insight_id
        assert ins1.rank == ins2.rank
        assert ins1.rank_score == ins2.rank_score
        assert ins1.confidence == ins2.confidence

    assert contract1.executive_summary["overview"] == contract2.executive_summary["overview"]


def test_json_serialization_safety():
    """Verify that InsightContract can be safely serialized to JSON without numpy/pandas artifacts."""
    analysis = make_comprehensive_analysis_contract()
    contract = run_aida_reasoning(analysis, max_iterations=3)

    json_str = contract.model_dump_json()
    parsed = json.loads(json_str)

    assert parsed["status"] == "SUCCESS"
    assert "insights" in parsed
    assert "executive_summary" in parsed
    assert "provenance_graph" in parsed
