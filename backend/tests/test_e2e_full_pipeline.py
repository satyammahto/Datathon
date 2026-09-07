"""
End-to-End Integration Test for Unified AIDA 4-Brain Architecture.
Validates that Person 1 (Data), Person 2 (ML), Person 3 (Agent), and Person 4 (UI)
execute in sequence without human intervention on unseen tabular datasets.
"""
import os
import pytest
from pathlib import Path

from backend.app.pipeline.orchestrator import run_full_aida_pipeline
from backend.app.schemas.enums import PipelineStatus, MLTaskType
from backend.app.schemas.dashboard_spec import DashboardSpec
from backend.app.schemas.insight_contract import InsightContract
from backend.app.schemas.analysis_contract import AnalysisContract

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures" / "data"


def test_full_pipeline_churn_classification():
    """
    Test 1: Full 4-Brain autonomous execution on unseen classification dataset.
    """
    file_path = str(FIXTURES_DIR / "churn_classification.csv")
    assert os.path.exists(file_path), f"Fixture not found at {file_path}"

    progress_events = []

    def on_progress(stage: str, msg: str):
        progress_events.append((stage, msg))

    results = run_full_aida_pipeline(
        file_path=file_path,
        dataset_name="churn_classification.csv",
        on_stage_progress=on_progress,
    )

    # 1. Validate structure of unified result dictionary
    assert "discovery" in results
    assert "analysis" in results
    assert "insights" in results
    assert "dashboard" in results

    # 2. Validate Person 1 (Discovery)
    p1 = results["discovery"]
    assert p1.file_info.row_count == 100
    assert p1.file_info.column_count == 6
    assert len(p1.target_candidates) > 0

    # 3. Validate Person 2 (ML / Accuracy Brain)
    analysis: AnalysisContract = results["analysis"]
    assert analysis.status == PipelineStatus.SUCCESS
    assert analysis.task == MLTaskType.CLASSIFICATION
    assert analysis.winner is not None
    assert analysis.winner.status == PipelineStatus.SUCCESS
    assert analysis.winner.model_name is not None
    assert analysis.winner.score is not None

    # 4. Validate Person 3 (Agent / Insight Brain)
    insights: InsightContract = results["insights"]
    assert insights.status == PipelineStatus.SUCCESS
    assert len(insights.insights) > 0
    # Every insight must have been checked by critic and verifier
    for ins in insights.insights:
        assert ins.critic is not None
        assert ins.verification is not None
        assert ins.verification.status.value in ["VERIFIED", "REFUTED", "UNVERIFIED"]

    # 5. Validate Person 4 (Dashboard Spec)
    dash: DashboardSpec = results["dashboard"]
    assert dash.dataset_name == "churn_classification.csv"
    assert len(dash.kpis) >= 4
    assert len(dash.insights) == len(insights.insights)
    assert dash.model_championship is not None
    assert dash.model_championship.champion_model_name == analysis.winner.model_name

    # 6. Verify Progress Event Callbacks
    stages_recorded = [s[0] for s in progress_events]
    assert "discovery" in stages_recorded
    assert "analysis" in stages_recorded
    assert "insights" in stages_recorded
    assert "dashboard" in stages_recorded
    assert "complete" in stages_recorded


def test_full_pipeline_housing_regression():
    """
    Test 2: Full 4-Brain autonomous execution on unseen regression dataset.
    """
    file_path = str(FIXTURES_DIR / "housing_regression.csv")
    assert os.path.exists(file_path), f"Fixture not found at {file_path}"

    results = run_full_aida_pipeline(
        file_path=file_path,
        dataset_name="housing_regression.csv",
    )

    analysis: AnalysisContract = results["analysis"]
    assert analysis.status == PipelineStatus.SUCCESS
    assert analysis.task == MLTaskType.REGRESSION
    assert analysis.winner is not None
    assert analysis.winner.status == PipelineStatus.SUCCESS

    dash: DashboardSpec = results["dashboard"]
    assert dash.dataset_name == "housing_regression.csv"
    assert len(dash.kpis) >= 4
    assert dash.model_championship is not None
