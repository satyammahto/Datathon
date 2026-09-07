"""
Tests for DashboardSpec & Presentation Adapter (Person 4 - Phase 4.0)
Verifies DiscoveryContract construction from DataFrames, DashboardSpec transformation,
NOT_APPLICABLE handling, and JSON serialization.
"""
import pytest
import pandas as pd
import numpy as np

from backend.app.schemas.enums import PipelineStatus, MLTaskType
from backend.app.schemas.dashboard_spec import DashboardSpec
from backend.app.api.adapter import (
    build_discovery_contract_for_df,
    build_dashboard_spec,
)
from backend.app.pipeline.ml_analysis import run_ml_pipeline
from backend.app.agents.graph import run_aida_reasoning


def test_build_discovery_contract_classification():
    """Verify that adapter constructs valid DiscoveryContract with classification routing."""
    df = pd.DataFrame({
        "user_id": [f"usr_{i}" for i in range(100)],
        "age": np.random.normal(35, 10, size=100),
        "income": np.random.uniform(20000, 100000, size=100),
        "tier": np.random.choice(["bronze", "silver", "gold"], size=100),
        "churn": np.random.choice([0, 1], size=100),
    })

    disc = build_discovery_contract_for_df(df, filename="churn_test.csv")

    assert disc.schema_version == "1.0"
    assert disc.fingerprint["row_count"] == 100
    assert disc.fingerprint["column_count"] == 5
    assert disc.schema_definition["age"] == "numeric"
    assert disc.schema_definition["tier"] == "categorical"
    assert "user_id" in disc.leakage_report["high_cardinality_identifiers"]
    assert "churn" in disc.target_candidates
    assert disc.router.classification is True
    assert disc.router.regression is False


def test_build_discovery_contract_regression():
    """Verify that adapter constructs valid DiscoveryContract with regression routing."""
    df = pd.DataFrame({
        "sqft": np.random.uniform(500, 3500, size=80),
        "bedrooms": np.random.choice([1, 2, 3, 4], size=80),
        "sale_price": np.random.uniform(150000, 800000, size=80),
    })

    disc = build_discovery_contract_for_df(df, filename="houses.csv")

    assert disc.router.regression is True
    assert disc.router.classification is False
    assert "sale_price" in disc.target_candidates


def test_build_discovery_contract_no_target_clustering():
    """Verify that adapter routes to clustering when no target is present."""
    df = pd.DataFrame({
        "feature_1": [1.0, 2.0, 3.0, 4.0],
    })

    disc = build_discovery_contract_for_df(df, filename="solo.csv")

    assert disc.router.clustering is True


def test_dashboard_spec_end_to_end_transformation():
    """Verify end-to-end transformation of pipeline results into validated DashboardSpec."""
    rng = np.random.default_rng(42)
    n = 100
    df = pd.DataFrame({
        "feature_a": rng.normal(10, 2, size=n),
        "feature_b": rng.uniform(0, 1, size=n),
        "category_c": rng.choice(["cat1", "cat2"], size=n),
        "target": rng.choice([0, 1], size=n),
    })

    discovery = build_discovery_contract_for_df(df, filename="binary_eval.csv")
    analysis = run_ml_pipeline(df, discovery)
    insights = run_aida_reasoning(analysis, max_iterations=3)

    dash = build_dashboard_spec(
        discovery=discovery,
        analysis=analysis,
        insights=insights,
        dataset_name="binary_eval.csv",
    )

    assert isinstance(dash, DashboardSpec)
    assert dash.status == PipelineStatus.SUCCESS
    assert dash.task_type.lower() == "classification"
    assert dash.target_column == "target"
    assert len(dash.kpis) >= 5
    assert dash.executive_summary.title != ""
    assert len(dash.insights) > 0

    # Model lab leaderboard
    assert len(dash.model_championship.leaderboard) > 0
    champion_rows = [r for r in dash.model_championship.leaderboard if r.is_champion]
    assert len(champion_rows) == 1

    # Error analysis
    assert dash.error_analysis.status == "SUCCESS"
    assert dash.error_analysis.confusion_matrix is not None

    # Charts
    chart_ids = [c.chart_id for c in dash.charts]
    assert "chart_model_benchmarks" in chart_ids
    assert "chart_confusion_matrix" in chart_ids

    # Report
    assert dash.report.dataset_name == "binary_eval.csv"
    assert dash.report.model_championship_summary["champion"] is not None

    # Serialization
    json_str = dash.model_dump_json()
    assert "chart_model_benchmarks" in json_str
    assert "dashboard_id" in json_str


def test_dashboard_spec_strictly_preserves_analytical_truth():
    """Prove that DashboardSpec acts solely as a presentation transformer without recalculating truth."""
    rng = np.random.default_rng(123)
    n = 60
    df = pd.DataFrame({
        "feat_x": rng.normal(5, 1, size=n),
        "feat_y": rng.uniform(0, 10, size=n),
        "target": rng.choice([0, 1], size=n),
    })

    discovery = build_discovery_contract_for_df(df, filename="truth_preserve_test.csv")
    analysis = run_ml_pipeline(df, discovery)
    insights = run_aida_reasoning(analysis, max_iterations=2)

    dash = build_dashboard_spec(
        discovery=discovery,
        analysis=analysis,
        insights=insights,
        dataset_name="truth_preserve_test.csv",
    )

    # 1. Champion model identity and score must be exact references
    assert dash.model_championship.champion_model_name == analysis.winner.model_name
    assert dash.model_championship.champion_score == pytest.approx(analysis.winner.score, rel=1e-5)

    # 2. Insight claims, methods, and source paths must be preserved verbatim
    assert len(dash.insights) == len(insights.insights)
    for dash_ins, true_ins in zip(dash.insights, insights.insights):
        assert dash_ins.claim == true_ins.claim
        assert dash_ins.method == true_ins.method
        assert dash_ins.source_paths == true_ins.source_paths

    # 3. Data quality metrics must match discovery report verbatim
    assert dash.data_quality.row_count == discovery.fingerprint["row_count"]
    assert dash.data_quality.column_count == discovery.fingerprint["column_count"]
    assert dash.data_quality.duplicate_rows == discovery.quality_report["duplicate_rows"]
    assert dash.data_quality.target_candidates == discovery.target_candidates

