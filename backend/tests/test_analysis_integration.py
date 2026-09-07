"""
End-to-End Analysis Contract Integration Tests (Person 2 - Phase 2.5)
Verifies the complete integration of Person 2's ML pipeline:
DiscoveryContract -> Statistical Analysis -> Validation Strategy -> Model Championship ->
OOF Error Analysis -> Controlled Experiments & Ensemble -> Validated AnalysisContract.
100% deterministic, zero LLM dependency.
"""
import json
import pytest
import numpy as np
import pandas as pd

from backend.app.schemas.enums import (
    PipelineStatus,
    MLTaskType,
    ValidationStrategy,
)
from backend.app.schemas.discovery_contract import DiscoveryContract, RouterDecision
from backend.app.schemas.analysis_contract import AnalysisContract
from backend.app.pipeline.ml_analysis import run_ml_pipeline, determine_ml_task


# ---------------------------------------------------------------------------
# Synthetic Dataset Generators for Testing
# ---------------------------------------------------------------------------

def make_synthetic_binary_classification(n: int = 120, seed: int = 42) -> pd.DataFrame:
    """Synthetic binary classification dataset with numeric, categorical, and missing values."""
    rng = np.random.default_rng(seed)
    num1 = rng.normal(10.0, 3.0, size=n)
    num2 = rng.exponential(2.0, size=n)
    cat1 = rng.choice(["tier_A", "tier_B", "tier_C"], size=n)
    cat2 = rng.choice(["region_north", "region_south"], size=n)

    # Probabilistic binary target based on features
    logits = 0.5 * (num1 - 10.0) - 0.4 * num2 + (cat1 == "tier_A") * 1.5
    probs = 1.0 / (1.0 + np.exp(-logits))
    target = (probs > 0.5).astype(int)

    df = pd.DataFrame({
        "feature_num1": num1,
        "feature_num2": num2,
        "feature_cat1": cat1,
        "feature_cat2": cat2,
        "target": target,
    })

    # Introduce some missing values
    df.loc[rng.choice(n, size=5, replace=False), "feature_num1"] = np.nan
    df.loc[rng.choice(n, size=4, replace=False), "feature_cat1"] = np.nan
    return df


def make_synthetic_multiclass_classification(n: int = 90, seed: int = 42) -> pd.DataFrame:
    """Synthetic multiclass classification dataset with 3 balanced classes."""
    rng = np.random.default_rng(seed)
    num1 = rng.normal(0.0, 1.0, size=n)
    num2 = rng.normal(0.0, 1.0, size=n)
    target = np.array([0] * (n // 3) + [1] * (n // 3) + [2] * (n - 2 * (n // 3)))

    # Shift features conditionally
    num1[target == 1] += 3.0
    num2[target == 2] += 3.0

    return pd.DataFrame({
        "feat_x": num1,
        "feat_y": num2,
        "category": rng.choice(["low", "medium", "high"], size=n),
        "target_class": target,
    })


def make_synthetic_regression(n: int = 100, seed: int = 42) -> pd.DataFrame:
    """Synthetic regression dataset with numeric, categorical, and continuous target."""
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(1.0, 20.0, size=n)
    x2 = rng.normal(5.0, 2.0, size=n)
    group = rng.choice(["grp1", "grp2"], size=n)
    noise = rng.normal(0.0, 0.5, size=n)

    y = 2.0 * x1 - 1.5 * x2 + (group == "grp1") * 3.0 + noise

    df = pd.DataFrame({
        "predictor_1": x1,
        "predictor_2": x2,
        "group_label": group,
        "target_value": y,
    })
    # Add a few NaNs in predictors
    df.loc[rng.choice(n, size=3, replace=False), "predictor_1"] = np.nan
    return df


# ---------------------------------------------------------------------------
# End-to-End Integration Tests
# ---------------------------------------------------------------------------

def test_1_classification_end_to_end():
    """1. Binary classification end-to-end integration."""
    df = make_synthetic_binary_classification(n=100, seed=42)
    discovery = DiscoveryContract(
        schema_version="1.0",
        schema={
            "feature_num1": "numeric",
            "feature_num2": "numeric",
            "feature_cat1": "categorical",
            "feature_cat2": "categorical",
            "target": "numeric",
        },
        target_candidates=["target"],
        router=RouterDecision(classification=True),
    )

    contract = run_ml_pipeline(df, discovery)

    assert contract.status == PipelineStatus.SUCCESS
    assert contract.task == MLTaskType.CLASSIFICATION
    assert contract.target_column == "target"
    assert contract.dataset_context["n_rows"] == 100
    assert contract.dataset_context["n_cols"] == 5

    # Statistical report
    assert contract.statistical_analysis.status == PipelineStatus.SUCCESS
    assert "feature_num1" in contract.statistical_analysis.numeric_summary

    # Validation & Championship
    assert contract.validation.status == PipelineStatus.SUCCESS
    assert contract.validation.strategy == ValidationStrategy.STRATIFIED_K_FOLD.value
    assert contract.winner.status == PipelineStatus.SUCCESS
    assert contract.winner.model_name in ["LogisticRegression", "DecisionTree", "RandomForest", "HistGradientBoosting"]
    assert contract.winner.score is not None
    assert contract.winner.score > 0.5

    # Error analysis
    assert contract.error_analysis.status == PipelineStatus.SUCCESS
    assert "confusion_matrix" in contract.error_analysis.classification

    # Experiments & Ensemble
    assert len(contract.experiments) > 0
    assert contract.ensemble.status == PipelineStatus.SUCCESS

    # Unrequested sections are NOT_APPLICABLE
    assert contract.time_analysis.status == PipelineStatus.NOT_APPLICABLE


def test_2_multiclass_end_to_end():
    """2. Multiclass classification end-to-end integration."""
    df = make_synthetic_multiclass_classification(n=90, seed=42)
    discovery = DiscoveryContract(
        schema_version="1.0",
        target_candidates=["target_class"],
        router=RouterDecision(classification=True),
    )

    contract = run_ml_pipeline(df, discovery)

    assert contract.status == PipelineStatus.SUCCESS
    assert contract.task == MLTaskType.CLASSIFICATION
    assert contract.winner.status == PipelineStatus.SUCCESS
    assert contract.error_analysis.status == PipelineStatus.SUCCESS
    per_class = contract.error_analysis.classification.get("class_metrics", [])
    assert len(per_class) == 3


def test_3_regression_end_to_end():
    """3. Regression end-to-end integration."""
    df = make_synthetic_regression(n=100, seed=42)
    discovery = DiscoveryContract(
        schema_version="1.0",
        schema={
            "predictor_1": "numeric",
            "predictor_2": "numeric",
            "group_label": "categorical",
            "target_value": "numeric",
        },
        target_candidates=["target_value"],
        router=RouterDecision(regression=True),
    )

    contract = run_ml_pipeline(df, discovery)

    assert contract.status == PipelineStatus.SUCCESS
    assert contract.task == MLTaskType.REGRESSION
    assert contract.validation.strategy == ValidationStrategy.K_FOLD.value
    assert contract.winner.status == PipelineStatus.SUCCESS
    assert contract.winner.primary_metric == "rmse"
    assert contract.error_analysis.status == PipelineStatus.SUCCESS
    assert contract.error_analysis.regression["residual_summary"]["rmse"] > 0.0
    assert contract.ensemble.status == PipelineStatus.SUCCESS


def test_4_no_target_routing():
    """4. Dataset with no supervised target (clustering/unsupervised only)."""
    df = pd.DataFrame({
        "dim1": [1.0, 2.0, 3.0, 4.0, 5.0],
        "dim2": [10.0, 20.0, 30.0, 40.0, 50.0],
    })
    discovery = DiscoveryContract(
        schema_version="1.0",
        target_candidates=[],
        router=RouterDecision(clustering=True),
    )

    contract = run_ml_pipeline(df, discovery)

    assert contract.status == PipelineStatus.SUCCESS
    assert contract.statistical_analysis.status == PipelineStatus.SUCCESS
    assert contract.validation.status == PipelineStatus.NOT_APPLICABLE
    assert contract.winner.status == PipelineStatus.NOT_APPLICABLE
    assert len(contract.models) == 0
    assert contract.error_analysis.status == PipelineStatus.NOT_APPLICABLE
    assert len(contract.experiments) == 0
    assert contract.ensemble.status == PipelineStatus.NOT_APPLICABLE


def test_5_temporal_routing():
    """5. Temporal dataset routing with time analysis enabled."""
    df = make_synthetic_regression(n=60, seed=42)
    discovery = DiscoveryContract(
        schema_version="1.0",
        target_candidates=["target_value"],
        router=RouterDecision(regression=True, time_analysis=True),
    )

    contract = run_ml_pipeline(df, discovery)

    assert contract.status == PipelineStatus.SUCCESS
    assert contract.validation.strategy == ValidationStrategy.TIME_SERIES_SPLIT.value
    assert contract.time_analysis.status == PipelineStatus.SUCCESS
    assert contract.winner.status == PipelineStatus.SUCCESS


def test_6_missing_values_end_to_end():
    """6. Preprocessing handles missing values inside folds without leakage."""
    df = make_synthetic_binary_classification(n=80, seed=99)
    # Check that missing values exist
    assert df.isna().sum().sum() > 0

    discovery = DiscoveryContract(
        schema_version="1.0",
        target_candidates=["target"],
        router=RouterDecision(classification=True),
    )

    contract = run_ml_pipeline(df, discovery)

    assert contract.status == PipelineStatus.SUCCESS
    assert contract.winner.status == PipelineStatus.SUCCESS
    assert contract.statistical_analysis.numeric_summary["feature_num1"]["missing_count"] > 0


def test_7_contract_pydantic_validation():
    """7. Full contract undergoes and passes strict Pydantic model validation."""
    df = make_synthetic_binary_classification(n=60, seed=42)
    discovery = DiscoveryContract(
        schema_version="1.0",
        target_candidates=["target"],
        router=RouterDecision(classification=True),
    )

    contract = run_ml_pipeline(df, discovery)
    dumped = contract.model_dump()
    reloaded = AnalysisContract.model_validate(dumped)

    assert reloaded.status == contract.status
    assert reloaded.winner.model_name == contract.winner.model_name
    assert reloaded.schema_version == "1.0"


def test_8_json_serialization():
    """8. Ensure contract can be serialized to JSON without numpy/pandas artifacts."""
    df = make_synthetic_binary_classification(n=60, seed=42)
    discovery = DiscoveryContract(
        schema_version="1.0",
        target_candidates=["target"],
        router=RouterDecision(classification=True),
    )

    contract = run_ml_pipeline(df, discovery)
    json_str = contract.model_dump_json()
    parsed = json.loads(json_str)

    assert parsed["status"] == "SUCCESS"
    assert parsed["schema_version"] == "1.0"
    assert "winner" in parsed
    assert "dataset_context" in parsed


def test_9_failure_isolation():
    """9. Target missing from dataset fails championship gracefully without pipeline crash."""
    df = pd.DataFrame({"col_a": [1, 2, 3], "col_b": [4, 5, 6]})
    discovery = DiscoveryContract(
        schema_version="1.0",
        target_candidates=["non_existent_target"],
        router=RouterDecision(classification=True),
    )

    contract = run_ml_pipeline(df, discovery)

    assert contract.status == PipelineStatus.FAILED
    assert "not found" in contract.reason.lower()
    assert contract.statistical_analysis.status == PipelineStatus.SUCCESS
    assert contract.winner.status == PipelineStatus.FAILED
    assert contract.error_analysis.status == PipelineStatus.NOT_APPLICABLE


def test_10_deterministic_repeatability():
    """10. Repeated pipeline executions on identical data produce identical results."""
    df = make_synthetic_binary_classification(n=70, seed=123)
    discovery = DiscoveryContract(
        schema_version="1.0",
        target_candidates=["target"],
        router=RouterDecision(classification=True),
    )

    run1 = run_ml_pipeline(df, discovery)
    run2 = run_ml_pipeline(df, discovery)

    assert run1.status == run2.status
    assert run1.winner.model_name == run2.winner.model_name
    assert run1.winner.score == run2.winner.score
    assert run1.ensemble.score == run2.ensemble.score
    assert len(run1.experiments) == len(run2.experiments)
    for exp1, exp2 in zip(run1.experiments, run2.experiments):
        assert exp1.feature == exp2.feature
        assert exp1.delta == exp2.delta


def test_11_oof_error_analysis_handoff():
    """11. OOF predictions from champion correctly flow into error analysis."""
    df = make_synthetic_regression(n=70, seed=42)
    discovery = DiscoveryContract(
        schema_version="1.0",
        target_candidates=["target_value"],
        router=RouterDecision(regression=True),
    )

    contract = run_ml_pipeline(df, discovery)

    assert contract.winner.status == PipelineStatus.SUCCESS
    assert "oof_predictions" in contract.winner.empirical_evidence
    assert contract.error_analysis.status == PipelineStatus.SUCCESS
    assert "residual_summary" in contract.error_analysis.regression
    assert contract.error_analysis.regression["residual_summary"]["rmse"] > 0
    assert contract.error_analysis.summary.get("total_samples") == 70


def test_12_experiment_handoff():
    """12. Experiment records and ensemble status are captured cleanly for Person 3."""
    df = make_synthetic_binary_classification(n=80, seed=42)
    discovery = DiscoveryContract(
        schema_version="1.0",
        target_candidates=["target"],
        router=RouterDecision(classification=True),
    )

    contract = run_ml_pipeline(df, discovery)

    assert len(contract.experiments) > 0
    first_exp = contract.experiments[0]
    assert first_exp.experiment_type == "feature_ablation"
    assert first_exp.baseline_score is not None
    assert first_exp.interpretation is not None

    assert contract.ensemble.status == PipelineStatus.SUCCESS
    assert isinstance(contract.ensemble.improves_champion, bool)


def test_13_not_applicable_behavior():
    """13. Sections that do not apply explicitly report NOT_APPLICABLE with reasons."""
    df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    discovery = DiscoveryContract(
        schema_version="1.0",
        target_candidates=[],
        router=RouterDecision(classification=False, regression=False, time_analysis=False),
    )

    contract = run_ml_pipeline(df, discovery)

    assert contract.validation.status == PipelineStatus.NOT_APPLICABLE
    assert contract.validation.reason is not None
    assert contract.winner.status == PipelineStatus.NOT_APPLICABLE
    assert contract.error_analysis.status == PipelineStatus.NOT_APPLICABLE
    assert contract.ensemble.status == PipelineStatus.NOT_APPLICABLE
    assert contract.time_analysis.status == PipelineStatus.NOT_APPLICABLE


def test_14_invalid_discovery_contract_handling():
    """14. Graceful handling of invalid or malformed DiscoveryContract inputs."""
    df = pd.DataFrame({"col": [1, 2, 3]})

    # None provided
    c_none = run_ml_pipeline(df, None)
    assert c_none.status == PipelineStatus.FAILED
    assert "none" in c_none.reason.lower()

    # Completely invalid string
    c_str = run_ml_pipeline(df, "not_a_contract")
    assert c_str.status == PipelineStatus.FAILED
    assert "invalid" in c_str.reason.lower()

    # Malformed dictionary missing required types
    c_dict = run_ml_pipeline(df, {"router": "invalid_not_a_dict"})
    assert c_dict.status == PipelineStatus.FAILED
    assert "validation" in c_dict.reason.lower() or "schema" in c_dict.reason.lower()
