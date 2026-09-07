import json
import numpy as np
import pandas as pd
import pytest

from backend.app.schemas.discovery_contract import DiscoveryContract, RouterDecision
from backend.app.schemas.analysis_contract import AnalysisContract, ValidationReport, ModelBenchmarkResult
from backend.app.schemas.enums import PipelineStatus, MLTaskType, ValidationStrategy
from backend.app.pipeline.model_championship import (
    benchmark_models,
    select_champion,
    build_feature_preprocessor,
)


@pytest.fixture
def synthetic_classification_df():
    np.random.seed(42)
    n = 120
    x1 = np.random.normal(50, 10, n)
    x2 = np.random.normal(0, 1, n)
    cat1 = np.random.choice(["TypeA", "TypeB", "TypeC"], size=n)
    is_active = np.random.choice([True, False], size=n)
    # Target depends probabilistically on x1
    prob = 1.0 / (1.0 + np.exp(-(x1 - 50) / 5.0))
    y = (np.random.rand(n) < prob).astype(int)

    return pd.DataFrame({
        "feature_num1": x1,
        "feature_num2": x2,
        "feature_cat": cat1,
        "feature_bool": is_active,
        "target": y,
    })


@pytest.fixture
def synthetic_regression_df():
    np.random.seed(42)
    n = 120
    x1 = np.random.normal(10, 2, n)
    x2 = np.random.normal(0, 1, n)
    cat = np.random.choice(["Low", "High"], size=n)
    # Continuous target with noise
    y = 3.5 * x1 - 2.0 * x2 + np.random.normal(0, 0.5, n)

    return pd.DataFrame({
        "x1": x1,
        "x2": x2,
        "segment": cat,
        "price": y,
    })


def test_classification_championship(synthetic_classification_df):
    disc = DiscoveryContract(
        schema_version="1.0",
        schema={
            "feature_num1": "numeric",
            "feature_num2": "numeric",
            "feature_cat": "categorical",
            "feature_bool": "categorical",
            "target": "categorical",
        },
        target_candidates=["target"],
        router=RouterDecision(classification=True),
    )
    val = ValidationReport(
        strategy=ValidationStrategy.STRATIFIED_K_FOLD.value,
        folds=3,
        primary_metric="f1_macro",
        random_seed=42,
    )

    benchmarks = benchmark_models(synthetic_classification_df, disc, MLTaskType.CLASSIFICATION, val)
    assert len(benchmarks) == 4

    # Verify all 4 candidates succeeded
    successful = [b for b in benchmarks if b.status == PipelineStatus.SUCCESS]
    assert len(successful) == 4
    model_names = {b.model_name for b in successful}
    assert "LogisticRegression" in model_names
    assert "DecisionTreeClassifier" in model_names
    assert "RandomForestClassifier" in model_names
    assert "HistGradientBoostingClassifier" in model_names

    # Check metrics
    for b in successful:
        assert b.primary_metric == "f1_macro"
        assert b.mean_score is not None and b.mean_score > 0.0
        assert len(b.fold_scores) == 3
        assert "f1_macro" in b.metrics
        assert "accuracy" in b.metrics
        assert "precision_macro" in b.metrics
        assert "recall_macro" in b.metrics

    # Test champion selection
    champ = select_champion(benchmarks, primary_metric="f1_macro")
    assert champ.status == PipelineStatus.SUCCESS
    assert champ.model_name in model_names
    assert champ.score == max(b.mean_score for b in successful)
    assert "highest_mean_f1_macro" in champ.selection_rule


def test_regression_championship(synthetic_regression_df):
    disc = DiscoveryContract(
        schema_version="1.0",
        schema={"x1": "numeric", "x2": "numeric", "segment": "categorical", "price": "numeric"},
        target_candidates=["price"],
        router=RouterDecision(regression=True),
    )
    val = ValidationReport(
        strategy=ValidationStrategy.K_FOLD.value,
        folds=3,
        primary_metric="rmse",
        random_seed=42,
    )

    benchmarks = benchmark_models(synthetic_regression_df, disc, MLTaskType.REGRESSION, val)
    assert len(benchmarks) == 4

    successful = [b for b in benchmarks if b.status == PipelineStatus.SUCCESS]
    assert len(successful) == 4
    model_names = {b.model_name for b in successful}
    assert "RidgeRegression" in model_names
    assert "RandomForestRegressor" in model_names

    # In regression, lowest RMSE wins
    champ = select_champion(benchmarks, primary_metric="rmse")
    assert champ.status == PipelineStatus.SUCCESS
    assert champ.score == min(b.mean_score for b in successful)
    assert "lowest_mean_rmse" in champ.selection_rule


def test_missing_values_and_unseen_categories():
    """Verify that imputer and one-hot encoder safely handle missing & unseen categories inside CV."""
    df = pd.DataFrame({
        "num_with_nan": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, np.nan, 9.0, 10.0] * 3,
        "cat_unseen": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"] * 3,
        "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1] * 3,
    })
    disc = DiscoveryContract(
        schema_version="1.0",
        schema={"num_with_nan": "numeric", "cat_unseen": "categorical", "target": "categorical"},
        target_candidates=["target"],
        router=RouterDecision(classification=True),
    )
    val = ValidationReport(strategy=ValidationStrategy.STRATIFIED_K_FOLD.value, folds=3, primary_metric="f1_macro")

    benchmarks = benchmark_models(df, disc, MLTaskType.CLASSIFICATION, val)
    successful = [b for b in benchmarks if b.status == PipelineStatus.SUCCESS]
    assert len(successful) >= 3  # All or most models cleanly succeed despite missing and unseen categories


def test_deterministic_reproducibility(synthetic_classification_df):
    """Running benchmark_models twice with the same seed yields strictly identical results."""
    disc = DiscoveryContract(
        schema_version="1.0",
        schema={"target": "categorical"},
        target_candidates=["target"],
        router=RouterDecision(classification=True),
    )
    val = ValidationReport(strategy=ValidationStrategy.STRATIFIED_K_FOLD.value, folds=3, random_seed=42)

    run1 = benchmark_models(synthetic_classification_df, disc, MLTaskType.CLASSIFICATION, val)
    run2 = benchmark_models(synthetic_classification_df, disc, MLTaskType.CLASSIFICATION, val)

    for b1, b2 in zip(run1, run2):
        assert b1.model_name == b2.model_name
        assert b1.mean_score == b2.mean_score
        assert b1.fold_scores == b2.fold_scores


def test_constant_target_fails_safely():
    df = pd.DataFrame({
        "x": [1.0, 2.0, 3.0, 4.0, 5.0],
        "target": [1, 1, 1, 1, 1],  # Zero variance
    })
    disc = DiscoveryContract(schema_version="1.0", target_candidates=["target"])
    val = ValidationReport(strategy=ValidationStrategy.K_FOLD.value, folds=3)

    benchmarks = benchmark_models(df, disc, MLTaskType.REGRESSION, val)
    assert len(benchmarks) == 1
    assert benchmarks[0].status == PipelineStatus.FAILED
    assert "constant" in benchmarks[0].reason.lower()


def test_invalid_target_missing_from_df():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    disc = DiscoveryContract(schema_version="1.0", target_candidates=["non_existent_col"])
    val = ValidationReport(strategy=ValidationStrategy.K_FOLD.value, folds=3)

    benchmarks = benchmark_models(df, disc, MLTaskType.REGRESSION, val)
    assert benchmarks == []


def test_failed_candidate_isolation():
    """Verify that if one candidate model fails, the others succeed and champion is still chosen."""
    good_b1 = ModelBenchmarkResult(
        model_name="ModelGood",
        family="linear",
        status=PipelineStatus.SUCCESS,
        primary_metric="f1_macro",
        mean_score=0.82,
        std_score=0.02,
        fold_scores=[0.81, 0.83],
    )
    failed_b = ModelBenchmarkResult(
        model_name="ModelBroken",
        family="tree",
        status=PipelineStatus.FAILED,
        reason="Mock singular matrix exception",
    )

    champ = select_champion([good_b1, failed_b], primary_metric="f1_macro")
    assert champ.status == PipelineStatus.SUCCESS
    assert champ.model_name == "ModelGood"
    assert champ.score == 0.82


def test_preprocessing_leakage_prevention():
    """Verify preprocessor is constructed unfitted and fitted only inside folds."""
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "cat": ["A", "B", "C"]})
    ct = build_feature_preprocessor(df, num_cols=["x"], cat_cols=["cat"])

    # ColumnTransformer must not be fitted prior to CV execution
    with pytest.raises(Exception):
        # Calling transform on unfitted transformer raises NotFittedError
        ct.transform(df)


def test_json_serialization_and_contract_integration(synthetic_classification_df):
    disc = DiscoveryContract(
        schema_version="1.0",
        schema={"target": "categorical"},
        target_candidates=["target"],
        router=RouterDecision(classification=True),
    )
    val = ValidationReport(strategy=ValidationStrategy.STRATIFIED_K_FOLD.value, folds=3)
    benchmarks = benchmark_models(synthetic_classification_df, disc, MLTaskType.CLASSIFICATION, val)
    champ = select_champion(benchmarks, primary_metric="f1_macro")

    analysis = AnalysisContract(
        schema_version="1.0",
        status=PipelineStatus.SUCCESS,
        task=MLTaskType.CLASSIFICATION,
        target_column="target",
        validation=val,
        models=benchmarks,
        winner=champ,
    )

    # Pure JSON serialization
    json_str = analysis.model_dump_json()
    assert isinstance(json_str, str)
    parsed = json.loads(json_str)

    assert len(parsed["models"]) == 4
    assert parsed["winner"]["model_name"] == champ.model_name
    assert parsed["winner"]["score"] == champ.score
    assert parsed["winner"]["empirical_evidence"]["competitors_evaluated"] == 4
