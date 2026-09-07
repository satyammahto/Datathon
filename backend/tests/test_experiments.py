import json
import numpy as np
import pandas as pd
import pytest

from backend.app.schemas.discovery_contract import DiscoveryContract, RouterDecision
from backend.app.schemas.analysis_contract import (
    AnalysisContract,
    ChampionResult,
    ValidationReport,
    ModelBenchmarkResult,
)
from backend.app.schemas.enums import PipelineStatus, MLTaskType, ValidationStrategy
from backend.app.pipeline.model_championship import benchmark_models, select_champion
from backend.app.pipeline.experiments import (
    run_feature_ablation,
    run_permutation_importance,
    run_ensemble_experiment,
    run_controlled_experiments,
)


@pytest.fixture
def synthetic_classification_dataset():
    np.random.seed(42)
    n = 100
    # informative feature
    x_vital = np.random.normal(50, 10, n)
    # random noise feature
    x_noise = np.random.normal(0, 1, n)
    # categorical feature
    cat_f = np.random.choice(["A", "B"], size=n)

    # Target depends strongly on x_vital
    prob = 1.0 / (1.0 + np.exp(-(x_vital - 50) / 4.0))
    y = (np.random.rand(n) < prob).astype(int)

    return pd.DataFrame({
        "vital_feature": x_vital,
        "noise_feature": x_noise,
        "group_cat": cat_f,
        "target": y,
    })


@pytest.fixture
def synthetic_regression_dataset():
    np.random.seed(42)
    n = 100
    x1 = np.random.normal(10, 2, n)
    x2 = np.random.normal(5, 1, n)
    x_noise = np.random.normal(0, 1, n)
    cat_f = np.random.choice(["Low", "High"], size=n)

    y = 4.0 * x1 - 2.5 * x2 + np.random.normal(0, 0.5, n)

    return pd.DataFrame({
        "feat_1": x1,
        "feat_2": x2,
        "feat_noise": x_noise,
        "segment": cat_f,
        "target": y,
    })


# ---------------------------------------------------------------------------
# 1. Feature Ablation Tests
# ---------------------------------------------------------------------------

def test_leave_one_feature_out_classification(synthetic_classification_dataset):
    disc = DiscoveryContract(
        schema_version="1.0",
        schema={"vital_feature": "numeric", "noise_feature": "numeric", "group_cat": "categorical", "target": "categorical"},
        target_candidates=["target"],
        router=RouterDecision(classification=True),
    )
    val = ValidationReport(strategy=ValidationStrategy.STRATIFIED_K_FOLD.value, folds=3, random_seed=42)

    benchmarks = benchmark_models(synthetic_classification_dataset, disc, MLTaskType.CLASSIFICATION, val)
    champ = select_champion(benchmarks, primary_metric="f1_macro")

    # Run feature ablation
    ablations = run_feature_ablation(
        synthetic_classification_dataset, disc, MLTaskType.CLASSIFICATION, val, champ
    )

    assert len(ablations) == 3
    features_ablated = {a.feature for a in ablations}
    assert "vital_feature" in features_ablated
    assert "noise_feature" in features_ablated

    vital_exp = next(a for a in ablations if a.feature == "vital_feature")
    assert vital_exp.status == PipelineStatus.SUCCESS
    assert vital_exp.baseline_score is not None
    assert vital_exp.candidate_metric is not None
    assert vital_exp.delta is not None
    # Removing vital feature causes performance decrease (negative delta)
    assert vital_exp.delta < 0.0
    assert vital_exp.interpretation == "performance_decreased"


def test_leave_one_feature_out_regression(synthetic_regression_dataset):
    disc = DiscoveryContract(
        schema_version="1.0",
        schema={"feat_1": "numeric", "feat_2": "numeric", "feat_noise": "numeric", "segment": "categorical", "target": "numeric"},
        target_candidates=["target"],
        router=RouterDecision(regression=True),
    )
    val = ValidationReport(strategy=ValidationStrategy.K_FOLD.value, folds=3, random_seed=42, primary_metric="rmse")

    benchmarks = benchmark_models(synthetic_regression_dataset, disc, MLTaskType.REGRESSION, val)
    champ = select_champion(benchmarks, primary_metric="rmse")

    ablations = run_feature_ablation(
        synthetic_regression_dataset, disc, MLTaskType.REGRESSION, val, champ
    )

    assert len(ablations) == 4
    feat1_exp = next(a for a in ablations if a.feature == "feat_1")
    assert feat1_exp.status == PipelineStatus.SUCCESS
    # In regression, higher RMSE is worse (positive delta)
    assert feat1_exp.delta > 0.0
    assert feat1_exp.interpretation == "performance_decreased"


def test_bounded_experiment_budget(synthetic_regression_dataset):
    disc = DiscoveryContract(
        schema_version="1.0",
        target_candidates=["target"],
        router=RouterDecision(regression=True),
    )
    val = ValidationReport(strategy=ValidationStrategy.K_FOLD.value, folds=3, random_seed=42)
    benchmarks = benchmark_models(synthetic_regression_dataset, disc, MLTaskType.REGRESSION, val)
    champ = select_champion(benchmarks, primary_metric="rmse")

    # Set budget = 2
    ablations = run_feature_ablation(
        synthetic_regression_dataset, disc, MLTaskType.REGRESSION, val, champ, max_ablation_features=2
    )
    assert len(ablations) == 2


def test_ablation_skipped_on_single_feature():
    df = pd.DataFrame({"single_feature": [1.0, 2.0, 3.0, 4.0], "target": [0, 1, 0, 1]})
    disc = DiscoveryContract(schema_version="1.0", target_candidates=["target"])
    val = ValidationReport(strategy=ValidationStrategy.K_FOLD.value, folds=2)
    champ = ChampionResult(status=PipelineStatus.SUCCESS, model_name="LogisticRegression", score=0.8)

    ablations = run_feature_ablation(df, disc, MLTaskType.CLASSIFICATION, val, champ)
    assert len(ablations) == 1
    assert ablations[0].status == PipelineStatus.NOT_APPLICABLE
    assert "at least 2 feature columns" in ablations[0].reason.lower() or "only 1 feature" in ablations[0].hypothesis.lower()


# ---------------------------------------------------------------------------
# 2. Permutation Importance Diagnostics Tests
# ---------------------------------------------------------------------------

def test_permutation_importance_diagnostic(synthetic_classification_dataset):
    disc = DiscoveryContract(schema_version="1.0", target_candidates=["target"])
    val = ValidationReport(strategy=ValidationStrategy.STRATIFIED_K_FOLD.value, folds=3)
    champ = ChampionResult(status=PipelineStatus.SUCCESS, model_name="RandomForestClassifier", score=0.85)

    diags = run_permutation_importance(
        synthetic_classification_dataset, disc, MLTaskType.CLASSIFICATION, val, champ
    )
    assert len(diags) > 0
    assert "disclaimer" in diags[0]
    assert "does not imply causal" in diags[0]["disclaimer"].lower()


# ---------------------------------------------------------------------------
# 3. Ensemble Comparison & Critical Champion Retention Tests
# ---------------------------------------------------------------------------

def test_ensemble_experiment_classification(synthetic_classification_dataset):
    disc = DiscoveryContract(
        schema_version="1.0",
        schema={"target": "categorical"},
        target_candidates=["target"],
        router=RouterDecision(classification=True),
    )
    val = ValidationReport(strategy=ValidationStrategy.STRATIFIED_K_FOLD.value, folds=3, random_seed=42)

    benchmarks = benchmark_models(synthetic_classification_dataset, disc, MLTaskType.CLASSIFICATION, val)
    champ = select_champion(benchmarks, primary_metric="f1_macro")

    ens_report = run_ensemble_experiment(
        synthetic_classification_dataset, disc, MLTaskType.CLASSIFICATION, val, champ, benchmarks, top_k=2
    )

    assert ens_report.status == PipelineStatus.SUCCESS
    assert len(ens_report.member_models) == 2
    assert ens_report.ensemble_score is not None
    assert ens_report.champion_score == champ.score
    assert isinstance(ens_report.improves_champion, bool)
    assert "identical_cross_validation_folds" in ens_report.details["selection_protocol"]


def test_ensemble_experiment_regression(synthetic_regression_dataset):
    disc = DiscoveryContract(
        schema_version="1.0",
        target_candidates=["target"],
        router=RouterDecision(regression=True),
    )
    val = ValidationReport(strategy=ValidationStrategy.K_FOLD.value, folds=3, random_seed=42, primary_metric="rmse")

    benchmarks = benchmark_models(synthetic_regression_dataset, disc, MLTaskType.REGRESSION, val)
    champ = select_champion(benchmarks, primary_metric="rmse")

    ens_report = run_ensemble_experiment(
        synthetic_regression_dataset, disc, MLTaskType.REGRESSION, val, champ, benchmarks, top_k=2
    )

    assert ens_report.status == PipelineStatus.SUCCESS
    assert ens_report.ensemble_type == "prediction_averaging"
    assert ens_report.delta is not None


def test_ensemble_insufficient_candidates_fails_safely():
    # Only 1 successful model
    b1 = ModelBenchmarkResult(
        model_name="OnlyOne",
        family="linear",
        status=PipelineStatus.SUCCESS,
        mean_score=0.8,
    )
    disc = DiscoveryContract(schema_version="1.0", target_candidates=["target"])
    val = ValidationReport(strategy=ValidationStrategy.K_FOLD.value, folds=3)
    champ = ChampionResult(status=PipelineStatus.SUCCESS, model_name="OnlyOne", score=0.8)

    ens_report = run_ensemble_experiment(
        pd.DataFrame({"x": [1, 2], "target": [0, 1]}), disc, MLTaskType.CLASSIFICATION, val, champ, [b1]
    )
    assert ens_report.status == PipelineStatus.NOT_APPLICABLE
    assert "at least 2" in ens_report.reason.lower()


# ---------------------------------------------------------------------------
# 4. Master Orchestration, Determinism & Contract Integration Tests
# ---------------------------------------------------------------------------

def test_controlled_experiments_master_orchestration(synthetic_classification_dataset):
    disc = DiscoveryContract(
        schema_version="1.0",
        schema={"target": "categorical"},
        target_candidates=["target"],
        router=RouterDecision(classification=True),
    )
    val = ValidationReport(strategy=ValidationStrategy.STRATIFIED_K_FOLD.value, folds=3, random_seed=42)

    benchmarks = benchmark_models(synthetic_classification_dataset, disc, MLTaskType.CLASSIFICATION, val)
    champ = select_champion(benchmarks, primary_metric="f1_macro")

    ablation_recs, ens_report, summary = run_controlled_experiments(
        synthetic_classification_dataset, disc, MLTaskType.CLASSIFICATION, val, champ, benchmarks
    )

    assert len(ablation_recs) >= 2
    assert ens_report.status == PipelineStatus.SUCCESS
    assert summary["status"] == "SUCCESS"
    assert summary["baseline_champion"] == champ.model_name

    # Contract integration
    contract = AnalysisContract(
        schema_version="1.0",
        status=PipelineStatus.SUCCESS,
        task=MLTaskType.CLASSIFICATION,
        target_column="target",
        validation=val,
        models=benchmarks,
        winner=champ,
        experiments=ablation_recs,
        ensemble=ens_report,
    )

    json_str = contract.model_dump_json()
    assert isinstance(json_str, str)
    parsed = json.loads(json_str)

    assert len(parsed["experiments"]) == len(ablation_recs)
    assert parsed["ensemble"]["status"] == "SUCCESS"
    assert parsed["ensemble"]["champion_score"] == champ.score


def test_deterministic_repeatability(synthetic_classification_dataset):
    disc = DiscoveryContract(schema_version="1.0", target_candidates=["target"], router=RouterDecision(classification=True))
    val = ValidationReport(strategy=ValidationStrategy.STRATIFIED_K_FOLD.value, folds=3, random_seed=42)
    benchmarks = benchmark_models(synthetic_classification_dataset, disc, MLTaskType.CLASSIFICATION, val)
    champ = select_champion(benchmarks, primary_metric="f1_macro")

    run1_ablations, run1_ens, _ = run_controlled_experiments(synthetic_classification_dataset, disc, MLTaskType.CLASSIFICATION, val, champ, benchmarks)
    run2_ablations, run2_ens, _ = run_controlled_experiments(synthetic_classification_dataset, disc, MLTaskType.CLASSIFICATION, val, champ, benchmarks)

    assert len(run1_ablations) == len(run2_ablations)
    for a1, a2 in zip(run1_ablations, run2_ablations):
        assert a1.feature == a2.feature
        assert a1.delta == a2.delta
        assert a1.candidate_metric == a2.candidate_metric

    assert run1_ens.ensemble_score == run2_ens.ensemble_score
    assert run1_ens.delta == run2_ens.delta
