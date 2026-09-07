import pytest
from backend.app.schemas.enums import PipelineStatus, MLTaskType, ValidationStrategy
from backend.app.schemas.discovery_contract import DiscoveryContract, RouterDecision
from backend.app.schemas.analysis_contract import ModelBenchmarkResult
from backend.app.pipeline.ml_analysis import determine_ml_task, run_ml_pipeline
from backend.app.pipeline.validation import select_validation_strategy
from backend.app.pipeline.model_championship import select_champion
from backend.app.pipeline.statistical import run_statistical_analysis
from backend.app.pipeline.time_analysis import run_time_analysis


def test_determine_ml_task_abstract(mock_discovery_contract):
    """Test task inference without any dataset-specific logic."""
    task = determine_ml_task(mock_discovery_contract)
    assert task == MLTaskType.CLASSIFICATION

    # Test regression routing
    reg_contract = DiscoveryContract(
        schema_version="1.0",
        schema={"feature1": "numeric", "target_val": "numeric"},
        target_candidates=["target_val"],
        router=RouterDecision(regression=True, classification=False),
    )
    assert determine_ml_task(reg_contract) == MLTaskType.REGRESSION

    # Test no target routing
    no_target_contract = DiscoveryContract(
        schema_version="1.0",
        schema={"feature1": "numeric"},
        target_candidates=[],
        router=RouterDecision(clustering=True),
    )
    assert determine_ml_task(no_target_contract) == MLTaskType.CLUSTERING


def test_validation_strategy_selection():
    """Verify validation strategy adapts to task and temporal characteristics."""
    contract = DiscoveryContract(
        schema_version="1.0",
        target_candidates=["y"],
        router=RouterDecision(classification=True),
    )
    val_report = select_validation_strategy(None, contract, MLTaskType.CLASSIFICATION)
    assert val_report.strategy == ValidationStrategy.STRATIFIED_K_FOLD.value
    assert val_report.folds == 5
    assert val_report.primary_metric == "f1"

    # Temporal dataset
    temporal_contract = DiscoveryContract(
        schema_version="1.0",
        target_candidates=["y"],
        router=RouterDecision(regression=True, time_analysis=True),
    )
    time_val = select_validation_strategy(None, temporal_contract, MLTaskType.REGRESSION)
    assert time_val.strategy == ValidationStrategy.TIME_SERIES_SPLIT.value
    assert time_val.primary_metric == "rmse"


def test_champion_selection_empirical_rule():
    """Verify champion selection is purely mathematical and deterministic."""
    benchmarks = [
        ModelBenchmarkResult(
            model_name="BaselineLogistic",
            family="linear",
            status=PipelineStatus.SUCCESS,
            metrics={"f1": 0.72},
        ),
        ModelBenchmarkResult(
            model_name="GradientBoosting",
            family="tree_ensemble",
            status=PipelineStatus.SUCCESS,
            metrics={"f1": 0.89},
        ),
        ModelBenchmarkResult(
            model_name="RandomForest",
            family="tree_ensemble",
            status=PipelineStatus.SUCCESS,
            metrics={"f1": 0.85},
        ),
    ]

    champion = select_champion(benchmarks, primary_metric="f1")
    assert champion.status == PipelineStatus.SUCCESS
    assert champion.model_name == "GradientBoosting"
    assert champion.score == 0.89
    assert champion.selection_rationale is not None

    # Verify lower-is-better for error metrics
    rmse_benchmarks = [
        ModelBenchmarkResult(
            model_name="ModelA",
            family="linear",
            status=PipelineStatus.SUCCESS,
            metrics={"rmse": 4.5},
        ),
        ModelBenchmarkResult(
            model_name="ModelB",
            family="tree",
            status=PipelineStatus.SUCCESS,
            metrics={"rmse": 2.1},
        ),
    ]
    rmse_champ = select_champion(rmse_benchmarks, primary_metric="rmse")
    assert rmse_champ.model_name == "ModelB"
    assert rmse_champ.score == 2.1


def test_champion_selection_empty_or_failed():
    """Verify graceful NOT_APPLICABLE and FAILED statuses."""
    empty_champ = select_champion([], primary_metric="f1")
    assert empty_champ.status == PipelineStatus.NOT_APPLICABLE
    assert empty_champ.reason is not None

    failed_models = [
        ModelBenchmarkResult(
            model_name="FailedModel",
            family="linear",
            status=PipelineStatus.FAILED,
            reason="Singular matrix",
            metrics={},
        )
    ]
    failed_champ = select_champion(failed_models, primary_metric="f1")
    assert failed_champ.status == PipelineStatus.FAILED


def test_run_ml_pipeline_with_mock_contract(mock_discovery_contract):
    """Verify that run_ml_pipeline generates an AnalysisContract meeting all Person 3 requirements."""
    analysis = run_ml_pipeline(None, mock_discovery_contract)
    assert analysis.schema_version == "1.0"
    assert analysis.status == PipelineStatus.SUCCESS
    assert analysis.task == MLTaskType.CLASSIFICATION
    assert analysis.target_column == "churn"
    assert analysis.validation.strategy == "StratifiedKFold"
    assert analysis.time_analysis.status == PipelineStatus.NOT_APPLICABLE
