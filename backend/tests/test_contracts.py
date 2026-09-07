import json
from pathlib import Path
import pytest
from backend.app.schemas.enums import PipelineStatus, MLTaskType, ValidationStrategy
from backend.app.schemas.discovery_contract import DiscoveryContract
from backend.app.schemas.analysis_contract import (
    AnalysisContract,
    ValidationReport,
    StatisticalReport,
    ModelBenchmarkResult,
    ChampionResult,
    TimeAnalysisReport,
)


def test_mock_discovery_contract_validation(mock_discovery_contract_data):
    """Test that the mock discovery contract complies with the schema."""
    contract = DiscoveryContract.model_validate(mock_discovery_contract_data)
    assert contract.schema_version == "1.0"
    assert "churn" in contract.target_candidates
    assert contract.router.classification is True
    assert contract.router.regression is False
    assert len(contract.schema_definition) == 5


def test_analysis_contract_serialization():
    """Test that AnalysisContract serializes to pure JSON without model objects."""
    report = AnalysisContract(
        schema_version="1.0",
        status=PipelineStatus.SUCCESS,
        task=MLTaskType.CLASSIFICATION,
        target_column="churn",
        validation=ValidationReport(
            status=PipelineStatus.SUCCESS,
            strategy=ValidationStrategy.STRATIFIED_K_FOLD.value,
            folds=5,
            primary_metric="f1",
            random_seed=42,
        ),
        statistical_analysis=StatisticalReport(status=PipelineStatus.SUCCESS),
        models=[
            ModelBenchmarkResult(
                model_name="RandomForest",
                family="tree_ensemble",
                status=PipelineStatus.SUCCESS,
                metrics={"f1": 0.85, "accuracy": 0.88},
            )
        ],
        winner=ChampionResult(
            status=PipelineStatus.SUCCESS,
            model_name="RandomForest",
            primary_metric="f1",
            score=0.85,
            selection_rationale="Achieved highest f1 score",
        ),
        time_analysis=TimeAnalysisReport(
            status=PipelineStatus.NOT_APPLICABLE,
            reason="No time column present in discovery contract",
        ),
    )

    json_str = report.model_dump_json()
    parsed = json.loads(json_str)

    assert parsed["schema_version"] == "1.0"
    assert parsed["status"] == "SUCCESS"
    assert parsed["task"] == "classification"
    assert parsed["winner"]["model_name"] == "RandomForest"
    assert parsed["time_analysis"]["status"] == "NOT_APPLICABLE"
    assert parsed["time_analysis"]["reason"] == "No time column present in discovery contract"


def test_status_values_supported():
    """Ensure all required status values (SUCCESS, NOT_APPLICABLE, FAILED, UNAVAILABLE) are valid."""
    statuses = [
        PipelineStatus.SUCCESS,
        PipelineStatus.NOT_APPLICABLE,
        PipelineStatus.FAILED,
        PipelineStatus.UNAVAILABLE,
    ]
    for st in statuses:
        report = StatisticalReport(status=st, reason=f"Testing {st.value}")
        assert report.status == st
        assert report.reason == f"Testing {st.value}"


def test_example_fixture_conformance():
    """Verify that example_analysis_contract.json validates against the Pydantic schema."""
    fixture_path = Path(__file__).resolve().parent / "fixtures" / "example_analysis_contract.json"
    with open(fixture_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    contract = AnalysisContract.model_validate(data)
    assert contract.status == PipelineStatus.SUCCESS
    assert contract.winner.model_name == "RandomForestClassifier"
    assert contract.time_analysis.status == PipelineStatus.NOT_APPLICABLE
