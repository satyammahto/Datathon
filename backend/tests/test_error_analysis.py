import json
import numpy as np
import pandas as pd
import pytest

from backend.app.schemas.discovery_contract import DiscoveryContract, RouterDecision
from backend.app.schemas.analysis_contract import (
    AnalysisContract,
    ChampionResult,
    ErrorAnalysisReport,
    ValidationReport,
)
from backend.app.schemas.enums import PipelineStatus, MLTaskType, ValidationStrategy
from backend.app.pipeline.error_analysis import (
    analyze_errors,
    compute_classification_error_analysis,
    compute_regression_error_analysis,
    compute_subgroup_slice_analysis,
    compute_numeric_error_associations,
)
from backend.app.pipeline.model_championship import benchmark_models, select_champion


# ---------------------------------------------------------------------------
# 1. Binary & Multiclass Classification Error Tests
# ---------------------------------------------------------------------------

def test_binary_classification_confusion_matrix_and_metrics():
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    y_pred = np.array([0, 0, 0, 1, 1, 1, 1, 1, 0, 0])

    diag = compute_classification_error_analysis(y_true, y_pred, top_k=5)
    assert diag["status"] == "SUCCESS"
    assert diag["summary"]["total_samples"] == 10
    assert diag["summary"]["error_count"] == 3
    assert diag["summary"]["error_rate"] == 0.3

    # Check 2x2 confusion matrix
    cm = diag["confusion_matrix"]
    assert len(cm) == 2
    assert len(cm[0]) == 2
    assert cm[0][0] == 3  # Class 0 TP
    assert cm[0][1] == 1  # Class 0 FN

    # Class metrics
    metrics = {m["class"]: m for m in diag["class_metrics"]}
    assert "0" in metrics
    assert "1" in metrics
    assert metrics["0"]["tp"] == 3
    assert metrics["0"]["fn"] == 1


def test_multiclass_metrics_and_weak_class_detection():
    # Class 0: strong (recall = 1.0)
    # Class 1: moderate (recall = 0.7)
    # Class 2: weak (recall = 0.20, support = 10)
    y_true = np.array([0] * 10 + [1] * 10 + [2] * 10)
    y_pred = np.array([0] * 10 + [1] * 7 + [0] * 3 + [2] * 2 + [0] * 8)

    diag = compute_classification_error_analysis(y_true, y_pred)
    assert len(diag["class_labels"]) == 3

    weak_classes = diag["weak_classes"]
    assert len(weak_classes) >= 1
    weak_class_names = [w["class"] for w in weak_classes]
    assert "2" in weak_class_names
    weak_entry = next(w for w in weak_classes if w["class"] == "2")
    assert weak_entry["recall"] == 0.2
    assert weak_entry["status"] == "WEAK"
    assert weak_entry["severity"] == "HIGH"


def test_worst_misclassifications_top_k_limiting():
    y_true = np.array([0] * 20)
    y_pred = np.array([1] * 20)  # 20 errors

    diag = compute_classification_error_analysis(y_true, y_pred, top_k=5)
    assert len(diag["worst_predictions"]) == 5
    assert diag["worst_predictions"][0]["actual"] == "0"
    assert diag["worst_predictions"][0]["predicted"] == "1"


# ---------------------------------------------------------------------------
# 2. Regression Residuals & Heteroscedasticity Tests
# ---------------------------------------------------------------------------

def test_regression_residual_statistics_and_worst_predictions():
    y_true = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    y_pred = np.array([11.0, 19.0, 35.0, 39.0, 60.0])

    diag = compute_regression_error_analysis(y_true, y_pred, top_k=3)
    assert diag["status"] == "SUCCESS"
    res_stats = diag["residual_summary"]
    assert res_stats["mae"] > 0
    assert res_stats["rmse"] > res_stats["mae"]
    assert res_stats["min_residual"] == -10.0  # 50 - 60 = -10
    assert res_stats["max_residual"] == 1.0   # 20 - 19 = 1.0

    worst = diag["worst_predictions"]
    assert len(worst) == 3
    # Rank 1 worst should have highest absolute error (10.0)
    assert worst[0]["absolute_error"] == 10.0
    assert worst[0]["sample_index"] == 4


def test_residual_systematic_bias_detection():
    # Model consistently underpredicts (actual > predicted)
    y_true = np.array([100.0, 110.0, 120.0, 130.0, 140.0, 150.0] * 5)
    y_pred = y_true - 15.0  # Always underpredicts by 15

    diag = compute_regression_error_analysis(y_true, y_pred)
    bias = diag["residual_summary"]["bias_diagnostic"]
    assert bias["bias_detected"] is True
    assert bias["direction"] == "systematic_underprediction"
    assert "underpredict" in bias["description"].lower()


def test_heteroscedasticity_diagnostic():
    # Error variance grows substantially with predicted value
    np.random.seed(42)
    n = 60
    y_pred = np.linspace(10, 100, n)
    # Noise scales with y_pred (heteroscedastic)
    noise = np.random.normal(0, scale=(y_pred / 10.0))
    y_true = y_pred + noise

    diag = compute_regression_error_analysis(y_true, y_pred)
    het = diag["heteroscedasticity"]
    assert het["status"] in ["POSSIBLE", "HOMOSCEDASTIC_LIKELY"]
    assert len(diag["error_by_prediction_bin"]) >= 3
    assert "disclaimer" in het or "evidence" in het


# ---------------------------------------------------------------------------
# 3. Subgroup & Slice Error Analysis Tests
# ---------------------------------------------------------------------------

def test_subgroup_slice_analysis_detection():
    # Subgroup 'Region South' has much worse accuracy
    n = 80
    regions = ["North"] * 40 + ["South"] * 40
    X = pd.DataFrame({"region": regions})

    # North: 100% correct; South: 20% correct
    y_true = np.array([1] * 40 + [1] * 40)
    y_pred = np.array([1] * 40 + [1] * 8 + [0] * 32)

    subgroups = compute_subgroup_slice_analysis(
        X, y_true, y_pred, task_type=MLTaskType.CLASSIFICATION, min_subgroup_size=10
    )

    assert len(subgroups) == 2
    south_slice = next(s for s in subgroups if s["subgroup"] == "South")
    assert south_slice["status"] == "WEAK_SLICE"
    assert south_slice["severity"] in ["HIGH", "MEDIUM"]
    assert south_slice["error_rate"] == 0.8


def test_subgroup_min_size_and_identifier_exclusion():
    # Feature with single unique row shouldn't trigger slice, nor should UUID identifier
    n = 40
    X = pd.DataFrame({
        "id_col": [f"ID_{i}" for i in range(n)],  # High cardinality identifier
        "rare_cat": ["A"] * 38 + ["Rare"] * 2,     # Rare category < min_group_size
        "valid_cat": ["Group1"] * 20 + ["Group2"] * 20,
    })
    y_true = np.array([1] * n)
    y_pred = np.array([1] * n)

    subgroups = compute_subgroup_slice_analysis(
        X, y_true, y_pred, task_type=MLTaskType.CLASSIFICATION, min_subgroup_size=5
    )

    analyzed_features = {s["feature"] for s in subgroups}
    analyzed_subgroups = {s["subgroup"] for s in subgroups}
    assert "id_col" not in analyzed_features  # Excluded due to high cardinality identifier
    assert "Rare" not in analyzed_subgroups   # Excluded because count (2) < min_group_size (5)
    assert "Group1" in analyzed_subgroups
    assert "Group2" in analyzed_subgroups


# ---------------------------------------------------------------------------
# 4. Numeric Feature / Error Associations
# ---------------------------------------------------------------------------

def test_numeric_feature_error_association():
    n = 50
    # Higher income correlates with higher prediction error
    income = np.linspace(20000, 100000, n)
    abs_errors = income * 0.001 + np.random.normal(0, 1, n)
    X = pd.DataFrame({"income": income, "constant_col": [42.0] * n})

    assocs = compute_numeric_error_associations(X, abs_errors)
    assert len(assocs) == 1
    assert assocs[0]["feature"] == "income"
    assert assocs[0]["direction"] == "positive"
    assert "Association does not imply causation." in assocs[0]["disclaimer"]


# ---------------------------------------------------------------------------
# 5. Robustness & Tail Outlier Sensitivity
# ---------------------------------------------------------------------------

def test_robustness_tail_sensitivity():
    # Mostly small errors (1.0) with 2 massive tail outliers (100.0)
    y_true = np.array([10.0] * 30)
    y_pred = np.array([11.0] * 28 + [110.0, 110.0])

    diag = compute_regression_error_analysis(y_true, y_pred)
    rob = diag["robustness"]
    assert rob["outlier_tail_reduction_pct"] > 30.0  # Removing worst 5% cuts MAE significantly


# ---------------------------------------------------------------------------
# 6. End-to-End analyze_errors & Contract Integration
# ---------------------------------------------------------------------------

def test_analyze_errors_with_champion_and_dataframe():
    np.random.seed(42)
    n = 60
    df = pd.DataFrame({
        "num_f": np.random.normal(10, 2, n),
        "cat_f": np.random.choice(["A", "B"], size=n),
        "target": np.random.choice([0, 1], size=n),
    })
    disc = DiscoveryContract(
        schema_version="1.0",
        schema={"num_f": "numeric", "cat_f": "categorical", "target": "categorical"},
        target_candidates=["target"],
        router=RouterDecision(classification=True),
    )
    val = ValidationReport(strategy=ValidationStrategy.STRATIFIED_K_FOLD.value, folds=3, random_seed=42)

    benchmarks = benchmark_models(df, disc, MLTaskType.CLASSIFICATION, val)
    champ = select_champion(benchmarks, primary_metric="f1_macro")

    # Run master error analysis
    err_report = analyze_errors(
        champion=champ,
        validation=val,
        df=df,
        discovery=disc,
        target_col="target",
    )

    assert err_report.status == PipelineStatus.SUCCESS
    assert err_report.classification["status"] == PipelineStatus.SUCCESS.value
    assert err_report.regression["status"] == PipelineStatus.NOT_APPLICABLE.value
    assert len(err_report.classification["class_metrics"]) == 2

    # Verify AnalysisContract integration
    contract = AnalysisContract(
        schema_version="1.0",
        status=PipelineStatus.SUCCESS,
        task=MLTaskType.CLASSIFICATION,
        target_column="target",
        validation=val,
        models=benchmarks,
        winner=champ,
        error_analysis=err_report,
    )

    json_str = contract.model_dump_json()
    assert isinstance(json_str, str)
    parsed = json.loads(json_str)
    assert parsed["error_analysis"]["status"] == "SUCCESS"
    assert parsed["error_analysis"]["model"]["name"] == champ.model_name


def test_analyze_errors_unavailable_when_no_predictions():
    champ = ChampionResult(status=PipelineStatus.SUCCESS, model_name="DummyModel")
    report = analyze_errors(champion=champ)
    assert report.status == PipelineStatus.UNAVAILABLE
    assert "predictions not available" in report.reason.lower()


def test_perfect_predictions_edge_case():
    y_true = np.array([0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1, 0, 1])

    diag = compute_classification_error_analysis(y_true, y_pred)
    assert diag["summary"]["error_count"] == 0
    assert diag["summary"]["error_rate"] == 0.0
    assert len(diag["worst_predictions"]) == 0
    assert len(diag["weak_classes"]) == 0
