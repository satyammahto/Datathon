"""
Person 1: Phase 1E Final Discovery Pipeline Validation & Person 2 Handoff Test Suite.

Validates:
1. 10-Family Dataset Matrix (Binary, Multiclass, Regression, Time-series, Exploratory, ID-heavy, Messy, XLSX, Adversarial Leakage, Ambiguous Targets)
2. Hidden/Unseen Dataset Structure (IoT Telemetry / Multi-sensor)
3. Discovery Contract v1.0.0 Pydantic roundtrip and JSON serialization
4. Deterministic repeated generation
5. Cross-module consistency (Fingerprint, Quality, Leakage, Router)
6. Feature partition disjointness (Usable vs Excluded vs Review vs Unsupported)
7. Cleaned data immutability & audit integrity
8. Robust failure mode handling
9. Performance sanity on 50,000-row dataset
10. Downstream Person 2 consumption contract validation
"""
import os
import io
import json
import time
import pytest
import numpy as np
import pandas as pd
from typing import Dict, Any

from app.contracts.discovery import (
    DiscoveryContract,
    TaskType,
    ValidationStrategy,
    InferredDtype,
    LeakageRiskLevel,
    LeakageType,
)
from app.pipeline.loaders import load_dataset
from app.pipeline.discovery import run_discovery
from app.pipeline.fingerprint import generate_fingerprint
from app.pipeline.quality import evaluate_quality
from app.pipeline.leakage import detect_leakage
from app.pipeline.router import route_dataset


@pytest.fixture
def tmp_dir(tmp_path):
    """Provide a temporary directory path for synthetic dataset files."""
    return str(tmp_path)


# ==============================================================================
# 1. 10-FAMILY DATASET MATRIX TESTS
# ==============================================================================

def test_matrix_family_a_binary_classification(tmp_dir):
    """Family A: Binary classification (customer churn / conversion)."""
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "customer_id": [f"CUST_{i:04d}" for i in range(n)],
        "age": np.random.randint(18, 70, size=n),
        "monthly_spend": np.random.uniform(20.0, 150.0, size=n),
        "contract_tier": np.random.choice(["basic", "standard", "premium"], size=n),
        "churn_flag": np.random.choice([0, 1], size=n, p=[0.75, 0.25]),
    })
    path = os.path.join(tmp_dir, "family_a_churn.csv")
    df.to_csv(path, index=False)

    contract = run_discovery(path, dataset_id="ds-family-a")
    assert contract.version == "1.0.0"
    assert contract.router.recommended_primary_task == TaskType.BINARY_CLASSIFICATION
    assert contract.router.validation_strategy in [ValidationStrategy.STRATIFIED_K_FOLD, ValidationStrategy.K_FOLD]
    assert "customer_id" in contract.router.excluded_identifier_columns
    assert "churn_flag" not in contract.router.usable_feature_columns
    assert "age" in contract.router.usable_feature_columns
    assert "monthly_spend" in contract.router.usable_feature_columns
    assert contract.fingerprint.class_balance is not None
    assert contract.fingerprint.class_balance.minority_class_percentage > 0


def test_matrix_family_b_multiclass_classification(tmp_dir):
    """Family B: Multiclass classification (risk tier / product category)."""
    np.random.seed(42)
    n = 300
    df = pd.DataFrame({
        "txn_id": [f"TX_{i:05d}" for i in range(n)],
        "amount": np.random.exponential(50.0, size=n),
        "velocity": np.random.poisson(3, size=n),
        "risk_level": np.random.choice(["low", "medium", "high", "critical"], size=n),
    })
    path = os.path.join(tmp_dir, "family_b_risk.csv")
    df.to_csv(path, index=False)

    contract = run_discovery(path, dataset_id="ds-family-b")
    assert contract.router.recommended_primary_task == TaskType.MULTICLASS_CLASSIFICATION
    assert "txn_id" in contract.router.excluded_identifier_columns
    assert "amount" in contract.router.usable_feature_columns
    assert "velocity" in contract.router.usable_feature_columns
    assert "risk_level" not in contract.router.usable_feature_columns


def test_matrix_family_c_regression(tmp_dir):
    """Family C: Continuous regression (house prices / revenue)."""
    np.random.seed(42)
    n = 250
    sqft = np.random.uniform(500, 3500, size=n)
    rooms = np.random.randint(1, 6, size=n)
    price = sqft * 150.0 + rooms * 20000.0 + np.random.normal(0, 10000, size=n)
    df = pd.DataFrame({
        "property_id": [f"PROP_{i:04d}" for i in range(n)],
        "sqft": sqft,
        "rooms": rooms,
        "sale_price": price,
    })
    path = os.path.join(tmp_dir, "family_c_housing.csv")
    df.to_csv(path, index=False)

    contract = run_discovery(path, dataset_id="ds-family-c")
    assert contract.router.recommended_primary_task == TaskType.REGRESSION
    assert "sale_price" not in contract.router.usable_feature_columns
    assert "sqft" in contract.router.usable_feature_columns
    assert "rooms" in contract.router.usable_feature_columns
    assert "property_id" in contract.router.excluded_identifier_columns


def test_matrix_family_d_time_series(tmp_dir):
    """Family D: Chronological regular time-series."""
    np.random.seed(42)
    dates = pd.date_range(start="2024-01-01", periods=120, freq="D")
    sales = 100 + np.arange(120) * 0.5 + np.random.normal(0, 5, size=120)
    df = pd.DataFrame({
        "timestamp": dates.strftime("%Y-%m-%d"),
        "foot_traffic": np.random.poisson(50, size=120),
        "daily_sales": sales,
    })
    path = os.path.join(tmp_dir, "family_d_timeseries.csv")
    df.to_csv(path, index=False)

    contract = run_discovery(path, dataset_id="ds-family-d")
    assert contract.fingerprint.temporal is not None
    assert contract.fingerprint.temporal.is_monotonic is True
    assert contract.router.validation_strategy == ValidationStrategy.TIME_SERIES_SPLIT
    assert "timestamp" in contract.fingerprint.datetime_columns or contract.fingerprint.temporal.temporal_column == "timestamp"


def test_matrix_family_e_no_target_exploratory(tmp_dir):
    """Family E: Exploratory / clustering dataset with no designated target."""
    np.random.seed(42)
    n = 150
    df = pd.DataFrame({
        "feature_a": np.random.normal(0, 1, size=n),
        "feature_b": np.random.normal(5, 2, size=n),
        "feature_c": np.random.uniform(-10, 10, size=n),
        "feature_d": np.random.exponential(1.5, size=n),
    })
    path = os.path.join(tmp_dir, "family_e_exploratory.csv")
    df.to_csv(path, index=False)

    contract = run_discovery(path, dataset_id="ds-family-e")
    assert contract.router.recommended_primary_task in [TaskType.CLUSTERING, TaskType.UNSUPERVISED_EDA]
    assert len(contract.router.usable_feature_columns) == 4


def test_matrix_family_f_id_heavy_dataset(tmp_dir):
    """Family F: ID-heavy dataset with multiple primary keys, UUIDs, and foreign keys."""
    n = 100
    df = pd.DataFrame({
        "uuid": [f"12345678-1234-5678-1234-{i:012d}" for i in range(n)],
        "account_id": [f"ACC_{i:06d}" for i in range(n)],
        "global_session_id": [f"SESS_{i*7:08d}" for i in range(n)],
        "metric_score": np.random.uniform(10, 100, size=n),
        "is_active": np.random.choice([0, 1], size=n),
    })
    path = os.path.join(tmp_dir, "family_f_id_heavy.csv")
    df.to_csv(path, index=False)

    contract = run_discovery(path, dataset_id="ds-family-f")
    assert "uuid" in contract.router.excluded_identifier_columns
    assert "account_id" in contract.router.excluded_identifier_columns
    assert "global_session_id" in contract.router.excluded_identifier_columns
    assert "uuid" not in contract.router.usable_feature_columns
    assert "account_id" not in contract.router.usable_feature_columns
    assert "metric_score" in contract.router.usable_feature_columns


def test_matrix_family_g_messy_missing_duplicates(tmp_dir):
    """Family G: Messy dataset with duplicate rows, missing values, and outliers."""
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "id": range(n),
        "clean_feature": np.random.uniform(10, 50, size=n),
        "mostly_null": [np.nan if i > 10 else 1.0 for i in range(n)],
        "noisy_category": [" Male " if i % 2 == 0 else "Female" for i in range(n)],
        "target": np.random.choice([0, 1], size=n),
    })
    # Add 10 duplicate rows
    df = pd.concat([df, df.iloc[:10]], ignore_index=True)
    path = os.path.join(tmp_dir, "family_g_messy.csv")
    df.to_csv(path, index=False)

    contract = run_discovery(path, dataset_id="ds-family-g")
    assert contract.quality_report.duplicate_row_count == 10
    assert contract.quality_report.total_missing_cells > 0
    assert "mostly_null" in contract.quality_report.columns_with_missing
    assert contract.cleaning_audit.cleaned_shape[0] == 100  # Duplicates dropped
    assert "mostly_null" in contract.cleaning_audit.dropped_columns or "mostly_null" in contract.router.unsupported_feature_columns


def test_matrix_family_h_xlsx_workbook():
    """Family H: Real XLSX file handling."""
    path = "backend/tests/fixtures/data/sample_workbook.xlsx"
    contract = run_discovery(path, dataset_id="ds-family-h-xlsx")
    assert contract.file_info.file_type == "xlsx"
    assert contract.file_info.row_count > 0
    assert len(contract.columns) > 0
    assert contract.quality_report.overall_quality_score >= 0


def test_matrix_family_i_adversarial_leakage(tmp_dir):
    """Family I: Adversarial dataset with perfect copy, derived target, and post-event flags."""
    np.random.seed(42)
    n = 150
    churn = np.random.choice([0, 1], size=n, p=[0.7, 0.3])
    df = pd.DataFrame({
        "customer_id": [f"ID_{i:04d}" for i in range(n)],
        "age": np.random.randint(20, 65, size=n),
        "monthly_charges": np.random.uniform(30, 120, size=n),
        "target_clone": churn,  # Exact duplicate of target
        "target_scaled": churn * 100.0,  # Perfectly linear derivation
        "refund_processed_post_churn": [1 if c == 1 else 0 for c in churn],  # Post-event leakage
        "constant_col": [42] * n,  # Zero variance
        "churn": churn,
    })
    path = os.path.join(tmp_dir, "family_i_leakage.csv")
    df.to_csv(path, index=False)

    contract = run_discovery(path, dataset_id="ds-family-i")
    assert contract.leakage_report.has_leakage_risk is True
    assert "target_clone" in contract.router.excluded_leakage_columns
    assert "target_scaled" in contract.router.excluded_leakage_columns
    assert "constant_col" in contract.router.excluded_leakage_columns
    assert "target_clone" not in contract.router.usable_feature_columns
    assert "target_scaled" not in contract.router.usable_feature_columns
    assert "age" in contract.router.usable_feature_columns
    assert "monthly_charges" in contract.router.usable_feature_columns


def test_matrix_family_j_ambiguous_target_candidates(tmp_dir):
    """Family J: Ambiguous dataset with multiple plausible target candidate columns."""
    np.random.seed(42)
    n = 150
    df = pd.DataFrame({
        "user_id": [f"U_{i:04d}" for i in range(n)],
        "tenure_months": np.random.randint(1, 48, size=n),
        "status_code": np.random.choice(["ACTIVE", "CANCELLED", "PENDING"], size=n),
        "final_outcome": np.random.choice([0, 1], size=n),
        "total_revenue": np.random.uniform(100, 5000, size=n),
    })
    path = os.path.join(tmp_dir, "family_j_ambiguous.csv")
    df.to_csv(path, index=False)

    contract = run_discovery(path, dataset_id="ds-family-j")
    assert len(contract.target_candidates) >= 2
    # Candidates should be sorted descending by confidence
    confidences = [c.confidence_score for c in contract.target_candidates]
    assert confidences == sorted(confidences, reverse=True)


# ==============================================================================
# 2. NOVEL HIDDEN / UNSEEN DATASET TEST
# ==============================================================================

def test_hidden_unseen_dataset_iot_telemetry(tmp_dir):
    """
    Novel unseen structural family: Industrial IoT multi-sensor telemetry log.
    Has cyclical features, physical units, sensor device nodes, vibration metrics.
    Demonstrates true dataset-agnostic intelligence without name-based rules.
    """
    np.random.seed(101)
    n = 300
    t = np.linspace(0, 50, n)
    df = pd.DataFrame({
        "device_guid": [f"DEV-{i%5:02d}-HEX" for i in range(n)],
        "sin_time_phase": np.sin(t),
        "cos_time_phase": np.cos(t),
        "bearing_vibration_hz": np.random.gamma(2.0, 15.0, size=n),
        "core_temperature_c": 65.0 + 10.0 * np.sin(t/5) + np.random.normal(0, 1.5, size=n),
        "operating_pressure_kpa": np.random.normal(101.3, 5.0, size=n),
        "battery_voltage_mv": np.random.uniform(3200, 4200, size=n),
        "maintenance_fault_code": np.random.choice(["E00", "E01", "E02", "OK"], size=n, p=[0.05, 0.05, 0.05, 0.85]),
    })
    path = os.path.join(tmp_dir, "unseen_industrial_iot.csv")
    df.to_csv(path, index=False)

    contract = run_discovery(path, dataset_id="ds-iot-unseen")
    assert contract.fingerprint.shape == [300, 8]
    assert "device_guid" in contract.fingerprint.categorical_columns or "device_guid" in contract.fingerprint.identifier_columns
    assert "bearing_vibration_hz" in contract.fingerprint.numeric_columns
    assert "core_temperature_c" in contract.fingerprint.numeric_columns
    assert contract.router.recommended_primary_task in [TaskType.MULTICLASS_CLASSIFICATION, TaskType.CLUSTERING, TaskType.UNSUPERVISED_EDA]
    assert len(contract.router.usable_feature_columns) >= 4


# ==============================================================================
# 3. CONTRACT VALIDATION & DETERMINISM
# ==============================================================================

def test_contract_pydantic_roundtrip_and_serialization(tmp_dir):
    """Verify DiscoveryContract serializes to JSON and re-validates into Pydantic model perfectly."""
    path = "backend/tests/fixtures/data/churn_classification.csv"
    contract = run_discovery(path, dataset_id="ds-churn-validate")

    # Serialize to JSON string
    json_str = contract.model_dump_json(indent=2)
    assert isinstance(json_str, str)
    assert len(json_str) > 500

    # Parse and re-validate
    data = json.loads(json_str)
    revalidated = DiscoveryContract.model_validate(data)
    assert revalidated.dataset_id == contract.dataset_id
    assert revalidated.version == "1.0.0"
    assert revalidated.router.recommended_primary_task == contract.router.recommended_primary_task
    assert revalidated.router.usable_feature_columns == contract.router.usable_feature_columns


def test_deterministic_repeated_generation(tmp_dir):
    """Running the pipeline twice on the exact same dataset must produce identical output."""
    path = "backend/tests/fixtures/data/housing_regression.csv"
    
    contract1 = run_discovery(path, dataset_id="ds-fixed-id")
    contract2 = run_discovery(path, dataset_id="ds-fixed-id")

    dict1 = contract1.model_dump(exclude={"created_at"})
    dict2 = contract2.model_dump(exclude={"created_at"})

    assert dict1 == dict2


# ==============================================================================
# 4. CROSS-MODULE CONSISTENCY & FEATURE PARTITION DISJOINTNESS
# ==============================================================================

def test_cross_module_consistency_and_partition_disjointness(tmp_dir):
    """
    Strictly verify:
    1. usable_feature_columns ∩ excluded_identifier_columns = ∅
    2. usable_feature_columns ∩ excluded_leakage_columns = ∅
    3. usable_feature_columns ∩ unsupported_feature_columns = ∅
    4. Target candidate is not in usable_feature_columns
    5. High-risk leakage is never placed in usable_feature_columns
    """
    np.random.seed(42)
    n = 200
    churn = np.random.choice([0, 1], size=n)
    df = pd.DataFrame({
        "user_uuid": [f"ID_{i:05d}" for i in range(n)],
        "age": np.random.randint(18, 70, size=n),
        "direct_leak": churn,
        "all_null_col": [np.nan] * n,
        "review_suspect": [c + np.random.normal(0, 0.05) for c in churn],
        "churn": churn,
    })
    path = os.path.join(tmp_dir, "consistency_check.csv")
    df.to_csv(path, index=False)

    contract = run_discovery(path, dataset_id="ds-consistency")
    r = contract.router

    usable = set(r.usable_feature_columns)
    excluded_id = set(r.excluded_identifier_columns)
    excluded_leak = set(r.excluded_leakage_columns)
    unsupported = set(r.unsupported_feature_columns)

    # Disjointness checks
    assert len(usable.intersection(excluded_id)) == 0, f"Overlap: {usable.intersection(excluded_id)}"
    assert len(usable.intersection(excluded_leak)) == 0, f"Overlap: {usable.intersection(excluded_leak)}"
    assert len(usable.intersection(unsupported)) == 0, f"Overlap: {usable.intersection(unsupported)}"

    # Specific placements
    assert "user_uuid" in excluded_id
    assert "direct_leak" in excluded_leak
    assert "all_null_col" in unsupported or "all_null_col" in excluded_leak
    assert "churn" not in usable
    assert "age" in usable


# ==============================================================================
# 5. CLEANED DATA CONSISTENCY & IMMUTABILITY
# ==============================================================================

def test_cleaned_data_consistency_and_immutability(tmp_dir):
    """Verify raw DataFrame is not mutated and cleaning audit matches transformed DataFrame."""
    np.random.seed(42)
    n = 60
    df = pd.DataFrame({
        "id": range(n),
        "num": [10.0 if i % 5 != 0 else np.nan for i in range(n)],
        "txt": ["  val  " for _ in range(n)],
    })
    raw_copy = df.copy()

    _, cols, _ = generate_fingerprint(df)
    report, audit, cleaned_df = evaluate_quality(df, cols)

    # Original DataFrame must remain untouched
    pd.testing.assert_frame_equal(df, raw_copy)

    # Cleaned DataFrame reflects audit
    assert cleaned_df.shape == tuple(audit.cleaned_shape)
    if "num" in audit.imputed_columns:
        assert cleaned_df["num"].isna().sum() == 0


# ==============================================================================
# 6. ROBUST FAILURE HANDLING
# ==============================================================================

def test_failure_invalid_extension(tmp_dir):
    """Reject unsupported file extensions with clean ValueError."""
    path = os.path.join(tmp_dir, "corrupt_data.docx")
    with open(path, "w") as f:
        f.write("Not a spreadsheet")
    with pytest.raises(ValueError, match="Unsupported file"):
        run_discovery(path)


def test_failure_nonexistent_file():
    """Reject missing file paths."""
    with pytest.raises(FileNotFoundError):
        run_discovery("backend/tests/fixtures/data/nonexistent_file_9999.csv")


def test_failure_empty_file(tmp_dir):
    """Reject empty 0-byte files gracefully."""
    path = os.path.join(tmp_dir, "empty.csv")
    with open(path, "w") as f:
        pass
    with pytest.raises(ValueError, match="empty"):
        run_discovery(path)


def test_failure_zero_rows_header_only(tmp_dir):
    """Reject files with headers but 0 rows gracefully."""
    path = os.path.join(tmp_dir, "zero_rows.csv")
    with open(path, "w") as f:
        f.write("col_a,col_b,col_c\n")
    with pytest.raises(ValueError, match="zero data rows"):
        run_discovery(path)


def test_all_null_dataset_handling(tmp_dir):
    """Dataset where all columns are completely null is routed to un-analyzable/unsupervised with warnings."""
    df = pd.DataFrame({
        "null_1": [np.nan, np.nan, np.nan, np.nan],
        "null_2": [np.nan, np.nan, np.nan, np.nan],
    })
    path = os.path.join(tmp_dir, "all_null.csv")
    df.to_csv(path, index=False)

    contract = run_discovery(path, dataset_id="ds-all-null")
    assert contract.quality_report.missing_cell_percentage == 100.0
    assert len(contract.router.usable_feature_columns) == 0
    assert contract.router.recommended_primary_task in [TaskType.UNSUPERVISED_EDA, TaskType.CLUSTERING]


# ==============================================================================
# 7. PERFORMANCE SANITY (50,000 ROWS)
# ==============================================================================

def test_performance_large_dataset_50k_rows(tmp_dir):
    """Verify P1 pipeline processes 50,000 rows across 15 columns efficiently without quadratic bottlenecks."""
    np.random.seed(42)
    n = 50000
    df = pd.DataFrame({
        "user_id": [f"U_{i:07d}" for i in range(n)],
        "age": np.random.randint(18, 80, size=n),
        "income": np.random.exponential(45000, size=n),
        "credit_score": np.random.normal(680, 50, size=n),
        "account_balance": np.random.uniform(0, 100000, size=n),
        "num_products": np.random.randint(1, 5, size=n),
        "is_active_member": np.random.choice([0, 1], size=n),
        "country": np.random.choice(["France", "Germany", "Spain"], size=n),
        "churn": np.random.choice([0, 1], size=n, p=[0.8, 0.2]),
    })
    path = os.path.join(tmp_dir, "perf_50k.csv")
    df.to_csv(path, index=False)

    t0 = time.time()
    contract = run_discovery(path, dataset_id="ds-perf-50k")
    elapsed = time.time() - t0

    assert contract.file_info.row_count == 50000
    assert contract.router.recommended_primary_task == TaskType.BINARY_CLASSIFICATION
    assert "user_id" in contract.router.excluded_identifier_columns
    assert "churn" not in contract.router.usable_feature_columns
    assert len(contract.router.usable_feature_columns) >= 5
    # Must complete in reasonable wall-clock time (< 5.0 seconds on standard CPU)
    assert elapsed < 5.0, f"Pipeline took too long: {elapsed:.2f}s"


# ==============================================================================
# 8. DOWNSTREAM PERSON 2 CONSUMPTION CONTRACT TEST
# ==============================================================================

def test_person_2_can_consume_mock_and_real_contracts():
    """Verify Person 2 can consume all contract fixtures cleanly without guessing."""
    fixture_paths = [
        "backend/app/contracts/fixtures/discovery_mock.json",
        "backend/app/contracts/fixtures/customer_churn_contract.json",
        "backend/app/contracts/fixtures/adversarial_leakage_contract.json",
    ]

    for p in fixture_paths:
        with open(p, "r") as f:
            data = json.load(f)
        contract = DiscoveryContract.model_validate(data)

        # Person 2 consumes these essential fields:
        task = contract.router.recommended_primary_task.value
        validation = contract.router.validation_strategy.value
        usable_features = contract.router.usable_feature_columns
        excluded_ids = contract.router.excluded_identifier_columns
        excluded_leaks = contract.router.excluded_leakage_columns

        assert isinstance(task, str)
        assert isinstance(validation, str)
        assert isinstance(usable_features, list)
        assert isinstance(excluded_ids, list)
        assert isinstance(excluded_leaks, list)

        # Confirm target candidates structure
        for tc in contract.target_candidates:
            assert isinstance(tc.column_name, str)
            assert isinstance(tc.task_type.value, str)
            assert 0.0 <= tc.confidence_score <= 1.0
