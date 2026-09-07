import json
import numpy as np
import pandas as pd
import pytest

from backend.app.schemas.discovery_contract import DiscoveryContract, RouterDecision
from backend.app.schemas.enums import PipelineStatus
from backend.app.pipeline.statistical import (
    run_statistical_analysis,
    compute_numeric_statistics,
    compute_categorical_statistics,
    compute_distribution_and_outliers,
    compute_correlations,
    compute_grouped_analysis,
    compute_two_group_test,
    compute_multi_group_test,
    compute_chi_square_test,
    compute_mean_confidence_intervals,
    sanitize_for_json,
)


@pytest.fixture
def synthetic_multivariate_df():
    """Synthetic dataset with numeric, categorical, skewed, outlier, and correlated features."""
    np.random.seed(42)
    n = 100
    x1 = np.random.normal(50, 10, n)
    x2 = x1 * 2.5 + np.random.normal(0, 2, n)  # Strongly correlated with x1
    x3 = np.random.exponential(scale=5, size=n)  # Skewed
    x3[0] = 500.0  # Obvious outlier

    groups = np.random.choice(["Control", "Treatment"], size=n, p=[0.5, 0.5])
    multi_groups = np.random.choice(["Low", "Medium", "High"], size=n)
    cat_outcome = np.random.choice(["Pass", "Fail"], size=n)

    df = pd.DataFrame({
        "num_x1": x1,
        "num_x2": x2,
        "num_skewed": x3,
        "group_two": groups,
        "group_multi": multi_groups,
        "cat_outcome": cat_outcome,
    })
    return df


@pytest.fixture
def synthetic_discovery():
    return DiscoveryContract(
        schema_version="1.0",
        schema={
            "num_x1": "numeric",
            "num_x2": "numeric",
            "num_skewed": "numeric",
            "group_two": "categorical",
            "group_multi": "categorical",
            "cat_outcome": "categorical",
        },
        target_candidates=["cat_outcome"],
        router=RouterDecision(classification=True),
    )


# ---------------------------------------------------------------------------
# Test Cases 1 - 17
# ---------------------------------------------------------------------------

def test_1_numeric_statistics():
    series = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
    res = compute_numeric_statistics(series)

    assert res["status"] == "SUCCESS"
    assert res["count"] == 5
    assert res["missing_count"] == 0
    assert res["mean"] == 30.0
    assert res["median"] == 30.0
    assert res["min"] == 10.0
    assert res["max"] == 50.0
    assert res["quantiles"]["25%"] == 20.0
    assert res["quantiles"]["75%"] == 40.0
    assert res["iqr"] == 20.0
    assert res["unique_count"] == 5
    assert not res["is_constant"]


def test_2_categorical_statistics():
    series = pd.Series(["A", "B", "A", "C", "A", "B", None])
    res = compute_categorical_statistics(series)

    assert res["status"] == "SUCCESS"
    assert res["count"] == 6
    assert res["missing_count"] == 1
    assert res["unique_count"] == 3
    assert res["top_categories"][0] == "A"
    assert res["frequency_distribution"]["A"] == 3
    assert res["category_proportions"]["A"] == 0.5
    assert not res["likely_identifier"]

    # Test high-cardinality identifier detection
    id_series = pd.Series([f"ID_{i}" for i in range(50)])
    id_res = compute_categorical_statistics(id_series)
    assert id_res["likely_identifier"] is True


def test_3_missing_values():
    series = pd.Series([1.0, 2.0, None, np.nan, 5.0])
    res = compute_numeric_statistics(series)
    assert res["count"] == 3
    assert res["missing_count"] == 2
    assert res["missing_percentage"] == 40.0
    assert res["mean"] == pytest.approx(8.0 / 3.0)


def test_4_iqr_outliers():
    # Normal cluster with 2 extreme outliers
    data = [10.0, 11.0, 12.0, 10.5, 11.5, 12.5, 11.0, 10.0, 100.0, -50.0]
    series = pd.Series(data)
    _, outlier_res = compute_distribution_and_outliers(series)

    assert outlier_res["status"] == "SUCCESS"
    assert outlier_res["method"] == "IQR"
    assert outlier_res["outlier_count"] >= 2
    assert outlier_res["outlier_percentage"] > 0
    assert outlier_res["lower_bound"] < 10.0
    assert outlier_res["upper_bound"] > 12.0


def test_5_pearson_correlation():
    df = pd.DataFrame({
        "a": [1.0, 2.0, 3.0, 4.0, 5.0],
        "b": [2.0, 4.0, 6.0, 8.0, 10.0],  # Perfect positive correlation r = 1.0
    })
    res = compute_correlations(df, ["a", "b"], min_observations=3)
    assert res["status"] == "SUCCESS"
    assert res["pearson"]["a"]["b"] == pytest.approx(1.0)
    assert len(res["strong_relationships"]) == 1
    assert res["strong_relationships"][0]["direction"] == "positive"
    assert "Correlation does not imply causation." in res["disclaimer"]


def test_6_spearman_correlation():
    # Monotonic non-linear relationship (Spearman = 1.0, Pearson < 1.0)
    df = pd.DataFrame({
        "a": [1.0, 2.0, 3.0, 4.0, 5.0],
        "b": [1.0, 8.0, 27.0, 64.0, 125.0],
    })
    res = compute_correlations(df, ["a", "b"], min_observations=3)
    assert res["status"] == "SUCCESS"
    assert res["spearman"]["a"]["b"] == pytest.approx(1.0)
    assert res["pearson"]["a"]["b"] < 1.0


def test_7_constant_columns():
    constant_series = pd.Series([42.0, 42.0, 42.0, 42.0])
    num_stats = compute_numeric_statistics(constant_series)
    assert num_stats["is_constant"] is True
    assert num_stats["std"] == 0.0

    dist_res, outlier_res = compute_distribution_and_outliers(constant_series)
    assert dist_res["status"] == "NOT_APPLICABLE"
    assert "Zero variance" in dist_res["reason"]

    # Correlation with constant column
    df = pd.DataFrame({
        "const": [42.0, 42.0, 42.0, 42.0],
        "vary": [1.0, 2.0, 3.0, 4.0],
    })
    corr_res = compute_correlations(df, ["const", "vary"], min_observations=3)
    assert corr_res["status"] == "NOT_APPLICABLE"


def test_8_grouped_analysis():
    df = pd.DataFrame({
        "category": ["A", "A", "B", "B", "B"],
        "metric": [10.0, 20.0, 100.0, 110.0, 120.0],
    })
    grouped = compute_grouped_analysis(df, ["category"], ["metric"], max_categories=10)
    assert len(grouped) == 1
    entry = grouped[0]
    assert entry["group_by"] == "category"
    assert entry["numeric_target"] == "metric"
    assert entry["groups"]["A"]["count"] == 2
    assert entry["groups"]["A"]["mean"] == 15.0
    assert entry["groups"]["B"]["mean"] == 110.0


def test_9_two_group_statistical_test():
    np.random.seed(42)
    group1 = np.random.normal(loc=10.0, scale=2.0, size=30)
    group2 = np.random.normal(loc=15.0, scale=2.0, size=30)  # Distinct shift

    test_res, effect_res = compute_two_group_test(
        group1, group2, "val", "group", "G1", "G2"
    )

    assert test_res is not None
    assert test_res["test_type"] == "two_group_comparison"
    assert test_res["welch_t_test"]["p_value"] < 0.001
    assert test_res["mann_whitney_u"]["p_value"] < 0.001

    assert effect_res is not None
    assert effect_res["metric"] == "Cohens_d"
    assert effect_res["magnitude"] == "large"


def test_10_multi_group_statistical_test():
    np.random.seed(42)
    g1 = np.random.normal(10, 1, 20)
    g2 = np.random.normal(15, 1, 20)
    g3 = np.random.normal(20, 1, 20)

    test_res, effect_res = compute_multi_group_test(
        [g1, g2, g3], ["G1", "G2", "G3"], "score", "cluster"
    )

    assert test_res is not None
    assert test_res["test_type"] == "multi_group_comparison"
    assert test_res["one_way_anova"]["p_value"] < 0.001
    assert test_res["kruskal_wallis"]["p_value"] < 0.001

    assert effect_res is not None
    assert effect_res["metric"] == "Eta_squared"
    assert effect_res["magnitude"] == "large"


def test_11_categorical_association():
    # Strong association between department and desk location
    s1 = pd.Series(["Eng"] * 30 + ["Sales"] * 30)
    s2 = pd.Series(["Floor2"] * 28 + ["Floor1"] * 2 + ["Floor1"] * 27 + ["Floor2"] * 3)

    test_res, effect_res = compute_chi_square_test(s1, s2, "dept", "floor")
    assert test_res is not None
    assert test_res["test_type"] == "chi_square_independence"
    assert test_res["p_value"] < 0.001

    assert effect_res is not None
    assert effect_res["metric"] == "Cramers_V"
    assert effect_res["magnitude"] == "large"


def test_12_effect_size_magnitude_evaluation():
    # Variance is large relative to mean difference (~0.2 difference with std ~ 3.0)
    g1 = np.array([10.0, 15.0, 8.0, 12.0, 7.0, 14.0])
    g2 = np.array([10.3, 15.2, 8.4, 12.1, 7.2, 14.2])

    _, effect_res = compute_two_group_test(g1, g2, "x", "grp", "A", "B")
    assert effect_res is not None
    assert effect_res["magnitude"] in ["small", "negligible"]
    assert "practical_significance_note" in effect_res


def test_13_confidence_intervals():
    series = pd.Series([10.0, 12.0, 14.0, 16.0, 18.0])
    ci = compute_mean_confidence_intervals(series, confidence=0.95)

    assert ci is not None
    assert ci["estimate"] == 14.0
    assert ci["confidence_level"] == 0.95
    assert ci["lower_bound"] < 14.0
    assert ci["upper_bound"] > 14.0
    assert ci["margin_of_error"] > 0
    assert ci["method"] == "t_distribution"


def test_14_insufficient_sample_size():
    small_series = pd.Series([1.0, 2.0])
    dist_res, outlier_res = compute_distribution_and_outliers(small_series)
    assert dist_res["status"] == "NOT_APPLICABLE"
    assert "Insufficient observations" in dist_res["reason"]

    # Two group test with < 5 elements
    g1 = np.array([1.0, 2.0])
    g2 = np.array([3.0, 4.0])
    t_res, e_res = compute_two_group_test(g1, g2, "x", "grp", "A", "B")
    assert t_res is None
    assert e_res is None


def test_15_all_null_column():
    null_series = pd.Series([None, np.nan, None])
    num_res = compute_numeric_statistics(null_series)
    assert num_res["status"] == "NOT_APPLICABLE"
    assert num_res["count"] == 0
    assert num_res["missing_percentage"] == 100.0

    cat_res = compute_categorical_statistics(null_series)
    assert cat_res["status"] == "NOT_APPLICABLE"
    assert cat_res["count"] == 0


def test_16_json_serialization(synthetic_multivariate_df, synthetic_discovery):
    """Verify that full pipeline result is strictly JSON-serializable."""
    report = run_statistical_analysis(synthetic_multivariate_df, synthetic_discovery)
    assert report.status == PipelineStatus.SUCCESS

    # Pydantic dump to JSON
    json_str = report.model_dump_json()
    assert isinstance(json_str, str)
    parsed = json.loads(json_str)

    # Check key sections exist and are non-empty
    assert "num_x1" in parsed["numeric_summary"]
    assert "group_two" in parsed["categorical_summary"]
    assert "num_skewed" in parsed["outliers"]
    assert "pearson" in parsed["correlations"]
    assert len(parsed["grouped_analysis"]) > 0
    assert len(parsed["confidence_intervals"]) > 0
    assert parsed["multiple_testing"]["comparisons"] >= 0


def test_17_deterministic_repeated_execution(synthetic_multivariate_df, synthetic_discovery):
    """Running twice on the same dataset produces mathematically identical outputs."""
    report1 = run_statistical_analysis(synthetic_multivariate_df, synthetic_discovery)
    report2 = run_statistical_analysis(synthetic_multivariate_df, synthetic_discovery)

    dump1 = report1.model_dump(mode="json")
    dump2 = report2.model_dump(mode="json")

    assert dump1 == dump2
