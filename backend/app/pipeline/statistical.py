"""
Deterministic Statistical Analysis Engine (Person 2 - ML / Accuracy Brain)
Operates on unknown datasets using pandas, numpy, and scipy.
Computes pure mathematical evidence with zero LLM hallucination and strict JSON safety.
"""
from typing import Dict, List, Any, Optional, Tuple
import math
import numpy as np
import pandas as pd
from scipy import stats

try:
    from backend.app.schemas.discovery_contract import DiscoveryContract
    from backend.app.schemas.analysis_contract import StatisticalReport
    from backend.app.schemas.enums import PipelineStatus
except ImportError:
    from app.schemas.discovery_contract import DiscoveryContract
    from app.schemas.analysis_contract import StatisticalReport
    from app.schemas.enums import PipelineStatus


# ---------------------------------------------------------------------------
# JSON Serialization Safety Helpers
# ---------------------------------------------------------------------------

def sanitize_for_json(val: Any) -> Any:
    """Recursively convert numpy/pandas scalars and collections into pure JSON-safe types."""
    if val is None:
        return None
    if isinstance(val, (bool, np.bool_)):
        return bool(val)
    if isinstance(val, (int, np.integer)):
        return int(val)
    if isinstance(val, (float, np.floating)):
        if math.isnan(val) or math.isinf(val) or np.isnan(val) or np.isinf(val):
            return None
        return round(float(val), 6)
    if isinstance(val, np.ndarray):
        return [sanitize_for_json(x) for x in val.tolist()]
    if isinstance(val, dict):
        return {str(k): sanitize_for_json(v) for k, v in val.items()}
    if isinstance(val, (list, tuple, set)):
        return [sanitize_for_json(x) for x in val]
    if pd.isna(val):
        return None
    return val


# ---------------------------------------------------------------------------
# 1. Numerical Column Statistics
# ---------------------------------------------------------------------------

def compute_numeric_statistics(series: pd.Series) -> Dict[str, Any]:
    """Compute exhaustive descriptive statistics for a numeric column."""
    total_count = len(series)
    valid_series = series.dropna()
    valid_count = len(valid_series)
    missing_count = total_count - valid_count
    missing_pct = (missing_count / total_count * 100.0) if total_count > 0 else 0.0

    if valid_count == 0:
        return {
            "count": 0,
            "missing_count": missing_count,
            "missing_percentage": round(missing_pct, 4),
            "status": PipelineStatus.NOT_APPLICABLE.value,
            "reason": "All values are missing or null",
        }

    # Deterministic calculations
    vals = valid_series.to_numpy(dtype=float)
    mean_val = float(np.mean(vals))
    median_val = float(np.median(vals))
    std_val = float(np.std(vals, ddof=1)) if valid_count > 1 else 0.0
    var_val = float(np.var(vals, ddof=1)) if valid_count > 1 else 0.0
    min_val = float(np.min(vals))
    max_val = float(np.max(vals))

    q25, q50, q75 = [float(x) for x in np.percentile(vals, [25, 50, 75])]
    iqr_val = q75 - q25

    # Skewness using scipy (bias=False for sample skewness)
    skew_val = float(stats.skew(vals, bias=False)) if valid_count >= 3 and std_val > 1e-12 else 0.0
    unique_count = int(len(np.unique(vals)))

    return sanitize_for_json({
        "status": PipelineStatus.SUCCESS.value,
        "count": valid_count,
        "missing_count": missing_count,
        "missing_percentage": round(missing_pct, 4),
        "mean": mean_val,
        "median": median_val,
        "std": std_val,
        "variance": var_val,
        "min": min_val,
        "max": max_val,
        "quantiles": {
            "25%": q25,
            "50%": q50,
            "75%": q75,
        },
        "iqr": iqr_val,
        "skewness": skew_val,
        "unique_count": unique_count,
        "is_constant": bool(unique_count <= 1 or std_val < 1e-12),
    })


# ---------------------------------------------------------------------------
# 2. Categorical Column Analysis
# ---------------------------------------------------------------------------

def compute_categorical_statistics(series: pd.Series, max_categories: int = 10) -> Dict[str, Any]:
    """Compute frequency distributions and cardinality characteristics for categorical columns."""
    total_count = len(series)
    valid_series = series.dropna().astype(str)
    valid_count = len(valid_series)
    missing_count = total_count - valid_count
    missing_pct = (missing_count / total_count * 100.0) if total_count > 0 else 0.0

    if valid_count == 0:
        return {
            "count": 0,
            "missing_count": missing_count,
            "missing_percentage": round(missing_pct, 4),
            "status": PipelineStatus.NOT_APPLICABLE.value,
            "reason": "All values are missing or null",
        }

    value_counts = valid_series.value_counts()
    unique_count = len(value_counts)
    unique_pct = (unique_count / valid_count * 100.0) if valid_count > 0 else 0.0

    # Flag likely high-cardinality identifiers (e.g. UUIDs, IDs)
    likely_identifier = bool(unique_count == valid_count and valid_count > 25)

    # Frequency distribution for top categories
    top_counts = value_counts.head(max_categories)
    frequency_distribution = {k: int(v) for k, v in top_counts.items()}
    category_proportions = {k: round(float(v / valid_count), 6) for k, v in top_counts.items()}

    return sanitize_for_json({
        "status": PipelineStatus.SUCCESS.value,
        "count": valid_count,
        "missing_count": missing_count,
        "missing_percentage": round(missing_pct, 4),
        "unique_count": unique_count,
        "unique_percentage": round(unique_pct, 4),
        "top_categories": list(top_counts.index),
        "frequency_distribution": frequency_distribution,
        "category_proportions": category_proportions,
        "likely_identifier": likely_identifier,
    })


# ---------------------------------------------------------------------------
# 3. Distribution & Outlier Analysis
# ---------------------------------------------------------------------------

def compute_distribution_and_outliers(series: pd.Series) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Compute distributional shape (kurtosis, skewness, percentiles) and IQR outlier bounds."""
    valid_series = series.dropna()
    valid_count = len(valid_series)

    if valid_count < 4:
        not_app = {
            "status": PipelineStatus.NOT_APPLICABLE.value,
            "reason": "Insufficient observations (< 4) for distribution and outlier analysis",
        }
        return not_app, not_app

    vals = valid_series.to_numpy(dtype=float)
    std_val = float(np.std(vals, ddof=1))

    if std_val < 1e-12:
        zero_var = {
            "status": PipelineStatus.NOT_APPLICABLE.value,
            "reason": "Zero variance / constant column",
        }
        return zero_var, zero_var

    # Percentiles
    p1, p5, p25, p50, p75, p95, p99 = [float(x) for x in np.percentile(vals, [1, 5, 25, 50, 75, 95, 99])]
    iqr_val = p75 - p25

    skew_val = float(stats.skew(vals, bias=False)) if valid_count >= 3 else 0.0
    kurt_val = float(stats.kurtosis(vals, bias=False)) if valid_count >= 4 else 0.0

    # Normality test: D'Agostino-Pearson if N >= 20, Shapiro-Wilk if 8 <= N < 5000
    normality_test: Dict[str, Any] = {"tested": False}
    if valid_count >= 20:
        try:
            stat, p_val = stats.normaltest(vals)
            normality_test = {
                "tested": True,
                "test_name": "DAgostino_Pearson",
                "statistic": float(stat),
                "p_value": float(p_val),
                "is_likely_normal": bool(p_val > 0.05),
            }
        except Exception as e:
            normality_test = {"tested": False, "reason": str(e)}
    elif valid_count >= 8:
        try:
            stat, p_val = stats.shapiro(vals)
            normality_test = {
                "tested": True,
                "test_name": "Shapiro_Wilk",
                "statistic": float(stat),
                "p_value": float(p_val),
                "is_likely_normal": bool(p_val > 0.05),
            }
        except Exception as e:
            normality_test = {"tested": False, "reason": str(e)}

    # IQR Outlier detection
    lower_bound = p25 - 1.5 * iqr_val
    upper_bound = p75 + 1.5 * iqr_val
    outliers_mask = (vals < lower_bound) | (vals > upper_bound)
    outlier_count = int(np.sum(outliers_mask))
    outlier_pct = float(outlier_count / valid_count * 100.0)

    # Extreme outliers (3.0 * IQR)
    extreme_lower = p25 - 3.0 * iqr_val
    extreme_upper = p75 + 3.0 * iqr_val
    extreme_count = int(np.sum((vals < extreme_lower) | (vals > extreme_upper)))

    dist_result = sanitize_for_json({
        "status": PipelineStatus.SUCCESS.value,
        "skewness": skew_val,
        "kurtosis": kurt_val,
        "quantiles": {
            "1%": p1,
            "5%": p5,
            "25%": p25,
            "50%": p50,
            "75%": p75,
            "95%": p95,
            "99%": p99,
        },
        "iqr": iqr_val,
        "normality_assessment": normality_test,
    })

    outlier_result = sanitize_for_json({
        "status": PipelineStatus.SUCCESS.value,
        "method": "IQR",
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "outlier_count": outlier_count,
        "outlier_percentage": round(outlier_pct, 4),
        "extreme_outlier_count": extreme_count,
        "extreme_bounds": {
            "lower": extreme_lower,
            "upper": extreme_upper,
        },
    })

    return dist_result, outlier_result


# ---------------------------------------------------------------------------
# 5. Correlation Engine (Pearson & Spearman)
# ---------------------------------------------------------------------------

def compute_correlations(
    df: pd.DataFrame,
    numeric_cols: List[str],
    min_observations: int = 5,
) -> Dict[str, Any]:
    """
    Compute Pearson and Spearman correlation matrices.
    Excludes constant columns and safely handles missing values.
    Identifies strong relationships (|r| >= 0.7) with non-causal disclaimer.
    """
    if len(numeric_cols) < 2:
        return {
            "status": PipelineStatus.NOT_APPLICABLE.value,
            "reason": "At least 2 numeric columns required for correlation analysis",
            "pearson": {},
            "spearman": {},
            "strong_relationships": [],
            "disclaimer": "Correlation does not imply causation.",
        }

    # Filter out constant or empty columns
    usable_cols = []
    for col in numeric_cols:
        clean = df[col].dropna()
        if len(clean) >= min_observations and clean.std(ddof=1) > 1e-12:
            usable_cols.append(col)

    if len(usable_cols) < 2:
        return {
            "status": PipelineStatus.NOT_APPLICABLE.value,
            "reason": "Fewer than 2 non-constant numeric columns with sufficient observations",
            "pearson": {},
            "spearman": {},
            "strong_relationships": [],
            "disclaimer": "Correlation does not imply causation.",
        }

    sub_df = df[usable_cols]
    pearson_df = sub_df.corr(method="pearson", min_periods=min_observations)
    spearman_df = sub_df.corr(method="spearman", min_periods=min_observations)

    pearson_mat = {col: {c2: (round(float(v), 6) if pd.notna(v) else None) for c2, v in row.items()} for col, row in pearson_df.iterrows()}
    spearman_mat = {col: {c2: (round(float(v), 6) if pd.notna(v) else None) for c2, v in row.items()} for col, row in spearman_df.iterrows()}

    # Identify strong relationships (|r| >= 0.7, excluding self-correlation)
    strong_relationships: List[Dict[str, Any]] = []
    seen_pairs = set()
    for i, col1 in enumerate(usable_cols):
        for col2 in usable_cols[i + 1:]:
            r_pearson = pearson_df.loc[col1, col2] if col1 in pearson_df.index and col2 in pearson_df.columns else np.nan
            r_spearman = spearman_df.loc[col1, col2] if col1 in spearman_df.index and col2 in spearman_df.columns else np.nan

            if pd.notna(r_pearson) and abs(r_pearson) >= 0.7:
                pair_key = tuple(sorted([col1, col2]))
                if pair_key not in seen_pairs:
                    seen_pairs.add(pair_key)
                    strong_relationships.append({
                        "feature_1": col1,
                        "feature_2": col2,
                        "pearson_r": round(float(r_pearson), 6),
                        "spearman_rho": round(float(r_spearman), 6) if pd.notna(r_spearman) else None,
                        "strength": "strong",
                        "direction": "positive" if r_pearson > 0 else "negative",
                    })

    return sanitize_for_json({
        "status": PipelineStatus.SUCCESS.value,
        "pearson": pearson_mat,
        "spearman": spearman_mat,
        "strong_relationships": strong_relationships,
        "disclaimer": "Correlation does not imply causation.",
    })


# ---------------------------------------------------------------------------
# 6. Grouped Summaries
# ---------------------------------------------------------------------------

def compute_grouped_analysis(
    df: pd.DataFrame,
    categorical_cols: List[str],
    numeric_cols: List[str],
    max_categories: int = 15,
) -> List[Dict[str, Any]]:
    """Compute generic grouped summaries (count, mean, median, std, min, max) without hardcoding."""
    grouped_results: List[Dict[str, Any]] = []

    for cat_col in categorical_cols:
        cat_series = df[cat_col].dropna().astype(str)
        unique_cats = cat_series.unique()
        # Only group by columns with 2 to max_categories distinct groups
        if len(unique_cats) < 2 or len(unique_cats) > max_categories:
            continue

        for num_col in numeric_cols:
            sub = df[[cat_col, num_col]].dropna()
            if len(sub) < 5:
                continue

            grouped = sub.groupby(cat_col)[num_col]
            group_stats: Dict[str, Any] = {}
            for name, grp in grouped:
                arr = grp.to_numpy(dtype=float)
                if len(arr) == 0:
                    continue
                group_stats[str(name)] = {
                    "count": int(len(arr)),
                    "mean": round(float(np.mean(arr)), 6),
                    "median": round(float(np.median(arr)), 6),
                    "std": round(float(np.std(arr, ddof=1)), 6) if len(arr) > 1 else 0.0,
                    "min": round(float(np.min(arr)), 6),
                    "max": round(float(np.max(arr)), 6),
                }

            if len(group_stats) >= 2:
                grouped_results.append({
                    "group_by": cat_col,
                    "numeric_target": num_col,
                    "groups": group_stats,
                })

    return sanitize_for_json(grouped_results)


# ---------------------------------------------------------------------------
# 7 & 8. Statistical Significance Tests & Effect Sizes
# ---------------------------------------------------------------------------

def compute_two_group_test(
    group1: np.ndarray,
    group2: np.ndarray,
    feature_name: str,
    group_col: str,
    group1_name: str,
    group2_name: str,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Run two-group numerical comparison (Welch's t-test and Mann-Whitney U) and Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    if n1 < 5 or n2 < 5:
        return None, None

    std1 = float(np.std(group1, ddof=1))
    std2 = float(np.std(group2, ddof=1))
    if std1 < 1e-12 and std2 < 1e-12:
        return None, None

    # Welch's t-test (robust against unequal variances)
    t_stat, t_pval = stats.ttest_ind(group1, group2, equal_var=False)

    # Mann-Whitney U non-parametric test
    u_stat, u_pval = stats.mannwhitneyu(group1, group2, alternative="two-sided")

    # Cohen's d effect size
    pooled_sd = math.sqrt(((n1 - 1) * std1 ** 2 + (n2 - 1) * std2 ** 2) / (n1 + n2 - 2)) if (n1 + n2 - 2) > 0 else 1.0
    cohens_d = (float(np.mean(group1)) - float(np.mean(group2))) / pooled_sd if pooled_sd > 1e-12 else 0.0

    d_abs = abs(cohens_d)
    d_magnitude = "negligible"
    if d_abs >= 0.8:
        d_magnitude = "large"
    elif d_abs >= 0.5:
        d_magnitude = "medium"
    elif d_abs >= 0.2:
        d_magnitude = "small"

    test_res = {
        "test_type": "two_group_comparison",
        "feature": feature_name,
        "group_by": group_col,
        "groups": [group1_name, group2_name],
        "sample_sizes": [n1, n2],
        "welch_t_test": {
            "statistic": float(t_stat) if pd.notna(t_stat) else None,
            "p_value": float(t_pval) if pd.notna(t_pval) else None,
        },
        "mann_whitney_u": {
            "statistic": float(u_stat) if pd.notna(u_stat) else None,
            "p_value": float(u_pval) if pd.notna(u_pval) else None,
        },
        "primary_p_value": float(t_pval) if pd.notna(t_pval) else float(u_pval),
        "assumptions_limitations": "Welch t-test does not assume equal variances; Mann-Whitney U is rank-based.",
    }

    effect_res = {
        "metric": "Cohens_d",
        "feature": feature_name,
        "group_by": group_col,
        "value": round(float(cohens_d), 6),
        "magnitude": d_magnitude,
        "practical_significance_note": f"Effect size is {d_magnitude} (d={cohens_d:.4f}). Statistical significance must be evaluated alongside magnitude.",
    }

    return test_res, effect_res


def compute_multi_group_test(
    groups: List[np.ndarray],
    group_names: List[str],
    feature_name: str,
    group_col: str,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Run One-Way ANOVA & Kruskal-Wallis test across 3+ groups and compute Eta-squared."""
    if len(groups) < 3 or any(len(g) < 5 for g in groups):
        return None, None

    try:
        f_stat, anova_pval = stats.f_oneway(*groups)
        h_stat, kruskal_pval = stats.kruskal(*groups)
    except Exception:
        return None, None

    # Eta-squared calculation: SS_between / SS_total
    all_vals = np.concatenate(groups)
    grand_mean = np.mean(all_vals)
    ss_total = np.sum((all_vals - grand_mean) ** 2)
    ss_between = np.sum([len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups])
    eta_sq = float(ss_between / ss_total) if ss_total > 1e-12 else 0.0

    eta_magnitude = "negligible"
    if eta_sq >= 0.14:
        eta_magnitude = "large"
    elif eta_sq >= 0.06:
        eta_magnitude = "medium"
    elif eta_sq >= 0.01:
        eta_magnitude = "small"

    test_res = {
        "test_type": "multi_group_comparison",
        "feature": feature_name,
        "group_by": group_col,
        "groups": group_names,
        "sample_sizes": [len(g) for g in groups],
        "one_way_anova": {
            "statistic": float(f_stat) if pd.notna(f_stat) else None,
            "p_value": float(anova_pval) if pd.notna(anova_pval) else None,
        },
        "kruskal_wallis": {
            "statistic": float(h_stat) if pd.notna(h_stat) else None,
            "p_value": float(kruskal_pval) if pd.notna(kruskal_pval) else None,
        },
        "primary_p_value": float(anova_pval) if pd.notna(anova_pval) else float(kruskal_pval),
        "assumptions_limitations": "ANOVA assumes approximate normality and homoscedasticity; Kruskal-Wallis is non-parametric.",
    }

    effect_res = {
        "metric": "Eta_squared",
        "feature": feature_name,
        "group_by": group_col,
        "value": round(float(eta_sq), 6),
        "magnitude": eta_magnitude,
        "practical_significance_note": f"Variance explained is {eta_sq * 100.0:.2f}% ({eta_magnitude}).",
    }

    return test_res, effect_res


def compute_chi_square_test(
    series1: pd.Series,
    series2: pd.Series,
    col1_name: str,
    col2_name: str,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Compute Chi-Square test of independence and Cramér's V effect size for categorical pairs."""
    sub = pd.DataFrame({col1_name: series1, col2_name: series2}).dropna()
    if len(sub) < 20:
        return None, None

    contingency = pd.crosstab(sub[col1_name], sub[col2_name])
    r, c = contingency.shape
    if r < 2 or c < 2 or r > 10 or c > 10:
        return None, None

    try:
        chi2, p_val, dof, _ = stats.chi2_contingency(contingency)
    except Exception:
        return None, None

    n = len(sub)
    min_dim = min(r - 1, c - 1)
    cramers_v = math.sqrt(chi2 / (n * min_dim)) if (n * min_dim) > 0 else 0.0

    v_magnitude = "negligible"
    if cramers_v >= 0.35:
        v_magnitude = "large"
    elif cramers_v >= 0.20:
        v_magnitude = "medium"
    elif cramers_v >= 0.10:
        v_magnitude = "small"

    test_res = {
        "test_type": "chi_square_independence",
        "variable_1": col1_name,
        "variable_2": col2_name,
        "contingency_shape": [int(r), int(c)],
        "sample_size": n,
        "degrees_of_freedom": int(dof),
        "statistic": round(float(chi2), 6),
        "p_value": round(float(p_val), 6),
        "primary_p_value": round(float(p_val), 6),
        "assumptions_limitations": "Requires expected cell frequencies generally >= 5.",
    }

    effect_res = {
        "metric": "Cramers_V",
        "variables": [col1_name, col2_name],
        "value": round(float(cramers_v), 6),
        "magnitude": v_magnitude,
        "practical_significance_note": f"Association strength is {v_magnitude} (V={cramers_v:.4f}).",
    }

    return test_res, effect_res


# ---------------------------------------------------------------------------
# 9. Confidence Intervals
# ---------------------------------------------------------------------------

def compute_mean_confidence_intervals(
    series: pd.Series,
    confidence: float = 0.95,
) -> Optional[Dict[str, Any]]:
    """Compute confidence interval for population mean using Student's t-distribution."""
    clean = series.dropna().to_numpy(dtype=float)
    n = len(clean)
    if n < 3:
        return None

    mean_val = float(np.mean(clean))
    std_err = float(stats.sem(clean))
    if std_err < 1e-12:
        return {
            "estimate": mean_val,
            "confidence_level": confidence,
            "lower_bound": mean_val,
            "upper_bound": mean_val,
            "margin_of_error": 0.0,
            "sample_size": n,
            "method": "zero_variance_exact",
        }

    # t-distribution critical value
    ci_lower, ci_upper = stats.t.interval(confidence, df=n - 1, loc=mean_val, scale=std_err)
    margin = float(ci_upper - mean_val)

    return sanitize_for_json({
        "estimate": round(mean_val, 6),
        "confidence_level": confidence,
        "lower_bound": round(float(ci_lower), 6),
        "upper_bound": round(float(ci_upper), 6),
        "margin_of_error": round(margin, 6),
        "sample_size": n,
        "method": "t_distribution",
    })


# ---------------------------------------------------------------------------
# 12. Multiple Testing Correction (Benjamini-Hochberg FDR)
# ---------------------------------------------------------------------------

def apply_benjamini_hochberg(
    test_records: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Apply Benjamini-Hochberg (FDR) multiple testing correction to gathered p-values."""
    valid_indices = []
    p_values = []
    for idx, t in enumerate(test_records):
        pval = t.get("primary_p_value")
        if pval is not None and not math.isnan(pval):
            valid_indices.append(idx)
            p_values.append(float(pval))

    m = len(p_values)
    if m == 0:
        return test_records, {"comparisons": 0, "correction": None}

    # Sort p-values to calculate FDR adjusted bounds
    sorted_pairs = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [1.0] * m

    cum_min = 1.0
    # Walk backwards from highest rank
    for rank_minus_1 in range(m - 1, -1, -1):
        original_pos, pval = sorted_pairs[rank_minus_1]
        rank = rank_minus_1 + 1
        adj = min(1.0, (pval * m) / rank)
        cum_min = min(cum_min, adj)
        adjusted[original_pos] = cum_min

    # Inject adjusted p-values back into test records
    for i, orig_idx in enumerate(valid_indices):
        test_records[orig_idx]["adjusted_p_value_fdr"] = round(adjusted[i], 6)
        test_records[orig_idx]["fdr_significant_at_05"] = bool(adjusted[i] < 0.05)

    multiple_testing_meta = {
        "comparisons": m,
        "correction": "Benjamini-Hochberg (FDR)",
        "alpha": 0.05,
    }

    return test_records, multiple_testing_meta


# ---------------------------------------------------------------------------
# Master Statistical Orchestrator
# ---------------------------------------------------------------------------

def run_statistical_analysis(
    df: Optional[Any],
    discovery: DiscoveryContract,
) -> StatisticalReport:
    """
    Execute full deterministic statistical profiling on unknown datasets.
    Zero LLM involvement; all numerical computations use pandas, numpy, and scipy.
    """
    if df is None:
        return StatisticalReport(
            status=PipelineStatus.NOT_APPLICABLE,
            reason="No data provided for statistical analysis",
        )

    # Convert to DataFrame if dictionary or list passed
    if not isinstance(df, pd.DataFrame):
        try:
            df = pd.DataFrame(df)
        except Exception as e:
            return StatisticalReport(
                status=PipelineStatus.FAILED,
                reason=f"Failed to convert input data to DataFrame: {str(e)}",
            )

    if df.empty or len(df) == 0:
        return StatisticalReport(
            status=PipelineStatus.NOT_APPLICABLE,
            reason="Input dataset contains zero rows",
        )

    schema_def = discovery.schema_definition or {}

    # Separate columns dynamically based on schema_def or infer from dtypes
    numeric_cols: List[str] = []
    categorical_cols: List[str] = []

    for col in df.columns:
        col_type = schema_def.get(col, "").lower()
        if col_type == "numeric" or (not col_type and pd.api.types.is_numeric_dtype(df[col])):
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)

    numeric_summary: Dict[str, Any] = {}
    categorical_summary: Dict[str, Any] = {}
    distributions: Dict[str, Any] = {}
    outliers: Dict[str, Any] = {}
    confidence_intervals: List[Dict[str, Any]] = []

    # 1. Numerical analysis & distribution/outlier checks
    for col in numeric_cols:
        series = pd.to_numeric(df[col], errors="coerce")
        stats_out = compute_numeric_statistics(series)
        numeric_summary[col] = stats_out

        dist_out, outlier_out = compute_distribution_and_outliers(series)
        distributions[col] = dist_out
        outliers[col] = outlier_out

        ci_out = compute_mean_confidence_intervals(series)
        if ci_out is not None:
            ci_out["column"] = col
            confidence_intervals.append(ci_out)

    # 2. Categorical analysis
    for col in categorical_cols:
        cat_out = compute_categorical_statistics(df[col])
        categorical_summary[col] = cat_out

    # 3. Correlations
    correlation_report = compute_correlations(df, numeric_cols)

    # 4. Grouped analysis
    grouped_report = compute_grouped_analysis(df, categorical_cols, numeric_cols)

    # 5. Statistical Significance Tests & Effect Sizes
    statistical_tests: List[Dict[str, Any]] = []
    effect_sizes: List[Dict[str, Any]] = []

    # Numerical grouped tests across valid categorical variables
    for cat_col in categorical_cols:
        cat_clean = df[cat_col].dropna().astype(str)
        unique_groups = cat_clean.unique()

        # Two-group comparison
        if len(unique_groups) == 2:
            g1_name, g2_name = unique_groups[0], unique_groups[1]
            for num_col in numeric_cols:
                sub = df[[cat_col, num_col]].dropna()
                g1_vals = sub[sub[cat_col] == g1_name][num_col].to_numpy(dtype=float)
                g2_vals = sub[sub[cat_col] == g2_name][num_col].to_numpy(dtype=float)

                t_res, e_res = compute_two_group_test(g1_vals, g2_vals, num_col, cat_col, g1_name, g2_name)
                if t_res is not None:
                    statistical_tests.append(t_res)
                if e_res is not None:
                    effect_sizes.append(e_res)

        # Multi-group comparison (3 to 6 groups to prevent noise explosion)
        elif 3 <= len(unique_groups) <= 6:
            for num_col in numeric_cols:
                sub = df[[cat_col, num_col]].dropna()
                group_arrays = [sub[sub[cat_col] == g][num_col].to_numpy(dtype=float) for g in unique_groups]
                t_res, e_res = compute_multi_group_test(group_arrays, list(unique_groups), num_col, cat_col)
                if t_res is not None:
                    statistical_tests.append(t_res)
                if e_res is not None:
                    effect_sizes.append(e_res)

    # Categorical pairs (Chi-Square test)
    for i, c1 in enumerate(categorical_cols[:5]):
        for c2 in categorical_cols[i + 1:6]:
            t_res, e_res = compute_chi_square_test(df[c1], df[c2], c1, c2)
            if t_res is not None:
                statistical_tests.append(t_res)
            if e_res is not None:
                effect_sizes.append(e_res)

    # 6. Multiple Testing Correction (FDR)
    corrected_tests, mult_meta = apply_benjamini_hochberg(statistical_tests)

    # Compile sanitized report
    return StatisticalReport(
        status=PipelineStatus.SUCCESS,
        numeric_summary=numeric_summary,
        categorical_summary=categorical_summary,
        distributions=distributions,
        outliers=outliers,
        correlations=correlation_report,
        grouped_analysis=grouped_report,
        statistical_tests=corrected_tests,
        effect_sizes=effect_sizes,
        confidence_intervals=confidence_intervals,
        multiple_testing=mult_meta,
        # Maintain backwards compatibility aliases
        summary_stats=numeric_summary,
        distribution_checks=distributions,
    )
