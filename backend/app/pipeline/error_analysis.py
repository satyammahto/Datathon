"""
Deterministic Error Analysis & Post-Hoc Diagnostics Engine (Person 2 - ML / Accuracy Brain)
Decomposes classification errors and regression residuals to generate structured diagnostic
evidence for Person 3 (Agent / Reasoning Brain).
Zero LLM involvement; strictly empirical, mathematical diagnostics.
"""
from typing import Dict, List, Any, Optional, Tuple
import math
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    mean_absolute_error,
    root_mean_squared_error,
)

try:
    from backend.app.schemas.discovery_contract import DiscoveryContract
    from backend.app.schemas.analysis_contract import (
        ErrorAnalysisReport,
        ChampionResult,
        ValidationReport,
    )
    from backend.app.schemas.enums import PipelineStatus, MLTaskType
except ImportError:
    from app.schemas.discovery_contract import DiscoveryContract
    from app.schemas.analysis_contract import (
        ErrorAnalysisReport,
        ChampionResult,
        ValidationReport,
    )
    from app.schemas.enums import PipelineStatus, MLTaskType

from .statistical import sanitize_for_json


# ---------------------------------------------------------------------------
# 1. Classification Error Decomposition
# ---------------------------------------------------------------------------

def compute_classification_error_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    indices: Optional[List[int]] = None,
    top_k: int = 10,
) -> Dict[str, Any]:
    """Decompose classification errors: confusion matrix, per-class metrics, weak classes, worst errors."""
    classes = np.unique(np.concatenate([y_true, y_pred]))
    cm = confusion_matrix(y_true, y_pred, labels=classes)

    total_samples = len(y_true)
    correct_count = int(np.sum(y_true == y_pred))
    error_count = total_samples - correct_count
    error_rate = float(error_count / total_samples) if total_samples > 0 else 0.0
    overall_acc = float(correct_count / total_samples) if total_samples > 0 else 0.0
    overall_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    # Per-class metrics
    class_metrics: List[Dict[str, Any]] = []
    weak_classes: List[Dict[str, Any]] = []

    for idx, c in enumerate(classes):
        tp = int(cm[idx, idx])
        fn = int(np.sum(cm[idx, :]) - tp)
        fp = int(np.sum(cm[:, idx]) - tp)
        tn = int(total_samples - tp - fp - fn)
        support = tp + fn

        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        c_str = str(c.item() if hasattr(c, "item") else c)
        c_entry = {
            "class": c_str,
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
        }
        class_metrics.append(c_entry)

        # Weak class rule: recall < 0.60 and support >= 2 (so not triggered by 1 single sample)
        if rec < 0.60 and support >= 2:
            weak_classes.append({
                "class": c_str,
                "support": support,
                "recall": round(rec, 4),
                "f1": round(f1, 4),
                "status": "WEAK",
                "severity": "HIGH" if rec < 0.35 else "MEDIUM",
                "evidence": f"Class '{c_str}' has low recall ({rec:.2f}) with support of {support} samples.",
            })

    # Sort weak classes by lowest recall then lowest F1
    weak_classes.sort(key=lambda x: (x["recall"], x["f1"]))

    # Misclassified samples (top_k)
    sample_idxs = indices if indices is not None and len(indices) == total_samples else list(range(total_samples))
    worst_misclassifications: List[Dict[str, Any]] = []
    for i in range(total_samples):
        if y_true[i] != y_pred[i]:
            act_val = y_true[i].item() if hasattr(y_true[i], "item") else y_true[i]
            pred_val = y_pred[i].item() if hasattr(y_pred[i], "item") else y_pred[i]
            worst_misclassifications.append({
                "sample_index": int(sample_idxs[i]),
                "actual": str(act_val),
                "predicted": str(pred_val),
            })
            if len(worst_misclassifications) >= top_k:
                break

    cm_serializable = [[int(val) for val in row] for row in cm]
    class_labels_serializable = [str(c.item() if hasattr(c, "item") else c) for c in classes]

    return {
        "status": PipelineStatus.SUCCESS.value,
        "summary": {
            "total_samples": total_samples,
            "error_count": error_count,
            "error_rate": round(error_rate, 4),
            "accuracy": round(overall_acc, 4),
            "f1_macro": round(overall_f1, 4),
        },
        "confusion_matrix": cm_serializable,
        "class_labels": class_labels_serializable,
        "class_metrics": class_metrics,
        "weak_classes": weak_classes,
        "worst_predictions": worst_misclassifications,
    }


# ---------------------------------------------------------------------------
# 2. Regression Residual Decomposition & Heteroscedasticity
# ---------------------------------------------------------------------------

def compute_regression_error_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    indices: Optional[List[int]] = None,
    top_k: int = 10,
) -> Dict[str, Any]:
    """Decompose regression residuals: MAE, RMSE, systematic bias, worst errors, and heteroscedasticity."""
    total_samples = len(y_true)
    y_true_f = y_true.astype(float)
    y_pred_f = y_pred.astype(float)

    residuals = y_true_f - y_pred_f
    abs_errors = np.abs(residuals)

    mae = float(mean_absolute_error(y_true_f, y_pred_f))
    rmse = float(root_mean_squared_error(y_true_f, y_pred_f))
    mean_res = float(np.mean(residuals))
    median_res = float(np.median(residuals))
    std_res = float(np.std(residuals, ddof=1)) if total_samples > 1 else 0.0
    min_res = float(np.min(residuals))
    max_res = float(np.max(residuals))
    median_abs_err = float(np.median(abs_errors))

    # Systematic Bias Detection
    bias_detected = False
    bias_direction = "neutral"
    bias_desc = "Residuals are approximately centered around zero."
    if (std_res < 1e-12 and abs(mean_res) > 1e-6) or (std_res >= 1e-12 and abs(mean_res) > 0.15 * std_res):
        bias_detected = True
        if mean_res > 0:
            bias_direction = "systematic_underprediction"
            bias_desc = f"Model shows positive residual bias (mean={mean_res:.3f}): it tends to systematically underpredict actual values."
        else:
            bias_direction = "systematic_overprediction"
            bias_desc = f"Model shows negative residual bias (mean={mean_res:.3f}): it tends to systematically overpredict actual values."

    # Top-K worst predictions by absolute error
    sample_idxs = indices if indices is not None and len(indices) == total_samples else list(range(total_samples))
    sorted_err_order = np.argsort(-abs_errors)
    worst_preds: List[Dict[str, Any]] = []
    for rank, idx in enumerate(sorted_err_order[:top_k]):
        worst_preds.append({
            "sample_index": int(sample_idxs[idx]),
            "actual_value": round(float(y_true_f[idx]), 4),
            "predicted_value": round(float(y_pred_f[idx]), 4),
            "residual": round(float(residuals[idx]), 4),
            "absolute_error": round(float(abs_errors[idx]), 4),
            "rank": rank + 1,
        })

    # Heteroscedasticity Diagnostic via Prediction Bins
    error_by_bin: List[Dict[str, Any]] = []
    heteroscedasticity_diag: Dict[str, Any] = {
        "status": "NOT_EVALUATED",
        "evidence": "Insufficient samples for quantile binning.",
    }

    if total_samples >= 15:
        n_bins = min(4, total_samples // 5)
        quantiles = np.linspace(0, 1, n_bins + 1)
        bin_edges = np.quantile(y_pred_f, quantiles)
        bin_edges[0] -= 1e-6
        bin_edges[-1] += 1e-6

        bin_maes = []
        for b_idx in range(n_bins):
            b_low, b_high = bin_edges[b_idx], bin_edges[b_idx + 1]
            mask = (y_pred_f >= b_low) & (y_pred_f <= b_high)
            bin_count = int(np.sum(mask))
            if bin_count >= 3:
                bin_mae = float(np.mean(abs_errors[mask]))
                bin_std = float(np.std(residuals[mask], ddof=1)) if bin_count > 1 else 0.0
                bin_maes.append(bin_mae)
                error_by_bin.append({
                    "bin_index": b_idx + 1,
                    "predicted_range": [round(float(b_low), 4), round(float(b_high), 4)],
                    "sample_count": bin_count,
                    "mean_absolute_error": round(bin_mae, 4),
                    "std_residual": round(bin_std, 4),
                })

        # Spearman correlation between predicted value and absolute error
        if np.std(y_pred_f) > 1e-12 and np.std(abs_errors) > 1e-12:
            rho, pval = stats.spearmanr(y_pred_f, abs_errors)
            rho_val = float(rho) if pd.notna(rho) else 0.0
            pval_val = float(pval) if pd.notna(pval) else 1.0
        else:
            rho_val, pval_val = 0.0, 1.0

        if len(bin_maes) >= 2 and min(bin_maes) > 1e-12:
            ratio = max(bin_maes) / min(bin_maes)
            if ratio >= 2.0 or (abs(rho_val) >= 0.35 and pval_val < 0.05):
                heteroscedasticity_diag = {
                    "status": "POSSIBLE",
                    "dispersion_ratio": round(float(ratio), 3),
                    "spearman_rho": round(rho_val, 4),
                    "p_value": round(pval_val, 4),
                    "evidence": (
                        f"Error magnitude varies substantially with predicted value "
                        f"(max/min bin MAE ratio = {ratio:.2f}, Spearman rho = {rho_val:.3f})."
                    ),
                    "disclaimer": "Association diagnostic only. Does not imply causal heteroscedasticity.",
                }
            else:
                heteroscedasticity_diag = {
                    "status": "HOMOSCEDASTIC_LIKELY",
                    "dispersion_ratio": round(float(ratio), 3),
                    "spearman_rho": round(rho_val, 4),
                    "evidence": f"Error dispersion remains relatively stable across prediction bins (ratio = {ratio:.2f}).",
                }

    # Robustness / Outlier Sensitivity
    robustness_info: Dict[str, Any] = {}
    if total_samples >= 20:
        # Exclude top 5% worst predictions
        cutoff_idx = max(1, int(total_samples * 0.05))
        trimmed_abs_errors = np.sort(abs_errors)[:-cutoff_idx]
        trimmed_mae = float(np.mean(trimmed_abs_errors))
        mae_reduction_pct = float((1.0 - trimmed_mae / mae) * 100.0) if mae > 1e-12 else 0.0
        robustness_info = {
            "overall_mae": round(mae, 4),
            "trimmed_mae_95pct": round(trimmed_mae, 4),
            "outlier_tail_reduction_pct": round(mae_reduction_pct, 2),
            "interpretation": f"Excluding the top 5% worst errors reduces MAE by {mae_reduction_pct:.1f}%.",
        }

    return {
        "status": PipelineStatus.SUCCESS.value,
        "residual_summary": {
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "mean_residual": round(mean_res, 4),
            "median_residual": round(median_res, 4),
            "std_residual": round(std_res, 4),
            "min_residual": round(min_res, 4),
            "max_residual": round(max_res, 4),
            "median_absolute_error": round(median_abs_err, 4),
            "bias_diagnostic": {
                "bias_detected": bias_detected,
                "direction": bias_direction,
                "description": bias_desc,
                "disclaimer": "Descriptive diagnostic only. Does not imply causation.",
            },
        },
        "worst_predictions": worst_preds,
        "error_by_prediction_bin": error_by_bin,
        "heteroscedasticity": heteroscedasticity_diag,
        "robustness": robustness_info,
    }


# ---------------------------------------------------------------------------
# 3. Subgroup / Slice Error Analysis
# ---------------------------------------------------------------------------

def compute_subgroup_slice_analysis(
    X: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task_type: MLTaskType,
    discovery: Optional[DiscoveryContract] = None,
    min_subgroup_size: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Compute subgroup error rates across valid categorical slices to flag underperforming cohorts."""
    n_samples = len(y_true)
    min_size = min_subgroup_size if min_subgroup_size is not None else max(5, int(math.ceil(0.01 * n_samples)))

    schema_def = (discovery.schema_definition if discovery else {}) or {}
    leakage_cols = set()
    if discovery and discovery.leakage_report:
        leakage_cols.update(discovery.leakage_report.get("high_cardinality_identifiers", []))

    subgroup_reports: List[Dict[str, Any]] = []

    # Overall baselines
    overall_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0)) if task_type == MLTaskType.CLASSIFICATION else None
    overall_mae = float(mean_absolute_error(y_true, y_pred)) if task_type == MLTaskType.REGRESSION else None

    for col in X.columns:
        if col in leakage_cols:
            continue

        series = X[col]
        # Only evaluate categorical-like columns
        is_cat = (
            schema_def.get(col, "").lower() == "categorical"
            or pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
            or str(series.dtype).startswith("category")
            or not pd.api.types.is_numeric_dtype(series)
        )
        if not is_cat:
            continue

        unique_vals = series.dropna().unique()
        # Protect against identifier / extreme cardinality
        if len(unique_vals) < 2 or len(unique_vals) > 20:
            continue

        for cat_val in unique_vals:
            mask = (series == cat_val).values
            group_count = int(np.sum(mask))
            if group_count < min_size:
                continue

            grp_true = y_true[mask]
            grp_pred = y_pred[mask]

            if task_type == MLTaskType.CLASSIFICATION:
                grp_f1 = float(f1_score(grp_true, grp_pred, average="macro", zero_division=0))
                grp_acc = float(accuracy_score(grp_true, grp_pred))
                err_rate = float(1.0 - grp_acc)
                f1_delta = grp_f1 - (overall_f1 or 0.0)

                # Deterministic severity rule
                is_weak = f1_delta < -0.15 or err_rate > 0.40
                severity = "LOW"
                if is_weak:
                    severity = "HIGH" if (f1_delta < -0.25 and group_count >= 10) else "MEDIUM"

                subgroup_reports.append({
                    "feature": str(col),
                    "subgroup": str(cat_val),
                    "sample_count": group_count,
                    "task": "classification",
                    "subgroup_f1": round(grp_f1, 4),
                    "overall_f1": round(overall_f1 or 0.0, 4),
                    "f1_delta": round(f1_delta, 4),
                    "error_rate": round(err_rate, 4),
                    "status": "WEAK_SLICE" if is_weak else "NORMAL",
                    "severity": severity if is_weak else "NONE",
                })
            else:
                grp_mae = float(mean_absolute_error(grp_true, grp_pred))
                mae_ratio = (grp_mae / overall_mae) if (overall_mae and overall_mae > 1e-12) else 1.0

                # Deterministic severity rule
                is_weak = mae_ratio >= 1.25
                severity = "LOW"
                if is_weak:
                    severity = "HIGH" if (mae_ratio >= 1.50 and group_count >= 10) else "MEDIUM"

                subgroup_reports.append({
                    "feature": str(col),
                    "subgroup": str(cat_val),
                    "sample_count": group_count,
                    "task": "regression",
                    "subgroup_mae": round(grp_mae, 4),
                    "overall_mae": round(overall_mae or 0.0, 4),
                    "mae_ratio": round(mae_ratio, 3),
                    "status": "WEAK_SLICE" if is_weak else "NORMAL",
                    "severity": severity if is_weak else "NONE",
                })

    return subgroup_reports


# ---------------------------------------------------------------------------
# 4. Numeric Feature vs Error Relationships
# ---------------------------------------------------------------------------

def compute_numeric_error_associations(
    X: pd.DataFrame,
    abs_errors: np.ndarray,
    discovery: Optional[DiscoveryContract] = None,
) -> List[Dict[str, Any]]:
    """Calculate statistical association between numeric feature values and absolute prediction error."""
    schema_def = (discovery.schema_definition if discovery else {}) or {}
    associations: List[Dict[str, Any]] = []

    for col in X.columns:
        series = pd.to_numeric(X[col], errors="coerce")
        clean_mask = series.notna().values
        if np.sum(clean_mask) < 10:
            continue

        col_type = schema_def.get(col, "").lower()
        if col_type == "categorical":
            continue

        clean_feat = series.values[clean_mask]
        clean_err = abs_errors[clean_mask]

        if np.std(clean_feat) < 1e-12 or np.std(clean_err) < 1e-12:
            continue

        rho, pval = stats.spearmanr(clean_feat, clean_err)
        if pd.notna(rho) and abs(rho) >= 0.20 and pd.notna(pval) and pval < 0.05:
            associations.append({
                "feature": str(col),
                "spearman_rho": round(float(rho), 4),
                "p_value": round(float(pval), 4),
                "direction": "positive" if rho > 0 else "negative",
                "interpretation": (
                    f"Higher {col} values are associated with "
                    f"{'larger' if rho > 0 else 'smaller'} prediction errors (Spearman rho={rho:.2f}, p={pval:.4f})."
                ),
                "disclaimer": "Association does not imply causation.",
            })

    return associations


# ---------------------------------------------------------------------------
# Master Error Analysis Orchestrator
# ---------------------------------------------------------------------------

def analyze_errors(
    champion: Optional[ChampionResult] = None,
    validation: Optional[ValidationReport] = None,
    df: Optional[Any] = None,
    discovery: Optional[DiscoveryContract] = None,
    target_col: Optional[str] = None,
    y_true: Optional[Any] = None,
    y_pred: Optional[Any] = None,
    X: Optional[Any] = None,
    task_type: Optional[MLTaskType] = None,
    top_k: int = 10,
    min_subgroup_size: Optional[int] = None,
) -> ErrorAnalysisReport:
    """
    Perform rigorous error and residual diagnostics on champion model out-of-fold validation predictions.
    Outputs structured diagnostic evidence consumed by Person 3.
    """
    # 1. Resolve predictions and ground truth
    resolved_y_true: Optional[np.ndarray] = None
    resolved_y_pred: Optional[np.ndarray] = None
    resolved_indices: Optional[List[int]] = None
    resolved_X: Optional[pd.DataFrame] = None

    if y_true is not None and y_pred is not None:
        resolved_y_true = np.asarray(y_true)
        resolved_y_pred = np.asarray(y_pred)
        if X is not None:
            resolved_X = pd.DataFrame(X)
    elif champion and champion.status == PipelineStatus.SUCCESS:
        evidence = champion.empirical_evidence or {}
        oof_preds = evidence.get("oof_predictions")
        oof_idxs = evidence.get("oof_indices")

        if oof_preds is not None and df is not None and target_col is not None:
            df_in = pd.DataFrame(df)
            if target_col in df_in.columns:
                valid_mask = df_in[target_col].notna()
                clean_df = df_in[valid_mask].copy()

                # Align with OoF indices
                if oof_idxs is not None and len(oof_idxs) > 0:
                    matched_idxs = [i for i in oof_idxs if i < len(clean_df) and oof_preds[i] is not None]
                    if matched_idxs:
                        resolved_y_true = clean_df[target_col].iloc[matched_idxs].values
                        resolved_y_pred = np.array([oof_preds[i] for i in matched_idxs])
                        resolved_indices = matched_idxs
                        resolved_X = clean_df.drop(columns=[target_col]).iloc[matched_idxs]

    if resolved_y_true is None or resolved_y_pred is None or len(resolved_y_true) == 0:
        return ErrorAnalysisReport(
            status=PipelineStatus.UNAVAILABLE,
            reason="Validation out-of-fold predictions not available for error analysis",
        )

    # 2. Resolve task type
    resolved_task: MLTaskType = MLTaskType.CLASSIFICATION
    if task_type is not None:
        resolved_task = task_type
    elif champion and champion.primary_metric:
        m = champion.primary_metric.lower()
        if any(term in m for term in ["rmse", "mae", "mse", "r2"]):
            resolved_task = MLTaskType.REGRESSION
        else:
            resolved_task = MLTaskType.CLASSIFICATION
    elif discovery and discovery.router:
        if discovery.router.regression:
            resolved_task = MLTaskType.REGRESSION
        else:
            resolved_task = MLTaskType.CLASSIFICATION
    else:
        # Infer from data types
        if pd.api.types.is_numeric_dtype(resolved_y_true) and len(np.unique(resolved_y_true)) > 10:
            resolved_task = MLTaskType.REGRESSION

    # 3. Perform Task-Specific Error Decomposition
    class_diag: Dict[str, Any] = {"status": PipelineStatus.NOT_APPLICABLE.value}
    reg_diag: Dict[str, Any] = {"status": PipelineStatus.NOT_APPLICABLE.value}
    summary_meta: Dict[str, Any] = {}
    abs_errors_for_assoc: Optional[np.ndarray] = None

    if resolved_task == MLTaskType.CLASSIFICATION:
        class_diag = compute_classification_error_analysis(
            resolved_y_true,
            resolved_y_pred,
            indices=resolved_indices,
            top_k=top_k,
        )
        summary_meta = class_diag["summary"]
        abs_errors_for_assoc = (resolved_y_true != resolved_y_pred).astype(float)
    else:
        reg_diag = compute_regression_error_analysis(
            resolved_y_true,
            resolved_y_pred,
            indices=resolved_indices,
            top_k=top_k,
        )
        summary_meta = {
            "total_samples": len(resolved_y_true),
            "mae": reg_diag["residual_summary"]["mae"],
            "rmse": reg_diag["residual_summary"]["rmse"],
        }
        abs_errors_for_assoc = np.abs(resolved_y_true.astype(float) - resolved_y_pred.astype(float))

    # 4. Subgroup Analysis (if features X available)
    subgroup_reports: List[Dict[str, Any]] = []
    if resolved_X is not None and not resolved_X.empty:
        subgroup_reports = compute_subgroup_slice_analysis(
            resolved_X,
            resolved_y_true,
            resolved_y_pred,
            task_type=resolved_task,
            discovery=discovery,
            min_subgroup_size=min_subgroup_size,
        )

    # 5. Numeric Feature vs Error Associations
    feature_associations: List[Dict[str, Any]] = []
    if resolved_X is not None and not resolved_X.empty and abs_errors_for_assoc is not None:
        feature_associations = compute_numeric_error_associations(
            resolved_X,
            abs_errors_for_assoc,
            discovery=discovery,
        )

    model_meta = {
        "name": champion.model_name if champion else "ChampionEstimator",
        "task": resolved_task.value,
        "primary_metric": champion.primary_metric if champion else None,
        "score": champion.score if champion else None,
    }

    # Backward compatibility mappings
    legacy_residuals = reg_diag.get("residual_summary", {})
    legacy_worst = class_diag.get("worst_predictions") or reg_diag.get("worst_predictions") or []
    legacy_cm = class_diag.get("confusion_matrix")
    legacy_segments = [s for s in subgroup_reports if s.get("status") == "WEAK_SLICE"]

    sanitized_output = sanitize_for_json({
        "status": PipelineStatus.SUCCESS.value,
        "model": model_meta,
        "summary": summary_meta,
        "classification": class_diag,
        "regression": reg_diag,
        "subgroup_analysis": subgroup_reports,
        "feature_error_associations": feature_associations,
        "robustness": reg_diag.get("robustness", {}),
        "residual_summary": legacy_residuals,
        "worst_predictions": legacy_worst,
        "confusion_matrix": legacy_cm,
        "high_loss_segments": legacy_segments,
    })

    return ErrorAnalysisReport.model_validate(sanitized_output)
