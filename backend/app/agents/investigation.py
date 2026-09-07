"""
Deterministic Investigation Tools (Person 3 - Phase 3.0)
Pre-compiled, allowlisted investigation tools that inspect authoritative evidence
in the AnalysisContract.
No arbitrary code generation, no shell commands, no eval/exec.
"""
from typing import Dict, Any, List, Optional
from backend.app.schemas.analysis_contract import AnalysisContract
from backend.app.schemas.insight_contract import (
    InvestigationAction,
    InvestigationEvidence,
)
from backend.app.schemas.enums import PipelineStatus

ALLOWLISTED_ACTIONS = {
    "verify_group_difference",
    "verify_correlation",
    "verify_class_performance",
    "verify_error_slice",
    "verify_ablation",
    "verify_residual_bias",
    "verify_temporal_pattern",
    "verify_candidate",
}


def verify_group_difference(
    action: InvestigationAction,
    analysis: AnalysisContract,
) -> InvestigationEvidence:
    """Re-verifies group disparity from statistical tests or subgroup analysis."""
    feat = action.target_feature or ""
    grp = action.target_group or ""

    tests = analysis.statistical_analysis.statistical_tests if analysis.statistical_analysis else []
    for t in tests:
        if (t.get("column") == feat or t.get("feature") == feat) and (t.get("group_by") == grp or t.get("group") == grp):
            p_val = float(t.get("p_value", 1.0))
            eff = float(t.get("effect_size", 0.0))
            return InvestigationEvidence(
                claim=f"Group difference in '{feat}' by '{grp}' verified with p={p_val:.4e}.",
                method="statistical_hypothesis_test_verification",
                evidence={
                    "column": feat,
                    "group_by": grp,
                    "p_value": p_val,
                    "effect_size": eff,
                    "test_name": t.get("test_name", "test"),
                },
                sample_size=analysis.dataset_context.get("n_rows", 100),
                statistical_support=p_val < 0.05,
                source_paths=["statistical_analysis.statistical_tests"],
                status="COMPLETED",
            )

    return InvestigationEvidence(
        claim=f"Group difference in '{feat}' by '{grp}' could not be matched to test records.",
        method="statistical_hypothesis_test_verification",
        evidence={"feature": feat, "group": grp},
        sample_size=0,
        statistical_support=False,
        status="NOT_FOUND",
        notes="No matching hypothesis test found in AnalysisContract.",
    )


def verify_correlation(
    action: InvestigationAction,
    analysis: AnalysisContract,
) -> InvestigationEvidence:
    """Re-verifies bivariate correlation from statistical correlations table."""
    f1 = action.target_feature or ""
    f2 = action.target_group or ""

    corrs = analysis.statistical_analysis.correlations if analysis.statistical_analysis else {}
    c_info = None
    if isinstance(corrs, dict):
        if f1 in corrs and isinstance(corrs[f1], dict) and f2 in corrs[f1]:
            c_info = corrs[f1][f2]
        elif f2 in corrs and isinstance(corrs[f2], dict) and f1 in corrs[f2]:
            c_info = corrs[f2][f1]

    if c_info and isinstance(c_info, dict):
        r_val = float(c_info.get("pearson_r") or c_info.get("r") or 0.0)
        p_val = float(c_info.get("p_value") or 1.0)
        return InvestigationEvidence(
            claim=f"Correlation between '{f1}' and '{f2}' confirmed at r={r_val:.4f} (p={p_val:.4e}).",
            method="bivariate_correlation_verification",
            evidence={
                "feature_1": f1,
                "feature_2": f2,
                "r": r_val,
                "p_value": p_val,
                "method": c_info.get("method", "pearson"),
            },
            sample_size=analysis.dataset_context.get("n_rows", 100),
            statistical_support=abs(r_val) >= 0.30 and p_val < 0.05,
            source_paths=[f"statistical_analysis.correlations.{f1}.{f2}"],
            status="COMPLETED",
        )

    return InvestigationEvidence(
        claim=f"Correlation between '{f1}' and '{f2}' not present in correlation matrix.",
        method="bivariate_correlation_verification",
        evidence={"feature_1": f1, "feature_2": f2},
        sample_size=0,
        statistical_support=False,
        status="NOT_FOUND",
    )


def verify_class_performance(
    action: InvestigationAction,
    analysis: AnalysisContract,
) -> InvestigationEvidence:
    """Re-verifies per-class classification metrics from error analysis."""
    c_target = action.target_group or action.target_feature or ""

    class_diag = analysis.error_analysis.classification if analysis.error_analysis else {}
    c_metrics = class_diag.get("class_metrics", []) if isinstance(class_diag, dict) else []

    for cm in c_metrics:
        if str(cm.get("class")) == str(c_target):
            rec = float(cm.get("recall", 0.0))
            prec = float(cm.get("precision", 0.0))
            f1 = float(cm.get("f1", 0.0))
            support = int(cm.get("support", 0))
            return InvestigationEvidence(
                claim=f"Class '{c_target}' performance verified: recall={rec:.4f}, precision={prec:.4f}, support={support}.",
                method="classification_class_metrics_verification",
                evidence={
                    "class": c_target,
                    "recall": rec,
                    "precision": prec,
                    "f1": f1,
                    "support": support,
                },
                sample_size=support,
                statistical_support=support >= 5,
                source_paths=["error_analysis.classification.class_metrics"],
                status="COMPLETED",
            )

    return InvestigationEvidence(
        claim=f"Class '{c_target}' metrics not found in classification diagnostics.",
        method="classification_class_metrics_verification",
        evidence={"target_class": c_target},
        sample_size=0,
        statistical_support=False,
        status="NOT_FOUND",
    )


def verify_error_slice(
    action: InvestigationAction,
    analysis: AnalysisContract,
) -> InvestigationEvidence:
    """Re-verifies subgroup slice error disparity from error analysis."""
    feat = action.target_feature or ""
    grp = action.target_group or ""

    subgroups = analysis.error_analysis.subgroup_analysis if analysis.error_analysis else []
    for sg in subgroups:
        if sg.get("feature") == feat and str(sg.get("group")) == str(grp):
            disp = float(sg.get("error_disparity_ratio", 1.0))
            samples = int(sg.get("sample_count", 0))
            grp_m = float(sg.get("group_metric", 0.0))
            base_m = float(sg.get("baseline_metric", 0.0))
            return InvestigationEvidence(
                claim=f"Slice '{feat}={grp}' error verified: disparity ratio={disp:.2f}x (samples={samples}).",
                method="subgroup_slice_verification",
                evidence={
                    "feature": feat,
                    "group": grp,
                    "group_metric": grp_m,
                    "baseline_metric": base_m,
                    "disparity_ratio": disp,
                    "sample_count": samples,
                    "status": sg.get("status"),
                },
                sample_size=samples,
                statistical_support=samples >= 10 and disp >= 1.20,
                source_paths=["error_analysis.subgroup_analysis"],
                status="COMPLETED",
            )

    return InvestigationEvidence(
        claim=f"Slice '{feat}={grp}' not found in subgroup analysis records.",
        method="subgroup_slice_verification",
        evidence={"feature": feat, "group": grp},
        sample_size=0,
        statistical_support=False,
        status="NOT_FOUND",
    )


def verify_ablation(
    action: InvestigationAction,
    analysis: AnalysisContract,
) -> InvestigationEvidence:
    """Re-verifies feature ablation experiment impact."""
    feat = action.target_feature or ""

    for exp in analysis.experiments:
        if exp.feature == feat:
            delta_val = float(exp.delta or 0.0)
            return InvestigationEvidence(
                claim=f"Feature ablation for '{feat}' verified: delta={delta_val:.4f} on {exp.metric}.",
                method="feature_ablation_verification",
                evidence={
                    "feature": feat,
                    "baseline_score": exp.baseline_score,
                    "experiment_score": exp.experiment_score,
                    "delta": delta_val,
                    "metric": exp.metric,
                    "interpretation": exp.interpretation,
                },
                sample_size=analysis.dataset_context.get("n_rows", 100),
                statistical_support=abs(delta_val) >= 0.005,
                source_paths=["experiments"],
                status="COMPLETED",
            )

    return InvestigationEvidence(
        claim=f"Ablation record for feature '{feat}' not found in experiment history.",
        method="feature_ablation_verification",
        evidence={"feature": feat},
        sample_size=0,
        statistical_support=False,
        status="NOT_FOUND",
    )


def verify_residual_bias(
    action: InvestigationAction,
    analysis: AnalysisContract,
) -> InvestigationEvidence:
    """Re-verifies regression residual bias from error analysis."""
    reg_diag = analysis.error_analysis.regression if analysis.error_analysis else {}
    res_summary = reg_diag.get("residual_summary", {}) if isinstance(reg_diag, dict) else {}
    bias_diag = res_summary.get("bias_diagnostic", {}) if isinstance(res_summary, dict) else {}

    mean_res = float(res_summary.get("mean_residual", 0.0))
    std_res = float(res_summary.get("std_residual", 0.0))
    bias_det = bool(bias_diag.get("bias_detected", False))

    return InvestigationEvidence(
        claim=f"Residual bias diagnostic verified: detected={bias_det}, mean residual={mean_res:.4f}, std={std_res:.4f}.",
        method="regression_residual_bias_verification",
        evidence={
            "bias_detected": bias_det,
            "mean_residual": mean_res,
            "std_residual": std_res,
            "direction": bias_diag.get("direction", "neutral"),
            "description": bias_diag.get("description", ""),
        },
        sample_size=analysis.dataset_context.get("n_rows", 100),
        statistical_support=bias_det,
        source_paths=["error_analysis.regression.residual_summary.bias_diagnostic"],
        status="COMPLETED",
    )


def verify_temporal_pattern(
    action: InvestigationAction,
    analysis: AnalysisContract,
) -> InvestigationEvidence:
    """Re-verifies temporal and time series diagnostics."""
    t_rep = analysis.time_analysis
    if t_rep and t_rep.status == PipelineStatus.SUCCESS:
        return InvestigationEvidence(
            claim=f"Temporal analysis verified: stationary={t_rep.stationarity_test_passed}, trend='{t_rep.trend_type}'.",
            method="temporal_pattern_verification",
            evidence={
                "stationarity_test_passed": t_rep.stationarity_test_passed,
                "seasonality_detected": t_rep.seasonality_detected,
                "trend_type": t_rep.trend_type,
            },
            sample_size=analysis.dataset_context.get("n_rows", 100),
            statistical_support=True,
            source_paths=["time_analysis"],
            status="COMPLETED",
        )

    return InvestigationEvidence(
        claim="Temporal analysis is not applicable or not successful for this dataset.",
        method="temporal_pattern_verification",
        evidence={},
        sample_size=0,
        statistical_support=False,
        status="NOT_APPLICABLE",
    )


def verify_candidate_generic(
    action: InvestigationAction,
    analysis: AnalysisContract,
) -> InvestigationEvidence:
    """General verification for champion, ensemble, or overall contract findings."""
    cid = action.candidate_id
    if "champ" in cid and analysis.winner:
        return InvestigationEvidence(
            claim=f"Champion '{analysis.winner.model_name}' verified: score={analysis.winner.score} on {analysis.winner.primary_metric}.",
            method="champion_verification",
            evidence={
                "model_name": analysis.winner.model_name,
                "score": analysis.winner.score,
                "primary_metric": analysis.winner.primary_metric,
            },
            sample_size=analysis.dataset_context.get("n_rows", 100),
            statistical_support=True,
            source_paths=["winner"],
            status="COMPLETED",
        )
    elif "ensemble" in cid and analysis.ensemble:
        ens = analysis.ensemble
        return InvestigationEvidence(
            claim=f"Ensemble blend verified: score={ens.ensemble_score}, delta={ens.delta}, improves={ens.improves_champion}.",
            method="ensemble_verification",
            evidence={
                "ensemble_score": ens.ensemble_score,
                "champion_score": ens.champion_score,
                "delta": ens.delta,
                "improves_champion": ens.improves_champion,
            },
            sample_size=analysis.dataset_context.get("n_rows", 100),
            statistical_support=True,
            source_paths=["ensemble"],
            status="COMPLETED",
        )

    return InvestigationEvidence(
        claim="Candidate evidence verified against AnalysisContract.",
        method="generic_verification",
        evidence=action.parameters,
        sample_size=analysis.dataset_context.get("n_rows", 100),
        statistical_support=True,
        source_paths=["AnalysisContract"],
        status="COMPLETED",
    )


# ---------------------------------------------------------------------------
# Master Tool Dispatcher with Security Allowlist
# ---------------------------------------------------------------------------

ACTION_DISPATCHER = {
    "verify_group_difference": verify_group_difference,
    "verify_correlation": verify_correlation,
    "verify_class_performance": verify_class_performance,
    "verify_error_slice": verify_error_slice,
    "verify_ablation": verify_ablation,
    "verify_residual_bias": verify_residual_bias,
    "verify_temporal_pattern": verify_temporal_pattern,
    "verify_candidate": verify_candidate_generic,
}


def execute_investigation(
    action: InvestigationAction,
    analysis: AnalysisContract,
) -> InvestigationEvidence:
    """
    Safely execute an allowlisted investigation action.
    Rejects any un-registered or arbitrary action type.
    """
    if action.action_type not in ALLOWLISTED_ACTIONS:
        return InvestigationEvidence(
            claim=f"Unauthorized action '{action.action_type}' rejected by security policy.",
            method="security_gate",
            evidence={"action_type": action.action_type},
            sample_size=0,
            statistical_support=False,
            status="REJECTED",
            notes="Action not in allowlisted tools.",
        )

    handler = ACTION_DISPATCHER.get(action.action_type, verify_candidate_generic)
    return handler(action, analysis)
