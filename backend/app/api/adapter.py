"""
Presentation Adapter & Pipeline Bridge (Person 4 - Phase 4.0)
Constructs the DiscoveryContract from raw uploaded datasets and transforms
the authoritative DiscoveryContract, AnalysisContract, and InsightContract
into a validated DashboardSpec.
Presentation only; NEVER independently recalculates or modifies analytical truth.
"""
from typing import Dict, Any, List, Optional
import datetime
import math
import numpy as np
import pandas as pd

from backend.app.schemas.discovery_contract import DiscoveryContract, RouterDecision
from backend.app.schemas.analysis_contract import AnalysisContract
from backend.app.schemas.insight_contract import InsightContract
from backend.app.schemas.enums import PipelineStatus, MLTaskType
from backend.app.schemas.dashboard_spec import (
    DashboardSpec,
    KPIItem,
    ChartSpec,
    ChartSeries,
    InsightCardSpec,
    ModelChampionshipSpec,
    ModelLeaderboardRow,
    ErrorAnalysisSpec,
    DataQualitySpec,
    ExecutiveSummarySpec,
    ExecutiveReportSpec,
)


def build_discovery_contract_for_df(
    df: pd.DataFrame,
    filename: str = "dataset.csv",
) -> DiscoveryContract:
    """
    Constructs an authoritative DiscoveryContract for an uploaded DataFrame.
    Automatically fingerprints the data, detects quality and leakage risks,
    and infers candidate analytical tasks.
    """
    n_rows, n_cols = df.shape
    mem_mb = round(float(df.memory_usage(deep=True).sum() / (1024 * 1024)), 4)

    # 1. Infer Schema Data Types
    schema_dict: Dict[str, str] = {}
    leakage_ids: List[str] = []
    constant_cols: List[str] = []
    missing_dict: Dict[str, int] = {}
    has_datetime = False

    for col in df.columns:
        col_str = str(col)
        series = df[col]
        missing_count = int(series.isna().sum())
        if missing_count > 0:
            missing_dict[col_str] = missing_count

        n_unique = int(series.nunique(dropna=True))
        if n_unique <= 1:
            constant_cols.append(col_str)

        # Detect identifiers (high cardinality, string or named like id)
        col_lower = col_str.lower()
        if (n_unique == n_rows and n_rows > 10) or any(id_token in col_lower for id_token in ["_id", "id_", "uuid", "guid", "identifier"]):
            if not pd.api.types.is_numeric_dtype(series) or "id" in col_lower:
                leakage_ids.append(col_str)

        if pd.api.types.is_datetime64_any_dtype(series):
            schema_dict[col_str] = "datetime"
            has_datetime = True
        elif pd.api.types.is_numeric_dtype(series):
            schema_dict[col_str] = "numeric"
        elif pd.api.types.is_bool_dtype(series):
            schema_dict[col_str] = "categorical"
        else:
            # Check if strings can be parsed as dates
            sample_non_null = series.dropna().head(10)
            parsed_as_date = False
            if len(sample_non_null) > 0 and any(d_kw in col_lower for d_kw in ["date", "time", "year", "timestamp"]):
                try:
                    pd.to_datetime(sample_non_null)
                    parsed_as_date = True
                except Exception:
                    parsed_as_date = False
            if parsed_as_date:
                schema_dict[col_str] = "datetime"
                has_datetime = True
            else:
                schema_dict[col_str] = "categorical"

    # 2. Duplicate rows
    duplicate_count = int(df.duplicated().sum())

    # 3. Target candidate heuristic
    target_candidates: List[str] = []
    preferred_targets = ["target", "label", "class", "churn", "status", "price", "revenue", "salary", "outcome", "y"]
    col_names = [str(c) for c in df.columns]

    for pref in preferred_targets:
        matches = [c for c in col_names if c.lower() == pref or c.lower().endswith(f"_{pref}")]
        if matches:
            target_candidates.extend(matches)
            break

    # If no preferred name match, select the last non-identifier column if more than 1 column
    if not target_candidates and len(col_names) > 1:
        eligible = [c for c in col_names if c not in leakage_ids]
        if eligible:
            target_candidates.append(eligible[-1])

    # 4. Router Decision
    classification = False
    regression = False
    clustering = False
    anomaly_detection = True

    if target_candidates:
        target_col = target_candidates[0]
        target_series = df[target_col].dropna()
        n_unique_target = target_series.nunique()
        col_type = schema_dict.get(target_col, "categorical")

        if col_type == "categorical" or n_unique_target <= 15 or pd.api.types.is_bool_dtype(target_series):
            classification = True
        else:
            regression = True
    else:
        clustering = True

    router = RouterDecision(
        classification=classification,
        regression=regression,
        clustering=clustering,
        anomaly_detection=anomaly_detection,
        time_analysis=has_datetime,
    )

    quality_report = {
        "missing_values": missing_dict,
        "duplicate_rows": duplicate_count,
        "constant_columns": constant_cols,
        "total_missing_cells": int(sum(missing_dict.values())),
        "completeness_pct": round(float((1.0 - (sum(missing_dict.values()) / (n_rows * n_cols if n_rows * n_cols > 0 else 1))) * 100.0), 2),
    }

    fingerprint = {
        "row_count": n_rows,
        "column_count": n_cols,
        "memory_mb": mem_mb,
        "filename": filename,
    }

    leakage_report = {
        "high_cardinality_identifiers": leakage_ids,
        "constant_columns": constant_cols,
    }

    return DiscoveryContract(
        schema_version="1.0",
        schema=schema_dict,
        target_candidates=target_candidates,
        router=router,
        quality_report=quality_report,
        fingerprint=fingerprint,
        leakage_report=leakage_report,
    )


def build_dashboard_spec(
    discovery: DiscoveryContract,
    analysis: AnalysisContract,
    insights: InsightContract,
    dataset_name: str = "dataset.csv",
) -> DashboardSpec:
    """
    Transforms authoritative Discovery, Analysis, and Insight contracts into a DashboardSpec.
    Never invents or recomputes numerical truth.
    """
    n_rows = discovery.fingerprint.get("row_count") or (analysis.dataset_context.get("n_rows") if analysis.dataset_context else 0)
    n_cols = discovery.fingerprint.get("column_count") or (analysis.dataset_context.get("n_cols") if analysis.dataset_context else 0)
    task_name = analysis.task.value.capitalize() if analysis.task else "Analytical Task"
    target_col = analysis.target_column

    # 1. KPIs
    champ_score_str = "N/A"
    champ_score_sub = "Unsupervised analysis"
    if analysis.winner and analysis.winner.status == PipelineStatus.SUCCESS and analysis.winner.score is not None:
        champ_score_str = f"{analysis.winner.score:.4f}"
        champ_score_sub = f"{analysis.winner.model_name} ({analysis.winner.primary_metric})"

    completeness_pct = discovery.quality_report.get("completeness_pct", 100.0)

    kpis = [
        KPIItem(id="kpi_rows", label="Observations", value=f"{n_rows:,}", subtext="Rows ingested", icon="database"),
        KPIItem(id="kpi_cols", label="Feature Space", value=f"{n_cols}", subtext="Columns mapped", icon="columns"),
        KPIItem(id="kpi_task", label="Detected Task", value=task_name, subtext=f"Target: {target_col or 'None'}", icon="crosshair"),
        KPIItem(id="kpi_score", label="Champion Metric", value=champ_score_str, subtext=champ_score_sub, icon="trophy"),
        KPIItem(id="kpi_insights", label="Verified Insights", value=str(len(insights.insights)), subtext=f"{len(insights.rejected_insights)} rejected by critic", icon="sparkles"),
        KPIItem(id="kpi_quality", label="Data Quality", value=f"{completeness_pct}%", subtext=f"{discovery.quality_report.get('duplicate_rows', 0)} duplicates", icon="shield-check"),
    ]

    # 2. Executive Summary
    exec_summary_dict = insights.executive_summary or {}
    exec_spec = ExecutiveSummarySpec(
        title=exec_summary_dict.get("title", f"AIDA Intelligence Analysis: {dataset_name}"),
        overview=exec_summary_dict.get("overview", "AIDA completed end-to-end autonomous analysis."),
        key_takeaways=exec_summary_dict.get("key_takeaways", []),
        recommendations=exec_summary_dict.get("recommendations", []),
        warnings=discovery.quality_report.get("constant_columns", []),
        verified_insight_count=len(insights.insights),
        high_confidence_count=sum(1 for ins in insights.insights if str(ins.confidence) == "HIGH"),
    )

    # 3. Insights Cards with Provenance Chain
    insight_cards: List[InsightCardSpec] = []
    for ins in insights.insights:
        critic_issues = ins.critic.issues if ins.critic else []
        critic_summary = ins.critic.critique_summary if ins.critic else None

        # Build provenance steps
        provenance_chain = [
            {"step": "1. Evidence Source", "detail": f"Path: {', '.join(ins.source_paths or ['AnalysisContract'])}"},
            {"step": "2. Analytical Method", "detail": f"Method: {ins.method}"},
            {"step": "3. Adversarial Critic", "detail": f"Status: {ins.critic.status.value if hasattr(ins.critic.status, 'value') else ins.critic.status} ({len(critic_issues)} issues)"},
            {"step": "4. Numerical Verifier", "detail": f"Status: {ins.verification.status.value if hasattr(ins.verification.status, 'value') else ins.verification.status}"},
            {"step": "5. Ranking", "detail": f"Rank #{ins.rank or 1} (Score: {ins.rank_score or 0.0:.4f})"},
        ]

        card = InsightCardSpec(
            insight_id=ins.insight_id,
            rank=ins.rank,
            claim=ins.claim,
            importance=ins.importance,
            confidence=ins.confidence.value if hasattr(ins.confidence, "value") else str(ins.confidence),
            verification_status=ins.verification.status.value if hasattr(ins.verification.status, "value") else str(ins.verification.status),
            method=ins.method,
            evidence_highlights=ins.evidence,
            source_paths=ins.source_paths,
            critic_summary=critic_summary,
            critic_issues=critic_issues,
            provenance_chain=provenance_chain,
        )
        insight_cards.append(card)

    # 4. Model Championship Spec
    leaderboard_rows: List[ModelLeaderboardRow] = []
    champ_name = analysis.winner.model_name if analysis.winner else None
    champ_score = analysis.winner.score if analysis.winner else None
    rationale = analysis.winner.selection_rule if analysis.winner else None

    for m in analysis.models:
        if m.mean_score is not None:
            leaderboard_rows.append(ModelLeaderboardRow(
                rank=m.rank or 999,
                model_name=m.model_name,
                primary_metric_name=m.primary_metric or analysis.validation.primary_metric or "metric",
                primary_metric_score=round(float(m.mean_score), 4),
                secondary_metrics={k: round(float(v), 4) for k, v in (m.metrics or {}).items() if k != m.primary_metric},
                fold_scores=[round(float(s), 4) for s in (m.fold_scores or [])],
                status=m.status.value if hasattr(m.status, "value") else str(m.status),
                is_champion=(m.model_name == champ_name),
            ))

    leaderboard_rows.sort(key=lambda r: r.rank)

    ens_dict = {}
    if analysis.ensemble and analysis.ensemble.status == PipelineStatus.SUCCESS:
        ens_dict = {
            "ensemble_score": analysis.ensemble.ensemble_score,
            "champion_score": analysis.ensemble.champion_score,
            "delta": analysis.ensemble.delta,
            "improves_champion": analysis.ensemble.improves_champion,
            "models": analysis.ensemble.models,
            "reason": analysis.ensemble.reason,
        }

    model_champ_spec = ModelChampionshipSpec(
        task_type=analysis.task.value if analysis.task else "not_applicable",
        target_column=target_col,
        validation_strategy=analysis.validation.strategy or "CrossValidation",
        folds=analysis.validation.folds or 5,
        primary_metric=analysis.validation.primary_metric or "f1_macro",
        champion_model_name=champ_name,
        champion_score=champ_score,
        selection_rationale=rationale,
        leaderboard=leaderboard_rows,
        ensemble_result=ens_dict,
        status="SUCCESS" if analysis.winner and analysis.winner.status == PipelineStatus.SUCCESS else "NOT_APPLICABLE",
        reason="Supervised championship completed" if analysis.winner and analysis.winner.status == PipelineStatus.SUCCESS else "No supervised champion model available",
    )

    # 5. Error Analysis Spec
    err_rep = analysis.error_analysis
    class_diag = err_rep.classification if err_rep and isinstance(err_rep.classification, dict) else {}
    reg_diag = err_rep.regression if err_rep and isinstance(err_rep.regression, dict) else {}

    error_analysis_spec = ErrorAnalysisSpec(
        task_type=analysis.task.value if analysis.task else "not_applicable",
        status=err_rep.status.value if err_rep and hasattr(err_rep.status, "value") else "NOT_APPLICABLE",
        reason=err_rep.reason if err_rep else "Error analysis not executed",
        confusion_matrix=class_diag.get("confusion_matrix"),
        class_labels=class_diag.get("class_labels", []),
        class_metrics=class_diag.get("class_metrics", []),
        weak_classes=class_diag.get("weak_classes", []),
        residual_summary=reg_diag.get("residual_summary", {}),
        bias_diagnostic=reg_diag.get("residual_summary", {}).get("bias_diagnostic", {}) if isinstance(reg_diag.get("residual_summary"), dict) else {},
        heteroscedasticity=reg_diag.get("heteroscedasticity", {}),
        subgroup_slices=err_rep.subgroup_analysis if err_rep else [],
        worst_predictions=err_rep.worst_predictions if err_rep else [],
    )

    # 6. Data Quality Spec
    leakage_risks = [
        {"feature": id_col, "risk_type": "high_cardinality_identifier", "severity": "HIGH", "action": "Excluded from feature inputs"}
        for id_col in discovery.leakage_report.get("high_cardinality_identifiers", [])
    ]
    for const_col in discovery.quality_report.get("constant_columns", []):
        leakage_risks.append({"feature": const_col, "risk_type": "zero_variance_constant", "severity": "MEDIUM", "action": "Excluded due to lack of informative variance"})

    data_quality_spec = DataQualitySpec(
        row_count=n_rows,
        column_count=n_cols,
        memory_mb=discovery.fingerprint.get("memory_mb", 0.0),
        column_types=discovery.schema_definition,
        missing_value_summary=discovery.quality_report.get("missing_values", {}),
        duplicate_rows=discovery.quality_report.get("duplicate_rows", 0),
        constant_columns=discovery.quality_report.get("constant_columns", []),
        leakage_risks=leakage_risks,
        routing_decisions=discovery.router.model_dump() if discovery.router else {},
        target_candidates=discovery.target_candidates,
    )

    # 7. Automatic Charts
    charts: List[ChartSpec] = []

    # Chart A: Model Leaderboard Comparison
    if leaderboard_rows:
        charts.append(ChartSpec(
            chart_id="chart_model_benchmarks",
            title="Model Championship Leaderboard",
            chart_type="bar",
            description=f"Empirical cross-validation performance ({analysis.validation.primary_metric}) across tested architectures.",
            x_axis_label="Candidate Model",
            y_axis_label=analysis.validation.primary_metric or "Score",
            categories=[r.model_name for r in leaderboard_rows],
            series=[ChartSeries(
                name=analysis.validation.primary_metric or "Score",
                data=[r.primary_metric_score for r in leaderboard_rows],
                color="#6366f1",
            )],
        ))

    # Chart B: Classification Confusion Matrix
    if class_diag.get("confusion_matrix"):
        cm_data = class_diag["confusion_matrix"]
        labels = class_diag.get("class_labels") or [f"Class {i}" for i in range(len(cm_data))]
        charts.append(ChartSpec(
            chart_id="chart_confusion_matrix",
            title="Champion Confusion Matrix",
            chart_type="confusion_matrix",
            description="Out-of-fold predicted class vs actual class distributions.",
            categories=labels,
            series=[ChartSeries(name="Counts", data=cm_data)],
            metadata={"labels": labels},
        ))

    # Chart C: Correlation Heatmap / Key Associations
    corrs = analysis.statistical_analysis.correlations if analysis.statistical_analysis else {}
    if corrs and isinstance(corrs, dict):
        corr_pairs = []
        for f1, f1_corrs in list(corrs.items())[:6]:
            if isinstance(f1_corrs, dict):
                for f2, c_val in list(f1_corrs.items())[:6]:
                    if f1 != f2 and isinstance(c_val, dict):
                        r = float(c_val.get("r") or c_val.get("pearson_r") or 0.0)
                        corr_pairs.append({"f1": f1, "f2": f2, "r": round(r, 3)})
        if corr_pairs:
            charts.append(ChartSpec(
                chart_id="chart_correlations",
                title="Bivariate Correlation Overview",
                chart_type="bar",
                description="Strongest statistical correlations across feature pairs.",
                categories=[f"{p['f1']} ~ {p['f2']}" for p in corr_pairs[:8]],
                series=[ChartSeries(
                    name="Correlation (r)",
                    data=[p["r"] for p in corr_pairs[:8]],
                    color="#06b6d4",
                )],
            ))

    # Chart D: Subgroup Error Disparities
    if err_rep and err_rep.subgroup_analysis:
        slices = [s for s in err_rep.subgroup_analysis if s.get("status") == "WEAK_SLICE"][:6]
        if slices:
            charts.append(ChartSpec(
                chart_id="chart_weak_slices",
                title="Subgroup Error Disparities (Weak Slices)",
                chart_type="bar",
                description="Subgroup slices with highest error disparity ratio compared to baseline.",
                x_axis_label="Subgroup Slice",
                y_axis_label="Disparity Ratio (x)",
                categories=[f"{s.get('feature')}={s.get('group')}" for s in slices],
                series=[ChartSeries(
                    name="Disparity Ratio",
                    data=[round(float(s.get("error_disparity_ratio", 1.0)), 2) for s in slices],
                    color="#f43f5e",
                )],
            ))

    # 8. Executive Report Spec
    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
    report_spec = ExecutiveReportSpec(
        title=f"AIDA Autonomous Intelligence Report — {dataset_name}",
        generated_at=now_iso,
        dataset_name=dataset_name,
        dataset_overview={
            "observations": n_rows,
            "features": n_cols,
            "detected_task": task_name,
            "target_column": target_col,
            "status": "VALIDATED",
        },
        data_quality_summary=discovery.quality_report,
        analytical_routing=discovery.router.model_dump() if discovery.router else {},
        key_verified_insights=[{"rank": ins.rank, "claim": ins.claim, "confidence": ins.confidence} for ins in insight_cards],
        statistical_findings=[{"test": t.get("test_name"), "p_value": t.get("p_value")} for t in (analysis.statistical_analysis.statistical_tests if analysis.statistical_analysis else [])[:5]],
        model_championship_summary={
            "champion": champ_name,
            "score": champ_score,
            "metric": analysis.validation.primary_metric,
            "validation_strategy": analysis.validation.strategy,
            "models_benchmarked": len(analysis.models),
        },
        error_analysis_summary={
            "weak_slices_count": len([s for s in (err_rep.subgroup_analysis if err_rep else []) if s.get("status") == "WEAK_SLICE"]),
            "weak_classes_count": len(class_diag.get("weak_classes", [])),
        },
        controlled_experiments_summary={
            "experiments_run": len(analysis.experiments),
            "ensemble_improves": analysis.ensemble.improves_champion if analysis.ensemble else False,
        },
        warnings_and_risks=[w["feature"] + ": " + w["risk_type"] for w in leakage_risks],
        executive_recommendations=exec_spec.recommendations,
        provenance_evidence_summary=[{"insight": ins.claim, "source": ins.source_paths} for ins in insight_cards[:5]],
    )

    return DashboardSpec(
        dashboard_id=f"dash_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}",
        dataset_name=dataset_name,
        task_type=task_name,
        target_column=target_col,
        status=analysis.status,
        reason=analysis.reason,
        kpis=kpis,
        executive_summary=exec_spec,
        insights=insight_cards,
        rejected_insights=insights.rejected_insights,
        model_championship=model_champ_spec,
        error_analysis=error_analysis_spec,
        data_quality=data_quality_spec,
        charts=charts,
        report=report_spec,
    )
