"""
Deterministic Insight Discovery Engine (Person 3 - Phase 3.0)
Extracts candidate hypotheses strictly from authoritative numerical evidence
already recorded in the AnalysisContract.
Zero LLM hallucination; zero numerical invention.
"""
from typing import List, Dict, Any, Optional
import math

from backend.app.schemas.analysis_contract import AnalysisContract
from backend.app.schemas.insight_contract import CandidateHypothesis
from backend.app.schemas.enums import PipelineStatus, MLTaskType


def compute_deterministic_priority(
    base_priority: float,
    effect_magnitude: float = 0.0,
    sample_size: int = 100,
    is_significant: bool = True,
) -> float:
    """
    Deterministically score hypothesis priority based on empirical weight:
    - Base priority by finding category
    - Effect magnitude bonus (+0.0 to +0.25)
    - Sample size credibility bonus (+0.0 to +0.10)
    - Statistical significance (+0.10 if p < 0.05)
    Capped within [0.10, 0.99].
    """
    effect_bonus = min(0.25, abs(effect_magnitude) * 0.25)
    sample_bonus = min(0.10, math.log10(max(10, sample_size)) * 0.03)
    sig_bonus = 0.10 if is_significant else -0.10

    raw_priority = base_priority + effect_bonus + sample_bonus + sig_bonus
    return round(float(min(0.99, max(0.10, raw_priority))), 4)


def discover_candidate_hypotheses(analysis: AnalysisContract) -> List[CandidateHypothesis]:
    """
    Exhaustively scan the AnalysisContract to discover evidence-backed candidate hypotheses.
    Every hypothesis is anchored to an unambiguous source path and empirical numbers.
    """
    candidates: List[CandidateHypothesis] = []
    cand_counter = 1

    if not analysis:
        return []

    # -----------------------------------------------------------------------
    # 1. Model Championship & Baseline Performance
    # -----------------------------------------------------------------------
    if analysis.winner and analysis.winner.status == PipelineStatus.SUCCESS and analysis.winner.score is not None:
        champ = analysis.winner
        score_val = round(float(champ.score), 4)
        metric_name = champ.primary_metric or "metric"
        cid = f"cand_{cand_counter:03d}_champ_{champ.model_name.lower()}"
        cand_counter += 1

        prio = compute_deterministic_priority(
            base_priority=0.55,
            effect_magnitude=score_val if score_val <= 1.0 else 0.5,
            sample_size=analysis.dataset_context.get("n_rows", 100),
            is_significant=True,
        )

        candidates.append(CandidateHypothesis(
            candidate_id=cid,
            claim=f"Champion model '{champ.model_name}' established authoritative validation performance ({metric_name}={score_val}).",
            source_path="winner",
            evidence={
                "model_name": champ.model_name,
                "metric": metric_name,
                "score": score_val,
                "selection_rule": champ.selection_rule or "empirical_validation_score",
            },
            priority=prio,
            investigation_type="verify_candidate",
            feature=None,
            group=None,
        ))

    # -----------------------------------------------------------------------
    # 2. Classification Weak Classes (Error Analysis)
    # -----------------------------------------------------------------------
    class_diag = analysis.error_analysis.classification if analysis.error_analysis else {}
    weak_classes = class_diag.get("weak_classes", []) if isinstance(class_diag, dict) else []

    for idx, wc in enumerate(weak_classes):
        c_name = str(wc.get("class"))
        rec = float(wc.get("recall", 0.0))
        support = int(wc.get("support", 0))
        cid = f"cand_{cand_counter:03d}_weak_class_{c_name}"
        cand_counter += 1

        prio = compute_deterministic_priority(
            base_priority=0.75,
            effect_magnitude=1.0 - rec,
            sample_size=support,
            is_significant=True,
        )

        candidates.append(CandidateHypothesis(
            candidate_id=cid,
            claim=f"Class '{c_name}' demonstrates substantially degraded recall ({rec:.4f}) with {support} sample support.",
            source_path=f"error_analysis.classification.weak_classes[{idx}]",
            evidence={
                "class": c_name,
                "recall": round(rec, 4),
                "f1": round(float(wc.get("f1", 0.0)), 4),
                "support": support,
                "severity": wc.get("severity", "MEDIUM"),
            },
            priority=prio,
            investigation_type="verify_class_performance",
            feature=None,
            group=c_name,
        ))

    # -----------------------------------------------------------------------
    # 3. Subgroup Slice Disparities (Weak Slices)
    # -----------------------------------------------------------------------
    subgroups = analysis.error_analysis.subgroup_analysis if analysis.error_analysis else []
    for idx, sg in enumerate(subgroups):
        if sg.get("status") == "WEAK_SLICE":
            feat = sg.get("feature", "unknown")
            grp = str(sg.get("group", "unknown"))
            grp_m = float(sg.get("group_metric", 0.0))
            base_m = float(sg.get("baseline_metric", 0.0))
            disp = float(sg.get("error_disparity_ratio", 1.0))
            samples = int(sg.get("sample_count", 0))
            cid = f"cand_{cand_counter:03d}_slice_{feat}_{grp}"
            cand_counter += 1

            prio = compute_deterministic_priority(
                base_priority=0.80,
                effect_magnitude=min(1.0, (disp - 1.0) * 0.5),
                sample_size=samples,
                is_significant=True,
            )

            candidates.append(CandidateHypothesis(
                candidate_id=cid,
                claim=f"Subgroup '{feat}={grp}' exhibits elevated model error (disparity ratio {disp:.2f}x vs global baseline).",
                source_path=f"error_analysis.subgroup_analysis[{idx}]",
                evidence={
                    "feature": feat,
                    "group": grp,
                    "group_metric": round(grp_m, 4),
                    "baseline_metric": round(base_m, 4),
                    "disparity_ratio": round(disp, 4),
                    "sample_count": samples,
                    "metric_name": sg.get("metric_name", "error"),
                },
                priority=prio,
                investigation_type="verify_error_slice",
                feature=feat,
                group=grp,
            ))

    # -----------------------------------------------------------------------
    # 4. Feature Ablation Performance Drops (Controlled Experiments)
    # -----------------------------------------------------------------------
    for idx, exp in enumerate(analysis.experiments):
        if exp.status == PipelineStatus.SUCCESS and exp.delta is not None:
            # Check for material performance drop
            delta_val = float(exp.delta)
            metric_name = exp.metric or "metric"
            interp = exp.interpretation or ""
            if interp == "performance_decreased" or (analysis.task == MLTaskType.CLASSIFICATION and delta_val < -0.01) or (analysis.task == MLTaskType.REGRESSION and delta_val > 0.01):
                feat = exp.feature or f"feature_{idx}"
                cid = f"cand_{cand_counter:03d}_ablation_{feat}"
                cand_counter += 1

                prio = compute_deterministic_priority(
                    base_priority=0.70,
                    effect_magnitude=abs(delta_val) * 2.0,
                    sample_size=analysis.dataset_context.get("n_rows", 100),
                    is_significant=True,
                )

                candidates.append(CandidateHypothesis(
                    candidate_id=cid,
                    claim=f"Removing feature '{feat}' materially degraded model performance (delta={delta_val:.4f} {metric_name}).",
                    source_path=f"experiments[{idx}]",
                    evidence={
                        "feature": feat,
                        "baseline_score": round(float(exp.baseline_score or 0.0), 4),
                        "experiment_score": round(float(exp.experiment_score or 0.0), 4),
                        "delta": round(delta_val, 4),
                        "metric": metric_name,
                        "interpretation": interp,
                    },
                    priority=prio,
                    investigation_type="verify_ablation",
                    feature=feat,
                    group=None,
                ))

    # -----------------------------------------------------------------------
    # 5. Regression Residual Bias & Heteroscedasticity
    # -----------------------------------------------------------------------
    reg_diag = analysis.error_analysis.regression if analysis.error_analysis else {}
    if isinstance(reg_diag, dict):
        res_summary = reg_diag.get("residual_summary", {})
        bias_diag = res_summary.get("bias_diagnostic", {}) if isinstance(res_summary, dict) else {}
        if bias_diag.get("bias_detected"):
            direction = bias_diag.get("direction", "residual_bias")
            mean_res = float(res_summary.get("mean_residual", 0.0))
            cid = f"cand_{cand_counter:03d}_residual_bias"
            cand_counter += 1

            prio = compute_deterministic_priority(
                base_priority=0.72,
                effect_magnitude=abs(mean_res),
                sample_size=analysis.dataset_context.get("n_rows", 100),
                is_significant=True,
            )

            candidates.append(CandidateHypothesis(
                candidate_id=cid,
                claim=f"Regression champion exhibits {direction} (mean residual = {mean_res:.4f}).",
                source_path="error_analysis.regression.residual_summary.bias_diagnostic",
                evidence={
                    "direction": direction,
                    "mean_residual": round(mean_res, 4),
                    "std_residual": round(float(res_summary.get("std_residual", 0.0)), 4),
                    "description": bias_diag.get("description", ""),
                },
                priority=prio,
                investigation_type="verify_residual_bias",
                feature=None,
                group=None,
            ))

    # -----------------------------------------------------------------------
    # 6. Statistical Correlations (|r| >= 0.5, p < 0.05)
    # -----------------------------------------------------------------------
    stats_rep = analysis.statistical_analysis
    corrs = stats_rep.correlations if stats_rep else {}
    if isinstance(corrs, dict):
        seen_pairs = set()
        for f1, f1_corrs in corrs.items():
            if not isinstance(f1_corrs, dict):
                continue
            for f2, c_info in f1_corrs.items():
                if f1 == f2 or not isinstance(c_info, dict):
                    continue
                pair_key = tuple(sorted([f1, f2]))
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)

                r_val = float(c_info.get("pearson_r") or c_info.get("r") or 0.0)
                p_val = float(c_info.get("p_value") or 1.0)

                if abs(r_val) >= 0.50 and p_val < 0.05:
                    cid = f"cand_{cand_counter:03d}_corr_{f1}_{f2}"
                    cand_counter += 1

                    prio = compute_deterministic_priority(
                        base_priority=0.60,
                        effect_magnitude=abs(r_val),
                        sample_size=analysis.dataset_context.get("n_rows", 100),
                        is_significant=True,
                    )

                    candidates.append(CandidateHypothesis(
                        candidate_id=cid,
                        claim=f"Significant correlation between '{f1}' and '{f2}' (r={r_val:.4f}, p={p_val:.4e}).",
                        source_path=f"statistical_analysis.correlations.{f1}.{f2}",
                        evidence={
                            "feature_1": f1,
                            "feature_2": f2,
                            "r": round(r_val, 4),
                            "p_value": round(p_val, 6),
                            "method": c_info.get("method", "pearson"),
                        },
                        priority=prio,
                        investigation_type="verify_correlation",
                        feature=f1,
                        group=f2,
                    ))

    # -----------------------------------------------------------------------
    # 7. Statistical Hypothesis Tests (p < 0.05)
    # -----------------------------------------------------------------------
    tests = stats_rep.statistical_tests if stats_rep else []
    for idx, t_info in enumerate(tests):
        p_val = float(t_info.get("p_value", 1.0))
        if p_val < 0.05:
            col = t_info.get("column") or t_info.get("feature", "unknown")
            grp = t_info.get("group_by") or t_info.get("group", "unknown")
            t_name = t_info.get("test_name", "test")
            eff = float(t_info.get("effect_size", 0.0) or 0.0)
            cid = f"cand_{cand_counter:03d}_test_{col}_{grp}"
            cand_counter += 1

            prio = compute_deterministic_priority(
                base_priority=0.65,
                effect_magnitude=abs(eff),
                sample_size=analysis.dataset_context.get("n_rows", 100),
                is_significant=True,
            )

            candidates.append(CandidateHypothesis(
                candidate_id=cid,
                claim=f"Significant group disparity in '{col}' grouped by '{grp}' ({t_name}, p={p_val:.4e}).",
                source_path=f"statistical_analysis.statistical_tests[{idx}]",
                evidence={
                    "column": col,
                    "group_by": grp,
                    "test_name": t_name,
                    "p_value": round(p_val, 6),
                    "effect_size": round(eff, 4),
                },
                priority=prio,
                investigation_type="verify_group_difference",
                feature=col,
                group=grp,
            ))

    # -----------------------------------------------------------------------
    # 8. Ensemble Validation Result
    # -----------------------------------------------------------------------
    ens = analysis.ensemble
    if ens and ens.status == PipelineStatus.SUCCESS and ens.champion_score is not None:
        cid = f"cand_{cand_counter:03d}_ensemble"
        cand_counter += 1
        delta_val = float(ens.delta or 0.0)
        improves = ens.improves_champion

        claim = (
            f"Top-K ensemble blend improved champion score by {abs(delta_val):.4f} under identical CV."
            if improves
            else f"Top-K ensemble blend did not outperform single champion model (delta={delta_val:.4f})."
        )

        prio = compute_deterministic_priority(
            base_priority=0.60,
            effect_magnitude=abs(delta_val) * 2.0,
            sample_size=analysis.dataset_context.get("n_rows", 100),
            is_significant=improves,
        )

        candidates.append(CandidateHypothesis(
            candidate_id=cid,
            claim=claim,
            source_path="ensemble",
            evidence={
                "champion_score": round(float(ens.champion_score), 4),
                "ensemble_score": round(float(ens.ensemble_score or ens.score or 0.0), 4),
                "delta": round(delta_val, 4),
                "improves_champion": improves,
                "models": ens.models,
            },
            priority=prio,
            investigation_type="verify_candidate",
            feature=None,
            group=None,
        ))

    # Sort candidates deterministically by descending priority
    candidates.sort(key=lambda c: c.priority, reverse=True)
    return candidates
