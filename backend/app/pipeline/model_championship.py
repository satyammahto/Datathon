"""
Deterministic Model Championship Engine (Person 2 - ML / Accuracy Brain)
Empirically benchmarks candidate machine learning models using leakage-free cross-validation
and selects the empirical champion based strictly on mathematical evidence.
Zero LLM involvement; zero hallucination.
"""
from typing import List, Dict, Any, Optional, Tuple
import time
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    root_mean_squared_error,
    mean_absolute_error,
    r2_score,
)

# Classification Estimators
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

# Regression Estimators
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor

try:
    from backend.app.schemas.discovery_contract import DiscoveryContract
    from backend.app.schemas.analysis_contract import (
        ModelBenchmarkResult,
        ChampionResult,
        ValidationReport,
    )
    from backend.app.schemas.enums import PipelineStatus, MLTaskType
except ImportError:
    from app.schemas.discovery_contract import DiscoveryContract
    from app.schemas.analysis_contract import (
        ModelBenchmarkResult,
        ChampionResult,
        ValidationReport,
    )
    from app.schemas.enums import PipelineStatus, MLTaskType

from .validation import build_cv_splitter
from .statistical import sanitize_for_json


# ---------------------------------------------------------------------------
# Preprocessing Pipeline Builder (Strictly Leakage-Free)
# ---------------------------------------------------------------------------

def build_feature_preprocessor(
    X: pd.DataFrame,
    num_cols: List[str],
    cat_cols: List[str],
) -> Optional[ColumnTransformer]:
    """
    Construct an unfitted ColumnTransformer.
    Preprocessing is fitted exclusively inside each CV training fold.
    """
    transformers = []

    if num_cols:
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("num", num_pipeline, num_cols))

    if cat_cols:
        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        transformers.append(("cat", cat_pipeline, cat_cols))

    if not transformers:
        return None

    return ColumnTransformer(transformers=transformers, remainder="drop")


# ---------------------------------------------------------------------------
# Candidate Model Factories
# ---------------------------------------------------------------------------

def get_classification_candidates(random_state: int = 42) -> List[Tuple[str, str, Any, Dict[str, Any]]]:
    """Return the candidate classification models with reproducible parameters."""
    return [
        (
            "LogisticRegression",
            "linear",
            LogisticRegression(max_iter=1000, random_state=random_state),
            {"max_iter": 1000, "random_state": random_state},
        ),
        (
            "DecisionTreeClassifier",
            "tree",
            DecisionTreeClassifier(max_depth=5, random_state=random_state),
            {"max_depth": 5, "random_state": random_state},
        ),
        (
            "RandomForestClassifier",
            "tree_ensemble",
            RandomForestClassifier(n_estimators=50, max_depth=6, random_state=random_state, n_jobs=1),
            {"n_estimators": 50, "max_depth": 6, "random_state": random_state},
        ),
        (
            "HistGradientBoostingClassifier",
            "gradient_boosting",
            HistGradientBoostingClassifier(max_iter=50, random_state=random_state),
            {"max_iter": 50, "random_state": random_state},
        ),
    ]


def get_regression_candidates(random_state: int = 42) -> List[Tuple[str, str, Any, Dict[str, Any]]]:
    """Return the candidate regression models with reproducible parameters."""
    return [
        (
            "RidgeRegression",
            "linear",
            Ridge(alpha=1.0, random_state=random_state),
            {"alpha": 1.0, "random_state": random_state},
        ),
        (
            "DecisionTreeRegressor",
            "tree",
            DecisionTreeRegressor(max_depth=5, random_state=random_state),
            {"max_depth": 5, "random_state": random_state},
        ),
        (
            "RandomForestRegressor",
            "tree_ensemble",
            RandomForestRegressor(n_estimators=50, max_depth=6, random_state=random_state, n_jobs=1),
            {"n_estimators": 50, "max_depth": 6, "random_state": random_state},
        ),
        (
            "HistGradientBoostingRegressor",
            "gradient_boosting",
            HistGradientBoostingRegressor(max_iter=50, random_state=random_state),
            {"max_iter": 50, "random_state": random_state},
        ),
    ]


# ---------------------------------------------------------------------------
# Fold Evaluation Engine
# ---------------------------------------------------------------------------

def evaluate_classification_candidate(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: np.ndarray,
    cv_splitter: Any,
) -> Tuple[List[float], Dict[str, float], List[Any], List[int]]:
    """Evaluate classification pipeline fold-by-fold strictly inside training folds and collect OoF predictions."""
    fold_f1_scores = []
    fold_accuracies = []
    fold_precisions = []
    fold_recalls = []
    fold_roc_aucs = []

    classes = np.unique(y)
    is_binary = len(classes) == 2

    oof_predictions = [None] * len(X)
    oof_indices: List[int] = []

    for train_idx, val_idx in cv_splitter.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_val)

        for i_pos, v_idx in enumerate(val_idx):
            val_int = int(v_idx)
            pred_scalar = y_pred[i_pos]
            if hasattr(pred_scalar, "item"):
                pred_scalar = pred_scalar.item()
            oof_predictions[val_int] = pred_scalar
            oof_indices.append(val_int)

        # Core Metrics
        f1 = float(f1_score(y_val, y_pred, average="macro", zero_division=0))
        acc = float(accuracy_score(y_val, y_pred))
        prec = float(precision_score(y_val, y_pred, average="macro", zero_division=0))
        rec = float(recall_score(y_val, y_pred, average="macro", zero_division=0))

        fold_f1_scores.append(f1)
        fold_accuracies.append(acc)
        fold_precisions.append(prec)
        fold_recalls.append(rec)

        # ROC-AUC (optional / when valid)
        if hasattr(pipeline.named_steps["estimator"], "predict_proba"):
            try:
                y_proba = pipeline.predict_proba(X_val)
                if is_binary and len(np.unique(y_val)) == 2:
                    auc = float(roc_auc_score(y_val, y_proba[:, 1]))
                    fold_roc_aucs.append(auc)
                elif not is_binary and len(np.unique(y_val)) == len(classes):
                    auc = float(roc_auc_score(y_val, y_proba, multi_class="ovr", average="macro"))
                    fold_roc_aucs.append(auc)
            except Exception:
                pass

    metrics = {
        "f1_macro": float(np.mean(fold_f1_scores)),
        "f1": float(np.mean(fold_f1_scores)),  # Alias for backward compatibility
        "accuracy": float(np.mean(fold_accuracies)),
        "precision_macro": float(np.mean(fold_precisions)),
        "recall_macro": float(np.mean(fold_recalls)),
    }
    if fold_roc_aucs:
        metrics["roc_auc"] = float(np.mean(fold_roc_aucs))

    return fold_f1_scores, metrics, oof_predictions, oof_indices


def evaluate_regression_candidate(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: np.ndarray,
    cv_splitter: Any,
) -> Tuple[List[float], Dict[str, float], List[Any], List[int]]:
    """Evaluate regression pipeline fold-by-fold strictly inside training folds and collect OoF predictions."""
    fold_rmses = []
    fold_maes = []
    fold_r2s = []

    oof_predictions = [None] * len(X)
    oof_indices: List[int] = []

    for train_idx, val_idx in cv_splitter.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_val)

        for i_pos, v_idx in enumerate(val_idx):
            val_int = int(v_idx)
            pred_scalar = y_pred[i_pos]
            if hasattr(pred_scalar, "item"):
                pred_scalar = pred_scalar.item()
            oof_predictions[val_int] = float(pred_scalar)
            oof_indices.append(val_int)

        rmse = float(root_mean_squared_error(y_val, y_pred))
        mae = float(mean_absolute_error(y_val, y_pred))
        r2 = float(r2_score(y_val, y_pred))

        fold_rmses.append(rmse)
        fold_maes.append(mae)
        fold_r2s.append(r2)

    metrics = {
        "rmse": float(np.mean(fold_rmses)),
        "mae": float(np.mean(fold_maes)),
        "r2": float(np.mean(fold_r2s)),
    }
    return fold_rmses, metrics, oof_predictions, oof_indices


# ---------------------------------------------------------------------------
# Master Benchmark Orchestrator
# ---------------------------------------------------------------------------

def benchmark_models(
    df: Optional[Any],
    discovery: DiscoveryContract,
    task_type: MLTaskType,
    validation: ValidationReport,
    target_column: Optional[str] = None,
) -> List[ModelBenchmarkResult]:
    """
    Benchmark multiple candidate models against the validation strategy.
    Isolates faults so broken models do not crash the tournament.
    """
    if df is None:
        return []

    if not isinstance(df, pd.DataFrame):
        try:
            df = pd.DataFrame(df)
        except Exception:
            return []

    if df.empty or len(df) == 0:
        return []

    if task_type not in (MLTaskType.CLASSIFICATION, MLTaskType.REGRESSION):
        return []

    # 1. Resolve target column
    target_col = target_column or (discovery.target_candidates[0] if discovery.target_candidates else None)
    if not target_col or target_col not in df.columns:
        return []

    # 2. Extract and validate target
    target_series = df[target_col]
    valid_mask = target_series.notna()
    if not valid_mask.any():
        return [ModelBenchmarkResult(
            model_name="AllCandidates",
            name="AllCandidates",
            family="none",
            status=PipelineStatus.FAILED,
            reason="Target column contains only null values",
        )]

    clean_df = df[valid_mask].copy()
    y_raw = clean_df[target_col].values

    # Check target variance
    if len(np.unique(y_raw)) <= 1:
        return [ModelBenchmarkResult(
            model_name="AllCandidates",
            name="AllCandidates",
            family="none",
            status=PipelineStatus.FAILED,
            reason="Target column is constant (zero variance)",
        )]

    # 3. Identify feature columns
    schema_def = discovery.schema_definition or {}
    feature_cols = [c for c in clean_df.columns if c != target_col]
    if not feature_cols:
        return [ModelBenchmarkResult(
            model_name="AllCandidates",
            name="AllCandidates",
            family="none",
            status=PipelineStatus.FAILED,
            reason="No usable feature columns detected in dataset",
        )]

    num_cols = []
    cat_cols = []
    for col in feature_cols:
        col_type = schema_def.get(col, "").lower()
        if col_type == "numeric" or (not col_type and pd.api.types.is_numeric_dtype(clean_df[col])):
            num_cols.append(col)
        else:
            cat_cols.append(col)

    X = clean_df[feature_cols]
    n_samples = len(X)

    # 4. Build CV Splitter
    cv_splitter, cv_error = build_cv_splitter(validation, n_samples=n_samples, y=y_raw)
    if cv_splitter is None:
        return [ModelBenchmarkResult(
            model_name="AllCandidates",
            name="AllCandidates",
            family="none",
            status=PipelineStatus.NOT_APPLICABLE,
            reason=cv_error or "Unable to construct validation splitter",
        )]

    # 5. Build candidates based on task
    seed = validation.random_seed or 42
    if task_type == MLTaskType.CLASSIFICATION:
        candidates = get_classification_candidates(random_state=seed)
        primary_metric = validation.primary_metric or "f1_macro"
        is_lower_better = False
    else:
        candidates = get_regression_candidates(random_state=seed)
        primary_metric = validation.primary_metric or "rmse"
        is_lower_better = True

    results: List[ModelBenchmarkResult] = []

    # 6. Evaluate each candidate with fault isolation
    for name, family, estimator, params in candidates:
        preprocessor = build_feature_preprocessor(X, num_cols, cat_cols)
        if preprocessor is None:
            results.append(ModelBenchmarkResult(
                model_name=name,
                name=name,
                family=family,
                status=PipelineStatus.FAILED,
                reason="Failed to build feature preprocessor",
                parameters=params,
            ))
            continue

        full_pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("estimator", estimator),
        ])

        start_time = time.perf_counter()
        try:
            if task_type == MLTaskType.CLASSIFICATION:
                fold_scores, metrics, oof_preds, oof_idxs = evaluate_classification_candidate(full_pipeline, X, y_raw, cv_splitter)
            else:
                fold_scores, metrics, oof_preds, oof_idxs = evaluate_regression_candidate(full_pipeline, X, y_raw, cv_splitter)

            elapsed = round(time.perf_counter() - start_time, 4)
            mean_sc = float(np.mean(fold_scores))
            std_sc = float(np.std(fold_scores, ddof=1)) if len(fold_scores) > 1 else 0.0

            results.append(ModelBenchmarkResult(
                model_name=name,
                name=name,
                family=family,
                status=PipelineStatus.SUCCESS,
                primary_metric=primary_metric,
                mean_score=round(mean_sc, 6),
                std_score=round(std_sc, 6),
                fold_scores=[round(s, 6) for s in fold_scores],
                metrics={k: round(v, 6) for k, v in metrics.items()},
                train_time_seconds=elapsed,
                parameters=params,
                oof_predictions=oof_preds,
                oof_indices=oof_idxs,
            ))
        except Exception as exc:
            # Fault isolation: record failure and continue
            results.append(ModelBenchmarkResult(
                model_name=name,
                name=name,
                family=family,
                status=PipelineStatus.FAILED,
                reason=f"Model evaluation failed: {str(exc)}",
                parameters=params,
            ))

    # 7. Rank successful candidates
    successful = [r for r in results if r.status == PipelineStatus.SUCCESS and r.mean_score is not None]
    successful_sorted = sorted(successful, key=lambda r: r.mean_score, reverse=not is_lower_better)
    for rank_idx, r in enumerate(successful_sorted, 1):
        r.rank = rank_idx

    return results


# ---------------------------------------------------------------------------
# Empirical Champion Selection
# ---------------------------------------------------------------------------

def select_champion(
    benchmarks: List[Any],
    primary_metric: str = "f1_macro",
) -> ChampionResult:
    """
    Select the best-performing model strictly based on validation metrics.
    Deterministic, empirical, and traceable.
    """
    if not benchmarks:
        return ChampionResult(
            status=PipelineStatus.NOT_APPLICABLE,
            reason="No benchmark results available to determine champion",
        )

    # Convert dicts back to ModelBenchmarkResult if passed as raw dictionaries
    parsed_benchmarks: List[ModelBenchmarkResult] = []
    for b in benchmarks:
        if isinstance(b, dict):
            parsed_benchmarks.append(ModelBenchmarkResult.model_validate(b))
        elif isinstance(b, ModelBenchmarkResult):
            parsed_benchmarks.append(b)

    valid_models = []
    for b in parsed_benchmarks:
        if b.status == PipelineStatus.SUCCESS:
            if b.mean_score is None:
                # Fallback to metrics dictionary (e.g. for Phase 2.0 test compatibility)
                metric_val = b.metrics.get(primary_metric)
                if metric_val is None:
                    # check alternate aliases (e.g. f1 vs f1_macro)
                    if "f1" in primary_metric:
                        metric_val = b.metrics.get("f1_macro") or b.metrics.get("f1")
                    elif "rmse" in primary_metric:
                        metric_val = b.metrics.get("rmse")
                if metric_val is not None:
                    b.mean_score = float(metric_val)
            if b.mean_score is not None:
                valid_models.append(b)

    if not valid_models:
        failed_reasons = [b.reason for b in parsed_benchmarks if b.reason]
        aggregate_reason = "; ".join(failed_reasons[:2]) if failed_reasons else "All candidate models failed evaluation"
        return ChampionResult(
            status=PipelineStatus.FAILED,
            reason=f"No models successfully evaluated: {aggregate_reason}",
        )

    # Determine metric direction
    metric_name = primary_metric.lower()
    is_lower_better = any(err_term in metric_name for err_term in ["rmse", "mae", "mse", "loss"])
    selection_rule = f"lowest_mean_{primary_metric}" if is_lower_better else f"highest_mean_{primary_metric}"

    sorted_models = sorted(
        valid_models,
        key=lambda m: m.mean_score,
        reverse=not is_lower_better,
    )
    best = sorted_models[0]

    std_part = f" (std: {best.std_score:.4f})" if best.std_score is not None else ""
    return ChampionResult(
        status=PipelineStatus.SUCCESS,
        model_name=best.model_name,
        name=best.model_name,
        primary_metric=primary_metric,
        metric=primary_metric,
        score=best.mean_score,
        selection_rule=selection_rule,
        selection_rationale=(
            f"Model {best.model_name} ranked #1 with {selection_rule} of {best.mean_score:.4f}{std_part} "
            f"across {len(best.fold_scores)} validation folds."
        ),
        empirical_evidence={
            "metric": primary_metric,
            "score": best.mean_score,
            "std_score": best.std_score,
            "fold_scores": best.fold_scores,
            "competitors_evaluated": len(valid_models),
            "selection_rule": selection_rule,
            "oof_predictions": best.oof_predictions,
            "oof_indices": best.oof_indices,
        },
    )
