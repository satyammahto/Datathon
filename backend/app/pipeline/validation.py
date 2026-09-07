"""
Validation Strategy Module (Person 2 - ML / Accuracy Brain)
Responsible for selecting and configuring data validation (CV) strategies
without data leakage (StratifiedKFold, KFold, TimeSeriesSplit).
"""
from typing import Optional, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold, TimeSeriesSplit

try:
    from backend.app.schemas.discovery_contract import DiscoveryContract
    from backend.app.schemas.analysis_contract import ValidationReport
    from backend.app.schemas.enums import PipelineStatus, MLTaskType, ValidationStrategy
except ImportError:
    from app.schemas.discovery_contract import DiscoveryContract
    from app.schemas.analysis_contract import ValidationReport
    from app.schemas.enums import PipelineStatus, MLTaskType, ValidationStrategy


def select_validation_strategy(
    df: Optional[Any],
    discovery: DiscoveryContract,
    task_type: MLTaskType,
) -> ValidationReport:
    """
    Deterministically choose validation strategy based on discovery router
    and fingerprint directives (e.g. Stratified for classification, TimeSeries for temporal).
    """
    if task_type == MLTaskType.NOT_APPLICABLE:
        return ValidationReport(
            status=PipelineStatus.NOT_APPLICABLE,
            reason="No applicable predictive ML task detected",
        )

    # 1. Check temporal requirement
    if discovery.router.time_analysis:
        return ValidationReport(
            status=PipelineStatus.SUCCESS,
            strategy=ValidationStrategy.TIME_SERIES_SPLIT.value,
            folds=5,
            primary_metric="rmse" if task_type == MLTaskType.REGRESSION else "f1_macro",
            random_seed=42,
            split_details={"temporal_ordering": True, "shuffle": False},
        )

    # 2. Classification default to StratifiedKFold
    if task_type == MLTaskType.CLASSIFICATION:
        return ValidationReport(
            status=PipelineStatus.SUCCESS,
            strategy=ValidationStrategy.STRATIFIED_K_FOLD.value,
            folds=5,
            primary_metric="f1",
            random_seed=42,
            split_details={"shuffle": True},
        )

    # 3. Regression default to KFold
    if task_type == MLTaskType.REGRESSION:
        return ValidationReport(
            status=PipelineStatus.SUCCESS,
            strategy=ValidationStrategy.K_FOLD.value,
            folds=5,
            primary_metric="rmse",
            random_seed=42,
            split_details={"shuffle": True},
        )

    return ValidationReport(
        status=PipelineStatus.NOT_APPLICABLE,
        reason=f"Validation strategy not defined for task {task_type}",
    )


def build_cv_splitter(
    validation_report: ValidationReport,
    n_samples: int,
    y: Optional[Any] = None,
) -> Tuple[Optional[Any], Optional[str]]:
    """
    Construct the scikit-learn CV splitter with small-data safety.
    Dynamically bounds fold count if mathematically necessary or returns
    an explicit reason if cross-validation cannot be safely performed.
    """
    if n_samples < 2:
        return None, f"Insufficient observations ({n_samples}) for cross-validation; minimum 2 required."

    strategy = validation_report.strategy
    requested_folds = validation_report.folds or 5
    seed = validation_report.random_seed or 42

    # --- TimeSeriesSplit ---
    if strategy == ValidationStrategy.TIME_SERIES_SPLIT.value:
        if n_samples < 3:
            return None, f"Insufficient observations ({n_samples}) for TimeSeriesSplit; minimum 3 required."
        effective_folds = min(requested_folds, n_samples - 1)
        effective_folds = max(2, effective_folds)
        if effective_folds >= n_samples:
            return None, f"Cannot create TimeSeriesSplit with {effective_folds} folds on {n_samples} samples."
        return TimeSeriesSplit(n_splits=effective_folds), None

    # --- StratifiedKFold ---
    if strategy == ValidationStrategy.STRATIFIED_K_FOLD.value:
        if y is None:
            return None, "StratifiedKFold requires target array y."
        # Count classes
        unique_classes, counts = np.unique(y, return_counts=True)
        if len(unique_classes) < 2:
            return None, f"Target contains only {len(unique_classes)} unique class. StratifiedKFold requires at least 2 classes."
        min_class_count = int(np.min(counts))
        if min_class_count < 2:
            return None, (
                f"Smallest class has only {min_class_count} sample(s); "
                f"stratified cross-validation requires at least 2 samples per class."
            )
        effective_folds = min(requested_folds, min_class_count)
        effective_folds = max(2, effective_folds)
        return StratifiedKFold(n_splits=effective_folds, shuffle=True, random_state=seed), None

    # --- Standard KFold ---
    effective_folds = min(requested_folds, n_samples)
    effective_folds = max(2, effective_folds)
    return KFold(n_splits=effective_folds, shuffle=True, random_state=seed), None
