"""
ML Analysis Master Orchestrator (Person 2 - ML / Accuracy Brain)
Coordinates Discovery validation, statistical analysis, validation strategy selection,
model championship, out-of-fold error diagnostics, controlled experiments, and ensemble
validation to produce the immutable, verified AnalysisContract for Person 3.
Zero LLM numerical dependency; 100% deterministic mathematical execution.
"""
from typing import Optional, Any, Dict, List
import pandas as pd

try:
    from backend.app.schemas.discovery_contract import DiscoveryContract
    from backend.app.schemas.analysis_contract import (
        AnalysisContract,
        ValidationReport,
        StatisticalReport,
        ModelBenchmarkResult,
        ChampionResult,
        ErrorAnalysisReport,
        ExperimentRecord,
        EnsembleReport,
        TimeAnalysisReport,
    )
    from backend.app.schemas.enums import PipelineStatus, MLTaskType
except ImportError:
    from app.schemas.discovery_contract import DiscoveryContract
    from app.schemas.analysis_contract import (
        AnalysisContract,
        ValidationReport,
        StatisticalReport,
        ModelBenchmarkResult,
        ChampionResult,
        ErrorAnalysisReport,
        ExperimentRecord,
        EnsembleReport,
        TimeAnalysisReport,
    )
    from app.schemas.enums import PipelineStatus, MLTaskType

from .statistical import run_statistical_analysis, sanitize_for_json
from .validation import select_validation_strategy
from .model_championship import benchmark_models, select_champion
from .error_analysis import analyze_errors
from .time_analysis import run_time_analysis
from .experiments import run_controlled_experiments


def determine_ml_task(discovery: Any) -> MLTaskType:
    """
    Deterministically infer the applicable ML task type from the Discovery router
    and target candidates without dataset-specific hardcoding.
    """
    if discovery is None:
        return MLTaskType.NOT_APPLICABLE

    router = getattr(discovery, "router", None)
    if router:
        # 1. Person 1 canonical router check
        rec_task = getattr(router, "recommended_primary_task", None)
        if rec_task is not None:
            val = getattr(rec_task, "value", str(rec_task)).lower()
            if "classification" in val:
                return MLTaskType.CLASSIFICATION
            elif "regression" in val:
                return MLTaskType.REGRESSION
            elif any(t in val for t in ("time", "temporal", "series", "forecast")):
                return MLTaskType.TIME_ANALYSIS
            elif "cluster" in val:
                return MLTaskType.CLUSTERING
            elif "anomaly" in val:
                return MLTaskType.ANOMALY_DETECTION

        # 2. Person 2 schema boolean check
        has_targets = bool(getattr(discovery, "target_candidates", None))
        if getattr(router, "classification", False) and has_targets:
            return MLTaskType.CLASSIFICATION
        elif getattr(router, "regression", False) and has_targets:
            return MLTaskType.REGRESSION
        elif getattr(router, "time_analysis", False):
            return MLTaskType.TIME_ANALYSIS
        elif getattr(router, "clustering", False):
            return MLTaskType.CLUSTERING
        elif getattr(router, "anomaly_detection", False):
            return MLTaskType.ANOMALY_DETECTION

    return MLTaskType.NOT_APPLICABLE


def run_ml_pipeline(
    df: Optional[Any],
    discovery: Any,
) -> AnalysisContract:
    """
    Orchestrate Person 2's end-to-end analytical pass based on Discovery Contract.
    Produces the standard Analysis Contract consumed by Person 3.

    Follows strict pipeline discipline:
    1. Validation of DiscoveryContract
    2. Input structure validation & dataset context extraction
    3. Deterministic task routing
    4. Statistical analysis
    5. Time analysis (when applicable)
    6. Supervised ML championship, error diagnostics, & controlled experiments
    7. Pydantic validation & JSON serialization verification
    """
    # -----------------------------------------------------------------------
    # Step 1: Validate DiscoveryContract input
    # -----------------------------------------------------------------------
    if discovery is None:
        return AnalysisContract(
            schema_version="1.0",
            status=PipelineStatus.FAILED,
            reason="Invalid DiscoveryContract: input is None",
            task=MLTaskType.NOT_APPLICABLE,
            validation=ValidationReport(status=PipelineStatus.FAILED, reason="DiscoveryContract is missing"),
            statistical_analysis=StatisticalReport(status=PipelineStatus.FAILED, reason="DiscoveryContract is missing"),
            winner=ChampionResult(status=PipelineStatus.FAILED, reason="DiscoveryContract is missing"),
            error_analysis=ErrorAnalysisReport(status=PipelineStatus.FAILED, reason="DiscoveryContract is missing"),
            ensemble=EnsembleReport(status=PipelineStatus.FAILED, reason="DiscoveryContract is missing"),
            time_analysis=TimeAnalysisReport(status=PipelineStatus.FAILED, reason="DiscoveryContract is missing"),
        )

    # Check if Person 1 canonical contract (object or dict)
    if hasattr(discovery, "contract_type") or hasattr(discovery, "file_info"):
        discovery = DiscoveryContract.from_person1(discovery)
    elif isinstance(discovery, dict) and ("file_info" in discovery or "contract_type" in discovery):
        discovery = DiscoveryContract.from_person1(discovery)
    elif isinstance(discovery, dict):
        try:
            discovery = DiscoveryContract.model_validate(discovery)
        except Exception:
            discovery = DiscoveryContract.from_person1(discovery)
    elif not isinstance(discovery, DiscoveryContract):
        try:
            discovery = DiscoveryContract.from_person1(discovery)
        except Exception as exc:
            return AnalysisContract(
                schema_version="1.0",
                status=PipelineStatus.FAILED,
                reason=f"Invalid DiscoveryContract: expected DiscoveryContract or dict, got {type(discovery).__name__}: {str(exc)}",
                task=MLTaskType.NOT_APPLICABLE,
                validation=ValidationReport(status=PipelineStatus.FAILED, reason="Invalid DiscoveryContract type"),
                statistical_analysis=StatisticalReport(status=PipelineStatus.FAILED, reason="Invalid DiscoveryContract type"),
                winner=ChampionResult(status=PipelineStatus.FAILED, reason="Invalid DiscoveryContract type"),
                error_analysis=ErrorAnalysisReport(status=PipelineStatus.FAILED, reason="Invalid DiscoveryContract type"),
                ensemble=EnsembleReport(status=PipelineStatus.FAILED, reason="Invalid DiscoveryContract type"),
                time_analysis=TimeAnalysisReport(status=PipelineStatus.FAILED, reason="Invalid DiscoveryContract type"),
            )

    # -----------------------------------------------------------------------
    # Step 2: Validate dataset structure
    # -----------------------------------------------------------------------
    clean_df = None
    if df is not None:
        if isinstance(df, pd.DataFrame):
            clean_df = df
        else:
            try:
                clean_df = pd.DataFrame(df)
            except Exception as exc:
                return AnalysisContract(
                    schema_version="1.0",
                    status=PipelineStatus.FAILED,
                    reason=f"Unusable or invalid dataset structure: cannot convert input to DataFrame: {str(exc)}",
                    task=MLTaskType.NOT_APPLICABLE,
                    validation=ValidationReport(status=PipelineStatus.FAILED, reason="Invalid dataset structure"),
                    statistical_analysis=StatisticalReport(status=PipelineStatus.FAILED, reason="Invalid dataset structure"),
                    winner=ChampionResult(status=PipelineStatus.FAILED, reason="Invalid dataset structure"),
                    error_analysis=ErrorAnalysisReport(status=PipelineStatus.FAILED, reason="Invalid dataset structure"),
                    ensemble=EnsembleReport(status=PipelineStatus.FAILED, reason="Invalid dataset structure"),
                    time_analysis=TimeAnalysisReport(status=PipelineStatus.FAILED, reason="Invalid dataset structure"),
                )

    # -----------------------------------------------------------------------
    # Step 3: Determine task and target
    # -----------------------------------------------------------------------
    task_type = determine_ml_task(discovery)
    target_col = None
    if discovery.target_candidates:
        first_t = discovery.target_candidates[0]
        if hasattr(first_t, "column_name"):
            target_col = first_t.column_name
        elif isinstance(first_t, dict) and "column_name" in first_t:
            target_col = first_t["column_name"]
        else:
            target_col = str(first_t)

    dataset_context = {
        "target_column": target_col,
        "task_type": task_type.value,
        "n_rows": int(len(clean_df)) if clean_df is not None else 0,
        "n_cols": int(len(clean_df.columns)) if clean_df is not None else 0,
        "feature_columns": [str(c) for c in clean_df.columns if c != target_col] if clean_df is not None else [],
        "router": discovery.router.model_dump() if discovery.router else {},
    }

    # -----------------------------------------------------------------------
    # Step 4: Statistical Analysis (Fault-Isolated)
    # -----------------------------------------------------------------------
    try:
        stats_report = run_statistical_analysis(clean_df, discovery)
    except Exception as exc:
        stats_report = StatisticalReport(
            status=PipelineStatus.FAILED,
            reason=f"Statistical analysis failed: {str(exc)}",
        )

    # -----------------------------------------------------------------------
    # Step 5: Time Analysis (Fault-Isolated)
    # -----------------------------------------------------------------------
    try:
        time_report = run_time_analysis(clean_df, discovery)
    except Exception as exc:
        time_report = TimeAnalysisReport(
            status=PipelineStatus.FAILED,
            reason=f"Time analysis failed: {str(exc)}",
        )

    # -----------------------------------------------------------------------
    # Step 6: Supervised ML Routing & Modeling
    # -----------------------------------------------------------------------
    is_supervised = task_type in (MLTaskType.CLASSIFICATION, MLTaskType.REGRESSION)

    if not is_supervised or not target_col:
        # Case A: Unsupervised, Clustering, Time Analysis only, or No Target
        validation_report = ValidationReport(
            status=PipelineStatus.NOT_APPLICABLE,
            reason="No valid supervised target candidate detected for cross-validation",
        )
        benchmarks: List[ModelBenchmarkResult] = []
        champion = ChampionResult(
            status=PipelineStatus.NOT_APPLICABLE,
            reason="No supervised target detected for model championship",
        )
        error_report = ErrorAnalysisReport(
            status=PipelineStatus.NOT_APPLICABLE,
            reason="No supervised champion model available for error analysis",
        )
        experiments_records: List[ExperimentRecord] = []
        ensemble_report = EnsembleReport(
            status=PipelineStatus.NOT_APPLICABLE,
            reason="No supervised candidate models available for ensemble evaluation",
        )
    elif clean_df is not None and target_col not in clean_df.columns:
        # Case B: Target column missing from dataset
        validation_report = ValidationReport(
            status=PipelineStatus.FAILED,
            reason=f"Target column '{target_col}' not found in dataset columns",
        )
        benchmarks = []
        champion = ChampionResult(
            status=PipelineStatus.FAILED,
            reason=f"Target column '{target_col}' not found in dataset columns",
        )
        error_report = ErrorAnalysisReport(
            status=PipelineStatus.NOT_APPLICABLE,
            reason="Supervised championship failed: target column missing",
        )
        experiments_records = []
        ensemble_report = EnsembleReport(
            status=PipelineStatus.NOT_APPLICABLE,
            reason="No champion model available for ensemble evaluation",
        )
    elif clean_df is None:
        # Case C: DataFrame is None (e.g. abstract interface / dry run)
        validation_report = select_validation_strategy(None, discovery, task_type)
        benchmarks = []
        champion = ChampionResult(
            status=PipelineStatus.NOT_APPLICABLE,
            reason="No benchmark results available to determine champion",
        )
        error_report = ErrorAnalysisReport(
            status=PipelineStatus.UNAVAILABLE,
            reason="Validation out-of-fold predictions not available for error analysis",
        )
        experiments_records = []
        ensemble_report = EnsembleReport(
            status=PipelineStatus.NOT_APPLICABLE,
            reason="No champion model available for ensemble benchmarking",
        )
    else:
        # Case D: Supervised ML execution with full fault isolation
        # 6a. Validation strategy
        try:
            validation_report = select_validation_strategy(clean_df, discovery, task_type)
        except Exception as exc:
            validation_report = ValidationReport(
                status=PipelineStatus.FAILED,
                reason=f"Validation selection failed: {str(exc)}",
            )

        # 6b. Benchmark models & select champion
        try:
            benchmarks = benchmark_models(clean_df, discovery, task_type, validation_report, target_column=target_col)
            primary_metric = validation_report.primary_metric or ("f1" if task_type == MLTaskType.CLASSIFICATION else "rmse")
            champion = select_champion(benchmarks, primary_metric=primary_metric)
        except Exception as exc:
            benchmarks = []
            champion = ChampionResult(
                status=PipelineStatus.FAILED,
                reason=f"Model championship execution failed: {str(exc)}",
            )

        # 6c. Error analysis on champion
        try:
            error_report = analyze_errors(
                champion=champion,
                validation=validation_report,
                df=clean_df,
                discovery=discovery,
                target_col=target_col,
                task_type=task_type,
            )
        except Exception as exc:
            error_report = ErrorAnalysisReport(
                status=PipelineStatus.FAILED,
                reason=f"Error analysis execution failed: {str(exc)}",
            )

        # 6d. Controlled experiments & ensemble validation
        try:
            experiments_records, ensemble_report, _ = run_controlled_experiments(
                df=clean_df,
                discovery=discovery,
                task_type=task_type,
                validation=validation_report,
                champion=champion,
                benchmarks=benchmarks,
            )
        except Exception as exc:
            experiments_records = []
            ensemble_report = EnsembleReport(
                status=PipelineStatus.FAILED,
                reason=f"Controlled experiments execution failed: {str(exc)}",
            )

    # -----------------------------------------------------------------------
    # Step 7: Determine overall pipeline status
    # -----------------------------------------------------------------------
    if stats_report.status == PipelineStatus.FAILED and (is_supervised and champion.status == PipelineStatus.FAILED):
        overall_status = PipelineStatus.FAILED
        overall_reason = "Critical pipeline failure in both statistical analysis and model championship"
    elif is_supervised and clean_df is not None and target_col not in clean_df.columns:
        overall_status = PipelineStatus.FAILED
        overall_reason = f"Supervised target column '{target_col}' not found in dataset"
    else:
        overall_status = PipelineStatus.SUCCESS
        overall_reason = None

    # -----------------------------------------------------------------------
    # Step 8: Assemble, Validate Pydantic Schema, & Verify JSON Serialization
    # -----------------------------------------------------------------------
    contract_data = {
        "schema_version": "1.0",
        "status": overall_status,
        "reason": overall_reason,
        "task": task_type,
        "target_column": target_col,
        "dataset_context": sanitize_for_json(dataset_context),
        "validation": validation_report,
        "statistical_analysis": stats_report,
        "models": benchmarks,
        "winner": champion,
        "error_analysis": error_report,
        "experiments": experiments_records,
        "ensemble": ensemble_report,
        "time_analysis": time_report,
    }

    try:
        contract = AnalysisContract.model_validate(contract_data)
    except Exception as exc:
        return AnalysisContract(
            schema_version="1.0",
            status=PipelineStatus.FAILED,
            reason=f"AnalysisContract Pydantic validation failed: {str(exc)}",
            task=task_type,
            target_column=target_col,
        )

    # Strict JSON serialization guarantee
    try:
        contract.model_dump_json()
        contract.model_dump()
    except Exception as exc:
        return AnalysisContract(
            schema_version="1.0",
            status=PipelineStatus.FAILED,
            reason=f"AnalysisContract JSON serialization check failed: {str(exc)}",
            task=task_type,
            target_column=target_col,
        )

    return contract
