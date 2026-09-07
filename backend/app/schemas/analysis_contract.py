from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field, ConfigDict
from backend.app.schemas.enums import PipelineStatus, MLTaskType, ValidationStrategy


class ComponentStatus(BaseModel):
    model_config = ConfigDict(extra="allow")

    status: PipelineStatus = PipelineStatus.SUCCESS
    reason: Optional[str] = None


class ValidationReport(ComponentStatus):
    strategy: Optional[str] = ValidationStrategy.STRATIFIED_K_FOLD.value
    folds: Optional[int] = 5
    primary_metric: Optional[str] = "f1"
    random_seed: Optional[int] = 42
    split_details: Dict[str, Any] = Field(default_factory=dict)


class StatisticalReport(ComponentStatus):
    numeric_summary: Dict[str, Any] = Field(default_factory=dict)
    categorical_summary: Dict[str, Any] = Field(default_factory=dict)
    distributions: Dict[str, Any] = Field(default_factory=dict)
    outliers: Dict[str, Any] = Field(default_factory=dict)
    correlations: Dict[str, Any] = Field(default_factory=dict)
    grouped_analysis: List[Dict[str, Any]] = Field(default_factory=list)
    statistical_tests: List[Dict[str, Any]] = Field(default_factory=list)
    effect_sizes: List[Dict[str, Any]] = Field(default_factory=list)
    confidence_intervals: List[Dict[str, Any]] = Field(default_factory=list)
    multiple_testing: Dict[str, Any] = Field(default_factory=lambda: {"comparisons": 0, "correction": None})
    summary_stats: Dict[str, Any] = Field(default_factory=dict)
    distribution_checks: Dict[str, Any] = Field(default_factory=dict)


class ModelBenchmarkResult(BaseModel):
    model_config = ConfigDict(extra="allow")

    model_name: str
    name: Optional[str] = None
    family: str = "unknown"
    status: PipelineStatus = PipelineStatus.SUCCESS
    reason: Optional[str] = None
    primary_metric: Optional[str] = None
    mean_score: Optional[float] = None
    std_score: Optional[float] = None
    fold_scores: List[float] = Field(default_factory=list)
    metrics: Dict[str, float] = Field(default_factory=dict)
    rank: Optional[int] = None
    train_time_seconds: Optional[float] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)
    oof_predictions: Optional[List[Any]] = None
    oof_indices: Optional[List[int]] = None

    def model_post_init(self, __context: Any) -> None:
        if self.name is None:
            self.name = self.model_name
        if not self.model_name:
            self.model_name = self.name or ""


class ChampionResult(ComponentStatus):
    model_name: Optional[str] = None
    name: Optional[str] = None
    primary_metric: Optional[str] = None
    metric: Optional[str] = None
    score: Optional[float] = None
    empirical_evidence: Dict[str, Any] = Field(default_factory=dict)
    selection_rule: Optional[str] = None
    selection_rationale: Optional[str] = None

    def model_post_init(self, __context: Any) -> None:
        if self.name is None and self.model_name is not None:
            self.name = self.model_name
        if self.model_name is None and self.name is not None:
            self.model_name = self.name
        if self.metric is None and self.primary_metric is not None:
            self.metric = self.primary_metric
        if self.primary_metric is None and self.metric is not None:
            self.primary_metric = self.metric
        if self.selection_rule is None and self.selection_rationale is not None:
            self.selection_rule = self.selection_rationale
        if self.selection_rationale is None and self.selection_rule is not None:
            self.selection_rationale = self.selection_rule


class ErrorAnalysisReport(ComponentStatus):
    model: Dict[str, Any] = Field(default_factory=dict)
    summary: Dict[str, Any] = Field(default_factory=dict)
    classification: Dict[str, Any] = Field(default_factory=dict)
    regression: Dict[str, Any] = Field(default_factory=dict)
    subgroup_analysis: List[Dict[str, Any]] = Field(default_factory=list)
    feature_error_associations: List[Dict[str, Any]] = Field(default_factory=list)
    robustness: Dict[str, Any] = Field(default_factory=dict)

    # Backwards compatibility fields:
    residual_summary: Dict[str, Any] = Field(default_factory=dict)
    worst_predictions: List[Dict[str, Any]] = Field(default_factory=list)
    confusion_matrix: Optional[List[List[int]]] = None
    high_loss_segments: List[Dict[str, Any]] = Field(default_factory=list)


class ExperimentRecord(BaseModel):
    model_config = ConfigDict(extra="allow")

    experiment_id: str
    experiment_type: str = "feature_ablation"
    hypothesis: str = ""
    feature: Optional[str] = None
    status: PipelineStatus = PipelineStatus.SUCCESS
    reason: Optional[str] = None
    metric: Optional[str] = None
    baseline_metric: Optional[float] = None
    baseline_score: Optional[float] = None
    candidate_metric: Optional[float] = None
    experiment_score: Optional[float] = None
    delta: Optional[float] = None
    interpretation: Optional[str] = None
    accepted: bool = False
    details: Dict[str, Any] = Field(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        if self.baseline_score is None and self.baseline_metric is not None:
            self.baseline_score = self.baseline_metric
        if self.baseline_metric is None and self.baseline_score is not None:
            self.baseline_metric = self.baseline_score
        if self.experiment_score is None and self.candidate_metric is not None:
            self.experiment_score = self.candidate_metric
        if self.candidate_metric is None and self.experiment_score is not None:
            self.candidate_metric = self.experiment_score


class EnsembleReport(ComponentStatus):
    ensemble_type: Optional[str] = None
    member_models: List[str] = Field(default_factory=list)
    models: List[str] = Field(default_factory=list)
    champion_score: Optional[float] = None
    ensemble_score: Optional[float] = None
    score: Optional[float] = None
    improvement_over_champion: Optional[float] = None
    delta: Optional[float] = None
    improves_champion: bool = False
    details: Dict[str, Any] = Field(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        if not self.models and self.member_models:
            self.models = self.member_models
        if not self.member_models and self.models:
            self.member_models = self.models
        if self.score is None and self.ensemble_score is not None:
            self.score = self.ensemble_score
        if self.ensemble_score is None and self.score is not None:
            self.ensemble_score = self.score
        if self.delta is None and self.improvement_over_champion is not None:
            self.delta = self.improvement_over_champion
        if self.improvement_over_champion is None and self.delta is not None:
            self.improvement_over_champion = self.delta


class TimeAnalysisReport(ComponentStatus):
    seasonality_detected: Optional[bool] = None
    stationarity_test_passed: Optional[bool] = None
    trend_type: Optional[str] = None
    autocorrelation_summary: Dict[str, Any] = Field(default_factory=dict)


class AnalysisContract(BaseModel):
    """
    Standard output contract produced by Person 2 (ML / Accuracy Brain)
    for consumption by Person 3 (Agent / Reasoning Brain).
    Completely decoupled from internal model instances.
    """
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    schema_version: str = "1.0"
    status: PipelineStatus = PipelineStatus.SUCCESS
    reason: Optional[str] = None
    task: MLTaskType = MLTaskType.CLASSIFICATION
    target_column: Optional[str] = None
    dataset_context: Dict[str, Any] = Field(default_factory=dict)

    validation: ValidationReport = Field(default_factory=ValidationReport)
    statistical_analysis: StatisticalReport = Field(default_factory=StatisticalReport)
    models: List[ModelBenchmarkResult] = Field(default_factory=list)
    winner: Optional[ChampionResult] = None
    error_analysis: ErrorAnalysisReport = Field(default_factory=ErrorAnalysisReport)
    experiments: List[ExperimentRecord] = Field(default_factory=list)
    ensemble: EnsembleReport = Field(default_factory=EnsembleReport)
    time_analysis: TimeAnalysisReport = Field(default_factory=TimeAnalysisReport)
