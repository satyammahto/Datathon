"""
DashboardSpec Schema (Person 4 - Product & Presentation Layer)
Defines the presentation-layer contract consumed by the AIDA frontend.
Decouples visual presentation from internal execution details.
Contains presentation data only; never recalculates analytical truth.
"""
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field, ConfigDict
from backend.app.schemas.enums import PipelineStatus, MLTaskType


class KPIItem(BaseModel):
    id: str
    label: str
    value: str
    subtext: Optional[str] = None
    status: str = "normal"  # normal, success, warning, alert
    icon: Optional[str] = None


class ChartSeries(BaseModel):
    name: str
    data: List[Any]
    color: Optional[str] = None


class ChartSpec(BaseModel):
    chart_id: str
    title: str
    chart_type: str  # bar, line, scatter, heatmap, confusion_matrix, residual_plot
    description: Optional[str] = None
    x_axis_label: Optional[str] = None
    y_axis_label: Optional[str] = None
    categories: List[str] = Field(default_factory=list)
    series: List[ChartSeries] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    status: str = "SUCCESS"
    reason: Optional[str] = None


class InsightCardSpec(BaseModel):
    insight_id: str
    rank: Optional[int] = None
    claim: str
    importance: float = 0.5
    confidence: str = "MEDIUM"  # HIGH, MEDIUM, LOW, REJECTED
    verification_status: str = "VERIFIED"  # VERIFIED, REJECTED, CONTRADICTED
    method: str
    evidence_highlights: Dict[str, Any] = Field(default_factory=dict)
    source_paths: List[str] = Field(default_factory=list)
    critic_summary: Optional[str] = None
    critic_issues: List[str] = Field(default_factory=list)
    provenance_chain: List[Dict[str, Any]] = Field(default_factory=list)


class ModelLeaderboardRow(BaseModel):
    rank: int
    model_name: str
    primary_metric_name: str
    primary_metric_score: float
    secondary_metrics: Dict[str, float] = Field(default_factory=dict)
    fold_scores: List[float] = Field(default_factory=list)
    status: str = "SUCCESS"
    is_champion: bool = False


class ModelChampionshipSpec(BaseModel):
    task_type: str
    target_column: Optional[str] = None
    validation_strategy: str
    folds: int
    primary_metric: str
    champion_model_name: Optional[str] = None
    champion_score: Optional[float] = None
    selection_rationale: Optional[str] = None
    leaderboard: List[ModelLeaderboardRow] = Field(default_factory=list)
    ensemble_result: Dict[str, Any] = Field(default_factory=dict)
    status: str = "SUCCESS"
    reason: Optional[str] = None


class ErrorAnalysisSpec(BaseModel):
    task_type: str
    status: str = "SUCCESS"
    reason: Optional[str] = None
    # Classification
    confusion_matrix: Optional[List[List[int]]] = None
    class_labels: List[str] = Field(default_factory=list)
    class_metrics: List[Dict[str, Any]] = Field(default_factory=list)
    weak_classes: List[Dict[str, Any]] = Field(default_factory=list)
    # Regression
    residual_summary: Dict[str, Any] = Field(default_factory=dict)
    bias_diagnostic: Dict[str, Any] = Field(default_factory=dict)
    heteroscedasticity: Dict[str, Any] = Field(default_factory=dict)
    # Both
    subgroup_slices: List[Dict[str, Any]] = Field(default_factory=list)
    worst_predictions: List[Dict[str, Any]] = Field(default_factory=list)


class DataQualitySpec(BaseModel):
    row_count: int
    column_count: int
    memory_mb: float = 0.0
    column_types: Dict[str, str] = Field(default_factory=dict)
    missing_value_summary: Dict[str, int] = Field(default_factory=dict)
    duplicate_rows: int = 0
    constant_columns: List[str] = Field(default_factory=list)
    leakage_risks: List[Dict[str, Any]] = Field(default_factory=list)
    routing_decisions: Dict[str, bool] = Field(default_factory=dict)
    target_candidates: List[str] = Field(default_factory=list)


class ExecutiveSummarySpec(BaseModel):
    title: str
    overview: str
    key_takeaways: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    verified_insight_count: int = 0
    high_confidence_count: int = 0


class ExecutiveReportSpec(BaseModel):
    title: str
    generated_at: str
    dataset_name: str
    dataset_overview: Dict[str, Any] = Field(default_factory=dict)
    data_quality_summary: Dict[str, Any] = Field(default_factory=dict)
    analytical_routing: Dict[str, Any] = Field(default_factory=dict)
    key_verified_insights: List[Dict[str, Any]] = Field(default_factory=list)
    statistical_findings: List[Dict[str, Any]] = Field(default_factory=list)
    model_championship_summary: Dict[str, Any] = Field(default_factory=dict)
    error_analysis_summary: Dict[str, Any] = Field(default_factory=dict)
    controlled_experiments_summary: Dict[str, Any] = Field(default_factory=dict)
    warnings_and_risks: List[str] = Field(default_factory=list)
    executive_recommendations: List[str] = Field(default_factory=list)
    provenance_evidence_summary: List[Dict[str, Any]] = Field(default_factory=list)


class DashboardSpec(BaseModel):
    """
    Top-level presentation specification contract for the AIDA user interface.
    """
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    schema_version: str = "1.0"
    dashboard_id: str
    dataset_name: str
    task_type: str
    target_column: Optional[str] = None
    status: PipelineStatus = PipelineStatus.SUCCESS
    reason: Optional[str] = None

    kpis: List[KPIItem] = Field(default_factory=list)
    executive_summary: ExecutiveSummarySpec = Field(default_factory=lambda: ExecutiveSummarySpec(title="", overview=""))
    insights: List[InsightCardSpec] = Field(default_factory=list)
    rejected_insights: List[Dict[str, Any]] = Field(default_factory=list)
    model_championship: ModelChampionshipSpec
    error_analysis: ErrorAnalysisSpec
    data_quality: DataQualitySpec
    charts: List[ChartSpec] = Field(default_factory=list)
    report: ExecutiveReportSpec
