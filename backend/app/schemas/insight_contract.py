"""
Insight Contract Schemas (Person 3 - Autonomous Reasoning & Investigation Brain)
Defines Pydantic v2 schemas for candidate hypotheses, investigation actions,
investigation evidence, critic reports, verification reports, verified insights,
and the final InsightContract.
Purely JSON serializable; zero model/pipeline object leakage.
"""
from typing import Dict, List, Any, Optional
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict
from backend.app.schemas.enums import PipelineStatus


class ConfidenceLevel(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    REJECTED = "REJECTED"


class VerificationStatus(str, Enum):
    VERIFIED = "VERIFIED"
    REJECTED = "REJECTED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    CONTRADICTED = "CONTRADICTED"


class CriticStatus(str, Enum):
    PASSED = "PASSED"
    CHALLENGED = "CHALLENGED"
    REJECTED = "REJECTED"


class CandidateHypothesis(BaseModel):
    """
    A potential insight discovered deterministically from the AnalysisContract.
    Contains authoritative source path, initial evidence, and an empirical priority score.
    """
    model_config = ConfigDict(extra="allow")

    candidate_id: str
    claim: str
    source_path: str
    evidence: Dict[str, Any] = Field(default_factory=dict)
    priority: float = 0.5
    investigation_type: str = "verify_candidate"
    feature: Optional[str] = None
    group: Optional[str] = None
    investigated: bool = False


class InvestigationAction(BaseModel):
    """
    A structured, allowlisted investigation action scheduled by the planner.
    Strictly constrained; no arbitrary code or tool execution.
    """
    model_config = ConfigDict(extra="allow")

    action_id: str
    action_type: str
    candidate_id: str
    target_feature: Optional[str] = None
    target_group: Optional[str] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)
    reasoning: Optional[str] = None


class InvestigationEvidence(BaseModel):
    """
    Empirical output returned by a deterministic investigation tool.
    Never modifies AnalysisContract.
    """
    model_config = ConfigDict(extra="allow")

    claim: str
    method: str
    evidence: Dict[str, Any] = Field(default_factory=dict)
    sample_size: Optional[int] = None
    statistical_support: bool = False
    source_paths: List[str] = Field(default_factory=list)
    status: str = "COMPLETED"
    notes: Optional[str] = None


class CriticReport(BaseModel):
    """
    Adversarial critique report evaluating candidate validity across 10 safety checks.
    """
    model_config = ConfigDict(extra="allow")

    status: CriticStatus = CriticStatus.PASSED
    issues: List[str] = Field(default_factory=list)
    severity: str = "NONE"  # NONE, LOW, MEDIUM, HIGH
    critique_summary: str = ""
    checks_evaluated: Dict[str, bool] = Field(default_factory=dict)


class VerificationReport(BaseModel):
    """
    Hard numerical gate verifying cited numbers and directions against AnalysisContract.
    """
    model_config = ConfigDict(extra="allow")

    status: VerificationStatus = VerificationStatus.VERIFIED
    checks_passed: List[str] = Field(default_factory=list)
    checks_failed: List[str] = Field(default_factory=list)
    discrepancies: List[Dict[str, Any]] = Field(default_factory=list)
    verification_notes: Optional[str] = None


class VerifiedInsight(BaseModel):
    """
    A fully vetted, verified analytical insight with end-to-end numerical provenance.
    """
    model_config = ConfigDict(extra="allow")

    insight_id: str
    claim: str
    importance: float = 0.5
    confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM
    evidence: Dict[str, Any] = Field(default_factory=dict)
    source_paths: List[str] = Field(default_factory=list)
    method: str = "deterministic_analysis"
    critic: CriticReport = Field(default_factory=CriticReport)
    verification: VerificationReport = Field(default_factory=VerificationReport)
    rank: Optional[int] = None
    rank_score: Optional[float] = None
    ranking_rationale: Optional[str] = None


class InsightContract(BaseModel):
    """
    Master contract produced by Person 3 (Autonomous Reasoning & Investigation Brain).
    Provides structured, ranked, and verified insights for consumption by the UI/Dashboard.
    """
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    schema_version: str = "1.0"
    status: PipelineStatus = PipelineStatus.SUCCESS
    reason: Optional[str] = None
    dataset_context: Dict[str, Any] = Field(default_factory=dict)
    candidate_count: int = 0
    investigations_count: int = 0
    insights: List[VerifiedInsight] = Field(default_factory=list)
    rejected_insights: List[Dict[str, Any]] = Field(default_factory=list)
    executive_summary: Dict[str, Any] = Field(default_factory=dict)
    provenance_graph: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
