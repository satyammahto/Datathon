"""
LangGraph State Definition (Person 3 - Phase 3.0)
Defines the typed graph state for AIDA's Autonomous Reasoning & Investigation loop.
Strictly JSON-safe; no DataFrames, estimators, or un-serializable objects.
"""
from typing import TypedDict, List, Dict, Any, Optional
from backend.app.schemas.analysis_contract import AnalysisContract
from backend.app.schemas.insight_contract import (
    CandidateHypothesis,
    InvestigationAction,
    InvestigationEvidence,
    CriticReport,
    VerificationReport,
    VerifiedInsight,
)


class AIDAState(TypedDict, total=False):
    """
    Typed graph state for LangGraph execution of AIDA's Person 3 reasoning agent.
    """
    analysis: AnalysisContract
    candidate_hypotheses: List[CandidateHypothesis]
    investigation_queue: List[InvestigationAction]
    current_action: Optional[InvestigationAction]
    completed_investigations: List[InvestigationEvidence]
    evidence: List[Dict[str, Any]]
    critic_findings: List[CriticReport]
    verification_results: List[VerificationReport]
    validated_insights: List[VerifiedInsight]
    rejected_insights: List[Dict[str, Any]]
    iteration: int
    max_iterations: int
    status: str
    errors: List[str]
    summary: Dict[str, Any]


def initialize_aida_state(
    analysis: AnalysisContract,
    max_iterations: int = 5,
) -> AIDAState:
    """
    Helper to cleanly construct an initial AIDAState from an AnalysisContract.
    """
    return {
        "analysis": analysis,
        "candidate_hypotheses": [],
        "investigation_queue": [],
        "current_action": None,
        "completed_investigations": [],
        "evidence": [],
        "critic_findings": [],
        "verification_results": [],
        "validated_insights": [],
        "rejected_insights": [],
        "iteration": 0,
        "max_iterations": max_iterations,
        "status": "INITIALIZED",
        "errors": [],
        "summary": {},
    }
