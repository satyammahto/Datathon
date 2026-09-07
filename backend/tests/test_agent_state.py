"""
Tests for LangGraph AIDAState (Person 3 - Phase 3.0)
Verifies state initialization, typed fields, iteration bounding, and error handling.
"""
import pytest
from backend.app.schemas.analysis_contract import AnalysisContract
from backend.app.schemas.enums import PipelineStatus, MLTaskType
from backend.app.agents.state import initialize_aida_state, AIDAState


def test_state_initialization():
    """Verify clean initialization of AIDAState from an AnalysisContract."""
    analysis = AnalysisContract(
        schema_version="1.0",
        status=PipelineStatus.SUCCESS,
        task=MLTaskType.CLASSIFICATION,
    )
    state = initialize_aida_state(analysis, max_iterations=5)

    assert state["status"] == "INITIALIZED"
    assert state["iteration"] == 0
    assert state["max_iterations"] == 5
    assert state["candidate_hypotheses"] == []
    assert state["investigation_queue"] == []
    assert state["completed_investigations"] == []
    assert state["validated_insights"] == []
    assert state["rejected_insights"] == []
    assert state["errors"] == []
    assert state["analysis"].status == PipelineStatus.SUCCESS


def test_state_iteration_limit_and_mutation():
    """Verify iteration bounding and safe state updates."""
    analysis = AnalysisContract(
        schema_version="1.0",
        status=PipelineStatus.SUCCESS,
    )
    state = initialize_aida_state(analysis, max_iterations=3)

    state["iteration"] += 1
    assert state["iteration"] == 1
    assert state["iteration"] <= state["max_iterations"]

    # Append errors without mutating analysis contract
    state["errors"].append("Transient warning")
    assert len(state["errors"]) == 1
    assert state["analysis"].status == PipelineStatus.SUCCESS
