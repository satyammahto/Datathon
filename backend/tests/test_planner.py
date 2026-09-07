"""
Tests for Next-Best-Investigation Planner (Person 3 - Phase 3.0)
Verifies priority selection, allowlist enforcement, duplicate prevention, and termination.
"""
import pytest
from backend.app.schemas.insight_contract import CandidateHypothesis, InvestigationAction
from backend.app.agents.planner import plan_next_investigation
from backend.app.agents.investigation import ALLOWLISTED_ACTIONS


def test_planner_selects_highest_priority():
    """Verify that planner strictly selects the highest priority candidate."""
    cands = [
        CandidateHypothesis(
            candidate_id="cand_low",
            claim="Low priority finding",
            source_path="path1",
            priority=0.35,
            investigation_type="verify_correlation",
        ),
        CandidateHypothesis(
            candidate_id="cand_high",
            claim="High priority slice finding",
            source_path="path2",
            priority=0.88,
            investigation_type="verify_error_slice",
            feature="region",
            group="South",
        ),
        CandidateHypothesis(
            candidate_id="cand_med",
            claim="Medium priority ablation finding",
            source_path="path3",
            priority=0.62,
            investigation_type="verify_ablation",
            feature="income",
        ),
    ]

    action = plan_next_investigation(
        candidate_hypotheses=cands,
        investigated_candidate_ids=set(),
    )

    assert action is not None
    assert action.candidate_id == "cand_high"
    assert action.action_type == "verify_error_slice"
    assert action.target_feature == "region"
    assert action.target_group == "South"


def test_planner_duplicate_prevention():
    """Verify that investigated candidates are never scheduled again."""
    cands = [
        CandidateHypothesis(
            candidate_id="cand_high",
            claim="High priority finding",
            source_path="path",
            priority=0.90,
            investigation_type="verify_ablation",
        ),
        CandidateHypothesis(
            candidate_id="cand_next",
            claim="Next finding",
            source_path="path",
            priority=0.75,
            investigation_type="verify_residual_bias",
        ),
    ]

    # First investigation
    action1 = plan_next_investigation(
        candidate_hypotheses=cands,
        investigated_candidate_ids=set(),
    )
    assert action1.candidate_id == "cand_high"

    # Second investigation with cand_high marked investigated
    action2 = plan_next_investigation(
        candidate_hypotheses=cands,
        investigated_candidate_ids={"cand_high"},
    )
    assert action2.candidate_id == "cand_next"
    assert action2.action_type == "verify_residual_bias"


def test_planner_allowlist_enforcement():
    """Verify that action types are strictly within ALLOWLISTED_ACTIONS."""
    cands = [
        CandidateHypothesis(
            candidate_id="cand_arbitrary",
            claim="Arbitrary tool attempt",
            source_path="path",
            priority=0.80,
            investigation_type="unauthorized_eval_code",
        ),
    ]

    action = plan_next_investigation(cands, set())
    assert action is not None
    assert action.action_type in ALLOWLISTED_ACTIONS
    assert action.action_type == "verify_candidate"  # mapped to safe fallback


def test_planner_terminates_when_exhausted():
    """Verify that planner returns None when all candidates are investigated or below threshold."""
    cands = [
        CandidateHypothesis(
            candidate_id="cand_1",
            claim="Low priority candidate",
            source_path="path",
            priority=0.10,  # Below default min_priority_threshold=0.20
            investigation_type="verify_correlation",
        )
    ]

    action = plan_next_investigation(cands, set())
    assert action is None

    # Empty candidates
    assert plan_next_investigation([], set()) is None
