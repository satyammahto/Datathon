"""
Next-Best-Investigation Planner (Person 3 - Phase 3.0)
Selects the highest-value uninvestigated candidate hypothesis, maps it to an allowlisted
action, prevents duplicate investigations, and terminates when candidates are exhausted.
Zero arbitrary code generation; 100% deterministic fallback.
"""
from typing import List, Optional, Set, Dict, Any
import os

from backend.app.schemas.insight_contract import (
    CandidateHypothesis,
    InvestigationAction,
)
from backend.app.agents.investigation import ALLOWLISTED_ACTIONS


def plan_next_investigation(
    candidate_hypotheses: List[CandidateHypothesis],
    investigated_candidate_ids: Set[str],
    action_counter: int = 1,
    min_priority_threshold: float = 0.20,
    use_llm: bool = False,
) -> Optional[InvestigationAction]:
    """
    Plan the next highest-priority investigation action.
    Guarantees:
    - Only uninvestigated candidates are selected
    - Only allowlisted actions are generated
    - Prevents duplicates
    - Stops when candidates fall below priority threshold
    """
    # Filter uninvestigated candidates above threshold
    eligible = [
        c for c in candidate_hypotheses
        if c.candidate_id not in investigated_candidate_ids
        and not c.investigated
        and c.priority >= min_priority_threshold
    ]

    if not eligible:
        return None

    # Sort descending by priority
    eligible_sorted = sorted(eligible, key=lambda c: c.priority, reverse=True)
    best_candidate = eligible_sorted[0]

    # Map candidate to allowlisted action type
    act_type = best_candidate.investigation_type
    if act_type not in ALLOWLISTED_ACTIONS:
        act_type = "verify_candidate"

    action = InvestigationAction(
        action_id=f"act_{action_counter:03d}_{act_type}",
        action_type=act_type,
        candidate_id=best_candidate.candidate_id,
        target_feature=best_candidate.feature,
        target_group=best_candidate.group,
        parameters=best_candidate.evidence,
        reasoning=f"Selected highest-priority candidate '{best_candidate.candidate_id}' (priority={best_candidate.priority:.2f}).",
    )

    if not use_llm or not os.environ.get("OPENAI_API_KEY"):
        return action

    # Optional LLM-guided prioritization
    try:
        # If LLM reasoning were to re-rank among eligible candidates, validate strictly:
        # 1. Action type must be in ALLOWLISTED_ACTIONS
        # 2. Candidate ID must be in [c.candidate_id for c in eligible]
        return action
    except Exception:
        return action
