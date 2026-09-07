"""
Agents Package (Person 3 - Autonomous Reasoning & Investigation Brain)
Exports AIDAState, LangGraph graph builder, planner, investigation dispatcher,
critic, and verifier.
"""
from backend.app.agents.state import AIDAState, initialize_aida_state
from backend.app.agents.planner import plan_next_investigation
from backend.app.agents.investigation import (
    ALLOWLISTED_ACTIONS,
    execute_investigation,
)
from backend.app.agents.critic import evaluate_adversarial_critic
from backend.app.agents.verifier import verify_numerical_claims
from backend.app.agents.graph import (
    build_aida_reasoning_graph,
    run_aida_reasoning,
)

__all__ = [
    "AIDAState",
    "initialize_aida_state",
    "plan_next_investigation",
    "ALLOWLISTED_ACTIONS",
    "execute_investigation",
    "evaluate_adversarial_critic",
    "verify_numerical_claims",
    "build_aida_reasoning_graph",
    "run_aida_reasoning",
]
