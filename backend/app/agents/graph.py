"""
LangGraph Autonomous Reasoning Graph (Person 3 - Phase 3.0)
Assembles and compiles AIDA's multi-step investigation loop:
discover -> plan -> investigate -> critic -> verify -> decide_continue -> rank -> summarize -> END
Bounded iterations, early stopping, zero infinite loops.
"""
from typing import Dict, Any, List, Optional
from langgraph.graph import StateGraph, END

from backend.app.schemas.analysis_contract import AnalysisContract
from backend.app.schemas.insight_contract import (
    InsightContract,
    VerifiedInsight,
    VerificationStatus,
    ConfidenceLevel,
)
from backend.app.schemas.enums import PipelineStatus
from backend.app.agents.state import AIDAState, initialize_aida_state
from backend.app.pipeline.insight_discovery import discover_candidate_hypotheses
from backend.app.agents.planner import plan_next_investigation
from backend.app.agents.investigation import execute_investigation
from backend.app.agents.critic import evaluate_adversarial_critic
from backend.app.agents.verifier import verify_numerical_claims
from backend.app.pipeline.confidence import evaluate_insight_confidence
from backend.app.pipeline.insight_ranking import rank_verified_insights
from backend.app.pipeline.insight_validator import build_provenance_graph
from backend.app.pipeline.summary import generate_executive_summary


# ---------------------------------------------------------------------------
# Node 1: Discover Candidates
# ---------------------------------------------------------------------------
def discover_node(state: AIDAState) -> Dict[str, Any]:
    analysis = state["analysis"]
    candidates = discover_candidate_hypotheses(analysis)
    return {
        "candidate_hypotheses": candidates,
        "status": "DISCOVERY_COMPLETE",
    }


# ---------------------------------------------------------------------------
# Node 2: Plan Next Investigation
# ---------------------------------------------------------------------------
def plan_node(state: AIDAState) -> Dict[str, Any]:
    candidates = state.get("candidate_hypotheses", [])
    completed_actions = state.get("completed_investigations", [])
    investigated_ids = {
        ins.insight_id for ins in state.get("validated_insights", [])
    }
    for rej in state.get("rejected_insights", []):
        if "candidate_id" in rej:
            investigated_ids.add(rej["candidate_id"])

    current_iter = state.get("iteration", 0) + 1
    action = plan_next_investigation(
        candidate_hypotheses=candidates,
        investigated_candidate_ids=investigated_ids,
        action_counter=current_iter,
        min_priority_threshold=0.20,
    )

    return {
        "current_action": action,
        "iteration": current_iter,
        "status": "PLANNING_COMPLETE",
    }


# ---------------------------------------------------------------------------
# Node 3: Execute Investigation
# ---------------------------------------------------------------------------
def investigate_node(state: AIDAState) -> Dict[str, Any]:
    action = state.get("current_action")
    analysis = state["analysis"]

    if not action:
        return {"status": "NO_ACTION"}

    evidence = execute_investigation(action, analysis)
    completed = list(state.get("completed_investigations", []))
    completed.append(evidence)

    return {
        "completed_investigations": completed,
        "status": "INVESTIGATION_COMPLETE",
    }


# ---------------------------------------------------------------------------
# Node 4: Adversarial Critic
# ---------------------------------------------------------------------------
def critic_node(state: AIDAState) -> Dict[str, Any]:
    action = state.get("current_action")
    if not action:
        return {"status": "NO_ACTION"}

    candidates = state.get("candidate_hypotheses", [])
    cand_match = next((c for c in candidates if c.candidate_id == action.candidate_id), None)

    completed = state.get("completed_investigations", [])
    latest_evidence = completed[-1] if completed else None

    if not cand_match:
        return {"status": "CANDIDATE_NOT_FOUND"}

    critic_report = evaluate_adversarial_critic(
        candidate=cand_match,
        evidence=latest_evidence,
        analysis=state.get("analysis"),
    )

    findings = list(state.get("critic_findings", []))
    findings.append(critic_report)

    return {
        "critic_findings": findings,
        "status": "CRITIQUE_COMPLETE",
    }


# ---------------------------------------------------------------------------
# Node 5: Numerical Verifier Gate
# ---------------------------------------------------------------------------
def verify_node(state: AIDAState) -> Dict[str, Any]:
    action = state.get("current_action")
    if not action:
        return {"status": "NO_ACTION"}

    candidates = state.get("candidate_hypotheses", [])
    cand_match = next((c for c in candidates if c.candidate_id == action.candidate_id), None)
    if not cand_match:
        return {"status": "CANDIDATE_NOT_FOUND"}

    completed = state.get("completed_investigations", [])
    latest_evidence = completed[-1] if completed else None

    critic_findings = state.get("critic_findings", [])
    latest_critic = critic_findings[-1] if critic_findings else None

    ver_report = verify_numerical_claims(
        candidate=cand_match,
        investigation_evidence=latest_evidence,
        analysis=state["analysis"],
    )

    ver_results = list(state.get("verification_results", []))
    ver_results.append(ver_report)

    validated = list(state.get("validated_insights", []))
    rejected = list(state.get("rejected_insights", []))

    # Mark candidate investigated
    cand_match.investigated = True

    if ver_report.status == VerificationStatus.VERIFIED:
        # Evaluate confidence
        conf = evaluate_insight_confidence(
            verification=ver_report,
            critic=latest_critic,
            sample_size=latest_evidence.sample_size if latest_evidence else None,
            statistical_support=latest_evidence.statistical_support if latest_evidence else True,
        )

        insight = VerifiedInsight(
            insight_id=cand_match.candidate_id,
            claim=cand_match.claim,
            importance=cand_match.priority,
            confidence=conf,
            evidence=latest_evidence.evidence if latest_evidence else cand_match.evidence,
            source_paths=latest_evidence.source_paths if latest_evidence and latest_evidence.source_paths else [cand_match.source_path],
            method=latest_evidence.method if latest_evidence else cand_match.investigation_type,
            critic=latest_critic,
            verification=ver_report,
        )
        validated.append(insight)
    else:
        rejected.append({
            "candidate_id": cand_match.candidate_id,
            "claim": cand_match.claim,
            "status": ver_report.status.value,
            "rejection_reason": ver_report.verification_notes,
            "discrepancies": ver_report.discrepancies,
        })

    return {
        "verification_results": ver_results,
        "validated_insights": validated,
        "rejected_insights": rejected,
        "status": "VERIFICATION_COMPLETE",
    }


# ---------------------------------------------------------------------------
# Node 6: Decide Continue (Conditional Edge Evaluator)
# ---------------------------------------------------------------------------
def decide_continue(state: AIDAState) -> str:
    iteration = state.get("iteration", 0)
    max_iters = state.get("max_iterations", 5)

    if iteration >= max_iters:
        return "rank"

    # Check if any uninvestigated candidates above priority threshold remain
    candidates = state.get("candidate_hypotheses", [])
    investigated_ids = {
        ins.insight_id for ins in state.get("validated_insights", [])
    }
    for rej in state.get("rejected_insights", []):
        if "candidate_id" in rej:
            investigated_ids.add(rej["candidate_id"])

    has_more = any(
        c.candidate_id not in investigated_ids and not c.investigated and c.priority >= 0.20
        for c in candidates
    )

    if has_more:
        return "plan"
    return "rank"


# ---------------------------------------------------------------------------
# Node 7: Rank Insights
# ---------------------------------------------------------------------------
def rank_node(state: AIDAState) -> Dict[str, Any]:
    raw_validated = state.get("validated_insights", [])
    ranked = rank_verified_insights(raw_validated)
    return {
        "validated_insights": ranked,
        "status": "RANKING_COMPLETE",
    }


# ---------------------------------------------------------------------------
# Node 8: Executive Summary & Graph Assembly
# ---------------------------------------------------------------------------
def summarize_node(state: AIDAState) -> Dict[str, Any]:
    insights = state.get("validated_insights", [])
    analysis = state.get("analysis")
    dataset_context = analysis.dataset_context if analysis else {}

    summary = generate_executive_summary(
        insights=insights,
        dataset_context=dataset_context,
        use_llm=False,
    )
    return {
        "summary": summary,
        "status": "COMPLETE",
    }


# ---------------------------------------------------------------------------
# Master Graph Construction
# ---------------------------------------------------------------------------
def build_aida_reasoning_graph() -> Any:
    """Build and compile the LangGraph StateGraph for AIDA's autonomous investigation loop."""
    workflow = StateGraph(AIDAState)

    workflow.add_node("discover", discover_node)
    workflow.add_node("plan", plan_node)
    workflow.add_node("investigate", investigate_node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("verify", verify_node)
    workflow.add_node("rank", rank_node)
    workflow.add_node("summarize", summarize_node)

    workflow.set_entry_point("discover")
    workflow.add_edge("discover", "plan")
    workflow.add_edge("plan", "investigate")
    workflow.add_edge("investigate", "critic")
    workflow.add_edge("critic", "verify")

    workflow.add_conditional_edges(
        "verify",
        decide_continue,
        {
            "plan": "plan",
            "rank": "rank",
        },
    )

    workflow.add_edge("rank", "summarize")
    workflow.add_edge("summarize", END)

    return workflow.compile()


def run_aida_reasoning(
    analysis: AnalysisContract,
    max_iterations: int = 5,
) -> InsightContract:
    """
    Master entrypoint to execute Person 3's Autonomous Reasoning & Investigation Brain.
    Produces an immutable, verified InsightContract for Person 4 / UI.
    """
    initial_state = initialize_aida_state(analysis, max_iterations=max_iterations)
    graph = build_aida_reasoning_graph()

    # Execute graph
    final_state = graph.invoke(initial_state)

    validated_insights = final_state.get("validated_insights", [])
    rejected_insights = final_state.get("rejected_insights", [])
    candidates = final_state.get("candidate_hypotheses", [])
    completed_inv = final_state.get("completed_investigations", [])
    summary = final_state.get("summary", {})

    provenance_graph = build_provenance_graph(validated_insights)

    return InsightContract(
        schema_version="1.0",
        status=PipelineStatus.SUCCESS if analysis.status == PipelineStatus.SUCCESS else analysis.status,
        reason=analysis.reason,
        dataset_context=analysis.dataset_context,
        candidate_count=len(candidates),
        investigations_count=len(completed_inv),
        insights=validated_insights,
        rejected_insights=rejected_insights,
        executive_summary=summary,
        provenance_graph=provenance_graph,
        metadata={
            "iterations_executed": final_state.get("iteration", 0),
            "max_iterations": max_iterations,
            "status": final_state.get("status", "COMPLETE"),
        },
    )
