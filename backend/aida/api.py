"""FastAPI Application for AIDA Trust Layer.

Serves verified, deterministic insights and executive summary payloads
to Person 4 (Frontend / UI Dashboard).
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, APIRouter, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

try:
    from backend.aida.contracts.discovery_contract import DiscoveryContract
    from backend.aida.contracts.analysis_contract import AnalysisContract
    from backend.aida.contracts.insight_contract import InsightContract, VerifiedInsight, RejectedInsight
    from backend.aida.pipeline.summary import generate_deterministic_summary
    from backend.aida.graph import run_aida_pipeline
except ImportError:
    from aida.contracts.discovery_contract import DiscoveryContract
    from aida.contracts.analysis_contract import AnalysisContract
    from aida.contracts.insight_contract import InsightContract, VerifiedInsight, RejectedInsight
    from aida.pipeline.summary import generate_deterministic_summary
    from aida.graph import run_aida_pipeline


router = APIRouter(tags=["AIDA Trust Layer"])

app = FastAPI(
    title="AIDA Trust Layer API",
    description="Autonomous Intelligence & Data Analyst - Strictly Verified Insight Engine",
    version="1.0.0",
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalysisRequest(BaseModel):
    """Request model accepting upstream Discovery and Analysis contracts."""
    discovery_contract: Dict[str, Any] = Field(
        ..., description="Raw discovery JSON contract from Person 1"
    )
    analysis_contract: Dict[str, Any] = Field(
        ..., description="ML analysis JSON contract from Person 2"
    )


class InsightsResponse(BaseModel):
    """Unified response model delivered to Person 4 (UI / Dashboard)."""
    dataset_id: str
    status: str
    summary: Dict[str, Any]
    verified_insights: List[VerifiedInsight]
    rejected_insights: List[RejectedInsight]
    pipeline_errors: List[str] = Field(default_factory=list)


@router.get("/health", tags=["Health"])
@app.get("/health", tags=["Health"])
def health_check() -> Dict[str, str]:
    """Connectivity verification endpoint for Person 4."""
    return {"status": "Trust Layer Online"}


@router.post("/v1/insights", response_model=InsightsResponse, tags=["Insights"])
@router.post("/insights", response_model=InsightsResponse, tags=["Insights"])
@app.post("/api/v1/insights", response_model=InsightsResponse, tags=["Insights"])
def generate_insights(request: AnalysisRequest) -> InsightsResponse:
    """Ingest Discovery and Analysis contracts, execute autonomous LangGraph investigation,

    and return strictly verified insights and executive summary.
    """
    errors: List[str] = []
    discovery_obj: Optional[DiscoveryContract] = None
    analysis_obj: Optional[AnalysisContract] = None

    # Parse Discovery Contract safely
    try:
        discovery_obj = DiscoveryContract.model_validate(request.discovery_contract)
    except Exception as exc:
        errors.append(f"Discovery contract validation warning: {str(exc)}")

    # Parse Analysis Contract safely
    try:
        analysis_obj = AnalysisContract.model_validate(request.analysis_contract)
    except Exception as exc:
        errors.append(f"Analysis contract validation warning: {str(exc)}")

    # Prepare initial state for the LangGraph pipeline
    initial_state = {
        "discovery_contract": discovery_obj,
        "analysis_contract": analysis_obj,
        "pipeline_errors": errors,
        "errors": errors,
    }

    # Run the hardened 5-node pipeline
    final_state = run_aida_pipeline(initial_state)

    contract: Optional[InsightContract] = final_state.get("insight_contract")
    verified = final_state.get("verified_insights", [])
    rejected = final_state.get("rejected_insights", [])
    critiques = final_state.get("critique_evaluations", [])
    all_errors = list(final_state.get("pipeline_errors", []))

    dataset_id = (
        discovery_obj.dataset_id
        if discovery_obj
        else (analysis_obj.dataset_id if analysis_obj else "unknown_dataset")
    )

    if contract is None:
        contract = InsightContract(
            dataset_id=dataset_id,
            verified_insights=verified,
            rejected_insights=rejected,
            summary={"status": "COMPLETED"},
        )

    # Generate or extract deterministic executive summary
    executive_summary = (
        contract.summary.get("executive_summary")
        or generate_deterministic_summary(contract, critique_evaluations=critiques)
    )

    return InsightsResponse(
        dataset_id=dataset_id,
        status="SUCCESS" if not all_errors else "COMPLETED_WITH_WARNINGS",
        summary=executive_summary,
        verified_insights=verified,
        rejected_insights=rejected,
        pipeline_errors=all_errors,
    )
