"""
AIDA Master Pipeline Orchestrator (Person 1, 2, 3, 4 Unification)
Coordinates the 4-brain autonomous data investigation lifecycle:
1. Person 1 (Data Brain): Dataset ingestion, semantic type inference, fingerprinting, quality audit, leakage detection, routing.
2. Person 2 (ML / Accuracy Brain): Statistical hypothesis testing, CV strategy selection, model championship, error diagnostics.
3. Person 3 (Agent / Insight Brain): Multi-step LangGraph reasoning loop (discover -> plan -> investigate -> critic -> verify -> rank -> summarize).
4. Person 4 (Product / UI Brain): Authoritative contract synthesis into dynamic DashboardSpec & Plotly visualizations.
"""
from typing import Dict, Any, Optional, Callable
import os
import pandas as pd

# Person 1
from app.pipeline.discovery import run_discovery
from app.pipeline.loaders import load_dataset

# Person 2 & 3 Schemas
from backend.app.schemas.discovery_contract import DiscoveryContract
from backend.app.schemas.analysis_contract import AnalysisContract
from backend.app.schemas.insight_contract import InsightContract
from backend.app.schemas.dashboard_spec import DashboardSpec

# Person 2
from backend.app.pipeline.ml_analysis import run_ml_pipeline

# Person 3
from backend.app.agents.graph import run_aida_reasoning

# Person 4
from backend.app.api.adapter import build_dashboard_spec
from app.pipeline.visualization import AutoChartEngine


def run_full_aida_pipeline(
    file_path: str,
    dataset_name: Optional[str] = None,
    dataset_id: Optional[str] = None,
    on_stage_progress: Optional[Callable[[str, str], None]] = None,
) -> Dict[str, Any]:
    """
    Executes the complete 4-stage AIDA pipeline on an unseen CSV/XLSX dataset.
    Returns:
        Dict with keys:
            - 'discovery': Person 1 DiscoveryContract
            - 'analysis': Person 2 AnalysisContract
            - 'insights': Person 3 InsightContract
            - 'dashboard': Person 4 DashboardSpec
    """
    d_name = dataset_name or os.path.basename(file_path)

    # -------------------------------------------------------------------------
    # STAGE 1: Person 1 - Data Brain (Discovery, Quality, Leakage, Routing)
    # -------------------------------------------------------------------------
    if on_stage_progress:
        on_stage_progress("discovery", "Executing discovery fingerprinting and data quality audit...")

    df, _ = load_dataset(file_path)
    p1_discovery = run_discovery(
        file_path=file_path,
        dataset_id=dataset_id,
        dataset_name=d_name,
    )

    # Adapt into standard Person 2 consumption contract
    p2_discovery = DiscoveryContract.from_person1(p1_discovery)

    # -------------------------------------------------------------------------
    # STAGE 2: Person 2 - ML / Accuracy Brain (Statistics, Championship, Errors)
    # -------------------------------------------------------------------------
    if on_stage_progress:
        on_stage_progress("analysis", "Benchmarking candidate models and running statistical diagnostics...")

    analysis_contract = run_ml_pipeline(
        df=df,
        discovery=p2_discovery,
    )

    # -------------------------------------------------------------------------
    # STAGE 3: Person 3 - Agent / Insight Brain (LangGraph Multi-Step Loop)
    # -------------------------------------------------------------------------
    if on_stage_progress:
        on_stage_progress("insights", "Executing LangGraph adversarial critic and deterministic verifier...")

    insight_contract = run_aida_reasoning(
        analysis=analysis_contract,
        max_iterations=5,
    )

    # -------------------------------------------------------------------------
    # STAGE 4: Person 4 - Product / UI Brain (Dashboard Spec & Visualizations)
    # -------------------------------------------------------------------------
    if on_stage_progress:
        on_stage_progress("dashboard", "Synthesizing executive presentation spec and Plotly charts...")

    dashboard_spec = build_dashboard_spec(
        discovery=p2_discovery,
        analysis=analysis_contract,
        insights=insight_contract,
        dataset_name=d_name,
    )

    # If dataset has records and analysis was completed, augment with Plotly charts
    try:
        chart_engine = AutoChartEngine()
        plotly_charts = chart_engine.generate_chart_suite(df, analysis_contract)
        if plotly_charts:
            dashboard_spec.charts = plotly_charts
    except Exception as exc:
        # Gracefully preserve core charts from adapter
        pass

    if on_stage_progress:
        on_stage_progress("complete", "AIDA pipeline execution complete.")

    return {
        "discovery": p1_discovery,
        "analysis": analysis_contract,
        "insights": insight_contract,
        "dashboard": dashboard_spec,
    }
