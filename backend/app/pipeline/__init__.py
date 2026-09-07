"""
AIDA Unified Pipeline Package (Person 1, 2, 3, and 4).
"""
# Person 1 (Data Brain)
from .discovery import run_discovery

# Person 2 (ML / Accuracy Brain)
from .statistical import run_statistical_analysis
from .validation import select_validation_strategy
from .model_championship import benchmark_models, select_champion
from .error_analysis import analyze_errors
from .time_analysis import run_time_analysis
from .experiments import (
    run_feature_ablation,
    run_ensemble_experiment,
    run_controlled_experiments,
)
from .ml_analysis import determine_ml_task, run_ml_pipeline

# Person 3 (Agent / Insight Brain)
from .insight_discovery import discover_candidate_hypotheses
from .confidence import evaluate_insight_confidence
from .insight_ranking import rank_verified_insights
from .insight_validator import build_provenance_graph
from .summary import generate_executive_summary

# Person 4 (Product & UI Brain)
from .visualization import AutoChartEngine
from .dashboard import DashboardAssembler

__all__ = [
    "run_discovery",
    "run_statistical_analysis",
    "select_validation_strategy",
    "benchmark_models",
    "select_champion",
    "analyze_errors",
    "run_time_analysis",
    "run_feature_ablation",
    "run_ensemble_experiment",
    "run_controlled_experiments",
    "determine_ml_task",
    "run_ml_pipeline",
    "discover_candidate_hypotheses",
    "evaluate_insight_confidence",
    "rank_verified_insights",
    "build_provenance_graph",
    "generate_executive_summary",
    "AutoChartEngine",
    "DashboardAssembler",
]
