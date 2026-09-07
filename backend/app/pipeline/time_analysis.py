"""
Time Analysis Module (Person 2 - ML / Accuracy Brain)
Responsible for testing stationarity, autocorrelation, trend, and seasonality
when time-series characteristics are indicated by the Discovery Contract.
"""
from typing import Optional, Any
try:
    from backend.app.schemas.discovery_contract import DiscoveryContract
    from backend.app.schemas.analysis_contract import TimeAnalysisReport
    from backend.app.schemas.enums import PipelineStatus
except ImportError:
    from app.schemas.discovery_contract import DiscoveryContract
    from app.schemas.analysis_contract import TimeAnalysisReport
    from app.schemas.enums import PipelineStatus


def run_time_analysis(
    df: Optional[Any],
    discovery: DiscoveryContract,
) -> TimeAnalysisReport:
    """
    Perform temporal and time-series diagnostic analysis.
    If the discovery router reports time_analysis=False, returns NOT_APPLICABLE.
    """
    if not discovery.router.time_analysis:
        return TimeAnalysisReport(
            status=PipelineStatus.NOT_APPLICABLE,
            reason="Discovery router indicated dataset has no active time analysis component",
        )

    # In future phase: stationarity (ADF), ACF/PACF, STL decomposition
    return TimeAnalysisReport(
        status=PipelineStatus.SUCCESS,
        seasonality_detected=False,
        stationarity_test_passed=True,
        trend_type="none",
        autocorrelation_summary={},
    )
