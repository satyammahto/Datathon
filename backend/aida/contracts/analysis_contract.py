"""Analysis Contract Schema (Input from Person 2 - ML Analysis).

Encapsulates machine learning model evaluations, feature importances, clustering
metrics, anomaly detection results, and formal statistical hypothesis tests.
"""

from __future__ import annotations
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, ConfigDict, model_validator


class ModelSummary(BaseModel):
    """Summary of a trained machine learning model."""
    model_config = ConfigDict(extra="ignore")

    model_name: str = Field(..., description="Unique model name or run ID")
    algorithm: str = Field(..., description="Algorithm name, e.g. RandomForest, XGBoost, LinearRegression")
    task_type: Literal["classification", "regression", "clustering", "other"] = Field(
        default="classification", description="Supervised or unsupervised task type"
    )
    evaluation_metrics: Dict[str, float] = Field(
        default_factory=dict,
        description="Standard metrics (e.g. {'accuracy': 0.88, 'f1': 0.86, 'r2': 0.74, 'rmse': 12.3})",
    )
    feature_importances: Optional[Dict[str, float]] = Field(
        None,
        description="Mapping of feature name to normalized importance score or coefficient",
    )
    hyperparameters: Optional[Dict[str, Any]] = Field(
        None, description="Key hyperparameters used for training"
    )

    def get_ranked_features(self, top_n: Optional[int] = None) -> List[tuple[str, float]]:
        """Return features sorted descending by importance value."""
        if not self.feature_importances:
            return []
        sorted_items = sorted(
            self.feature_importances.items(), key=lambda item: abs(item[1]), reverse=True
        )
        return sorted_items[:top_n] if top_n else sorted_items


class StatisticalTestResult(BaseModel):
    """Result of a statistical hypothesis test."""
    model_config = ConfigDict(extra="ignore")

    test_name: str = Field(..., description="e.g. chi_square, anova, independent_t_test, mann_whitney")
    variables: List[str] = Field(..., description="Columns / variables tested")
    statistic: float = Field(..., description="Computed test statistic")
    p_value: float = Field(..., ge=0.0, le=1.0, description="Exact p-value")
    alpha: float = Field(default=0.05, description="Significance threshold")
    is_significant: bool = Field(..., description="True if p_value < alpha")
    interpretation: Optional[str] = Field(None, description="Pre-computed statistical interpretation")


class ClusteringSummary(BaseModel):
    """Unsupervised clustering analysis findings."""
    model_config = ConfigDict(extra="ignore")

    algorithm: str = Field(..., description="Clustering algorithm (e.g. KMeans, DBSCAN)")
    n_clusters: int = Field(..., ge=1, description="Number of discovered clusters")
    silhouette_score: Optional[float] = Field(
        None, ge=-1.0, le=1.0, description="Overall silhouette score"
    )
    cluster_sizes: Dict[str, int] = Field(
        default_factory=dict, description="Cluster ID -> count of instances"
    )


class AnomalySummary(BaseModel):
    """Anomaly / Outlier detection results."""
    model_config = ConfigDict(extra="ignore")

    algorithm: str = Field(..., description="Detector name (e.g. IsolationForest, LocalOutlierFactor, IQR)")
    anomaly_count: int = Field(..., ge=0, description="Total anomalous records flagged")
    anomaly_percentage: float = Field(..., ge=0.0, le=100.0, description="Percentage of dataset flagged")
    features_analyzed: List[str] = Field(default_factory=list)


class AnalysisContract(BaseModel):
    """Complete Analysis Contract ingested from Person 2 (ML Analysis)."""
    model_config = ConfigDict(extra="allow")

    dataset_id: str = Field(default="dataset", description="Corresponding dataset identifier")
    target_column: Optional[str] = Field(None, description="Supervised target variable if specified")
    models: List[ModelSummary] = Field(default_factory=list, description="Trained models and their metrics")
    statistical_tests: List[StatisticalTestResult] = Field(
        default_factory=list, description="Computed statistical tests"
    )
    clustering: Optional[ClusteringSummary] = Field(None, description="Clustering results if run")
    anomalies: Optional[AnomalySummary] = Field(None, description="Anomaly detection results if run")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Execution and pipeline metadata")

    @model_validator(mode="before")
    @classmethod
    def adapt_person2_input(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            if hasattr(data, "model_dump"):
                data = data.model_dump()
            elif hasattr(data, "__dict__"):
                data = dict(data.__dict__)
            else:
                return data

        adapted = dict(data)

        # Dataset ID
        if not adapted.get("dataset_id"):
            meta = adapted.get("metadata") or {}
            adapted["dataset_id"] = meta.get("dataset_id") or "dataset"

        # Models adaptation: handle ModelBenchmarkResult from Person 2
        raw_models = adapted.get("models") or []
        adapted_models = []
        for m in raw_models:
            m_dict = m if isinstance(m, dict) else (m.model_dump() if hasattr(m, "model_dump") else getattr(m, "__dict__", {}))
            m_name = m_dict.get("model_name") or m_dict.get("name") or "Model"
            m_algo = m_dict.get("family") or m_dict.get("algorithm") or "unknown"
            raw_metrics = m_dict.get("metrics") or {}
            if not raw_metrics and m_dict.get("mean_score") is not None:
                raw_metrics = {"score": float(m_dict["mean_score"])}

            # Normalize metrics values to float
            clean_metrics = {}
            for k, v in raw_metrics.items():
                try:
                    clean_metrics[str(k)] = float(v)
                except (ValueError, TypeError):
                    pass

            adapted_models.append({
                "model_name": m_name,
                "algorithm": m_algo,
                "task_type": m_dict.get("task_type", "classification"),
                "evaluation_metrics": clean_metrics,
                "feature_importances": m_dict.get("feature_importances"),
                "hyperparameters": m_dict.get("hyperparameters"),
            })
        adapted["models"] = adapted_models

        # Statistical tests adaptation
        if not adapted.get("statistical_tests"):
            stat_rep = adapted.get("statistical_analysis") or {}
            stat_dict = stat_rep if isinstance(stat_rep, dict) else (stat_rep.model_dump() if hasattr(stat_rep, "model_dump") else getattr(stat_rep, "__dict__", {}))
            raw_tests = stat_dict.get("statistical_tests") or []
            adapted_tests = []
            for t in raw_tests:
                t_dict = t if isinstance(t, dict) else (t.model_dump() if hasattr(t, "model_dump") else getattr(t, "__dict__", {}))
                t_name = t_dict.get("test_name") or t_dict.get("test") or "statistical_test"
                p_val = t_dict.get("p_value")
                stat_val = t_dict.get("statistic") or t_dict.get("stat") or 0.0
                if p_val is not None:
                    try:
                        p_float = max(0.0, min(1.0, float(p_val)))
                        adapted_tests.append({
                            "test_name": str(t_name),
                            "variables": t_dict.get("variables") or [t_dict.get("feature", "var")],
                            "statistic": float(stat_val),
                            "p_value": p_float,
                            "alpha": float(t_dict.get("alpha", 0.05)),
                            "is_significant": bool(t_dict.get("is_significant", p_float < 0.05)),
                            "interpretation": t_dict.get("interpretation"),
                        })
                    except (ValueError, TypeError):
                        pass
            adapted["statistical_tests"] = adapted_tests

        return adapted

    def get_model(self, model_name_or_algo: str) -> Optional[ModelSummary]:
        """Lookup model by name or algorithm (case-insensitive)."""
        target = model_name_or_algo.lower()
        for m in self.models:
            if m.model_name.lower() == target or m.algorithm.lower() == target:
                return m
        return None

    def get_best_model(self, metric: str = "f1", higher_is_better: bool = True) -> Optional[ModelSummary]:
        """Deterministically identify the best performing model for a chosen metric."""
        eligible = [m for m in self.models if metric in m.evaluation_metrics]
        if not eligible:
            return None
        return sorted(
            eligible,
            key=lambda m: m.evaluation_metrics[metric],
            reverse=higher_is_better,
        )[0]
