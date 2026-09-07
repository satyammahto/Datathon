"""Discovery Contract Schema (Input from Person 1 - Data Discovery).

Encapsulates dataset profiling, dynamic column schemas, statistical distributions,
missingness reports, and correlation matrices without any hardcoded domain logic.
"""

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, ConfigDict, model_validator


class DistributionStats(BaseModel):
    """Statistical distribution metrics for a numeric column."""
    model_config = ConfigDict(extra="ignore")

    count: int = Field(..., description="Number of non-null observations")
    mean: float = Field(..., description="Arithmetic mean")
    std: float = Field(..., description="Standard deviation")
    min: float = Field(..., description="Minimum value")
    q25: float = Field(..., description="25th percentile (Q1)")
    median: float = Field(..., description="50th percentile (Median)")
    q75: float = Field(..., description="75th percentile (Q3)")
    max: float = Field(..., description="Maximum value")
    skewness: Optional[float] = Field(None, description="Fisher-Pearson skewness")
    kurtosis: Optional[float] = Field(None, description="Excess kurtosis")


class ColumnProfile(BaseModel):
    """Profiling summary for an individual dataset column."""
    model_config = ConfigDict(extra="ignore")

    name: str = Field(..., description="Column identifier / name")
    semantic_type: Literal[
        "numeric", "categorical", "datetime", "boolean", "text", "identifier", "unknown"
    ] = Field(default="unknown", description="Inferred semantic type")
    raw_dtype: str = Field(..., description="Native pandas/SQL data type")
    null_count: int = Field(default=0, ge=0, description="Total missing/null values")
    null_percentage: float = Field(default=0.0, ge=0.0, le=100.0, description="Missing rate [0-100]")
    unique_count: int = Field(default=0, ge=0, description="Count of distinct values")
    cardinality_ratio: float = Field(default=0.0, ge=0.0, le=1.0, description="Unique values / total rows")
    distribution: Optional[DistributionStats] = Field(None, description="Numeric distribution stats if applicable")
    top_categories: Optional[Dict[str, int]] = Field(None, description="Frequency counts for top categories")
    is_constant: bool = Field(default=False, description="True if column contains only 1 distinct value")


class CorrelationMatrix(BaseModel):
    """Pairwise correlation matrix for numeric columns."""
    model_config = ConfigDict(extra="ignore")

    method: Literal["pearson", "spearman", "kendall"] = Field(
        default="pearson", description="Correlation calculation method"
    )
    matrix: Dict[str, Dict[str, float]] = Field(
        default_factory=dict,
        description="Nested mapping: column_a -> {column_b: correlation_coefficient}",
    )


class MissingnessReport(BaseModel):
    """Overall dataset missingness breakdown."""
    model_config = ConfigDict(extra="ignore")

    total_missing_cells: int = Field(default=0, ge=0)
    overall_missing_percentage: float = Field(default=0.0, ge=0.0, le=100.0)
    columns_with_missing: List[str] = Field(default_factory=list)


class DiscoveryContract(BaseModel):
    """Complete Discovery Contract ingested from Person 1 (Data Discovery)."""
    model_config = ConfigDict(extra="allow")

    dataset_id: str = Field(..., description="Unique dataset identifier")
    dataset_name: str = Field(..., description="Filename or resource name")
    row_count: int = Field(default=0, ge=0, description="Total number of records")
    column_count: int = Field(default=0, ge=0, description="Total number of features")
    columns: Dict[str, ColumnProfile] = Field(
        default_factory=dict,
        description="Dictionary mapping column names to their statistical profile",
    )
    correlations: Optional[CorrelationMatrix] = Field(
        None, description="Numeric pairwise correlation matrix"
    )
    missingness: Optional[MissingnessReport] = Field(
        None, description="Comprehensive missingness report"
    )
    sample_rows: Optional[List[Dict[str, Any]]] = Field(
        None, description="First few rows for qualitative reference"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Arbitrary pipeline or dataset metadata"
    )

    @model_validator(mode="before")
    @classmethod
    def adapt_person1_input(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            if hasattr(data, "model_dump"):
                data = data.model_dump()
            elif hasattr(data, "__dict__"):
                data = dict(data.__dict__)
            else:
                return data

        adapted = dict(data)
        fi = adapted.get("file_info") or {}
        fp = adapted.get("fingerprint") or {}
        qr = adapted.get("quality_report") or {}

        # 1. row_count & column_count
        if "row_count" not in adapted or adapted["row_count"] is None:
            if "row_count" in fi:
                adapted["row_count"] = fi["row_count"]
            elif "shape" in fp and isinstance(fp["shape"], (list, tuple)) and len(fp["shape"]) > 0:
                adapted["row_count"] = fp["shape"][0]
            else:
                adapted["row_count"] = 0

        if "column_count" not in adapted or adapted["column_count"] is None:
            if "column_count" in fi:
                adapted["column_count"] = fi["column_count"]
            elif "shape" in fp and isinstance(fp["shape"], (list, tuple)) and len(fp["shape"]) > 1:
                adapted["column_count"] = fp["shape"][1]
            else:
                raw_c = adapted.get("columns", [])
                adapted["column_count"] = len(raw_c) if isinstance(raw_c, (list, dict)) else 0

        # 2. columns: transform list of ColumnMetadata into Dict[str, ColumnProfile]
        raw_cols = adapted.get("columns")
        if isinstance(raw_cols, list):
            col_profiles = {}
            for col in raw_cols:
                c_dict = col if isinstance(col, dict) else (col.model_dump() if hasattr(col, "model_dump") else getattr(col, "__dict__", {}))
                c_name = c_dict.get("name", "unknown")
                inf_type = str(c_dict.get("inferred_type", "unknown")).lower()
                sem_type = "unknown"
                if "numeric" in inf_type or "int" in inf_type or "float" in inf_type:
                    sem_type = "numeric"
                elif "cat" in inf_type:
                    sem_type = "categorical"
                elif "date" in inf_type or "time" in inf_type:
                    sem_type = "datetime"
                elif "bool" in inf_type:
                    sem_type = "boolean"
                elif "id" in inf_type:
                    sem_type = "identifier"
                elif "text" in inf_type:
                    sem_type = "text"

                stats = c_dict.get("stats") or {}
                if hasattr(stats, "model_dump"):
                    stats = stats.model_dump()
                elif hasattr(stats, "__dict__"):
                    stats = stats.__dict__

                dist_obj = None
                if isinstance(stats, dict) and any(k in stats for k in ("mean", "min", "max", "std", "median")):
                    dist_obj = {
                        "count": stats.get("count", adapted.get("row_count", 0)),
                        "mean": float(stats.get("mean") or 0.0),
                        "std": float(stats.get("std") or 1.0),
                        "min": float(stats.get("min") or 0.0),
                        "q25": float(stats.get("q25") or 0.0),
                        "median": float(stats.get("median") or 0.0),
                        "q75": float(stats.get("q75") or 0.0),
                        "max": float(stats.get("max") or 0.0),
                        "skewness": float(stats.get("skewness")) if stats.get("skewness") is not None else None,
                        "kurtosis": float(stats.get("kurtosis")) if stats.get("kurtosis") is not None else None,
                    }

                col_profiles[c_name] = {
                    "name": c_name,
                    "semantic_type": sem_type,
                    "raw_dtype": str(c_dict.get("raw_dtype", "object")),
                    "null_count": int(c_dict.get("null_count", 0) or 0),
                    "null_percentage": float(c_dict.get("null_percentage", 0.0) or 0.0),
                    "unique_count": int(c_dict.get("unique_count", 0) or 0),
                    "cardinality_ratio": float(c_dict.get("cardinality_ratio", 0.0) or 0.0),
                    "distribution": dist_obj,
                    "top_categories": stats.get("top_frequencies") if isinstance(stats, dict) else None,
                    "is_constant": bool(c_dict.get("is_constant", False)),
                }
            adapted["columns"] = col_profiles

        # 3. correlations
        if not adapted.get("correlations"):
            corrs = fp.get("correlations") if isinstance(fp, dict) else getattr(fp, "correlations", None)
            if corrs and isinstance(corrs, dict):
                adapted["correlations"] = {
                    "method": "pearson",
                    "matrix": corrs,
                }

        # 4. missingness
        if not adapted.get("missingness"):
            if qr:
                qr_dict = qr if isinstance(qr, dict) else (qr.model_dump() if hasattr(qr, "model_dump") else getattr(qr, "__dict__", {}))
                adapted["missingness"] = {
                    "total_missing_cells": int(qr_dict.get("total_missing_cells", 0) or 0),
                    "overall_missing_percentage": float(qr_dict.get("missing_cell_percentage", 0.0) or 0.0),
                    "columns_with_missing": qr_dict.get("columns_with_missing", []),
                }

        return adapted

    def get_column(self, col_name: str) -> Optional[ColumnProfile]:
        """Safely fetch profile for a column name."""
        return self.columns.get(col_name)

    def get_correlation(self, col_a: str, col_b: str) -> Optional[float]:
        """Safely retrieve correlation between two columns if calculated."""
        if not self.correlations or not self.correlations.matrix:
            return None
        # Check both (col_a, col_b) and symmetric (col_b, col_a)
        if col_a in self.correlations.matrix and col_b in self.correlations.matrix[col_a]:
            return self.correlations.matrix[col_a][col_b]
        if col_b in self.correlations.matrix and col_a in self.correlations.matrix[col_b]:
            return self.correlations.matrix[col_b][col_a]
        return None
