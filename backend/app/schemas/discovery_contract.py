from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field, ConfigDict


class RouterDecision(BaseModel):
    model_config = ConfigDict(extra="allow")

    classification: bool = False
    regression: bool = False
    clustering: bool = False
    anomaly_detection: bool = False
    time_analysis: bool = False


class DiscoveryContract(BaseModel):
    """
    Contract received from Person 1 (Data Brain).
    Person 2 consumes this contract to orchestrate statistical & ML modeling.
    """
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    schema_version: str = "1.0"
    schema_definition: Dict[str, str] = Field(default_factory=dict, alias="schema")
    target_candidates: List[str] = Field(default_factory=list)
    router: RouterDecision = Field(default_factory=RouterDecision)
    quality_report: Dict[str, Any] = Field(default_factory=dict)
    fingerprint: Dict[str, Any] = Field(default_factory=dict)
    leakage_report: Dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_person1(cls, p1: Any) -> "DiscoveryContract":
        """Adapt a Person 1 DiscoveryContract into the Person 2/3 format."""
        if isinstance(p1, cls):
            return p1
        if isinstance(p1, dict):
            # If already in Person 2 format
            if "schema" in p1 or "schema_definition" in p1:
                return cls.model_validate(p1)

        # Extract columns mapping
        columns = getattr(p1, "columns", {}) if not isinstance(p1, dict) else p1.get("columns", {})
        schema_dict = {}
        if isinstance(columns, dict):
            for col_name, col_meta in columns.items():
                dtype = getattr(col_meta, "declared_type", None) if not isinstance(col_meta, dict) else col_meta.get("declared_type")
                if dtype is not None:
                    dtype_str = getattr(dtype, "value", str(dtype)).lower()
                else:
                    dtype_str = "numeric"
                schema_dict[str(col_name)] = dtype_str
        elif isinstance(columns, (list, tuple)):
            for col in columns:
                c_name = getattr(col, "name", None) if not isinstance(col, dict) else col.get("name")
                c_type = getattr(col, "inferred_type", None) or getattr(col, "raw_dtype", "numeric")
                if isinstance(col, dict):
                    c_type = col.get("inferred_type") or col.get("raw_dtype", "numeric")
                c_type_str = getattr(c_type, "value", str(c_type)).lower()
                if "numeric" in c_type_str or "int" in c_type_str or "float" in c_type_str:
                    clean_type = "numeric"
                elif "cat" in c_type_str or "str" in c_type_str:
                    clean_type = "categorical"
                elif "date" in c_type_str or "time" in c_type_str:
                    clean_type = "datetime"
                elif "bool" in c_type_str:
                    clean_type = "boolean"
                else:
                    clean_type = c_type_str
                if c_name:
                    schema_dict[str(c_name)] = clean_type

        # Extract target candidates
        raw_targets = getattr(p1, "target_candidates", []) if not isinstance(p1, dict) else p1.get("target_candidates", [])
        targets = []
        for t in raw_targets:
            if hasattr(t, "column_name"):
                targets.append(t.column_name)
            elif isinstance(t, dict) and "column_name" in t:
                targets.append(t["column_name"])
            else:
                targets.append(str(t))

        # Extract router tasks
        raw_router = getattr(p1, "router", None) if not isinstance(p1, dict) else p1.get("router", {})
        primary_task = getattr(raw_router, "recommended_primary_task", None) if not isinstance(raw_router, dict) else raw_router.get("recommended_primary_task")
        primary_val = str(getattr(primary_task, "value", primary_task) or "").lower()
        decisions = getattr(raw_router, "task_decisions", {}) if not isinstance(raw_router, dict) else raw_router.get("task_decisions", {})

        is_clf = "classification" in primary_val
        is_reg = "regression" in primary_val
        is_clus = "clustering" in primary_val
        is_anom = "anomaly" in primary_val
        is_time = any(x in primary_val for x in ["time", "temporal", "series", "forecast"])

        if isinstance(decisions, dict):
            for k, v in decisions.items():
                status = getattr(v, "status", None) if not isinstance(v, dict) else v.get("status")
                if status == "applicable":
                    k_low = str(k).lower()
                    if "classification" in k_low: is_clf = True
                    if "regression" in k_low: is_reg = True
                    if "clustering" in k_low: is_clus = True
                    if "anomaly" in k_low: is_anom = True
                    if "time" in k_low: is_time = True

        router_dec = RouterDecision(
            classification=is_clf,
            regression=is_reg,
            clustering=is_clus,
            anomaly_detection=is_anom,
            time_analysis=is_time,
        )

        # Quality & Fingerprint
        raw_qual = getattr(p1, "quality_report", {}) if not isinstance(p1, dict) else p1.get("quality_report", {})
        qual_dict = raw_qual.model_dump() if hasattr(raw_qual, "model_dump") else (raw_qual if isinstance(raw_qual, dict) else {})

        raw_fp = getattr(p1, "fingerprint", {}) if not isinstance(p1, dict) else p1.get("fingerprint", {})
        fp_dict = raw_fp.model_dump() if hasattr(raw_fp, "model_dump") else (raw_fp if isinstance(raw_fp, dict) else {})
        if "shape" in fp_dict and isinstance(fp_dict["shape"], list) and len(fp_dict["shape"]) >= 2:
            fp_dict["row_count"] = fp_dict["shape"][0]
            fp_dict["column_count"] = fp_dict["shape"][1]

        raw_leak = getattr(p1, "leakage_report", {}) if not isinstance(p1, dict) else p1.get("leakage_report", {})
        leak_dict = raw_leak.model_dump() if hasattr(raw_leak, "model_dump") else (raw_leak if isinstance(raw_leak, dict) else {})

        return cls(
            schema_version="1.0",
            schema_definition=schema_dict,
            target_candidates=targets,
            router=router_dec,
            quality_report=qual_dict,
            fingerprint=fp_dict,
            leakage_report=leak_dict,
        )
