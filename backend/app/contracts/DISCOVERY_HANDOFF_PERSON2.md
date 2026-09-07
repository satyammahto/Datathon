# AIDA Person 1 (Data Brain) → Person 2 (ML Brain) Discovery Handoff Guide

**Contract Version**: `v1.0.0`  
**Contract Schema Definition**: [`backend/app/contracts/discovery.py`](file:///Users/rickyhemulpatel/datathon/backend/app/contracts/discovery.py)  
**Fixtures**:
- Minimal Mock: [`backend/app/contracts/fixtures/discovery_mock.json`](file:///Users/rickyhemulpatel/datathon/backend/app/contracts/fixtures/discovery_mock.json)
- Real Normal Example: [`backend/app/contracts/fixtures/customer_churn_contract.json`](file:///Users/rickyhemulpatel/datathon/backend/app/contracts/fixtures/customer_churn_contract.json)
- Real Adversarial Example: [`backend/app/contracts/fixtures/adversarial_leakage_contract.json`](file:///Users/rickyhemulpatel/datathon/backend/app/contracts/fixtures/adversarial_leakage_contract.json)

---

## 1. Pipeline Overview

```
Raw CSV / XLSX / JSON
        ↓
  Safe Loader (Encoding, Delimiter, Multiformat)
        ↓
  Semantic Type Inference & Dataset Fingerprint
        ↓
  Data Quality Auditing & Safe Cleaning Engine
        ↓
  Target Candidate Ranking & Temporal Detection
        ↓
  Leakage Hunter & Identifier Isolation
        ↓
  Intelligent Dataset & Task Router
        ↓
  DiscoveryContract v1.0.0 (Deterministic, Machine-Readable JSON)
```

Person 1 produces a fully validated, deterministic, JSON-serializable `DiscoveryContract`.  
Person 2 (ML Brain) can consume this contract directly to train baseline models, execute model tournaments, cross-validate safely, and extract feature importances without manual data wrangling.

---

## 2. Core Contract Sections

### 2.1 Top-Level Contract (`DiscoveryContract`)
| Field | Type | Description |
| :--- | :--- | :--- |
| `version` | `str` | `"1.0.0"` |
| `contract_type` | `str` | `"discovery"` |
| `dataset_id` | `str` | Unique dataset identifier (e.g. `"ds-mock-churn-001"`) |
| `dataset_name` | `str` | Human-readable dataset name or file name |
| `file_info` | `FileInfo` | Row count, column count, encoding, delimiter, size |
| `columns` | `List[ColumnMetadata]` | Per-column semantic types, distributions, null counts, stats |
| `fingerprint` | `DatasetFingerprint` | Structural breakdown, entity keys, class balance, temporal properties |
| `quality_report` | `QualityReport` | Overall score (0-100), duplicate counts, missingness, outliers |
| `leakage_report` | `LeakageReport` | Excluded leakage columns, proxy warnings, identifier isolation |
| `router` | `RouterDecision` | Recommended primary task, validation strategy, partitioned feature lists |
| `target_candidates` | `List[TargetCandidate]` | Ranked candidate target columns with confidence & rationale |
| `cleaning_audit` | `CleaningAudit` | Record of duplicate drops, column drops, and missingness imputations |

---

## 3. Semantic Types Dictionary (`InferredDtype`)

Person 1 classifies every column into one of the following semantic types:

1. `numeric_continuous`: Real-valued measurements, currency, percentages, float features with fractional spread.
2. `numeric_discrete`: Non-negative integers, discrete counts, frequencies, low-range integer sequences.
3. `categorical_nominal`: Unordered discrete categories (e.g. `['France', 'Germany', 'Spain']`).
4. `categorical_ordinal`: Ordered ratings, tier ranks, Likert scales (e.g. `['low', 'medium', 'high']`).
5. `datetime`: Timestamps, calendar dates, ISO-8601 strings, UNIX epoch timestamps.
6. `boolean`: Binary flags (`0/1`, `True/False`, `Y/N`).
7. `identifier`: Primary keys, database IDs, UUIDs, monotonic auto-increments, token strings.
8. `text`: Short human-readable text tokens or names.
9. `free_text`: Long unstructured text comments, descriptions (avg length > 40 chars, high whitespace).
10. `unknown`: Columns with 100% missing values or un-parseable data.

---

## 4. Feature Partition Architecture

Person 1 enforces **strict disjointness** across feature lists in `contract.router`. Person 2 should use these lists directly:

```
Total Columns
 ├── Target Candidate(s) (Excluded from all feature sets)
 ├── usable_feature_columns           --> [Person 2 ML Features]
 ├── excluded_identifier_columns     --> [DO NOT FEED TO MODEL: Entity/PKs]
 ├── excluded_leakage_columns        --> [DO NOT FEED TO MODEL: Target-derived/Proxy/Constant]
 ├── review_feature_columns          --> [WARNING: Check if feature is available at inference]
 └── unsupported_feature_columns     --> [DO NOT FEED TO MODEL: 100% null or unstructured text]
```

### Precedence Invariants:
- `usable_feature_columns ∩ excluded_identifier_columns = ∅`
- `usable_feature_columns ∩ excluded_leakage_columns = ∅`
- `usable_feature_columns ∩ unsupported_feature_columns = ∅`
- The primary `target_column` is **never** in `usable_feature_columns`.

---

## 5. Router Task Decisions & Validation Strategies

### 5.1 Primary Tasks (`contract.router.recommended_primary_task`)
- `binary_classification`: Target has 2 classes.
- `multiclass_classification`: Target has 3–50 discrete classes.
- `regression`: Target is continuous real-valued or high-cardinality numeric magnitude.
- `time_series_forecasting`: Temporal column detected with regular spacing, monotonicity, and continuous metric.
- `clustering` / `unsupervised_eda`: No clear supervised target candidate exists; dataset has continuous/categorical features.

### 5.2 Recommended Validation Strategies (`contract.router.validation_strategy`)
- `StratifiedKFold`: Recommended for classification, especially when `class_balance.is_imbalanced == True`.
- `KFold`: Recommended for regression on standard cross-sectional tabular data.
- `TimeSeriesSplit` / `WalkForward`: Recommended when `fingerprint.temporal` has `is_monotonic == True` or `temporal_column` is present.
- `GroupKFold`: Recommended when `router.group_column` is present to prevent intra-group entity contamination.

---

## 6. How Person 2 Should Consume the Contract (Python Example)

```python
import json
from app.contracts.discovery import DiscoveryContract, TaskType, ValidationStrategy

# 1. Load Contract
with open("discovery_contract.json", "r") as f:
    contract_data = json.load(f)
contract = DiscoveryContract.model_validate(contract_data)

# 2. Extract ML Setup
task_type: TaskType = contract.router.recommended_primary_task
val_strategy: ValidationStrategy = contract.router.validation_strategy
features = contract.router.usable_feature_columns

# 3. Extract Target Column (if supervised)
if task_type in [TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION, TaskType.REGRESSION]:
    target_col = contract.target_candidates[0].column_name
    print(f"Training {task_type.value} model on target '{target_col}' with {len(features)} features.")
    print(f"Usable features: {features}")
    print(f"Excluded identifiers: {contract.router.excluded_identifier_columns}")
    print(f"Excluded leakage: {contract.router.excluded_leakage_columns}")
else:
    print(f"Executing unsupervised analysis: {task_type.value}")
```

---

## 7. Edge Cases & Unsupported States

1. **Zero-target / Exploratory Dataset**:
   `target_candidates` is empty (`[]`), `recommended_primary_task` is `unsupervised_eda` or `clustering`.
2. **All-Null Features**:
   Placed in `unsupported_feature_columns` or dropped in `cleaning_audit.dropped_columns`. Never included in `usable_feature_columns`.
3. **Severe Class Imbalance**:
   `fingerprint.class_balance.is_imbalanced == True`. Person 2 should use PR-AUC, F1-macro, and stratified splits.
4. **Adversarial / Proxy Leakage**:
   Leakage features with correlation > 0.98 or post-event signals are cataloged under `leakage_report.leakage_candidates` and automatically excluded from `usable_feature_columns`.

---

## 8. Summary Checklist for Downstream Engineers

- [x] All outputs are deterministic and reproducible.
- [x] Schema is 100% Pydantic v2 compatible.
- [x] Zero LLM hallucination for statistical/numerical metrics.
- [x] Tested across 10 distinct tabular families and unseen multi-sensor architectures.
