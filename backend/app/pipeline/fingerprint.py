"""
Person 1: Deep Semantic Type Inference & Dataset Fingerprint Engine.
Detects semantic data types, cardinality, distributions, identifiers,
special formats (currency, percentage, postal codes), temporal structures,
class balances, and target candidates across unseen datasets.
"""
import re
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field
import pandas as pd
import numpy as np

from app.contracts.discovery import (
    DatasetFingerprint,
    ColumnMetadata,
    ColumnStats,
    InferredDtype,
    ClassBalance,
    TemporalProperties,
    TargetCandidate,
    TaskType,
)


# ─── Regular Expressions & Constants ──────────────────────────────

UUID_REGEX = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)

US_ZIP_REGEX = re.compile(r"^\d{5}(-\d{4})?$")
UK_POSTCODE_REGEX = re.compile(r"^[A-Z]{1,2}\d[A-Z\d]? ?\d[A-Z]{2}$", re.IGNORECASE)
CA_POSTCODE_REGEX = re.compile(r"^[A-Z]\d[A-Z] ?\d[A-Z]\d$", re.IGNORECASE)
PIN_REGEX = re.compile(r"^\d{6}$")

CURRENCY_REGEX = re.compile(
    r"^\s*([$€£₹¥₩]|USD|EUR|INR|GBP|CAD|AUD|CHF|JPY|CNY)?\s*-?\(?\s*([$€£₹¥₩]|USD|EUR|INR|GBP|CAD|AUD|CHF|JPY|CNY)?\s*[\d,]+(\.\d+)?\s*\)?\s*([$€£₹¥₩]|USD|EUR|INR|GBP|CAD|AUD|CHF|JPY|CNY)?\s*$",
    re.IGNORECASE,
)

PERCENTAGE_REGEX = re.compile(r"^\s*-?[\d,]+(\.\d+)?\s*%\s*$")

ISO_DATE_REGEX = re.compile(
    r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}([T ]\d{1,2}:\d{2}(:\d{2}(\.\d+)?)?(Z|[+-]\d{2}:?\d{2})?)?$"
)

# Known Ordinal Scales (low to high mappings)
KNOWN_ORDINAL_SETS = [
    {"low", "medium", "high"},
    {"low", "med", "high"},
    {"low", "medium", "high", "very high", "critical"},
    {"minimal", "low", "medium", "high", "critical"},
    {"poor", "fair", "good", "very good", "excellent"},
    {"poor", "fair", "good", "excellent"},
    {"bad", "average", "good"},
    {"xs", "s", "m", "l", "xl", "xxl"},
    {"small", "medium", "large"},
    {"small", "medium", "large", "x-large"},
    {"never", "rarely", "sometimes", "often", "always"},
    {"strongly disagree", "disagree", "neutral", "agree", "strongly agree"},
    {"tier 1", "tier 2", "tier 3"},
    {"tier 1", "tier 2", "tier 3", "tier 4"},
    {"level 1", "level 2", "level 3"},
    {"level 1", "level 2", "level 3", "level 4", "level 5"},
    {"p1", "p2", "p3", "p4", "p5"},
    {"bronze", "silver", "gold", "platinum", "diamond"},
    {"high school", "associate", "bachelor", "master", "doctorate", "phd"},
]

BOOLEAN_PAIRS = [
    {0, 1},
    {"0", "1"},
    {True, False},
    {"true", "false"},
    {"yes", "no"},
    {"y", "n"},
    {"t", "f"},
    {"active", "inactive"},
    {"pass", "fail"},
    {"enabled", "disabled"},
    {"positive", "negative"},
]

ID_KEYWORDS = {
    "id", "uuid", "guid", "key", "ssn", "sin", "hash", "code", "token",
    "account_number", "account_no", "cust_id", "customer_id", "user_id",
    "order_id", "transaction_id", "ticket_id", "employee_id", "member_id",
    "patient_id", "product_id", "item_id", "device_id", "session_id",
}

DISCRETE_NUMERIC_KEYWORDS = {
    "age", "count", "number", "orders", "quantity", "items", "days", "visits",
    "steps", "year", "month", "day", "rank", "priority", "bedrooms",
    "bathrooms", "floors", "seats", "children", "dependents", "tenure",
    "products", "calls", "transactions", "attempts",
}

CONTINUOUS_NUMERIC_KEYWORDS = {
    "salary", "revenue", "price", "cost", "amount", "balance", "income",
    "rate", "ratio", "percentage", "score", "temperature", "distance",
    "weight", "height", "speed", "duration", "latency", "area", "sqft",
    "volume", "loss", "profit", "fare", "charges",
}

TARGET_NOMENCLATURE_KEYWORDS = {
    "target", "label", "outcome", "class", "churn", "churned", "exited",
    "survived", "status", "default", "defaulted", "fraud", "fraudulent",
    "converted", "conversion", "response", "purchased", "bought", "price",
    "revenue", "sales", "cost", "salary", "charges", "fare", "delay",
    "duration", "growth", "risk_level",
}


@dataclass
class SemanticInference:
    inferred_type: InferredDtype
    subtype: Optional[str] = None
    confidence: float = 1.0
    reasons: List[str] = field(default_factory=list)
    is_identifier: bool = False
    is_constant: bool = False
    parsed_numeric_series: Optional[pd.Series] = None


# ─── Semantic Detectors ──────────────────────────────────────────

def _check_constant(series: pd.Series, non_null: pd.Series) -> Optional[SemanticInference]:
    """Detect single-value constant columns."""
    unique_count = non_null.nunique()
    if unique_count <= 1:
        val = non_null.iloc[0] if len(non_null) > 0 else "None"
        return SemanticInference(
            inferred_type=InferredDtype.CATEGORICAL_NOMINAL if pd.api.types.is_object_dtype(series) else InferredDtype.NUMERIC_DISCRETE,
            subtype="constant_single_value",
            confidence=1.0,
            reasons=[f"Single unique value across non-null entries ('{val}')"],
            is_constant=True,
        )
    return None


def _check_uuid(series: pd.Series, non_null: pd.Series, col_name: str) -> Optional[SemanticInference]:
    """Detect UUID / GUID formatted identifiers."""
    if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
        return None
    
    sample = non_null.head(30).astype(str)
    uuid_matches = sum(bool(UUID_REGEX.match(s.strip())) for s in sample)
    if len(sample) > 0 and (uuid_matches / len(sample)) >= 0.85:
        return SemanticInference(
            inferred_type=InferredDtype.IDENTIFIER,
            subtype="uuid",
            confidence=0.99,
            reasons=[f"Values match 128-bit hexadecimal UUID pattern (e.g., '{sample.iloc[0]}')"],
            is_identifier=True,
        )
    return None


def _check_postal_code(series: pd.Series, non_null: pd.Series, col_name: str) -> Optional[SemanticInference]:
    """
    Detect Postal/ZIP codes (e.g. US 5-digit with leading zeros '00123', UK, Canada, PIN).
    Crucial: Must NOT be treated as continuous numeric features.
    """
    name_lower = col_name.lower()
    zip_keywords = ["zip", "postal", "postcode", "pin_code", "pincode", "zipcode", "zip_code"]
    has_zip_name = any(kw in name_lower for kw in zip_keywords)

    sample = non_null.head(50).astype(str).str.strip()
    if len(sample) == 0:
        return None

    # US 5-digit ZIP
    us_matches = sum(bool(US_ZIP_REGEX.match(s)) for s in sample)
    uk_matches = sum(bool(UK_POSTCODE_REGEX.match(s)) for s in sample)
    ca_matches = sum(bool(CA_POSTCODE_REGEX.match(s)) for s in sample)
    pin_matches = sum(bool(PIN_REGEX.match(s)) for s in sample) if has_zip_name else 0

    total_s = len(sample)
    if (us_matches / total_s >= 0.85) or (uk_matches / total_s >= 0.85) or (ca_matches / total_s >= 0.85) or (pin_matches / total_s >= 0.85) or (has_zip_name and us_matches / total_s >= 0.5):
        reasons = [f"Values conform to geographic postal/ZIP code format (sample: '{sample.iloc[0]}')"]
        if has_zip_name:
            reasons.append(f"Column name '{col_name}' explicitly denotes postal/ZIP identifier")
        return SemanticInference(
            inferred_type=InferredDtype.IDENTIFIER if non_null.nunique() / max(len(series), 1) > 0.5 else InferredDtype.CATEGORICAL_NOMINAL,
            subtype="postal_code",
            confidence=0.95 if has_zip_name else 0.85,
            reasons=reasons,
            is_identifier=True,
        )
    return None


def _check_currency(series: pd.Series, non_null: pd.Series, col_name: str) -> Optional[SemanticInference]:
    """
    Detect formatted currency strings (e.g. '₹50,000', '$1,234.50', '€100', '(500 USD)').
    Safely extracts numeric representation for statistics.
    """
    if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
        return None

    sample = non_null.head(30).astype(str).str.strip()
    if len(sample) == 0:
        return None

    curr_matches = sum(bool(CURRENCY_REGEX.match(s)) for s in sample)
    if curr_matches / len(sample) >= 0.75:
        # Attempt safe numeric parsing for the entire series
        cleaned = non_null.astype(str).str.replace(r"[^\d.-]", "", regex=True)
        # Handle parenthesized negative currency, e.g. (100) -> -100
        neg_mask = non_null.astype(str).str.contains(r"\(.*\)", regex=True)
        num_series = pd.to_numeric(cleaned, errors="coerce")
        num_series[neg_mask] = -num_series[neg_mask]
        
        valid_ratio = num_series.notnull().sum() / len(non_null)
        if valid_ratio >= 0.80:
            return SemanticInference(
                inferred_type=InferredDtype.NUMERIC_CONTINUOUS,
                subtype="currency",
                confidence=0.95,
                reasons=[f"Parsed string values to currency numbers (matched currency symbol / financial notation in sample '{sample.iloc[0]}')"],
                parsed_numeric_series=num_series,
            )
    return None


def _check_percentage(series: pd.Series, non_null: pd.Series, col_name: str) -> Optional[SemanticInference]:
    """
    Detect percentage strings (e.g. '75%', '12.5%').
    Safely extracts float representation for statistics.
    """
    if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
        return None

    sample = non_null.head(30).astype(str).str.strip()
    if len(sample) == 0:
        return None

    pct_matches = sum(bool(PERCENTAGE_REGEX.match(s)) for s in sample)
    if pct_matches / len(sample) >= 0.75:
        cleaned = non_null.astype(str).str.replace("%", "", regex=False).str.strip()
        num_series = pd.to_numeric(cleaned, errors="coerce")
        if num_series.notnull().sum() / len(non_null) >= 0.80:
            return SemanticInference(
                inferred_type=InferredDtype.NUMERIC_CONTINUOUS,
                subtype="percentage",
                confidence=0.96,
                reasons=[f"Parsed percentage formatted values (sample '{sample.iloc[0]}') to numeric rates"],
                parsed_numeric_series=num_series,
            )
    return None


def _check_boolean(series: pd.Series, non_null: pd.Series, col_name: str) -> Optional[SemanticInference]:
    """Detect boolean flags and 2-state indicators."""
    name_lower = col_name.lower()
    unique_count = non_null.nunique()

    # Explicit bool dtype
    if pd.api.types.is_bool_dtype(series):
        return SemanticInference(
            inferred_type=InferredDtype.BOOLEAN,
            subtype="boolean_flag",
            confidence=1.0,
            reasons=["Native boolean data type"],
        )

    if unique_count <= 2:
        unique_vals = set(non_null.unique())
        # Check normalized string / numeric representations
        normalized_vals = {str(v).strip().lower() for v in unique_vals}
        for pair in BOOLEAN_PAIRS:
            pair_norm = {str(p).lower() for p in pair}
            if normalized_vals.issubset(pair_norm):
                reasons = [f"Binary values match standard boolean pair {pair_norm}"]
                bool_name_prefixes = ["is_", "has_", "flag_", "should_", "can_", "was_", "did_"]
                if any(name_lower.startswith(pre) for pre in bool_name_prefixes) or name_lower.endswith("_flag"):
                    reasons.append(f"Column name '{col_name}' signifies a boolean flag")
                return SemanticInference(
                    inferred_type=InferredDtype.BOOLEAN,
                    subtype="boolean_binary",
                    confidence=0.98 if len(reasons) > 1 else 0.90,
                    reasons=reasons,
                )
    return None


def _check_identifier(series: pd.Series, non_null: pd.Series, col_name: str, total_rows: int) -> Optional[SemanticInference]:
    """
    Detect record identifiers, primary keys, and entity tokens.
    Uses naming signals, cardinality ratio, monotonic integer checks, and pattern prefixes.
    """
    name_lower = col_name.lower()
    unique_count = non_null.nunique()
    cardinality_ratio = unique_count / max(total_rows, 1)

    # 1. Exact ID keyword matches or prefix/suffix
    has_id_name = any(
        name_lower == kw
        or name_lower.startswith(f"{kw}_")
        or name_lower.endswith(f"_{kw}")
        or name_lower.endswith("id")
        for kw in ID_KEYWORDS
    )

    # 2. Sequential / monotonic increasing integer ID (e.g. 1, 2, 3, 4, ...)
    if pd.api.types.is_numeric_dtype(series) and cardinality_ratio == 1.0 and total_rows >= 10:
        numeric_s = pd.to_numeric(non_null, errors="coerce")
        if (numeric_s % 1 == 0).all():
            diffs = numeric_s.sort_values().diff().dropna()
            if len(diffs) > 0 and (diffs == 1).all():
                return SemanticInference(
                    inferred_type=InferredDtype.IDENTIFIER,
                    subtype="sequential_id",
                    confidence=0.99,
                    reasons=["100% unique strictly sequential integer series", "Auto-increment primary key pattern"],
                    is_identifier=True,
                )

    # 3. High-cardinality string identifiers with alphanumeric code prefix (e.g. USR_001, TXN-9982)
    if (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)) and cardinality_ratio >= 0.80 and total_rows >= 15:
        sample = non_null.head(30).astype(str).str.strip()
        code_pattern = re.compile(r"^[A-Za-z]{2,5}[-_#]?\d{3,10}$")
        if sum(bool(code_pattern.match(s)) for s in sample) / len(sample) >= 0.80:
            return SemanticInference(
                inferred_type=InferredDtype.IDENTIFIER,
                subtype="record_code_id",
                confidence=0.95,
                reasons=[f"Values follow alphanumeric code identifier pattern (e.g. '{sample.iloc[0]}')", f"High uniqueness ratio ({cardinality_ratio:.2%})"],
                is_identifier=True,
            )

    # 4. Keyword match with high cardinality
    if has_id_name and cardinality_ratio >= 0.60:
        return SemanticInference(
            inferred_type=InferredDtype.IDENTIFIER,
            subtype="id_like",
            confidence=0.95,
            reasons=[f"Column name contains identifier keyword '{col_name}'", f"High unique cardinality ratio ({cardinality_ratio:.2%})"],
            is_identifier=True,
        )

    # 5. Near-unique string identifier without keyword
    if (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)) and cardinality_ratio >= 0.98 and total_rows >= 30:
        avg_len = non_null.astype(str).str.len().mean()
        if avg_len <= 32:  # Not long free text
            return SemanticInference(
                inferred_type=InferredDtype.IDENTIFIER,
                subtype="near_unique_token",
                confidence=0.85,
                reasons=[f"Near-unique distinct values ({unique_count}/{total_rows})", "Short token length consistent with entity keys"],
                is_identifier=True,
            )

    return None


def _check_datetime(series: pd.Series, non_null: pd.Series, col_name: str) -> Optional[SemanticInference]:
    """Detect datetime dtypes, ISO strings, parseable dates, and epoch timestamps."""
    name_lower = col_name.lower()
    date_keywords = ["date", "time", "timestamp", "datetime", "created_at", "updated_at", "dob", "birth_date", "expiry", "due_date"]
    has_date_name = any(kw in name_lower for kw in date_keywords)

    # 1. Native pandas datetime
    if pd.api.types.is_datetime64_any_dtype(series):
        return SemanticInference(
            inferred_type=InferredDtype.DATETIME,
            subtype="native_datetime",
            confidence=1.0,
            reasons=["Native pandas datetime64 dtype"],
        )

    # 2. String parseable dates
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        sample = non_null.head(50).astype(str).str.strip()
        if len(sample) > 0:
            iso_matches = sum(bool(ISO_DATE_REGEX.match(s)) for s in sample)
            if iso_matches / len(sample) >= 0.80:
                return SemanticInference(
                    inferred_type=InferredDtype.DATETIME,
                    subtype="iso_datetime",
                    confidence=0.98,
                    reasons=[f"Strings conform to ISO/standard calendar date format (sample '{sample.iloc[0]}')"],
                )

            # Try parsing sample with pd.to_datetime
            if has_date_name or iso_matches > 0:
                try:
                    parsed = pd.to_datetime(sample, errors="coerce")
                    if parsed.notnull().sum() / len(sample) >= 0.85 and parsed.nunique() > 1:
                        return SemanticInference(
                            inferred_type=InferredDtype.DATETIME,
                            subtype="date_string",
                            confidence=0.92,
                            reasons=[f"Successfully parsed date strings with pd.to_datetime", f"Name '{col_name}' suggests temporal attribute"],
                        )
                except Exception:
                    pass

    # 3. Epoch timestamps (integer/float seconds or milliseconds between 2000 and 2035)
    if pd.api.types.is_numeric_dtype(series) and has_date_name:
        numeric_s = pd.to_numeric(non_null, errors="coerce").dropna()
        if len(numeric_s) > 0:
            min_val, max_val = numeric_s.min(), numeric_s.max()
            # Seconds epoch range (2000-01-01 to 2035-01-01: ~9.4e8 to ~2.05e9)
            if 946684800 <= min_val and max_val <= 2051222400:
                return SemanticInference(
                    inferred_type=InferredDtype.DATETIME,
                    subtype="epoch_seconds",
                    confidence=0.90,
                    reasons=["Numeric range matches UNIX epoch seconds (2000-2035)", f"Column name '{col_name}' indicates timestamp"],
                )
            # Milliseconds epoch range (9.4e11 to 2.05e12)
            if 946684800000 <= min_val and max_val <= 2051222400000:
                return SemanticInference(
                    inferred_type=InferredDtype.DATETIME,
                    subtype="epoch_milliseconds",
                    confidence=0.90,
                    reasons=["Numeric range matches UNIX epoch milliseconds (2000-2035)", f"Column name '{col_name}' indicates timestamp"],
                )

    return None


def _check_ordinal_categorical(series: pd.Series, non_null: pd.Series, col_name: str) -> Optional[SemanticInference]:
    """Detect ordinal categorical data with credible ordering evidence."""
    unique_count = non_null.nunique()
    if unique_count > 15 or unique_count <= 1:
        return None

    unique_vals = {str(v).strip().lower() for v in non_null.unique()}
    
    for ordered_set in KNOWN_ORDINAL_SETS:
        if unique_vals.issubset(ordered_set):
            return SemanticInference(
                inferred_type=InferredDtype.CATEGORICAL_ORDINAL,
                subtype="ordinal_scale",
                confidence=0.92,
                reasons=[f"Distinct values {sorted(list(unique_vals))} match recognized ordinal hierarchy {sorted(list(ordered_set))}"],
            )

    # Numeric ratings 1-5 or 1-10 where column name indicates rating/score/grade/tier/level
    name_lower = col_name.lower()
    ordinal_keywords = ["rating", "grade", "tier", "level", "stage", "severity", "priority", "rank"]
    if any(kw in name_lower for kw in ordinal_keywords) and pd.api.types.is_numeric_dtype(series):
        numeric_s = pd.to_numeric(non_null, errors="coerce").dropna()
        if (numeric_s % 1 == 0).all() and unique_count <= 10:
            min_v, max_v = numeric_s.min(), numeric_s.max()
            if (min_v >= 1 and max_v <= 5) or (min_v >= 1 and max_v <= 10) or (min_v >= 0 and max_v <= 5):
                return SemanticInference(
                    inferred_type=InferredDtype.CATEGORICAL_ORDINAL,
                    subtype="ordinal_rating",
                    confidence=0.88,
                    reasons=[f"Discrete scale {min_v:.0f}-{max_v:.0f} with ordinal keyword '{col_name}'"],
                )

    return None


def _check_numeric(series: pd.Series, non_null: pd.Series, col_name: str, total_rows: int) -> Optional[SemanticInference]:
    """
    Distinguish between continuous measurements and discrete count quantities.
    Conservative heuristics grounded in mathematical properties and column nomenclature.
    """
    if not pd.api.types.is_numeric_dtype(series):
        return None

    name_lower = col_name.lower()
    numeric_s = pd.to_numeric(non_null, errors="coerce").dropna()
    if len(numeric_s) == 0:
        return None

    unique_count = numeric_s.nunique()
    cardinality_ratio = unique_count / max(total_rows, 1)

    has_discrete_keyword = any(kw in name_lower for kw in DISCRETE_NUMERIC_KEYWORDS)
    has_continuous_keyword = any(kw in name_lower for kw in CONTINUOUS_NUMERIC_KEYWORDS)

    # Check if numbers are all integers
    all_integers = bool((numeric_s % 1 == 0).all())

    # Case 1: Float with actual fractional values -> Continuous
    if not all_integers:
        return SemanticInference(
            inferred_type=InferredDtype.NUMERIC_CONTINUOUS,
            subtype="continuous_measurement",
            confidence=0.95,
            reasons=["Floating-point values with non-zero fractional components", "Continuous distribution spectrum"],
        )

    # Case 2: Discrete counts / quantities (strictly integers)
    if all_integers:
        if has_discrete_keyword or unique_count <= 30 or cardinality_ratio < 0.20:
            reasons = ["Strict integer values representing discrete counts/frequencies"]
            if has_discrete_keyword:
                reasons.append(f"Column name '{col_name}' indicates count-like metric")
            return SemanticInference(
                inferred_type=InferredDtype.NUMERIC_DISCRETE,
                subtype="count_discrete",
                confidence=0.90 if has_discrete_keyword else 0.80,
                reasons=reasons,
            )

        if has_continuous_keyword or cardinality_ratio >= 0.50 or numeric_s.max() > 10000:
            reasons = ["High-range integer values (e.g. currency, salary, or continuous measure stored as integer)"]
            if has_continuous_keyword:
                reasons.append(f"Column name '{col_name}' signifies continuous magnitude")
            return SemanticInference(
                inferred_type=InferredDtype.NUMERIC_CONTINUOUS,
                subtype="continuous_measurement",
                confidence=0.88,
                reasons=reasons,
            )

        return SemanticInference(
            inferred_type=InferredDtype.NUMERIC_DISCRETE,
            subtype="integer_quantity",
            confidence=0.80,
            reasons=["Integer sequence with moderate cardinality"],
        )

    return None


def _check_text_or_nominal(series: pd.Series, non_null: pd.Series, col_name: str, total_rows: int) -> SemanticInference:
    """
    Distinguish between nominal categories, short structured text, and free-form text.
    Ensures short categorical strings do NOT erroneously become free text.
    """
    unique_count = non_null.nunique()
    cardinality_ratio = unique_count / max(total_rows, 1)

    str_series = non_null.astype(str).str.strip()
    avg_len = float(str_series.str.len().mean()) if len(str_series) > 0 else 0.0
    max_len = int(str_series.str.len().max()) if len(str_series) > 0 else 0
    avg_spaces = float(str_series.str.count(" ").mean()) if len(str_series) > 0 else 0.0

    # 1. Free-form text (long sentences, multi-word comments/descriptions)
    if avg_len >= 40 or max_len >= 120 or avg_spaces >= 3.5:
        return SemanticInference(
            inferred_type=InferredDtype.FREE_TEXT,
            subtype="free_text_comment",
            confidence=0.92,
            reasons=[f"High average string length ({avg_len:.1f} chars)", f"Multi-word sentence structure (avg {avg_spaces:.1f} spaces/entry)"],
        )

    # 2. Nominal category (low cardinality or short categorical strings)
    if unique_count <= 60 or cardinality_ratio < 0.10:
        return SemanticInference(
            inferred_type=InferredDtype.CATEGORICAL_NOMINAL,
            subtype="nominal_category",
            confidence=0.90,
            reasons=[f"Discrete categorical domain ({unique_count} distinct categories, cardinality ratio: {cardinality_ratio:.2%})"],
        )

    # 3. Short structured text (e.g. personal names, product titles, street addresses)
    return SemanticInference(
        inferred_type=InferredDtype.TEXT,
        subtype="structured_text",
        confidence=0.85,
        reasons=[f"Short structured text entries (avg length: {avg_len:.1f} chars, {unique_count} distinct values)"],
    )


# ─── Master Column Semantic Inference ────────────────────────────

def infer_column_semantics(series: pd.Series, col_name: str, total_rows: int) -> SemanticInference:
    """
    Infer the deep semantic type of a column using multi-evidence signals.
    """
    non_null = series.dropna()
    if len(non_null) == 0:
        return SemanticInference(
            inferred_type=InferredDtype.UNKNOWN,
            subtype="all_missing",
            confidence=1.0,
            reasons=["All values in column are null / missing"],
        )

    # 1. Check Constant
    res = _check_constant(series, non_null)
    if res:
        return res

    # 2. Check UUID
    res = _check_uuid(series, non_null, col_name)
    if res:
        return res

    # 3. Check Postal / ZIP codes
    res = _check_postal_code(series, non_null, col_name)
    if res:
        return res

    # 4. Check Special String Formats (Currency & Percentage)
    res = _check_currency(series, non_null, col_name)
    if res:
        return res

    res = _check_percentage(series, non_null, col_name)
    if res:
        return res

    # 5. Check Boolean
    res = _check_boolean(series, non_null, col_name)
    if res:
        return res

    # 6. Check Date / Datetime
    res = _check_datetime(series, non_null, col_name)
    if res:
        return res

    # 7. Check Identifiers / Entity Keys
    res = _check_identifier(series, non_null, col_name, total_rows)
    if res:
        return res

    # 8. Check Ordinal Categorical
    res = _check_ordinal_categorical(series, non_null, col_name)
    if res:
        return res

    # 9. Check Numeric (Continuous vs Discrete)
    res = _check_numeric(series, non_null, col_name, total_rows)
    if res:
        return res

    # 10. Fallback: Text or Nominal Categorical
    return _check_text_or_nominal(series, non_null, col_name, total_rows)


# ─── Statistics Calculation ──────────────────────────────────────

def compute_column_stats(
    series: pd.Series,
    inferred_type: InferredDtype,
    parsed_numeric_series: Optional[pd.Series] = None,
) -> Optional[ColumnStats]:
    """Calculate statistical summaries for a column."""
    eval_series = parsed_numeric_series if parsed_numeric_series is not None else series.dropna()
    if len(eval_series) == 0:
        return None

    stats = ColumnStats()
    if inferred_type in (InferredDtype.NUMERIC_CONTINUOUS, InferredDtype.NUMERIC_DISCRETE) or parsed_numeric_series is not None:
        numeric_s = pd.to_numeric(eval_series, errors="coerce").dropna()
        if len(numeric_s) > 0:
            stats.min = float(numeric_s.min())
            stats.max = float(numeric_s.max())
            stats.mean = round(float(numeric_s.mean()), 4)
            stats.std = round(float(numeric_s.std()), 4) if len(numeric_s) > 1 else 0.0
            stats.median = float(numeric_s.median())
            stats.q25 = float(numeric_s.quantile(0.25))
            stats.q75 = float(numeric_s.quantile(0.75))
            stats.skewness = round(float(numeric_s.skew()), 4) if len(numeric_s) > 2 else 0.0
            stats.kurtosis = round(float(numeric_s.kurtosis()), 4) if len(numeric_s) > 3 else 0.0
            mode_val = numeric_s.mode()
            stats.mode = float(mode_val.iloc[0]) if len(mode_val) > 0 else None
    else:
        top_counts = eval_series.value_counts().head(10).to_dict()
        stats.top_frequencies = {str(k): int(v) for k, v in top_counts.items()}
        mode_val = eval_series.mode()
        stats.mode = str(mode_val.iloc[0]) if len(mode_val) > 0 else None

    return stats


# ─── Temporal Structure Analysis ─────────────────────────────────

def detect_temporal_structure(
    df: pd.DataFrame,
    datetime_cols: List[str]
) -> Optional[TemporalProperties]:
    """
    Analyze datetime columns to evaluate time span, ordering, frequency, and completeness.
    """
    if not datetime_cols:
        return None

    primary_col = datetime_cols[0]
    s = df[primary_col].dropna()
    if len(s) < 2:
        return TemporalProperties(
            temporal_column=primary_col,
            candidate_date_columns=datetime_cols,
            is_sorted=False,
            is_monotonic=False,
        )

    parsed_dates = pd.to_datetime(s, errors="coerce").dropna()
    if len(parsed_dates) < 2:
        return TemporalProperties(
            temporal_column=primary_col,
            candidate_date_columns=datetime_cols,
            is_sorted=False,
            is_monotonic=False,
        )

    # Sorted / Monotonic check
    is_sorted = bool(parsed_dates.is_monotonic_increasing)
    distinct_dates = int(parsed_dates.nunique())
    start_date = parsed_dates.min().isoformat()
    end_date = parsed_dates.max().isoformat()

    # Inferred frequency
    inferred_freq = None
    has_gaps = False
    is_regular = False
    completeness_ratio = 1.0

    if is_sorted and distinct_dates >= 4:
        try:
            freq = pd.infer_freq(parsed_dates.drop_duplicates())
            if freq:
                inferred_freq = str(freq)
                is_regular = True
            else:
                # Check approximate delta
                diffs = parsed_dates.drop_duplicates().diff().dropna()
                median_delta = diffs.median()
                if median_delta == pd.Timedelta(days=1):
                    inferred_freq = "D (Daily)"
                    is_regular = True
                elif median_delta == pd.Timedelta(days=7):
                    inferred_freq = "W (Weekly)"
                    is_regular = True
                elif pd.Timedelta(days=28) <= median_delta <= pd.Timedelta(days=31):
                    inferred_freq = "M (Monthly)"
                    is_regular = True
                elif median_delta == pd.Timedelta(hours=1):
                    inferred_freq = "H (Hourly)"
                    is_regular = True
                else:
                    inferred_freq = "irregular"
                    has_gaps = True
        except Exception:
            inferred_freq = "irregular"

    return TemporalProperties(
        temporal_column=primary_col,
        candidate_date_columns=datetime_cols,
        is_sorted=is_sorted,
        is_monotonic=is_sorted,
        inferred_frequency=inferred_freq,
        start_date=start_date,
        end_date=end_date,
        distinct_dates_count=distinct_dates,
        has_gaps=has_gaps,
        temporal_completeness_ratio=round(completeness_ratio, 4),
        is_regularly_spaced=is_regular,
    )


# ─── Class Balance Analysis ──────────────────────────────────────

def calculate_class_balance(df: pd.DataFrame, target_col: str) -> Optional[ClassBalance]:
    """Calculate class distributions and imbalance ratios for a categorical target."""
    if target_col not in df.columns:
        return None

    counts = df[target_col].dropna().value_counts().to_dict()
    total_valid = sum(counts.values())
    if total_valid == 0:
        return None

    proportions = {str(k): round(v / total_valid, 4) for k, v in counts.items()}
    class_counts_int = {str(k): int(v) for k, v in counts.items()}
    
    # Identify minority class
    sorted_classes = sorted(counts.items(), key=lambda x: x[1])
    min_class_label, min_count = sorted_classes[0]
    max_count = sorted_classes[-1][1]

    minority_pct = round((min_count / total_valid) * 100, 2)
    is_imbalanced = minority_pct < 25.0
    imbalance_ratio = round(max_count / max(min_count, 1), 2)

    return ClassBalance(
        target_column=target_col,
        proportions=proportions,
        class_counts=class_counts_int,
        minority_class=str(min_class_label),
        minority_class_percentage=minority_pct,
        is_imbalanced=is_imbalanced,
        imbalance_ratio=imbalance_ratio,
    )


# ─── Target Candidate Ranking ────────────────────────────────────

def rank_target_candidates(
    df: pd.DataFrame,
    columns_meta: List[ColumnMetadata],
    total_rows: int
) -> List[TargetCandidate]:
    """
    Identify and rank plausible target variables across binary classification,
    multiclass classification, and continuous regression tasks.
    Dataset-agnostic: does not assume fixed column position.
    """
    candidates: List[TargetCandidate] = []

    for col_meta in columns_meta:
        col_name = col_meta.name
        name_lower = col_name.lower()

        # Disqualify identifiers and constant columns
        if col_meta.is_identifier or col_meta.is_constant:
            continue

        # Disqualify severe missingness (> 60% nulls)
        if col_meta.null_percentage > 60.0:
            continue

        # Check nomenclature signals
        has_kw = any(
            name_lower == kw
            or name_lower.startswith(f"{kw}_")
            or name_lower.endswith(f"_{kw}")
            or kw in name_lower
            for kw in TARGET_NOMENCLATURE_KEYWORDS
        )

        score = 0.0
        reasons: List[str] = []

        # 1. Binary Classification Candidates
        if col_meta.inferred_type == InferredDtype.BOOLEAN or (
            col_meta.unique_count == 2
            and col_meta.inferred_type not in (InferredDtype.NUMERIC_CONTINUOUS, InferredDtype.DATETIME)
        ):
            score = 0.55
            reasons.append("Binary cardinality (2 distinct classes)")
            if has_kw:
                score += 0.40
                reasons.append(f"Column name '{col_name}' strongly matches standard target keywords")
            if col_meta.name == df.columns[-1]:
                score += 0.05
                reasons.append("Located at terminal column position")

            candidates.append(TargetCandidate(
                column_name=col_name,
                task_type=TaskType.BINARY_CLASSIFICATION,
                confidence_score=min(round(score, 2), 0.99),
                reasons=reasons,
                class_count=col_meta.unique_count,
            ))
            col_meta.is_potential_target = True

        # 2. Multiclass Classification Candidates
        elif (
            col_meta.inferred_type in (InferredDtype.CATEGORICAL_NOMINAL, InferredDtype.CATEGORICAL_ORDINAL)
            and 2 < col_meta.unique_count <= 20
        ):
            score = 0.40
            reasons.append(f"Discrete categorical domain ({col_meta.unique_count} classes)")
            if has_kw:
                score += 0.45
                reasons.append(f"Column name '{col_name}' matches target keywords")
            if col_meta.name == df.columns[-1]:
                score += 0.05

            candidates.append(TargetCandidate(
                column_name=col_name,
                task_type=TaskType.MULTICLASS_CLASSIFICATION,
                confidence_score=min(round(score, 2), 0.98),
                reasons=reasons,
                class_count=col_meta.unique_count,
            ))
            col_meta.is_potential_target = True

        # 3. Continuous Regression Candidates
        elif (
            col_meta.inferred_type in (InferredDtype.NUMERIC_CONTINUOUS, InferredDtype.NUMERIC_DISCRETE)
            and col_meta.unique_count > 20
            and col_meta.subtype not in ("postal_code", "sequential_id", "epoch_seconds")
        ):
            score = 0.40 if has_kw else 0.25
            if has_kw:
                reasons.append(f"Continuous numeric variable with target keyword match '{col_name}'")
                if col_meta.name == df.columns[-1]:
                    score += 0.10
                candidates.append(TargetCandidate(
                    column_name=col_name,
                    task_type=TaskType.REGRESSION,
                    confidence_score=min(round(score + 0.45, 2), 0.95),
                    reasons=reasons,
                    class_count=None,
                ))
                col_meta.is_potential_target = True

    # Filter by confidence threshold (>= 0.40) and sort descending
    viable_candidates = [c for c in candidates if c.confidence_score >= 0.40]
    viable_candidates.sort(key=lambda tc: tc.confidence_score, reverse=True)
    return viable_candidates


# ─── Dataset Fingerprint Assembly ────────────────────────────────

def generate_fingerprint(df: pd.DataFrame) -> Tuple[DatasetFingerprint, List[ColumnMetadata], List[TargetCandidate]]:
    """
    Perform deep semantic analysis and emit comprehensive DatasetFingerprint.
    """
    total_rows = len(df)
    columns_meta: List[ColumnMetadata] = []
    numeric_cols: List[str] = []
    cat_cols: List[str] = []
    datetime_cols: List[str] = []
    id_cols: List[str] = []
    text_cols: List[str] = []
    constant_cols: List[str] = []
    high_card_cats: List[str] = []
    cardinality_summary: Dict[str, int] = {}
    quality_warnings: List[str] = []

    for col in df.columns:
        s = df[col]
        null_count = int(s.isnull().sum())
        null_pct = round((null_count / max(total_rows, 1)) * 100, 2)
        unique_count = int(s.nunique())
        card_ratio = round(unique_count / max(total_rows, 1), 4)
        cardinality_summary[col] = unique_count

        # Perform deep semantic inference
        semantic = infer_column_semantics(s, col, total_rows)
        stats = compute_column_stats(s, semantic.inferred_type, semantic.parsed_numeric_series)

        col_meta = ColumnMetadata(
            name=col,
            raw_dtype=str(s.dtype),
            inferred_type=semantic.inferred_type,
            subtype=semantic.subtype,
            confidence=semantic.confidence,
            reasons=semantic.reasons,
            null_count=null_count,
            null_percentage=null_pct,
            unique_count=unique_count,
            cardinality_ratio=card_ratio,
            is_identifier=semantic.is_identifier,
            is_potential_target=False,
            is_constant=semantic.is_constant,
            sample_values=s.dropna().head(5).tolist(),
            stats=stats,
        )
        columns_meta.append(col_meta)

        if semantic.is_constant:
            constant_cols.append(col)
            quality_warnings.append(f"Column '{col}' has zero variance (constant).")
        elif semantic.is_identifier:
            id_cols.append(col)
        elif semantic.inferred_type in (InferredDtype.NUMERIC_CONTINUOUS, InferredDtype.NUMERIC_DISCRETE):
            numeric_cols.append(col)
        elif semantic.inferred_type in (InferredDtype.CATEGORICAL_NOMINAL, InferredDtype.CATEGORICAL_ORDINAL, InferredDtype.BOOLEAN):
            cat_cols.append(col)
        elif semantic.inferred_type == InferredDtype.DATETIME:
            datetime_cols.append(col)
        elif semantic.inferred_type in (InferredDtype.TEXT, InferredDtype.FREE_TEXT):
            text_cols.append(col)
            if card_ratio > 0.30:
                high_card_cats.append(col)

    # Primary entity key: first identifier column if available
    primary_key = id_cols[0] if id_cols else None

    # Temporal structure detection
    temporal_props = detect_temporal_structure(df, datetime_cols)

    # Target Candidates discovery and ranking
    target_candidates = rank_target_candidates(df, columns_meta, total_rows)

    # Class balance for top classification candidate
    class_bal = None
    if target_candidates and target_candidates[0].task_type in (TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION):
        class_bal = calculate_class_balance(df, target_candidates[0].column_name)

    # Candidate task types
    task_candidates_set: Set[TaskType] = {tc.task_type for tc in target_candidates}
    if not task_candidates_set:
        task_candidates_set = {TaskType.CLUSTERING, TaskType.UNSUPERVISED_EDA}
    else:
        task_candidates_set.add(TaskType.CLUSTERING)

    if temporal_props and temporal_props.is_sorted and temporal_props.distinct_dates_count and temporal_props.distinct_dates_count >= 10:
        task_candidates_set.add(TaskType.TIME_SERIES_FORECASTING)

    # Dataset Summary string
    summary_parts = [f"Dataset shape {df.shape[0]} rows x {df.shape[1]} columns"]
    if primary_key:
        summary_parts.append(f"primary key '{primary_key}'")
    if target_candidates:
        top_tc = target_candidates[0]
        summary_parts.append(f"primary target '{top_tc.column_name}' ({top_tc.task_type.value})")
    else:
        summary_parts.append("unsupervised / clustering task")
    dataset_summary = "; ".join(summary_parts)

    missing_cells = int(df.isnull().sum().sum())
    missing_summary = {
        "total_missing_cells": missing_cells,
        "missing_percentage": round((missing_cells / max(df.size, 1)) * 100, 2),
        "columns_with_missing": [c for c in df.columns if df[c].isnull().sum() > 0],
    }

    fingerprint = DatasetFingerprint(
        shape=[int(df.shape[0]), int(df.shape[1])],
        memory_mb=round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
        dataset_summary=dataset_summary,
        numeric_columns=numeric_cols,
        categorical_columns=cat_cols,
        datetime_columns=datetime_cols,
        identifier_columns=id_cols,
        text_columns=text_cols,
        constant_columns=constant_cols,
        high_cardinality_categoricals=high_card_cats,
        primary_entity_key=primary_key,
        temporal=temporal_props,
        class_balance=class_bal,
        missingness_summary=missing_summary,
        cardinality_summary=cardinality_summary,
        quality_warnings=quality_warnings,
        task_type_candidates=list(task_candidates_set),
    )

    return fingerprint, columns_meta, target_candidates
