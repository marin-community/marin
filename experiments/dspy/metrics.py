import dspy
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict

MATCH_THRESHOLD = 0.7

# State abbreviation to full name mapping
STATE_ABBR_TO_NAME = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas",
    "CA": "California", "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware",
    "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho",
    "IL": "Illinois", "IN": "Indiana", "IA": "Iowa", "KS": "Kansas",
    "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
    "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi",
    "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada",
    "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York",
    "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma",
    "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina",
    "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah",
    "VT": "Vermont", "VA": "Virginia", "WA": "Washington", "WV": "West Virginia",
    "WI": "Wisconsin", "WY": "Wyoming",
}

# Cache for gold standard data
_gold_data_cache: Dict[int, Dict] | None = None


def _load_gold_data(gold_path: str | Path = "data/gold.json") -> Dict[int, Dict]:
    """Load and cache gold standard data."""
    global _gold_data_cache
    if _gold_data_cache is not None:
        return _gold_data_cache

    gold_path = Path(gold_path)
    if not gold_path.exists():
        # Try downloading from the original repository
        url = "https://raw.githubusercontent.com/prrao87/structured-outputs/main/data/gold.json"
        os.makedirs(gold_path.parent, exist_ok=True)
        os.system(f"curl -k -L {url} -o {gold_path}")

    if not gold_path.exists():
        raise FileNotFoundError(f"Gold standard file not found at {gold_path}")

    with open(gold_path, "r") as f:
        gold_data = json.load(f)

    # Create lookup by record_id
    _gold_data_cache = {record["record_id"]: record for record in gold_data}
    return _gold_data_cache


def _safe_get_nested(data: Dict, path: str, default=None) -> Any:
    """Safely get nested dictionary value using dot notation."""
    keys = path.split(".")
    current = data
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current


def _normalize_state(state_val: Any) -> str | None:
    """Normalize state value to full state name."""
    if state_val is None:
        return None

    state_str = str(state_val).strip()
    if not state_str:
        return None

    state_upper = state_str.upper()
    if state_upper in STATE_ABBR_TO_NAME:
        return STATE_ABBR_TO_NAME[state_upper].lower()

    state_lower = state_str.lower()
    for abbr, full_name in STATE_ABBR_TO_NAME.items():
        if full_name.lower() == state_lower:
            return full_name.lower()

    return state_lower


def _extract_date_part(val: Any) -> str | None:
    """Extract date part from datetime string, ignoring time component."""
    if val is None:
        return None
    return str(val).split("T")[0].strip()


def _normalize_date(val: Any) -> str | None:
    """Normalize date to ISO format (YYYY-MM-DD) from various formats."""
    if val is None:
        return None

    date_str = str(val).strip()
    if not date_str:
        return None

    date_str = date_str.split("T")[0].strip()

    if len(date_str) == 10 and date_str.count("-") == 2:
        parts = date_str.split("-")
        if len(parts) == 3 and parts[0].isdigit() and len(parts[0]) == 4:
            return date_str

    try:
        from datetime import datetime
        for fmt in ["%B %d, %Y", "%b %d, %Y", "%m/%d/%Y", "%d/%m/%Y"]:
            try:
                dt = datetime.strptime(date_str, fmt)
                return dt.strftime("%Y-%m-%d")
            except ValueError:
                continue
    except Exception:
        pass

    return date_str


def _normalize_address_line(line: Any) -> str | None:
    """Normalize address line for comparison."""
    if line is None:
        return None
    if isinstance(line, list):
        if len(line) == 0:
            return None
        line = line[0]
    return str(line).lower().strip()


def _compare_values(val1: Any, val2: Any, field_name: str = "") -> bool:
    """Compare two values for equality, handling None and different types."""
    if val1 is None and val2 is None:
        return True
    if val1 is None or val2 is None:
        return False

    # Special handling for state fields
    if "state" in field_name.lower():
        norm_val1 = _normalize_state(val1)
        norm_val2 = _normalize_state(val2)
        return norm_val1 == norm_val2

    # Special handling for date fields
    if "birthdate" in field_name.lower():
        date1 = _normalize_date(val1)
        date2 = _normalize_date(val2)
        return date1 == date2
    elif "encounter_period" in field_name.lower():
        date1 = _extract_date_part(val1)
        date2 = _extract_date_part(val2)
        return date1 == date2

    # Handle array vs string for address lines
    if isinstance(val1, list) and isinstance(val2, str):
        if len(val1) == 1:
            return val1[0].lower().strip() == val2.lower().strip()
        return False

    if isinstance(val2, list) and isinstance(val1, str):
        if len(val2) == 1:
            return val1.lower().strip() == val2[0].lower().strip()
        return False

    # Handle list comparison
    if isinstance(val1, list) and isinstance(val2, list):
        return val1 == val2

    # Handle string comparison (case-insensitive)
    return str(val1).lower().strip() == str(val2).lower().strip()


def _evaluate_patient_fields(gold_record: Dict, result_record: Dict) -> tuple[int, int]:
    """Evaluate patient-level fields. Returns (matches, total)."""
    result_patient = result_record.get("patient", {})
    matches = 0
    total = 0

    # Top-level patient fields
    patient_fields = ["gender", "phone", "email", "maritalStatus", "age", "birthDate"]
    for field in patient_fields:
        gold_val = gold_record.get(field)
        result_val = result_patient.get(field)
        total += 1
        if _compare_values(gold_val, result_val, f"patient.{field}"):
            matches += 1

    # Patient name fields
    gold_name = gold_record.get("name", {}) or {}
    result_name = result_patient.get("name", {}) or {}
    name_fields = ["family", "given", "prefix"]
    for field in name_fields:
        gold_val = gold_name.get(field) if gold_name else None
        result_val = result_name.get(field) if result_name else None
        total += 1
        if _compare_values(gold_val, result_val, f"patient.name.{field}"):
            matches += 1

    # Patient address fields
    gold_addr = gold_record.get("address", {})
    result_addr = result_patient.get("address", {})
    if gold_addr is not None or result_addr is not None:
        addr_fields = ["line", "city", "state", "postalCode", "country"]
        for field in addr_fields:
            gold_val = gold_addr.get(field) if gold_addr else None
            result_val = result_addr.get(field) if result_addr else None
            total += 1
            if _compare_values(gold_val, result_val, f"patient.address.{field}"):
                matches += 1

    return matches, total


def _evaluate_practitioner_fields(gold_record: Dict, result_record: Dict) -> tuple[int, int]:
    """Evaluate practitioner fields. Returns (matches, total)."""
    gold_practitioners = gold_record.get("practitioner")
    result_practitioners = result_record.get("practitioner", [])
    matches = 0
    total = 0

    # Compare counts
    gold_count = len(gold_practitioners) if gold_practitioners else 0
    result_count = len(result_practitioners) if result_practitioners else 0
    total += 1
    if gold_count == result_count:
        matches += 1

    if gold_count == 0 or result_count == 0:
        return matches, total

    if not gold_practitioners or not result_practitioners:
        return matches, total

    # Compare address lines (set comparison)
    gold_address_lines = set()
    for prac in gold_practitioners:
        addr = prac.get("address", {})
        if addr and addr.get("line"):
            normalized = _normalize_address_line(addr["line"])
            if normalized:
                gold_address_lines.add(normalized)

    result_address_lines = set()
    for prac in result_practitioners:
        addr = prac.get("address", {})
        if addr and addr.get("line"):
            normalized = _normalize_address_line(addr["line"])
            if normalized:
                result_address_lines.add(normalized)

    total += 1
    if gold_address_lines == result_address_lines:
        matches += 1

    # Compare first practitioner fields
    gold_prac = gold_practitioners[0]
    result_prac = result_practitioners[0]

    # Name fields
    gold_name = gold_prac.get("name", {}) or {}
    result_name = result_prac.get("name", {}) or {}
    name_fields = ["family", "given", "prefix"]
    for field in name_fields:
        gold_val = gold_name.get(field) if gold_name else None
        result_val = result_name.get(field) if result_name else None
        total += 1
        if _compare_values(gold_val, result_val, f"practitioner.name.{field}"):
            matches += 1

    # Direct practitioner fields
    prac_fields = ["phone", "email"]
    for field in prac_fields:
        gold_val = gold_prac.get(field)
        result_val = result_prac.get(field)
        total += 1
        if _compare_values(gold_val, result_val, f"practitioner.{field}"):
            matches += 1

    # Other address fields
    gold_addr = gold_prac.get("address", {})
    result_addr = result_prac.get("address", {})
    if gold_addr is not None or result_addr is not None:
        addr_fields = ["city", "state", "postalCode", "country"]
        for field in addr_fields:
            gold_val = gold_addr.get(field) if gold_addr else None
            result_val = result_addr.get(field) if result_addr else None
            total += 1
            if _compare_values(gold_val, result_val, f"practitioner.address.{field}"):
                matches += 1

    return matches, total


def _evaluate_immunization_fields(gold_record: Dict, result_record: Dict) -> tuple[int, int]:
    """Evaluate immunization fields. Returns (matches, total)."""
    gold_immunizations = gold_record.get("immunization")
    result_immunizations = result_record.get("immunization", [])

    gold_count = len(gold_immunizations) if gold_immunizations else 0
    result_count = len(result_immunizations) if result_immunizations else 0

    total = 1
    matches = 1 if gold_count == result_count else 0
    return matches, total


def _evaluate_allergy_fields(gold_record: Dict, result_record: Dict) -> tuple[int, int]:
    """Evaluate allergy fields. Returns (matches, total)."""
    gold_allergy = gold_record.get("allergy", {})
    result_allergy = result_record.get("patient", {}).get("allergy", [])

    gold_substances = []
    if gold_allergy and gold_allergy.get("substance"):
        gold_substances = gold_allergy["substance"]

    result_substances = []
    if result_allergy:
        for allergy_obj in result_allergy:
            if allergy_obj.get("substance"):
                result_substances.extend(allergy_obj["substance"])

    gold_count = len(gold_substances)
    result_count = len(result_substances)

    total = 1
    matches = 1 if gold_count == result_count else 0
    return matches, total


def claim_verification_metric(example: dspy.Example, prediction: dspy.Prediction, trace = None) -> float:
    return int(example.label == prediction.label_int)


def field_extraction_metric(example: dspy.Example, prediction: dspy.Prediction, trace = None) -> float:
    """
    Evaluate field extraction by comparing prediction with gold standard.
    Returns a score between 0.0 and 1.0 based on field-level accuracy.
    """
    try:
        # Get record_id from example
        record_id = getattr(example, "record_id", None)
        if record_id is None:
            return 0.0

        # Load gold standard data
        gold_data = _load_gold_data()
        if record_id not in gold_data:
            return 0.0

        gold_record = gold_data[record_id]

        # Get result from prediction
        result_record = getattr(prediction, "result", None)
        if result_record is None:
            # Try to get it from prediction dict if it's a dict-like object
            if hasattr(prediction, "toDict"):
                result_record = prediction.toDict().get("result")
            elif isinstance(prediction, dict):
                result_record = prediction.get("result")

        if result_record is None:
            return 0.0

        # Evaluate all field categories
        total_matches = 0
        total_fields = 0

        patient_matches, patient_total = _evaluate_patient_fields(gold_record, result_record)
        total_matches += patient_matches
        total_fields += patient_total

        practitioner_matches, practitioner_total = _evaluate_practitioner_fields(gold_record, result_record)
        total_matches += practitioner_matches
        total_fields += practitioner_total

        immunization_matches, immunization_total = _evaluate_immunization_fields(gold_record, result_record)
        total_matches += immunization_matches
        total_fields += immunization_total

        allergy_matches, allergy_total = _evaluate_allergy_fields(gold_record, result_record)
        total_matches += allergy_matches
        total_fields += allergy_total

        # Return accuracy as a float between 0.0 and 1.0
        if total_fields == 0:
            return 0.0

        return (total_matches / total_fields) > MATCH_THRESHOLD

    except Exception as e:
        # Return 0.0 on any error
        return 0.0
