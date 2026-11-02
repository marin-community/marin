"""Trusted reference contracts selected before generating a candidate."""

from collections.abc import Mapping
from typing import Any

REFERENCE_KINDS = frozenset({"semantic", "symbolic"})


def reference_kind(record: Mapping[str, Any]) -> str | None:
    """Validate explicit reference metadata without guessing from answer content."""
    kind = record.get("math_reference_kind")
    if "math_reference_kind" in record and (not isinstance(kind, str) or kind not in REFERENCE_KINDS):
        raise ValueError("math_reference_kind must be semantic or symbolic")
    reference = record.get("expected_answer")
    question = record.get("question")
    if not isinstance(reference, str) or not reference.strip():
        raise ValueError("Expected answer must be nonempty text")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must be nonempty text")
    for value in (reference, question):
        # JSON permits escaped lone surrogates and control characters, but these
        # cannot form a valid text contract for the judge transport.
        value.encode("utf-8", errors="strict")
        if any(ord(character) < 32 and character not in "\n\r\t" for character in value):
            raise ValueError("Reference contract contains control characters")
    return kind


def prepare_math_reference(record: Mapping[str, Any], kind: str) -> dict[str, Any]:
    """Copy a trusted record and attach its explicitly selected grading contract."""
    if "math_reference_kind" in record and record["math_reference_kind"] != kind:
        raise ValueError("Conflicting math reference contracts")
    prepared = dict(record)
    prepared["math_reference_kind"] = kind
    reference_kind(prepared)
    return prepared
