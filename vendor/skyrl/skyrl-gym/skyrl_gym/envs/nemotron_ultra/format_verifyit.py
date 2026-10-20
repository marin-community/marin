"""Source-format predicates registered as client IFEval constraints."""

from __future__ import annotations

import json
import math
import re
from typing import Any

from verifyit.execution.worker import call_bounded


def _line_regex(text: str, params: dict) -> tuple[bool, str]:
    patterns = [re.compile(pattern) for pattern in params["verify_regex"]]
    matching_lines = sum(any(pattern.search(line) for pattern in patterns) for line in text.split("\n"))
    minimum = params["verify_min_matches"]
    passed = matching_lines >= minimum
    return passed, json.dumps({"matching_lines": matching_lines, "min_matches": minimum, "passed": passed})


def _markers(text: str, params: dict) -> tuple[bool, str]:
    expected = params["expected_markers"]
    missing = [marker for marker in expected if marker not in text]
    expected_set = set(expected)
    spurious = [
        match.group(0)
        for pattern in params["patterns"]
        for match in re.finditer(pattern, text)
        if match.group(0) not in expected_set
    ]
    passed = not missing and not spurious
    return passed, json.dumps(
        {
            "expected": expected,
            "missing": missing,
            "spurious": spurious,
            "passed": passed,
        }
    )


def _grade_format_ifeval(text: str, verifier: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    """Validate task parameters before asking IFEval to evaluate the candidate."""
    try:
        from verifyit.modes.grade_ifeval import grade_ifeval_candidate
        from verifyit.spec import Constraint, EmptyOutputPolicy, IfevalSpec
    except ImportError:
        return 0.0, {
            "error_type": "verification_error",
            "error_message": "verifyit IFEval dependency unavailable",
        }

    try:
        if not isinstance(verifier, dict):
            raise ValueError("Format verifier must be an object")
        kind = verifier.get("type")
        if kind in {"regex", "inline_prose"}:
            name = "marin_skyrl:format_line_regex"
            params = {
                "verify_regex": verifier.get("verify_regex", []),
                "verify_min_matches": verifier.get("verify_min_matches", 1),
            }
            minimum = params["verify_min_matches"]
            if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum <= 0:
                raise ValueError("Match threshold must be a positive integer")
            patterns = params["verify_regex"]
            check = _line_regex
        elif kind == "string_match":
            name = "marin_skyrl:format_markers"
            params = {
                "expected_markers": verifier.get("expected_markers", []),
                "patterns": verifier.get("patterns", []),
            }
            if not isinstance(params["expected_markers"], list) or not all(
                isinstance(marker, str) and marker for marker in params["expected_markers"]
            ):
                raise ValueError("Expected markers must be strings")
            patterns = params["patterns"]
            check = _markers
        else:
            raise ValueError("Unsupported format verifier type")
        if not isinstance(patterns, list) or not all(isinstance(pattern, str) and pattern for pattern in patterns):
            raise ValueError("Regex patterns must be strings")
        if not patterns and (kind != "string_match" or not params["expected_markers"]):
            raise ValueError("Format policy must contain a nonempty constraint")
        for pattern in patterns:
            re.compile(pattern)
    except (ValueError, TypeError, re.error):
        return 0.0, {
            "error_type": "schema_error",
            "error_message": "Invalid format verifier configuration",
        }

    try:
        verdict = grade_ifeval_candidate(
            IfevalSpec((Constraint(name, params),), empty_output=EmptyOutputPolicy.GRADE),
            text,
            registry={name: check},
        )
        detail = json.loads(verdict.detail["constraints"][0]["detail"])
        return verdict.reward, detail
    except Exception:
        return 0.0, {
            "error_type": "verification_error",
            "error_message": "Format verification failed",
        }


def grade_format_verifyit(text: str, verifier: dict[str, Any], timeout: float = 5.0) -> tuple[float, dict[str, Any]]:
    """Bound source-format checks with the shared process deadline."""
    try:
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not 0 < timeout <= 300
            or not math.isfinite(timeout)
        ):
            return 0.0, {
                "error_type": "schema_error",
                "error_message": "Invalid format verification deadline",
            }
        return call_bounded(_grade_format_ifeval, text, verifier, timeout=timeout)
    except Exception:
        return 0.0, {
            "error_type": "verification_error",
            "error_message": "Format verification failed",
        }
