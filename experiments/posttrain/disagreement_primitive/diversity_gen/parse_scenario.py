# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""JSON-output parser for Stage 2 (scenario) responses.

The LM is asked to return a single JSON object with shape:
    {
      "scenario_text": str,
      "user_query": str,
      "system_prompt": str,        # may be empty
      "axis_values_embodied": {axis_name -> spectrum_value},
      "rubric": {
        "good_indicators": list[str],
        "bad_indicators": list[str],
        "key_tension": str,
      }
    }

Validates that shape strictly. On any deviation raises ValueError; the
orchestrator can then retry.
"""

from __future__ import annotations

import json
from typing import Any

REQUIRED_TOP_KEYS = {
    "scenario_text",
    "user_query",
    "system_prompt",
    "axis_values_embodied",
    "rubric",
}
REQUIRED_RUBRIC_KEYS = {"good_indicators", "bad_indicators", "key_tension"}


def parse_scenario_response(raw_text: str) -> dict[str, Any]:
    """Parse + validate one scenario JSON response. Raise ValueError on deviation."""
    if not raw_text or not raw_text.strip():
        raise ValueError("empty response")

    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"json parse error: {exc}") from exc

    if not isinstance(parsed, dict):
        raise ValueError(f"top-level value is not a dict: {type(parsed).__name__}")

    missing = REQUIRED_TOP_KEYS - set(parsed.keys())
    if missing:
        raise ValueError(f"missing top-level keys: {sorted(missing)}")

    if not isinstance(parsed["scenario_text"], str) or not parsed["scenario_text"].strip():
        raise ValueError("scenario_text must be a non-empty string")
    if not isinstance(parsed["user_query"], str) or not parsed["user_query"].strip():
        raise ValueError("user_query must be a non-empty string")
    if not isinstance(parsed["system_prompt"], str):
        # may be empty string but must be string
        raise ValueError("system_prompt must be a string (may be empty)")

    axis_values = parsed["axis_values_embodied"]
    if not isinstance(axis_values, dict):
        raise ValueError(f"axis_values_embodied must be a dict, got {type(axis_values).__name__}")
    for k, v in axis_values.items():
        if not isinstance(k, str) or not isinstance(v, str):
            raise ValueError(f"axis_values_embodied entry {k!r} -> {v!r} must be str -> str")

    rubric = parsed["rubric"]
    if not isinstance(rubric, dict):
        raise ValueError(f"rubric must be a dict, got {type(rubric).__name__}")
    missing_rb = REQUIRED_RUBRIC_KEYS - set(rubric.keys())
    if missing_rb:
        raise ValueError(f"rubric missing keys: {sorted(missing_rb)}")
    for list_key in ("good_indicators", "bad_indicators"):
        if not isinstance(rubric[list_key], list) or not rubric[list_key]:
            raise ValueError(f"rubric.{list_key} must be a non-empty list")
        for i, item in enumerate(rubric[list_key]):
            if not isinstance(item, str) or not item.strip():
                raise ValueError(f"rubric.{list_key}[{i}] must be non-empty string")
    if not isinstance(rubric["key_tension"], str) or not rubric["key_tension"].strip():
        raise ValueError("rubric.key_tension must be a non-empty string")

    return parsed
