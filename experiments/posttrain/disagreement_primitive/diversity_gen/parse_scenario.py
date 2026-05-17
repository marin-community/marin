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

# For single_call_diverse: response is one JSON object with `scenarios: [...]`.
# Each scenario has the keys above PLUS the per-strategy fields below.
SCD_REQUIRED_PER_SCENARIO = REQUIRED_TOP_KEYS | {
    "is_default_scenario",
    "varied_axis",
    "varied_value",
    "context_summary",
}


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


def parse_single_call_diverse_response(
    raw_text: str,
    expected_n_total: int,
    axes_names: list[str],
) -> list[dict[str, Any]]:
    """Parse + validate the `single_call_diverse` strategy's response.

    The LM returns one JSON object: {"scenarios": [N+1 items]}. Validates:
      - Exactly `expected_n_total` items.
      - First item has is_default_scenario=true, empty varied_axis/value.
      - Items 2..N have is_default_scenario=false and varied_axis covering
        each axis name from `axes_names` exactly once.
      - Each item has all required scenario fields (text, query, rubric, etc.).
      - context_summary fields are pairwise distinct (case-insensitive string
        match — a weak check but catches verbatim repeats).

    Returns the list of scenario dicts. Raises ValueError on deviation.
    """
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
    if "scenarios" not in parsed:
        raise ValueError("missing 'scenarios' key")
    scenarios = parsed["scenarios"]
    if not isinstance(scenarios, list):
        raise ValueError(f"scenarios must be a list, got {type(scenarios).__name__}")
    if len(scenarios) != expected_n_total:
        raise ValueError(f"expected {expected_n_total} scenarios, got {len(scenarios)}")

    seen_varied_axes: list[str] = []
    seen_context_summaries: list[str] = []
    for i, s in enumerate(scenarios):
        if not isinstance(s, dict):
            raise ValueError(f"scenarios[{i}] is not a dict")
        missing = SCD_REQUIRED_PER_SCENARIO - set(s.keys())
        if missing:
            raise ValueError(f"scenarios[{i}] missing keys: {sorted(missing)}")
        # field-by-field validation reuses the per-scenario parser semantics
        if not isinstance(s["scenario_text"], str) or not s["scenario_text"].strip():
            raise ValueError(f"scenarios[{i}].scenario_text must be non-empty string")
        if not isinstance(s["user_query"], str) or not s["user_query"].strip():
            raise ValueError(f"scenarios[{i}].user_query must be non-empty string")
        if not isinstance(s["system_prompt"], str):
            raise ValueError(f"scenarios[{i}].system_prompt must be string (may be empty)")
        if not isinstance(s["axis_values_embodied"], dict):
            raise ValueError(f"scenarios[{i}].axis_values_embodied must be dict")
        for k, v in s["axis_values_embodied"].items():
            if not isinstance(k, str) or not isinstance(v, str):
                raise ValueError(f"scenarios[{i}].axis_values_embodied[{k!r}] must be str->str")
        rubric = s["rubric"]
        if not isinstance(rubric, dict):
            raise ValueError(f"scenarios[{i}].rubric must be dict")
        rb_missing = REQUIRED_RUBRIC_KEYS - set(rubric.keys())
        if rb_missing:
            raise ValueError(f"scenarios[{i}].rubric missing: {sorted(rb_missing)}")
        for list_key in ("good_indicators", "bad_indicators"):
            if not isinstance(rubric[list_key], list) or not rubric[list_key]:
                raise ValueError(f"scenarios[{i}].rubric.{list_key} must be non-empty list")
        if not isinstance(rubric["key_tension"], str) or not rubric["key_tension"].strip():
            raise ValueError(f"scenarios[{i}].rubric.key_tension must be non-empty string")
        if not isinstance(s["is_default_scenario"], bool):
            raise ValueError(f"scenarios[{i}].is_default_scenario must be bool")
        if not isinstance(s["varied_axis"], str):
            raise ValueError(f"scenarios[{i}].varied_axis must be string")
        if not isinstance(s["varied_value"], str):
            raise ValueError(f"scenarios[{i}].varied_value must be string")
        if not isinstance(s["context_summary"], str) or not s["context_summary"].strip():
            raise ValueError(f"scenarios[{i}].context_summary must be non-empty string")

        if s["is_default_scenario"]:
            if s["varied_axis"] or s["varied_value"]:
                raise ValueError(f"scenarios[{i}] is default but varied_axis/value are not empty")
        else:
            if not s["varied_axis"]:
                raise ValueError(f"scenarios[{i}] is variation but varied_axis is empty")
            if s["varied_axis"] not in axes_names:
                raise ValueError(f"scenarios[{i}].varied_axis {s['varied_axis']!r} not in axes list {axes_names}")
            if s["varied_axis"] in seen_varied_axes:
                raise ValueError(f"scenarios[{i}].varied_axis {s['varied_axis']!r} used more than once")
            seen_varied_axes.append(s["varied_axis"])
        seen_context_summaries.append(s["context_summary"].strip().lower())

    # Each axis appears exactly once across the variations
    missing_axes = set(axes_names) - set(seen_varied_axes)
    if missing_axes:
        raise ValueError(f"axes missing from variations: {sorted(missing_axes)}")

    # Weak diversity check: no two context_summaries are byte-identical
    if len(seen_context_summaries) != len(set(seen_context_summaries)):
        raise ValueError("at least two context_summary fields are identical strings (insufficient diversity)")

    return scenarios


# For the repair pass (dart.md §11.11): the LM rewrites scenario_text /
# user_query / system_prompt / rubric, preserving these immutable fields:
REPAIR_IMMUTABLE_FIELDS = (
    "scenario_id",
    "scenario_n",
    "is_default_scenario",
    "varied_axis",
    "varied_value",
)
REPAIR_REQUIRED_PER_SCENARIO = set(REPAIR_IMMUTABLE_FIELDS) | {
    "scenario_text",
    "user_query",
    "system_prompt",
    "rubric",
}

# Strict-mode-only fields that the LM self-reports about its own output.
# Present when `--strict-schema` is used (response_format=json_schema). Absent
# otherwise — the parser tolerates their absence to keep backwards compat with
# the V1/V2/V2.5/V3 outputs already on disk.
REPAIR_PLACEHOLDER_FIELDS = {"contains_placeholder", "placeholder_notes"}


def parse_repair_response(
    raw_text: str,
    source_scenarios: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Parse + validate the single-pass repair response (dart.md §11.11).

    Returns the rewritten scenarios as a list, in the same order as
    `source_scenarios`. Raises ValueError on any deviation.

    Validation rules:
      - Top-level shape: {"scenarios": [...]} with exactly len(source) items.
      - Each item has all of `REPAIR_REQUIRED_PER_SCENARIO`.
      - Each item's immutable fields (`REPAIR_IMMUTABLE_FIELDS`) match the
        corresponding source scenario verbatim. Order is taken from the source
        list (matched by position), and `scenario_id` must align.
      - Rubric has the same shape required by `parse_scenario_response`.
      - `scenario_text` and `user_query` are non-empty strings AND differ from
        the source (i.e., a non-trivial edit happened — the LM did not echo).
        The threshold for "non-trivial" at this pilot stage is just
        string-inequality; we will tighten if we see paraphrase-only outputs.
    """
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
    if "scenarios" not in parsed:
        raise ValueError("missing 'scenarios' key")
    items = parsed["scenarios"]
    if not isinstance(items, list):
        raise ValueError(f"scenarios must be a list, got {type(items).__name__}")
    if len(items) != len(source_scenarios):
        raise ValueError(
            f"expected {len(source_scenarios)} rewritten scenarios, got {len(items)}"
        )

    for i, (rew, src) in enumerate(zip(items, source_scenarios, strict=True)):
        if not isinstance(rew, dict):
            raise ValueError(f"scenarios[{i}] is not a dict")
        missing = REPAIR_REQUIRED_PER_SCENARIO - set(rew.keys())
        if missing:
            raise ValueError(f"scenarios[{i}] missing keys: {sorted(missing)}")

        for k in REPAIR_IMMUTABLE_FIELDS:
            if rew[k] != src[k]:
                raise ValueError(
                    f"scenarios[{i}] immutable field {k!r} drifted: "
                    f"src={src[k]!r} rewrite={rew[k]!r}"
                )

        if not isinstance(rew["scenario_text"], str) or not rew["scenario_text"].strip():
            raise ValueError(f"scenarios[{i}].scenario_text must be non-empty string")
        if not isinstance(rew["user_query"], str) or not rew["user_query"].strip():
            raise ValueError(f"scenarios[{i}].user_query must be non-empty string")
        if not isinstance(rew["system_prompt"], str):
            raise ValueError(f"scenarios[{i}].system_prompt must be string (may be empty)")

        rubric = rew["rubric"]
        if not isinstance(rubric, dict):
            raise ValueError(f"scenarios[{i}].rubric must be a dict")
        rb_missing = REQUIRED_RUBRIC_KEYS - set(rubric.keys())
        if rb_missing:
            raise ValueError(f"scenarios[{i}].rubric missing: {sorted(rb_missing)}")
        for list_key in ("good_indicators", "bad_indicators"):
            if not isinstance(rubric[list_key], list) or not rubric[list_key]:
                raise ValueError(f"scenarios[{i}].rubric.{list_key} must be non-empty list")
            for j, item in enumerate(rubric[list_key]):
                if not isinstance(item, str) or not item.strip():
                    raise ValueError(f"scenarios[{i}].rubric.{list_key}[{j}] must be non-empty string")
        if not isinstance(rubric["key_tension"], str) or not rubric["key_tension"].strip():
            raise ValueError(f"scenarios[{i}].rubric.key_tension must be non-empty string")

        # Anti-echo: at least one of scenario_text OR user_query must differ from
        # source. Some rewrites legitimately keep the original user_query (which
        # is already a good prompt) while rewriting scenario_text. We require a
        # non-trivial change on AT LEAST ONE field, not both.
        text_echo = rew["scenario_text"].strip() == (src.get("scenario_text") or "").strip()
        query_echo = rew["user_query"].strip() == (src.get("user_query") or "").strip()
        if text_echo and query_echo:
            raise ValueError(
                f"scenarios[{i}] BOTH scenario_text AND user_query are byte-identical "
                f"to source — the LM echoed instead of rewriting"
            )

        # Strict-mode-only fields (present iff the orchestrator used the strict
        # json_schema response_format). Validate their types when present.
        if "contains_placeholder" in rew:
            if not isinstance(rew["contains_placeholder"], bool):
                raise ValueError(
                    f"scenarios[{i}].contains_placeholder must be bool, got {type(rew['contains_placeholder']).__name__}"
                )
            if "placeholder_notes" not in rew:
                raise ValueError(f"scenarios[{i}] has contains_placeholder but no placeholder_notes")
            if not isinstance(rew["placeholder_notes"], str):
                raise ValueError(f"scenarios[{i}].placeholder_notes must be string")

    return items
