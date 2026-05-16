# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""JSON-output parser for Stage 1 (understanding) responses.

The LM is asked to return a JSON object with shape:
    {
      "behavior_understanding": str,
      "scientific_motivation": str,
      "variation_axes": list[{axis, description, spectrum, why_it_matters}],
    }

This module validates that shape strictly. On any deviation it raises
ValueError with a specific message — the orchestrator can then retry.

Also provides `merge_demographic_axes`, which appends
STANDARD_DEMOGRAPHIC_AXES (from `prompts.py`) to the LM-produced
behavior-specific axes so downstream consumers see one unified list with
`standard: True` tagging on the demographic ones.
"""

from __future__ import annotations

import json
from typing import Any

REQUIRED_TOP_KEYS = {"behavior_understanding", "scientific_motivation", "variation_axes"}
REQUIRED_AXIS_KEYS = {"axis", "description", "spectrum", "why_it_matters", "default_spectrum_value"}
MIN_AXES = 4
MAX_AXES = 6
MIN_SPECTRUM = 4
MAX_SPECTRUM = 6


def parse_understanding_response(raw_text: str) -> dict[str, Any]:
    """Parse + validate the LM's JSON response. Raise ValueError on any deviation.

    Returns a dict with the three required top-level keys, where
    `variation_axes` is a list[dict] each containing the four required axis
    keys plus possibly extras (extras are tolerated).
    """
    if not raw_text or not raw_text.strip():
        raise ValueError("empty response")

    # Some models still wrap JSON in ```json ... ``` fences even in
    # json_object response_format. Strip defensively.
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
        raise ValueError(f"missing required top-level keys: {sorted(missing)}")

    for str_key in ("behavior_understanding", "scientific_motivation"):
        if not isinstance(parsed[str_key], str) or not parsed[str_key].strip():
            raise ValueError(f"{str_key} must be a non-empty string")

    axes = parsed["variation_axes"]
    if not isinstance(axes, list):
        raise ValueError(f"variation_axes must be a list, got {type(axes).__name__}")
    if not (MIN_AXES <= len(axes) <= MAX_AXES):
        raise ValueError(f"variation_axes has {len(axes)} entries; expected {MIN_AXES}-{MAX_AXES}")

    for i, ax in enumerate(axes):
        if not isinstance(ax, dict):
            raise ValueError(f"variation_axes[{i}] is not a dict")
        missing_ax = REQUIRED_AXIS_KEYS - set(ax.keys())
        if missing_ax:
            raise ValueError(f"variation_axes[{i}] missing keys: {sorted(missing_ax)}")
        if not isinstance(ax["axis"], str) or not ax["axis"].strip():
            raise ValueError(f"variation_axes[{i}].axis must be non-empty string")
        if not isinstance(ax["description"], str) or not ax["description"].strip():
            raise ValueError(f"variation_axes[{i}].description must be non-empty string")
        if not isinstance(ax["why_it_matters"], str) or not ax["why_it_matters"].strip():
            raise ValueError(f"variation_axes[{i}].why_it_matters must be non-empty string")
        spec = ax["spectrum"]
        if not isinstance(spec, list):
            raise ValueError(f"variation_axes[{i}].spectrum must be a list")
        if not (MIN_SPECTRUM <= len(spec) <= MAX_SPECTRUM):
            raise ValueError(
                f"variation_axes[{i}].spectrum has {len(spec)} values; " f"expected {MIN_SPECTRUM}-{MAX_SPECTRUM}"
            )
        for j, v in enumerate(spec):
            if not isinstance(v, str) or not v.strip():
                raise ValueError(f"variation_axes[{i}].spectrum[{j}] must be non-empty string")

        default_value = ax["default_spectrum_value"]
        if not isinstance(default_value, str) or not default_value.strip():
            raise ValueError(f"variation_axes[{i}].default_spectrum_value must be non-empty string")
        if default_value not in spec:
            raise ValueError(
                f"variation_axes[{i}].default_spectrum_value {default_value!r} is not "
                f"one of the listed spectrum values: {spec}"
            )

    return parsed
