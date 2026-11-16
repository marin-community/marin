# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exact-answer chemistry verifier ported from NVIDIA NeMo Gym."""

from __future__ import annotations

import math
import re
from typing import Any

_SUPPORTED_PROPERTY_TYPES = {"count", "bool", "presence", "fragment"}
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")
_BOXED_RE = re.compile(r"\\boxed\{([^}]+)\}")
_DOUBLE_PAREN_RE = re.compile(r"\(\(([^)]+)\)\)")


def _extract_number(text: str, pattern: re.Pattern[str]) -> float | None:
    matches = pattern.findall(text)
    if not matches:
        return None
    inner = matches[-1].strip()
    try:
        return float(inner)
    except (TypeError, ValueError):
        numbers = _NUMBER_RE.findall(inner)
        if not numbers:
            return None
        try:
            return float(numbers[-1])
        except ValueError:
            return None


def grade_rdkit_chemistry(text: str, record: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    """Require the row-selected answer wrapper and NVIDIA's rounded exact match."""
    property_type = record["property_type"]
    if property_type not in _SUPPORTED_PROPERTY_TYPES:
        raise ValueError(f"Unsupported property_type={property_type!r}")
    pattern = _BOXED_RE if record.get("use_box_format", False) else _DOUBLE_PAREN_RE
    predicted = _extract_number(text.strip(), pattern)
    actual = float(record["expected_answer"])
    correct = predicted is not None and not math.isnan(predicted) and round(predicted) == round(actual)
    return float(correct), {
        "predicted_value": predicted,
        "expected_value": actual,
        "correct": correct,
        "property": record.get("property"),
        "property_type": property_type,
        "method": record.get("method"),
    }
