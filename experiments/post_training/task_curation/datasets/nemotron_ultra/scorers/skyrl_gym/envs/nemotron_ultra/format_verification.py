# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Citation and freeform formatting rewards ported from NVIDIA NeMo Gym."""

from __future__ import annotations

import re
from typing import Any


def grade_format(text: str, verifier: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    verifier_type = verifier.get("type", "")
    if verifier_type in {"regex", "inline_prose"}:
        patterns = [re.compile(pattern) for pattern in verifier.get("verify_regex", [])]
        matching_lines = sum(any(pattern.search(line) for pattern in patterns) for line in text.split("\n"))
        minimum = verifier.get("verify_min_matches", 1)
        passed = matching_lines >= minimum
        return float(passed), {"matching_lines": matching_lines, "min_matches": minimum, "passed": passed}
    if verifier_type == "string_match":
        expected = verifier.get("expected_markers", [])
        missing = [marker for marker in expected if marker not in text]
        spurious: list[str] = []
        expected_set = set(expected)
        for pattern in verifier.get("patterns", []):
            spurious.extend(
                match.group(0) for match in re.finditer(pattern, text) if match.group(0) not in expected_set
            )
        passed = not missing and not spurious
        return float(passed), {"expected": expected, "missing": missing, "spurious": spurious, "passed": passed}
    raise NotImplementedError(f"Unsupported format verifier type {verifier_type!r}")
