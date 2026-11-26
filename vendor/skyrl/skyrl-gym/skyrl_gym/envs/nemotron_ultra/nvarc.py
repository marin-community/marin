# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ARC-AGI grid and transform verifiers ported from NVIDIA NeMo Gym."""

from __future__ import annotations

import json
import re
from typing import Any

from verifyit.adapters.skyrl import grade_grid_candidate

from skyrl_gym.envs.nemotron_ultra.answer_extraction import final_answer_text, last_boxed_answer


def valid_grid(value: Any) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and isinstance(value[0], list)
        and bool(value[0])
        and all(isinstance(row, list) and len(row) == len(value[0]) for row in value)
        and all(type(cell) is int and 0 <= cell <= 9 for row in value for cell in row)
    )


def parse_grid(text: str) -> list[list[int]] | None:
    """Parse a JSON array or NVIDIA Board.from_text digit rows, 0-9 palette."""
    text = final_answer_text(text)
    boxed = last_boxed_answer(text)
    if boxed is not None:
        text = boxed
    text = text.strip()
    if text.startswith("["):
        try:
            candidate = json.loads(text)
        except json.JSONDecodeError:
            return None
        return candidate if valid_grid(candidate) else None
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not re.fullmatch(r"[0-9\s]+", line):
            return None
        cells = line.split() if " " in line or "\t" in line else list(line)
        rows.append([int(cell) for cell in cells])
    return rows if valid_grid(rows) else None


def _unfenced_transform(text: str) -> str | None:
    match = re.search(r"^def transform\b", text, re.MULTILINE)
    if match is None:
        return None
    # Keep module-level imports from the preamble; drop prose around the code.
    imports = [
        line for line in text[: match.start()].splitlines() if re.match(r"\s*(?:import\s|from\s.+?\simport\b)", line)
    ]
    return "\n".join([*imports, text[match.start() :]]).strip()


def extract_python(text: str) -> str | None:
    text = final_answer_text(text)
    blocks = re.findall(r"```python\s*\n(.*?)```", text, re.DOTALL)
    if blocks:
        return blocks[-1].strip()
    blocks = re.findall(r"```\s*\n(.*?)```", text, re.DOTALL)
    if blocks:
        return blocks[-1].strip()
    return _unfenced_transform(text)


def grade_transductive_arc(text: str, record: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    predicted = parse_grid(text)
    correct = grade_grid_candidate(record["expected_output"], predicted).reward == 1.0
    return float(correct), {
        "agent_mode": "transductive",
        "extraction_successful": predicted is not None,
        "exact_match": correct,
        "predicted_output": predicted,
    }
