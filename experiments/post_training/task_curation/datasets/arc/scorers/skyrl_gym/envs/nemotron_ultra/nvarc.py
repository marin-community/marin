# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ARC-AGI grid and transform verifiers ported from NVIDIA NeMo Gym."""

from __future__ import annotations

import json
import re
from typing import Any

from skyrl_gym.envs.nemotron_ultra.answer_extraction import final_answer_text, last_boxed_answer
from skyrl_gym.envs.nemotron_ultra.sandbox import MAX_VERIFIER_OUTPUT_CHARACTERS, SandboxClient


def _valid_grid(value: Any) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and isinstance(value[0], list)
        and bool(value[0])
        and all(isinstance(row, list) and len(row) == len(value[0]) for row in value)
        and all(type(cell) is int and 0 <= cell <= 9 for row in value for cell in row)
    )


def parse_grid(text: str) -> list[list[int]] | None:
    """Match NVIDIA Board.from_text with the integer color palette."""
    text = final_answer_text(text)
    boxed = last_boxed_answer(text)
    if boxed is not None:
        text = boxed
    rows = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not re.fullmatch(r"[0-9\s]+", line):
            return None
        cells = line.split() if " " in line or "\t" in line else list(line)
        rows.append([int(cell) for cell in cells])
    return rows if _valid_grid(rows) else None


def _extract_python(text: str) -> str | None:
    text = final_answer_text(text)
    blocks = re.findall(r"```python\s*\n(.*?)```", text, re.DOTALL)
    if blocks:
        return blocks[-1].strip()
    blocks = re.findall(r"```\s*\n(.*?)```", text, re.DOTALL)
    if blocks:
        return blocks[-1].strip()
    return text.strip() if "def transform" in text else None


def _execute_python(code: str, input_grid: list[list[int]], timeout_seconds: int, sandbox: SandboxClient):
    script = (
        "import json\n"
        + code
        + "\n"
        + f"_arc_result = transform({input_grid!r})\n"
        + "print(json.dumps(_arc_result.tolist() if isinstance(_arc_result, __import__('numpy').ndarray) else _arc_result))"
    )
    result = sandbox.execute(
        script, language="python", timeout_seconds=timeout_seconds, max_output_characters=MAX_VERIFIER_OUTPUT_CHARACTERS
    )
    if result.get("process_status") in {"error", "unknown"}:
        raise RuntimeError(f"ARC sandbox unavailable: {result}")
    if result.get("process_status") != "completed":
        return None, result
    try:
        value = json.loads(result.get("stdout", "").strip().rsplit("\n", 1)[-1])
    except (json.JSONDecodeError, IndexError):
        return None, result
    return (value if _valid_grid(value) else None), result


def grade_transductive_arc(text: str, record: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    predicted = parse_grid(text)
    correct = predicted is not None and predicted == record["expected_output"]
    return float(correct), {
        "agent_mode": "transductive",
        "extraction_successful": predicted is not None,
        "exact_match": correct,
        "predicted_output": predicted,
    }


def grade_inductive_arc(
    text: str,
    record: dict[str, Any],
    *,
    sandbox: SandboxClient,
    python_timeout_seconds: int = 30,
) -> tuple[float, dict[str, Any]]:
    code = _extract_python(text)
    execution = None
    predicted = None
    if code is not None:
        predicted, execution = _execute_python(code, record["test_input"], python_timeout_seconds, sandbox)
    correct = predicted is not None and predicted == record["expected_output"]
    return float(correct), {
        "agent_mode": "inductive",
        "extraction_successful": predicted is not None,
        "exact_match": correct,
        "predicted_output": predicted,
        "execution_output": execution,
    }
