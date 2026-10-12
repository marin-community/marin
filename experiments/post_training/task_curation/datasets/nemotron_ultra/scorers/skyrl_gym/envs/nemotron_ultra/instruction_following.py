# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verifiable-instruction reward ported from NVIDIA NeMo Gym."""

from __future__ import annotations

import threading
from typing import Any

from verifiable_instructions import instructions_registry

_NLTK_LOCK = threading.Lock()
_NLTK_READY = False


def _ensure_nltk_data() -> None:
    global _NLTK_READY
    if _NLTK_READY:
        return
    with _NLTK_LOCK:
        if _NLTK_READY:
            return
        import nltk

        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            if not nltk.download("punkt_tab", quiet=True):
                raise RuntimeError("Failed to install the NLTK punkt_tab data required by instruction verification")
        _NLTK_READY = True


def grade_instruction_following(text: str, record: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    """Evaluate every per-row constraint with NVIDIA's pinned instruction registry."""
    _ensure_nltk_data()
    results: list[bool] = []
    errors: list[str | None] = []
    for instruction_id, kwargs in zip(record["instruction_id_list"], record["kwargs"]):
        try:
            instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
            instruction = instruction_cls(instruction_id)
            instruction.build_description(**{key: value for key, value in (kwargs or {}).items() if value is not None})
            results.append(bool(instruction.check_following(text)))
            errors.append(None)
        except Exception as error:
            results.append(False)
            errors.append(f"{type(error).__name__}: {error}")

    grading_mode = record.get("grading_mode", "binary")
    if grading_mode == "binary":
        reward = float(all(results))
    elif grading_mode == "fraction":
        reward = sum(results) / len(results) if results else 0.0
    else:
        raise ValueError(f"Invalid instruction-following grading mode: {grading_mode!r}")
    return reward, {
        "follow_all_instructions": all(results),
        "follow_instruction_list": results,
        "instruction_errors": errors,
        "grading_mode": grading_mode,
    }
