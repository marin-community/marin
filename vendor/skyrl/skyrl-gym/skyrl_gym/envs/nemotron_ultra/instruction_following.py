# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verifiable-instruction reward ported from NVIDIA NeMo Gym."""

from __future__ import annotations

import threading
import random
from typing import Any

from verifiable_instructions import instructions_registry, instructions_util

_NLTK_LOCK = threading.Lock()
_NLTK_READY = False


def _ensure_nltk_data() -> None:
    global _NLTK_READY
    from langdetect import DetectorFactory

    # Stable source-owned detector policy, shared by native and cutover grading.
    DetectorFactory.seed = 0
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


def build_instruction(instruction_id: str, arguments: dict[str, Any]):
    """Build source references while preserving an explicitly requested zero index."""
    instruction = instructions_registry.INSTRUCTION_DICT[instruction_id](instruction_id)
    zero_start = (
        instruction_id == "new:copy_span_idx" and type(arguments.get("n_start")) is int and arguments["n_start"] == 0
    )
    prepared = {**arguments, "n_start": 1} if zero_start else dict(arguments)
    if instruction_id == "count:count_increment_word":
        for key in ("keyword1", "keyword2"):
            if not prepared.get(key):
                # The source default returns a list where its checker needs a word.
                prepared[key] = instructions_util.generate_keywords(num_keywords=1)[0]
    instruction.build_description(**prepared)
    if zero_start:
        # The pinned dependency treats zero as missing and otherwise randomizes it.
        instruction._n_start = 0
    if instruction_id == "length_constraints:nth_paragraph_first_word" and arguments.get("nth_paragraph") is None:
        if instruction._nth_paragraph > instruction._num_paragraphs:
            # randint's upper bound is inclusive; the source samples one past the end.
            instruction._nth_paragraph = random.randint(1, instruction._num_paragraphs)
    return instruction


def grade_instruction_following(text: str, record: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    """Evaluate every per-row constraint with NVIDIA's pinned instruction registry."""
    instruction_ids = record["instruction_id_list"]
    kwargs_list = record["kwargs"]
    if not instruction_ids or len(instruction_ids) != len(kwargs_list):
        raise ValueError("Instruction IDs and kwargs must have the same nonzero length")
    _ensure_nltk_data()
    results: list[bool] = []
    errors: list[str | None] = []
    for instruction_id, kwargs in zip(instruction_ids, kwargs_list, strict=True):
        try:
            instruction = build_instruction(
                instruction_id,
                {key: value for key, value in (kwargs or {}).items() if value is not None},
            )
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
        "num_passed": sum(results),
        "num_total": len(instruction_ids),
    }
