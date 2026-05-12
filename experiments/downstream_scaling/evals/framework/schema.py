# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Artifact schemas and JSONL boundary helpers for downstream-scaling evals."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from typing import Any, Required, TypedDict, TypeGuard

import fsspec

PROMPTS_FILENAME = "prompts.jsonl.gz"
COMPLETIONS_FILENAME = "completions.jsonl.gz"
GRADES_FILENAME = "grades.jsonl.gz"


class PromptRow(TypedDict, total=False):
    id: Required[str]
    prompt: Required[str]
    ground_truth: Any
    metadata: dict[str, Any]


class Completion(TypedDict, total=False):
    text: Required[str]
    metadata: dict[str, Any]


class CompletionRow(TypedDict, total=False):
    id: Required[str]
    completions: Required[list[Completion]]
    metadata: dict[str, Any]


class Grade(TypedDict, total=False):
    score: Required[float]
    metadata: dict[str, Any]


class GradeRow(TypedDict, total=False):
    id: Required[str]
    grades: Required[list[Grade]]
    metadata: dict[str, Any]


def prompts_file(output_path: str) -> str:
    return os.path.join(output_path, PROMPTS_FILENAME)


def completions_file(output_path: str) -> str:
    return os.path.join(output_path, COMPLETIONS_FILENAME)


def grades_file(output_path: str) -> str:
    return os.path.join(output_path, GRADES_FILENAME)


def is_prompt_row(row: Any) -> TypeGuard[PromptRow]:
    return isinstance(row, dict) and isinstance(row.get("id"), str) and isinstance(row.get("prompt"), str)


def is_completion(value: Any) -> TypeGuard[Completion]:
    return isinstance(value, dict) and isinstance(value.get("text"), str)


def is_completion_row(row: Any) -> TypeGuard[CompletionRow]:
    return (
        isinstance(row, dict)
        and isinstance(row.get("id"), str)
        and isinstance(row.get("completions"), list)
        and all(is_completion(completion) for completion in row["completions"])
    )


def is_grade(value: Any) -> TypeGuard[Grade]:
    score = value.get("score") if isinstance(value, dict) else None
    return isinstance(value, dict) and not isinstance(score, bool) and isinstance(score, int | float)


def is_grade_row(row: Any) -> TypeGuard[GradeRow]:
    return (
        isinstance(row, dict)
        and isinstance(row.get("id"), str)
        and isinstance(row.get("grades"), list)
        and all(is_grade(grade) for grade in row["grades"])
    )


def read_prompt_rows(path: str) -> Iterator[PromptRow]:
    seen_ids: set[str] = set()
    with fsspec.open(path, "rt", compression="gzip") as f:
        for line in f:
            row = json.loads(line)
            if not is_prompt_row(row):
                raise TypeError(f"Invalid PromptRow: {path}")
            if row["id"] in seen_ids:
                raise ValueError(f"Duplicate PromptRow id {row['id']!r}: {path}")
            seen_ids.add(row["id"])
            yield row


def read_completion_rows(path: str) -> Iterator[CompletionRow]:
    seen_ids: set[str] = set()
    with fsspec.open(path, "rt", compression="gzip") as f:
        for line in f:
            row = json.loads(line)
            if not is_completion_row(row):
                raise TypeError(f"Invalid CompletionRow: {path}")
            if row["id"] in seen_ids:
                raise ValueError(f"Duplicate CompletionRow id {row['id']!r}: {path}")
            seen_ids.add(row["id"])
            yield row


def read_grade_rows(path: str) -> Iterator[GradeRow]:
    seen_ids: set[str] = set()
    with fsspec.open(path, "rt", compression="gzip") as f:
        for line in f:
            row = json.loads(line)
            if not is_grade_row(row):
                raise TypeError(f"Invalid GradeRow: {path}")
            if row["id"] in seen_ids:
                raise ValueError(f"Duplicate GradeRow id {row['id']!r}: {path}")
            seen_ids.add(row["id"])
            yield row
