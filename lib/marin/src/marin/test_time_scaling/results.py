# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from typing import Any

import fsspec
from fsspec.core import url_to_fs

from marin.test_time_scaling.config import SelectorName

CANDIDATES_FILENAME = "candidates.jsonl"
SELECTED_FILENAME = "selected.jsonl"
SUMMARY_FILENAME = "summary.json"


def _artifact_path(base_path: str, filename: str) -> str:
    return f"{base_path.rstrip('/')}/{filename}"


def _ensure_parent_dir(path: str) -> None:
    fs, fs_path = url_to_fs(path)
    parent = fs_path.rsplit("/", 1)[0] if "/" in fs_path else ""
    if parent:
        fs.mkdirs(parent, exist_ok=True)


def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _to_jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {key: _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_to_jsonable(item) for item in value]
    return value


@dataclass(frozen=True)
class CandidateRecord:
    """Single generated candidate saved to `candidates.jsonl`."""

    prompt_id: str
    candidate_id: str
    sample_index: int
    raw_text: str
    extracted_answer: str | None
    is_correct: bool | None
    parse_valid: bool | None
    prompt_tokens: int | None
    completion_tokens: int | None
    finish_reason: str | None
    request_latency_seconds: float
    generation_seed: int | None
    logprob_sum: float | None
    normalized_logprob: float | None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CandidateRecord:
        return cls(
            prompt_id=str(data["prompt_id"]),
            candidate_id=str(data["candidate_id"]),
            sample_index=int(data["sample_index"]),
            raw_text=str(data["raw_text"]),
            extracted_answer=data.get("extracted_answer"),
            is_correct=data.get("is_correct"),
            parse_valid=data.get("parse_valid"),
            prompt_tokens=data.get("prompt_tokens"),
            completion_tokens=data.get("completion_tokens"),
            finish_reason=data.get("finish_reason"),
            request_latency_seconds=float(data["request_latency_seconds"]),
            generation_seed=data.get("generation_seed"),
            logprob_sum=data.get("logprob_sum"),
            normalized_logprob=data.get("normalized_logprob"),
        )


@dataclass(frozen=True)
class SelectionRecord:
    """Single selector decision saved to `selected.jsonl`."""

    prompt_id: str
    selector_name: SelectorName
    chosen_candidate_id: str
    chosen_sample_index: int
    final_selected_answer: str
    correctness: bool | None
    oracle_correct: bool | None
    oracle_gap: bool | None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SelectionRecord:
        return cls(
            prompt_id=str(data["prompt_id"]),
            selector_name=SelectorName(data["selector_name"]),
            chosen_candidate_id=str(data["chosen_candidate_id"]),
            chosen_sample_index=int(data["chosen_sample_index"]),
            final_selected_answer=str(data["final_selected_answer"]),
            correctness=data.get("correctness"),
            oracle_correct=data.get("oracle_correct"),
            oracle_gap=data.get("oracle_gap"),
        )


@dataclass(frozen=True)
class SelectorSummary:
    """Aggregate metrics for a single selector."""

    selector_name: SelectorName
    num_prompts: int
    num_scored_prompts: int
    accuracy: float | None
    oracle_gap_rate: float | None


@dataclass(frozen=True)
class RunSummary:
    """Top-level run summary written to `summary.json`."""

    manifest_id: str
    task_name: str
    num_prompts: int
    total_candidates: int
    oracle_accuracy: float | None
    parse_valid_rate: float | None
    duplicate_rate: float | None
    total_prompt_tokens: int
    total_completion_tokens: int
    total_request_latency_seconds: float
    selector_summaries: tuple[SelectorSummary, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


def write_candidate_records(output_dir: str, candidates: list[CandidateRecord]) -> None:
    path = _artifact_path(output_dir, CANDIDATES_FILENAME)
    _ensure_parent_dir(path)
    with fsspec.open(path, "w", encoding="utf-8") as handle:
        for candidate in candidates:
            handle.write(json.dumps(_to_jsonable(candidate), sort_keys=True) + "\n")


def read_candidate_records(path: str) -> list[CandidateRecord]:
    candidates: list[CandidateRecord] = []
    with fsspec.open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                candidates.append(CandidateRecord.from_dict(json.loads(stripped)))
    return candidates


def write_selection_records(output_dir: str, selections: list[SelectionRecord]) -> None:
    path = _artifact_path(output_dir, SELECTED_FILENAME)
    _ensure_parent_dir(path)
    with fsspec.open(path, "w", encoding="utf-8") as handle:
        for selection in selections:
            handle.write(json.dumps(_to_jsonable(selection), sort_keys=True) + "\n")


def read_selection_records(path: str) -> list[SelectionRecord]:
    selections: list[SelectionRecord] = []
    with fsspec.open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                selections.append(SelectionRecord.from_dict(json.loads(stripped)))
    return selections


def write_run_summary(output_dir: str, summary: RunSummary) -> None:
    path = _artifact_path(output_dir, SUMMARY_FILENAME)
    _ensure_parent_dir(path)
    with fsspec.open(path, "w", encoding="utf-8") as handle:
        json.dump(_to_jsonable(summary), handle, indent=2, sort_keys=True)
