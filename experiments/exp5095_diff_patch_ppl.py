# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Diff/patch perplexity-gap slice registry and row builders.

This module keeps diff/patch evaluation opt-in by exposing a dedicated
``diff_patch/<slice>`` namespace without wiring into default eval bundles.
"""

from __future__ import annotations

import json
import posixpath
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

from marin.evaluation.perplexity_gap import RawTextEvaluationDataset, raw_text_dataset

DIFF_PATCH_PREFIX = "diff_patch"
ISSUE_5095 = 5095

SWE_BENCH_PROVENANCE_FIELDS: tuple[str, ...] = (
    "instance_id",
    "repo",
    "base_commit",
    "version",
    "FAIL_TO_PASS",
    "PASS_TO_PASS",
)
COMMITPACK_PROVENANCE_FIELDS: tuple[str, ...] = ("repo_name", "commit_hash", "url")


class DiffPatchMetric(StrEnum):
    PATCH_TEXT = "patch_text"
    CONTEXT_PLUS_PATCH = "context_plus_patch"


@dataclass(frozen=True)
class DiffPatchSlice:
    source: str
    name: str
    relative_path: str
    metrics: tuple[DiffPatchMetric, ...]
    held_out_sample_cap: int

    @property
    def tags(self) -> tuple[str, ...]:
        return ("diff_patch", f"issue:{ISSUE_5095}", f"source:{self.source}", f"slice:{self.name}")

    def dataset_key(self, metric: DiffPatchMetric) -> str:
        return f"{self.source}/{self.name}_{metric.value}"

    def to_raw_dataset(self, raw_root: str, metric: DiffPatchMetric) -> RawTextEvaluationDataset:
        path_stem = self.relative_path.removesuffix(".jsonl.gz")
        return raw_text_dataset(
            posixpath.join(raw_root, f"{path_stem}_{metric.value}.jsonl.gz"),
            tags=(*self.tags, f"metric:{metric.value}"),
        )


DIFF_PATCH_SLICES: tuple[DiffPatchSlice, ...] = (
    DiffPatchSlice(
        source="swe_bench",
        name="issue_to_patch",
        relative_path="swe_bench/issue_to_patch.jsonl.gz",
        metrics=(DiffPatchMetric.PATCH_TEXT, DiffPatchMetric.CONTEXT_PLUS_PATCH),
        held_out_sample_cap=256,
    ),
    DiffPatchSlice(
        source="swe_bench",
        name="raw_git_diff",
        relative_path="swe_bench/raw_git_diff.jsonl.gz",
        metrics=(DiffPatchMetric.PATCH_TEXT,),
        held_out_sample_cap=256,
    ),
    DiffPatchSlice(
        source="commitpack",
        name="commit_message_plus_diff",
        relative_path="commitpack/commit_message_plus_diff.jsonl.gz",
        metrics=(DiffPatchMetric.PATCH_TEXT, DiffPatchMetric.CONTEXT_PLUS_PATCH),
        held_out_sample_cap=512,
    ),
)


def strip_provenance_fields(
    row: Mapping[str, object],
    *,
    masked_fields: tuple[str, ...],
) -> dict[str, object]:
    """Return a shallow copy with masked provenance fields removed."""

    return {key: value for key, value in row.items() if key not in masked_fields}


def build_diff_patch_eval_text(
    row: Mapping[str, object],
    *,
    patch_field: str,
    context_fields: tuple[tuple[str, str], ...],
    masked_fields: tuple[str, ...] = (),
) -> dict[DiffPatchMetric, str]:
    """Build patch-only and context+patch eval text from a structured row."""

    sanitized = strip_provenance_fields(row, masked_fields=masked_fields)
    patch_text = _normalize_field(sanitized.get(patch_field))
    if not patch_text:
        raise ValueError(f"Expected non-empty patch field '{patch_field}'")

    sections: list[str] = []
    for label, field_name in context_fields:
        context = _normalize_field(sanitized.get(field_name))
        if context:
            sections.append(f"{label}:\n{context}")
    sections.append(f"Patch:\n{patch_text}")

    return {
        DiffPatchMetric.PATCH_TEXT: patch_text,
        DiffPatchMetric.CONTEXT_PLUS_PATCH: "\n\n".join(sections),
    }


def build_swe_bench_issue_to_patch_eval_text(row: Mapping[str, object]) -> dict[DiffPatchMetric, str]:
    """Linearize SWE-bench issue+patch rows for diff/patch PPL slices."""

    return build_diff_patch_eval_text(
        row,
        patch_field="patch",
        context_fields=(("Issue", "problem_statement"), ("Hints", "hints_text")),
        masked_fields=SWE_BENCH_PROVENANCE_FIELDS,
    )


def build_commitpack_commit_message_plus_diff_eval_text(row: Mapping[str, object]) -> dict[DiffPatchMetric, str]:
    """Linearize CommitPack rows for commit-message-plus-diff PPL slices."""

    return build_diff_patch_eval_text(
        row,
        patch_field="diff",
        context_fields=(("Commit Message", "commit_message"),),
        masked_fields=COMMITPACK_PROVENANCE_FIELDS,
    )


def build_diff_patch_raw_validation_sets(
    raw_root: str = "raw/diff_patch",
    *,
    slices: tuple[DiffPatchSlice, ...] = DIFF_PATCH_SLICES,
) -> dict[str, RawTextEvaluationDataset]:
    """Render raw-text eval datasets keyed by source + slice + metric."""

    datasets: dict[str, RawTextEvaluationDataset] = {}
    for slice_spec in slices:
        for metric in slice_spec.metrics:
            datasets[slice_spec.dataset_key(metric)] = slice_spec.to_raw_dataset(raw_root, metric)
    return datasets


def diff_patch_source_sampling_plan(
    *,
    slices: tuple[DiffPatchSlice, ...] = DIFF_PATCH_SLICES,
) -> dict[str, dict[str, object]]:
    """Small held-out sampling plan for source builders.

    The plan is intentionally metadata-only so source integration can cap
    downloads before data ingestion.
    """

    return {
        f"{slice_spec.source}/{slice_spec.name}": {
            "held_out_sample_cap": slice_spec.held_out_sample_cap,
            "split": "validation",
            "source": slice_spec.source,
        }
        for slice_spec in slices
    }


ACTIVE_DIFF_PATCH_DATASETS: dict[str, RawTextEvaluationDataset] = build_diff_patch_raw_validation_sets()


def prefixed_diff_patch_validation_sets(
    datasets: Mapping[str, RawTextEvaluationDataset],
) -> dict[str, RawTextEvaluationDataset]:
    """Prefix diff/patch slice names with ``diff_patch/``."""
    return {posixpath.join(DIFF_PATCH_PREFIX, slice_name): dataset for slice_name, dataset in datasets.items()}


def diff_patch_raw_validation_sets() -> dict[str, RawTextEvaluationDataset]:
    """Diff/patch evaluation slices keyed by ``diff_patch/<slice>``."""
    return prefixed_diff_patch_validation_sets(ACTIVE_DIFF_PATCH_DATASETS)


def _normalize_field(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return json.dumps(value, sort_keys=True, ensure_ascii=True)
