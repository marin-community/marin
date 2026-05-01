# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Opt-in GH Archive structured-output raw validation slices for issue #5098.

These slices are meant to be consumed by the shared perplexity-gap cache
builder, not by a dedicated one-off experiment entrypoint.
"""

from __future__ import annotations

import posixpath
from dataclasses import dataclass

from marin.datakit.download.gh_archive import (
    GH_ARCHIVE_OPTIONAL_EVENT_TYPES,
    GH_ARCHIVE_REQUIRED_EVENT_TYPES,
    gh_archive_step,
)
from marin.evaluation.perplexity_gap import RawTextEvaluationDataset, raw_text_dataset
from marin.execution.executor import ExecutorStep

EPIC_5005 = 5005
GH_ARCHIVE_STRUCTURED_OUTPUT_ISSUE = 5098

# Small held-out hour window to avoid broad GH Archive pulls.
GH_ARCHIVE_EVAL_START_DATE = "2024-02-01"
GH_ARCHIVE_EVAL_END_DATE = "2024-02-01"
GH_ARCHIVE_EVAL_START_HOUR = 0
GH_ARCHIVE_EVAL_END_HOUR = 1
GH_ARCHIVE_EVAL_MAX_EVENTS_PER_EVENT_TYPE = 512


@dataclass(frozen=True)
class GhArchiveStructuredOutputSlice:
    event_type: str
    optional: bool = False

    @property
    def registry_key(self) -> str:
        return posixpath.join("gh_archive_structured_output", self.event_type)

    @property
    def raw_relative_glob(self) -> str:
        return posixpath.join(self.event_type, "*.jsonl.gz")

    @property
    def tags(self) -> tuple[str, ...]:
        return (
            "gh_archive_structured_output",
            f"epic:{EPIC_5005}",
            f"issue:{GH_ARCHIVE_STRUCTURED_OUTPUT_ISSUE}",
            f"event_type:{self.event_type}",
        )


GH_ARCHIVE_STRUCTURED_OUTPUT_SLICES: tuple[GhArchiveStructuredOutputSlice, ...] = (
    *(GhArchiveStructuredOutputSlice(event_type=event_type) for event_type in GH_ARCHIVE_REQUIRED_EVENT_TYPES),
    *(
        GhArchiveStructuredOutputSlice(event_type=event_type, optional=True)
        for event_type in GH_ARCHIVE_OPTIONAL_EVENT_TYPES
    ),
)

gh_archive_structured_output_eval = gh_archive_step(
    name="raw/gh_archive/structured_output_eval_2024_02_01_h00_01",
    start_date=GH_ARCHIVE_EVAL_START_DATE,
    end_date=GH_ARCHIVE_EVAL_END_DATE,
    start_hour=GH_ARCHIVE_EVAL_START_HOUR,
    end_hour=GH_ARCHIVE_EVAL_END_HOUR,
    event_types=tuple(slice_.event_type for slice_ in GH_ARCHIVE_STRUCTURED_OUTPUT_SLICES),
    max_events_per_event_type=GH_ARCHIVE_EVAL_MAX_EVENTS_PER_EVENT_TYPE,
).as_executor_step()


def gh_archive_structured_output_raw_validation_sets(
    *,
    raw_root: str | None = None,
    gh_archive_raw: ExecutorStep | None = None,
    include_optional_event_types: bool = True,
) -> dict[str, RawTextEvaluationDataset]:
    """Materialize GH Archive structured-output slices as opt-in raw validation datasets."""
    if raw_root is not None and gh_archive_raw is not None:
        raise ValueError("Provide either raw_root or gh_archive_raw, not both.")

    if raw_root is None and gh_archive_raw is None:
        gh_archive_raw = gh_archive_structured_output_eval

    datasets: dict[str, RawTextEvaluationDataset] = {}
    for slice_ in GH_ARCHIVE_STRUCTURED_OUTPUT_SLICES:
        if slice_.optional and not include_optional_event_types:
            continue

        if raw_root is not None:
            source = posixpath.join(raw_root, slice_.raw_relative_glob)
        else:
            assert gh_archive_raw is not None
            source = gh_archive_raw.cd(slice_.raw_relative_glob)

        datasets[slice_.registry_key] = raw_text_dataset(source, tags=slice_.tags)
    return datasets
