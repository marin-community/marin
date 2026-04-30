# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Common Crawl raw-web PPL slices.

This materializes tiny, deterministic held-out samples from one pinned crawl and
exposes them as raw-text validation sets for perplexity-gap experiments.
"""

from __future__ import annotations

import posixpath
from pathlib import Path

from experiments.exp5056_raw_web_markup_ppl import (
    prefixed_raw_web_markup_validation_sets,
    raw_web_markup_slice_key,
    raw_web_markup_tags,
)
from marin.datakit.download.common_crawl_archives import (
    COMMON_CRAWL_CRAWL_ID,
    COMMON_CRAWL_OUTPUT_FILENAME,
    COMMON_CRAWL_WARC_SOURCE,
    COMMON_CRAWL_WAT_SOURCE,
    CommonCrawlSampleArtifact,
    common_crawl_sample_step,
)
from marin.evaluation.perplexity_gap import RawTextEvaluationDataset, raw_text_dataset
from marin.execution.artifact import Artifact
from marin.execution.executor import InputName
from marin.execution.step_spec import StepSpec

common_crawl_warc_raw = common_crawl_sample_step(COMMON_CRAWL_WARC_SOURCE)
common_crawl_wat_raw = common_crawl_sample_step(COMMON_CRAWL_WAT_SOURCE)


def _surface_tags(surface: str) -> tuple[str, ...]:
    return raw_web_markup_tags(source="common_crawl", surface=surface, extra_tags=(f"crawl:{COMMON_CRAWL_CRAWL_ID}",))


def _materialized_output_file(step: StepSpec) -> str | InputName:
    if not step.output_path.startswith("gs://") and Path(step.output_path, ".artifact").exists():
        return Artifact.load(step, CommonCrawlSampleArtifact).output_file
    return step.as_executor_step().cd(COMMON_CRAWL_OUTPUT_FILENAME)


def common_crawl_raw_validation_sets(
    *,
    raw_root: str | None = None,
    warc_raw: StepSpec | None = None,
    wat_raw: StepSpec | None = None,
) -> dict[str, RawTextEvaluationDataset]:
    """Return Common Crawl WARC/WAT slices keyed under ``raw_web_markup/``."""

    if raw_root is None and warc_raw is None:
        warc_raw = common_crawl_warc_raw
    if raw_root is None and wat_raw is None:
        wat_raw = common_crawl_wat_raw

    if raw_root is not None:
        warc_source = posixpath.join(raw_root, "web/common_crawl/warc.jsonl.gz")
        wat_source = posixpath.join(raw_root, "web/common_crawl/wat.jsonl.gz")
    else:
        assert warc_raw is not None
        assert wat_raw is not None
        warc_source = _materialized_output_file(warc_raw)
        wat_source = _materialized_output_file(wat_raw)

    return prefixed_raw_web_markup_validation_sets(
        {
            raw_web_markup_slice_key("common_crawl", "warc"): raw_text_dataset(
                warc_source,
                tags=_surface_tags("warc"),
            ),
            raw_web_markup_slice_key("common_crawl", "wat"): raw_text_dataset(
                wat_source,
                tags=_surface_tags("wat"),
            ),
        }
    )
