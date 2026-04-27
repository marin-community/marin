# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Common Crawl raw-web PPL slices for issue #5056.

This keeps the Common Crawl WARC/WAT work separate from the Hugging Face-backed
raw-web slices. It materializes tiny, deterministic held-out samples from one
pinned crawl and exposes them as raw-text validation sets for perplexity-gap
experiments.
"""

from __future__ import annotations

import posixpath

from marin.datakit.download.common_crawl_archives import (
    COMMON_CRAWL_CRAWL_ID,
    COMMON_CRAWL_WARC_SOURCE,
    COMMON_CRAWL_WAT_SOURCE,
    common_crawl_sample_step,
)
from marin.evaluation.perplexity_gap import RawTextEvaluationDataset, raw_text_dataset
from marin.execution.executor import ExecutorStep

RAW_WEB_MARKUP_PREFIX = "raw_web_markup"
RAW_WEB_MARKUP_ISSUE_TAG = "issue:5056"
COMMON_CRAWL_SOURCE_TAG = "source:common_crawl"

common_crawl_warc_raw = common_crawl_sample_step(COMMON_CRAWL_WARC_SOURCE)
common_crawl_wat_raw = common_crawl_sample_step(COMMON_CRAWL_WAT_SOURCE)


def _surface_tags(surface: str) -> tuple[str, ...]:
    return (
        RAW_WEB_MARKUP_PREFIX,
        RAW_WEB_MARKUP_ISSUE_TAG,
        COMMON_CRAWL_SOURCE_TAG,
        f"surface:{surface}",
        f"crawl:{COMMON_CRAWL_CRAWL_ID}",
    )


def common_crawl_raw_validation_sets(
    *,
    raw_root: str | None = None,
    warc_raw: ExecutorStep | None = None,
    wat_raw: ExecutorStep | None = None,
) -> dict[str, RawTextEvaluationDataset]:
    """Return Common Crawl WARC/WAT slices keyed under ``raw_web_markup/``."""

    if raw_root is None and warc_raw is None:
        warc_raw = common_crawl_warc_raw
    if raw_root is None and wat_raw is None:
        wat_raw = common_crawl_wat_raw

    if raw_root is not None:
        warc_source: str | ExecutorStep = posixpath.join(raw_root, "web/common_crawl/warc.jsonl.gz")
        wat_source: str | ExecutorStep = posixpath.join(raw_root, "web/common_crawl/wat.jsonl.gz")
    else:
        assert warc_raw is not None
        assert wat_raw is not None
        warc_source = warc_raw.cd("data.jsonl.gz")
        wat_source = wat_raw.cd("data.jsonl.gz")

    return {
        posixpath.join(RAW_WEB_MARKUP_PREFIX, "common_crawl_warc"): raw_text_dataset(
            warc_source,
            tags=_surface_tags("warc"),
        ),
        posixpath.join(RAW_WEB_MARKUP_PREFIX, "common_crawl_wat"): raw_text_dataset(
            wat_source,
            tags=_surface_tags("wat"),
        ),
    }
