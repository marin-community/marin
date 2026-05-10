# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared transform helpers for rollout dataset pipelines."""

import logging
from collections.abc import Iterator

import pyarrow.parquet as pq
from rigging.filesystem import open_url
from zephyr import counters

logger = logging.getLogger(__name__)


def load_parquet_batched(path: str) -> Iterator[dict]:
    """Read parquet via iter_batches to avoid OOM on large nested-struct columns."""
    with open_url(path, "rb") as f:
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(batch_size=16):
            try:
                rows = batch.to_pydict()
            except UnicodeDecodeError as e:
                counters.increment("load_parquet_batched/utf8_skip_batch")
                logger.warning("Skipping batch from %s due to invalid UTF-8: %s", path, e)
                continue
            n = len(next(iter(rows.values())))
            for i in range(n):
                yield {k: rows[k][i] for k in rows}


def strip_think_tags(text: str) -> str:
    return text.replace("<think>", "").replace("</think>", "").strip()
