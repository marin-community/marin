# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared transform helpers for datakit download pipelines."""

import hashlib
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


def text_document(text: str, source: str) -> dict:
    """Build a datakit document with a content-addressed ``id`` derived from ``text``.

    The ``id`` is the SHA-256 hex digest of the UTF-8-encoded text, so byte-identical
    documents share an id and collapse during exact-dedup normalization.
    """
    return {
        "id": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "text": text,
        "source": source,
    }


def render_role_message(msg: dict) -> str:
    """Render a single chat message as ``<role>\\ncontent\\n</role>``.

    Missing or null content renders as an empty body.
    """
    role = msg.get("role", "unknown")
    content = msg.get("content") or ""
    return f"<{role}>\n{content}\n</{role}>"
