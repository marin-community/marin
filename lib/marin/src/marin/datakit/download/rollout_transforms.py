# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared transform helpers for datakit download pipelines."""

import hashlib
import logging
from collections.abc import Callable, Iterator

import pyarrow.parquet as pq
from fray import ResourceConfig
from rigging.filesystem import open_url
from zephyr import Dataset, ZephyrContext, counters, load_parquet

logger = logging.getLogger(__name__)

PARQUET_SHARD_TEMPLATE = "data-{shard:05d}-of-{total:05d}.parquet"


def run_document_transform(
    *,
    input_path: str,
    output_path: str,
    row_to_doc: Callable[[dict], list[dict]],
    name: str,
    ram: str,
    loader: Callable[[str], Iterator[dict]] = load_parquet,
    file_glob: str = "**/*.parquet",
    cpu: int = 1,
) -> None:
    """Run the canonical datakit download→document Zephyr pipeline.

    Reads ``{input_path}/{file_glob}`` with ``loader``, expands each row into zero or
    more Dolma-shaped documents via ``row_to_doc``, and writes sharded parquet under
    ``output_path``. Most HF-backed sources share this exact shape; sources that need a
    bespoke pipeline (resharding, grouping, JSONL inputs) build their own.

    Args:
        input_path: Directory holding the downloaded files.
        output_path: Directory to write sharded parquet documents into.
        row_to_doc: Maps a raw row to zero or more output documents.
        name: Zephyr context name (used for caching/observability).
        ram: Per-worker RAM request, e.g. ``"8g"``.
        loader: Per-file decoder; defaults to Zephyr's ``load_parquet``.
        file_glob: Glob (relative to ``input_path``) selecting input files.
        cpu: Per-worker CPU request.
    """
    pipeline = (
        Dataset.from_files(f"{input_path}/{file_glob}")
        .flat_map(loader)
        .flat_map(row_to_doc)
        .write_parquet(f"{output_path}/{PARQUET_SHARD_TEMPLATE}", skip_existing=True)
    )
    ZephyrContext(name=name, resources=ResourceConfig(cpu=cpu, ram=ram)).execute(pipeline)


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
