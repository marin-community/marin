# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Convert Zeek connection/http/dns log records to Dolma-format JSONL.

Input is a directory of files (parquet or JSONL) where each row is a single
Zeek log record with one column per Zeek field. The transform batches rows
into canonical TSV log blocks, renders them with
:func:`render_zeek_tsv_log`, and writes each block as a Dolma record::

    {"id": "<zeek-path>-<shard>-<block>-<hash>", "text": "<rendered tsv block>",
     "source": "<source label>", "render": "zeek-tsv"}

Grouping records into blocks (rather than one record per Dolma row) keeps the
``#fields`` header context adjacent to the records the model reads, which is
what Zeek's on-disk format looks like.
"""

from __future__ import annotations

import hashlib
import logging
import posixpath
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any

from marin.transform.security_artifacts.renderers import (
    DEFAULT_ZEEK_EMPTY_FIELD,
    DEFAULT_ZEEK_SEPARATOR,
    DEFAULT_ZEEK_SET_SEPARATOR,
    DEFAULT_ZEEK_UNSET_FIELD,
    render_zeek_tsv_log,
)
from zephyr import Dataset, ZephyrContext, load_jsonl, load_parquet

logger = logging.getLogger(__name__)

ZEEK_RENDER_TAG = "zeek-tsv"
SUPPORTED_INPUT_FORMATS = ("parquet", "jsonl")


@dataclass(frozen=True)
class ZeekToDolmaConfig:
    """Configuration for converting Zeek records to Dolma JSONL.

    Attributes:
        input_path: Directory containing the Zeek-record files.
        output_path: Output directory for the produced Dolma shards.
        zeek_path: Zeek log path label (``"conn"``, ``"dns"``, ``"http"``, ...).
        fields: Ordered Zeek field names corresponding to the record columns.
        source_label: ``source`` field written into each Dolma record.
        input_format: One of ``"parquet"`` or ``"jsonl"``.
        input_glob: Glob for input files, relative to ``input_path``.
        types: Optional Zeek type strings; when set, a ``#types`` header is
            emitted alongside ``#fields``.
        records_per_block: Number of records grouped under a single
            ``#fields`` header. Lower = more header context per block,
            higher = longer contiguous log body for the model to read.
        separator / set_separator / empty_field / unset_field: Zeek delimiters.
        max_blocks_per_file: Optional cap on rendered blocks per input file.
            Combined with ``records_per_block``, this bounds each slice to a
            predictable on-disk size — matching the issue's "small and
            region-local" requirement for diagnostic eval slices.
    """

    input_path: str
    output_path: str
    zeek_path: str
    fields: tuple[str, ...]
    source_label: str
    input_format: str = "parquet"
    input_glob: str | None = None
    types: tuple[str, ...] | None = None
    records_per_block: int = 64
    separator: str = DEFAULT_ZEEK_SEPARATOR
    set_separator: str = DEFAULT_ZEEK_SET_SEPARATOR
    empty_field: str = DEFAULT_ZEEK_EMPTY_FIELD
    unset_field: str = DEFAULT_ZEEK_UNSET_FIELD
    max_blocks_per_file: int | None = None

    def __post_init__(self) -> None:
        if self.input_format not in SUPPORTED_INPUT_FORMATS:
            raise ValueError(f"input_format must be one of {SUPPORTED_INPUT_FORMATS}, got {self.input_format!r}")
        if not self.fields:
            raise ValueError("fields must be a non-empty sequence of Zeek field names")
        if self.types is not None and len(self.types) != len(self.fields):
            raise ValueError(f"types length ({len(self.types)}) must match fields length ({len(self.fields)})")
        if self.records_per_block <= 0:
            raise ValueError(f"records_per_block must be positive, got {self.records_per_block}")
        if self.max_blocks_per_file is not None and self.max_blocks_per_file <= 0:
            raise ValueError(f"max_blocks_per_file must be positive when set, got {self.max_blocks_per_file}")


def _default_input_glob(cfg: ZeekToDolmaConfig) -> str:
    if cfg.input_glob is not None:
        return cfg.input_glob
    if cfg.input_format == "parquet":
        return "**/*.parquet"
    return "**/*.jsonl*"


def _project_record(record: Any, fields: tuple[str, ...]) -> dict[str, Any]:
    """Extract only ``fields`` from ``record`` as a plain dict.

    Missing fields are preserved as ``None`` so the downstream renderer emits
    the configured ``unset_field``.
    """
    if not isinstance(record, dict):
        raise TypeError(f"Expected a dict-like Zeek record, got {type(record).__name__}")
    return {f: record.get(f) for f in fields}


def _render_block(
    block_records: list[dict[str, Any]],
    block_index: int,
    cfg: ZeekToDolmaConfig,
) -> dict[str, Any]:
    text = render_zeek_tsv_log(
        block_records,
        fields=cfg.fields,
        zeek_path=cfg.zeek_path,
        types=cfg.types,
        separator=cfg.separator,
        set_separator=cfg.set_separator,
        empty_field=cfg.empty_field,
        unset_field=cfg.unset_field,
    )
    # Hash the serialized text rather than the raw records so the id is stable
    # even when upstream row ordering is non-deterministic between runs.
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    record_id = f"{cfg.zeek_path}-{block_index:06d}-{digest}"
    return {
        "id": record_id,
        "text": text,
        "source": cfg.source_label,
        "render": ZEEK_RENDER_TAG,
    }


def _render_records_into_blocks(
    records: Iterable[Any],
    cfg: ZeekToDolmaConfig,
) -> Iterator[dict[str, Any]]:
    """Group ``records`` into rendered Zeek TSV Dolma blocks.

    Yields one dict per block. Caps at ``cfg.max_blocks_per_file`` when set.
    """
    buffer: list[dict[str, Any]] = []
    block_index = 0
    for record in records:
        buffer.append(_project_record(record, cfg.fields))
        if len(buffer) >= cfg.records_per_block:
            yield _render_block(buffer, block_index, cfg)
            block_index += 1
            buffer = []
            if cfg.max_blocks_per_file is not None and block_index >= cfg.max_blocks_per_file:
                return
    if buffer:
        yield _render_block(buffer, block_index, cfg)


def render_file_to_dolma_blocks(file_path: str, cfg: ZeekToDolmaConfig) -> list[dict[str, Any]]:
    """Load ``file_path`` and render its records into Dolma blocks.

    Used as the per-shard unit of work for the Zephyr pipeline.
    """
    loader = load_parquet if cfg.input_format == "parquet" else load_jsonl
    return list(_render_records_into_blocks(loader(file_path), cfg))


def render_records_to_dolma_blocks(
    records: Iterable[Any],
    cfg: ZeekToDolmaConfig,
) -> list[dict[str, Any]]:
    """In-memory variant of :func:`render_file_to_dolma_blocks` for tests."""
    return list(_render_records_into_blocks(records, cfg))


def convert_zeek_to_dolma(cfg: ZeekToDolmaConfig) -> None:
    """Transform Zeek log records to Dolma-format JSONL shards.

    Each input file becomes one Zephyr shard. Records in that shard are
    grouped into ``records_per_block`` chunks and rendered as canonical Zeek
    TSV log blocks.
    """
    input_pattern = posixpath.join(cfg.input_path, _default_input_glob(cfg))
    logger.info(
        "Rendering Zeek records from %s (format=%s, path=%s, fields=%d, records_per_block=%d)",
        input_pattern,
        cfg.input_format,
        cfg.zeek_path,
        len(cfg.fields),
        cfg.records_per_block,
    )

    pipeline = (
        Dataset.from_files(input_pattern)
        .flat_map(lambda file_path: render_file_to_dolma_blocks(file_path, cfg))
        .write_jsonl(f"{cfg.output_path}/data-{{shard:05d}}-of-{{total:05d}}.jsonl.gz")
    )
    ctx = ZephyrContext(name=f"zeek-to-dolma-{cfg.zeek_path}")
    ctx.execute(pipeline)


__all__ = [
    "SUPPORTED_INPUT_FORMATS",
    "ZEEK_RENDER_TAG",
    "ZeekToDolmaConfig",
    "convert_zeek_to_dolma",
    "render_file_to_dolma_blocks",
    "render_records_to_dolma_blocks",
]
