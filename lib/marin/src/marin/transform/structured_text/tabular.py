# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Byte-preserving staging for raw CSV/TSV source files.

The ingestion contract here is intentionally narrow: we treat the source
file as an opaque sequence of text lines and only ever split at newline
boundaries. No field is reparsed, no numeric literal is re-normalised, and
no whitespace is stripped. That is what makes these slices useful as
perplexity probes for structured data — when a model assigns worse bits
per byte to a cell, we want that to reflect the cell contents the source
wrote, not whatever our ingestion layer rewrote it to.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import posixpath
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field

from marin.datakit.ingestion_manifest import (
    IngestionSourceManifest,
    MaterializedOutputMetadata,
    write_ingestion_metadata_json,
)
from marin.utils import fsspec_mkdirs
from rigging.filesystem import open_url, url_to_fs
from zephyr.writers import atomic_rename

logger = logging.getLogger(__name__)

DEFAULT_MAX_BYTES_PER_SOURCE = 30 * 1024 * 1024
"""Default per-source byte cap for kept text. Sits inside the 20-40 MB/source
band agreed on for structured eval slices."""

DEFAULT_MAX_BYTES_PER_DOCUMENT = 32 * 1024
"""Default byte cap for a single emitted JSONL document. Small enough that a
single long table gets chunked into several documents (so context length
truncation is deterministic), large enough to carry meaningful local
structure."""


@dataclass(frozen=True)
class TabularStagingConfig:
    """Configuration for a byte-preserving tabular staging step.

    Attributes:
        input_path: fsspec URL (or glob) pointing at the raw source files.
            Files are read in sorted order to make the cap deterministic.
        output_path: fsspec URL for the staged JSONL output directory.
        source_label: Identifier written into each record's ``source`` field,
            e.g. ``"gittables:csv"``.
        file_extensions: Tuple of suffixes to keep when walking ``input_path``.
            Case-insensitive; matched against the final path segment.
        max_bytes_per_source: Stop reading once this many bytes of text have
            been staged from this source. Enforces the per-source budget
            agreed on in the parent issue.
        max_bytes_per_document: Emit a new JSONL document once the current
            chunk exceeds this many bytes. Chunk boundaries always fall on
            source line boundaries.
        preserve_header: If True, prepend the source file's first non-blank
            line to each emitted chunk of the same file. Makes every
            chunk self-describing.
        output_filename: Name of the single staged JSONL output file, written
            under ``output_path``. Relative path; must end in ``.jsonl`` or
            ``.jsonl.gz``.
        source_manifest: Optional typed source manifest used for writing a
            ``metadata.json`` sidecar.
        content_fingerprint: Optional explicit hash copied from the source
            manifest into the step config so text-projection changes
            participate in executor hashing.
    """

    input_path: str
    output_path: str
    source_label: str
    file_extensions: tuple[str, ...] = (".csv", ".tsv", ".txt")
    max_bytes_per_source: int = DEFAULT_MAX_BYTES_PER_SOURCE
    max_bytes_per_document: int = DEFAULT_MAX_BYTES_PER_DOCUMENT
    preserve_header: bool = True
    output_filename: str = "staged.jsonl.gz"
    extra_metadata: dict[str, str] = field(default_factory=dict)
    source_manifest: IngestionSourceManifest | None = None
    content_fingerprint: str = ""


def serialize_csv_document(header_line: str | None, body_lines: Iterable[str]) -> str:
    """Concatenate a header line and body lines verbatim.

    The only transformation is joining: ``header_line`` is emitted once
    (with its trailing newline already attached), then the ``body_lines``
    are emitted in order. Callers guarantee that each input line retains
    its original line terminator, so the returned string is byte-identical
    to the original source slice (modulo the optional prepended header).

    Args:
        header_line: Optional header line to prepend. Must already contain
            its trailing newline. ``None`` means no header is prepended.
        body_lines: Iterable of source body lines, each retaining its
            original line terminator.

    Returns:
        Concatenated document text.
    """
    parts: list[str] = []
    if header_line is not None:
        parts.append(header_line)
    parts.extend(body_lines)
    return "".join(parts)


def chunk_lines_by_bytes(
    lines: Iterable[str],
    *,
    max_bytes_per_chunk: int,
    header_line: str | None = None,
) -> Iterator[list[str]]:
    """Split ``lines`` into chunks whose UTF-8 byte sizes stay under a cap.

    Splits only at input line boundaries — a line that individually exceeds
    the cap becomes its own (over-cap) chunk rather than being broken up.
    This keeps the ingestion invariant that no input is ever split mid-row.

    The cap accounts for the bytes a preserved header would add: if a
    caller passes ``header_line``, each chunk gets budgeted as if the
    header were already spent, so emitted documents stay under the cap
    after the header is prepended.

    Args:
        lines: Input lines (retaining their line terminators).
        max_bytes_per_chunk: Target maximum size, in UTF-8 bytes, of each
            emitted chunk.
        header_line: Optional header line that callers plan to prepend to
            each emitted chunk; its byte size is reserved from the budget.

    Yields:
        Lists of source lines. Each list is one chunk. Empty chunks are
        never yielded.
    """
    if max_bytes_per_chunk <= 0:
        raise ValueError(f"max_bytes_per_chunk must be positive, got {max_bytes_per_chunk}")

    header_bytes = len(header_line.encode("utf-8")) if header_line is not None else 0
    body_budget = max_bytes_per_chunk - header_bytes
    if body_budget <= 0:
        raise ValueError(
            f"max_bytes_per_chunk={max_bytes_per_chunk} is smaller than header size {header_bytes}; "
            "increase the cap or disable header preservation."
        )

    current: list[str] = []
    current_bytes = 0
    for line in lines:
        line_bytes = len(line.encode("utf-8"))
        if current and current_bytes + line_bytes > body_budget:
            yield current
            current = []
            current_bytes = 0
        current.append(line)
        current_bytes += line_bytes
    if current:
        yield current


def _iter_source_files(input_path: str, file_extensions: tuple[str, ...]) -> list[str]:
    """Return a sorted list of files under ``input_path`` matching the extensions.

    Uses fsspec's ``find`` so the helper works for both local and remote URLs.
    Case-insensitive extension match.
    """
    fs, root = url_to_fs(input_path)
    if fs.isfile(root):
        candidates = [root]
    else:
        candidates = list(fs.find(root))

    allowed = tuple(ext.lower() for ext in file_extensions)

    def _matches(path: str) -> bool:
        return path.lower().endswith(allowed)

    selected = [path for path in candidates if _matches(path)]
    selected.sort()
    return selected


def _read_text_file_lines(fs, path: str) -> list[str]:
    """Read ``path`` via fsspec and return its lines with line terminators preserved.

    ``str.splitlines(keepends=True)`` keeps each line's original terminator,
    which is how we keep byte-for-byte fidelity. Non-UTF-8 inputs are rejected
    instead of repaired because replacement characters would corrupt the byte
    stream this eval is meant to measure.
    """
    with fs.open(path, "rb") as f:
        raw = f.read()
    text = raw.decode("utf-8")
    return text.splitlines(keepends=True)


def _first_non_blank_line(lines: Iterable[str]) -> str | None:
    for line in lines:
        if line.strip():
            return line
    return None


def _doc_id(source_label: str, file_path: str, chunk_index: int) -> str:
    """Deterministic document id derived from the source path and chunk index."""
    basename = os.path.basename(file_path)
    digest = hashlib.sha1(file_path.encode("utf-8")).hexdigest()[:8]
    return f"{source_label}:{basename}:{digest}:{chunk_index:04d}"


def stage_tabular_source(cfg: TabularStagingConfig) -> dict[str, int | str]:
    """Stage raw CSV/TSV files into a single JSONL with byte-preserved text.

    Each JSONL record has the following shape::

        {
          "id": "<source_label>:<basename>:<hash>:<chunk>",
          "text": "<raw bytes from the source, verbatim>",
          "source": "<source_label>",
          "provenance": {
              "file": "<source path>",
              "chunk_index": <int>,
              "header_preserved": <bool>,
              **cfg.extra_metadata,
          },
        }

    Returns a dict with ``record_count``, ``bytes_written``, and
    ``output_file`` for logging.
    """
    fs, _ = url_to_fs(cfg.input_path)
    source_files = _iter_source_files(cfg.input_path, cfg.file_extensions)
    if not source_files:
        raise ValueError(f"No source files with extensions {cfg.file_extensions} found under {cfg.input_path}")
    if cfg.source_manifest is not None and cfg.content_fingerprint:
        expected = cfg.source_manifest.fingerprint()
        if cfg.content_fingerprint != expected:
            raise ValueError(
                f"content_fingerprint mismatch: config has {cfg.content_fingerprint}, source manifest has {expected}"
            )

    output_path = cfg.output_path
    fsspec_mkdirs(output_path, exist_ok=True)

    out_file = posixpath.join(output_path, cfg.output_filename)
    compression = "gzip" if out_file.endswith(".gz") else None

    total_text_bytes = 0
    record_count = 0

    with atomic_rename(out_file) as temp_path:
        with open_url(temp_path, "wt", encoding="utf-8", compression=compression) as outfile:
            for path in source_files:
                if total_text_bytes >= cfg.max_bytes_per_source:
                    logger.info(
                        "Reached per-source cap (%d bytes) after %d records; stopping.",
                        cfg.max_bytes_per_source,
                        record_count,
                    )
                    break

                lines = _read_text_file_lines(fs, path)
                if not lines:
                    continue

                if cfg.preserve_header:
                    header = _first_non_blank_line(lines)
                    if header is not None:
                        header_idx = lines.index(header)
                        body_lines = lines[:header_idx] + lines[header_idx + 1 :]
                    else:
                        body_lines = lines
                else:
                    header = None
                    body_lines = lines

                for chunk_index, chunk in enumerate(
                    chunk_lines_by_bytes(
                        body_lines,
                        max_bytes_per_chunk=cfg.max_bytes_per_document,
                        header_line=header if cfg.preserve_header else None,
                    )
                ):
                    text = serialize_csv_document(header if cfg.preserve_header else None, chunk)
                    text_bytes = len(text.encode("utf-8"))

                    if total_text_bytes + text_bytes > cfg.max_bytes_per_source and record_count > 0:
                        # Only stop once we've written at least one document so we never emit 0 rows
                        logger.info(
                            "Would exceed per-source cap with next chunk (%d + %d > %d); stopping.",
                            total_text_bytes,
                            text_bytes,
                            cfg.max_bytes_per_source,
                        )
                        break

                    record = {
                        "id": _doc_id(cfg.source_label, path, chunk_index),
                        "text": text,
                        "source": cfg.source_label,
                        "provenance": {
                            "file": path,
                            "chunk_index": chunk_index,
                            "header_preserved": cfg.preserve_header and header is not None,
                            **cfg.extra_metadata,
                        },
                    }
                    json.dump(record, outfile, ensure_ascii=False)
                    outfile.write("\n")
                    total_text_bytes += text_bytes
                    record_count += 1
                else:
                    continue
                break  # propagate inner break when per-source cap was hit

    logger.info(
        "Staged %d records (%d bytes of text) to %s",
        record_count,
        total_text_bytes,
        out_file,
    )
    result: dict[str, int | str] = {
        "record_count": record_count,
        "bytes_written": total_text_bytes,
        "output_file": out_file,
    }
    if cfg.source_manifest is not None:
        result["metadata_file"] = write_ingestion_metadata_json(
            manifest=cfg.source_manifest,
            materialized_output=MaterializedOutputMetadata(
                input_path=cfg.input_path,
                output_path=cfg.output_path,
                output_file=out_file,
                record_count=record_count,
                bytes_written=total_text_bytes,
                metadata={"source_file_count": len(source_files)},
            ),
        )
    return result
