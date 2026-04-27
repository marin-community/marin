# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Byte-preserving serializers for HuggingFace-hosted table datasets.

``tabular.py`` handles raw CSV/TSV files where the ingestion contract is
"never reparse the bytes the source wrote". HF-hosted table datasets like
ToTTo and WikiTableQuestions arrive as pre-parsed Parquet with nested
structure (lists of cells, metadata fields). For those datasets the
byte-preservation concern collapses to one rule: **every cell value that
was a string in the source must survive into the emitted text verbatim**.
No ``float(cell)``, no ``" ".join(cell.split())``, no case folding.
"""

from __future__ import annotations

import json
import logging
import os
import posixpath
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from rigging.filesystem import open_url, url_to_fs
from zephyr.writers import atomic_rename

from marin.utils import fsspec_mkdirs

logger = logging.getLogger(__name__)

DEFAULT_MAX_BYTES_PER_SOURCE = 30 * 1024 * 1024
"""Mirror of the per-source cap in ``tabular.py`` — kept separate so tuning
the two is independent."""

TABLE_ROW_DELIMITER = "\n"
TABLE_CELL_DELIMITER = "\t"
"""Tab-separated rows, newline-separated rows. TSV is the serialization
format with the least whitespace-munging behavior in tokenizers and is
the standard choice across the structured-eval literature."""


@dataclass(frozen=True)
class TableRecordStagingConfig:
    """Configuration for staging HF table-record datasets into JSONL.

    Attributes:
        input_path: fsspec URL pointing at HF-exported parquet files.
            The loader globs split parquet shards and passes them through
            ``datasets.load_dataset("parquet", data_files=...)``.
        output_path: fsspec URL for the staged JSONL directory.
        source_label: Identifier written into each record's ``source`` field.
        serializer_name: Which serializer to invoke. Must match a key in
            :data:`SERIALIZERS`.
        split: Which HF split to stage.
        subset: Optional HF config/subset name.
        max_bytes_per_source: Stop once this many bytes of text have been
            staged. Enforces the 20-40 MB/source budget.
        output_filename: Name of the single JSONL output file.
        extra_metadata: Extra key/values baked into every record's
            ``provenance`` field.
    """

    input_path: str
    output_path: str
    source_label: str
    serializer_name: str
    split: str = "validation"
    subset: str | None = None
    max_bytes_per_source: int = DEFAULT_MAX_BYTES_PER_SOURCE
    output_filename: str = "staged.jsonl.gz"
    extra_metadata: dict[str, str] = field(default_factory=dict)


def _format_cell(cell_value: Any) -> str:
    """Format a single cell value for serialization.

    Preserves strings verbatim. For non-string scalars we use ``str(...)``,
    which for Python ``int``/``float`` produces the usual textual form. If
    an upstream dataset pre-formatted numerics as strings (the common
    case for table datasets sourced from HTML), we keep those strings
    byte-identical.
    """
    if cell_value is None:
        return ""
    if isinstance(cell_value, str):
        return cell_value
    return str(cell_value)


def serialize_totto_example(example: dict[str, Any]) -> str:
    """Serialize a ToTTo example into a single-string PPL document.

    ToTTo records look like::

        {
          "table_page_title": str,
          "table_section_title": str,
          "table": [[{"value": str, "is_header": bool, ...}, ...], ...],
          "highlighted_cells": [[row_idx, col_idx], ...],
          "sentence_annotations": {"final_sentence": [str, ...], ...},
          ...
        }

    The emitted document preserves the table's row/column structure in
    TSV form (so tokenizers see natural delimiters), followed by a blank
    line and the target sentence. Both the table cells and the sentence
    are byte-preserved.

    The ``final_sentence`` field is usually a list (one per annotator);
    we take the first non-empty string. This is the form that
    perplexity-gap experiments over ToTTo need.
    """
    page_title = _format_cell(example.get("table_page_title", ""))
    section_title = _format_cell(example.get("table_section_title", ""))

    rows = example.get("table", [])
    serialized_rows: list[str] = []
    for row in rows:
        cells = [_format_cell(cell.get("value", "")) if isinstance(cell, dict) else _format_cell(cell) for cell in row]
        serialized_rows.append(TABLE_CELL_DELIMITER.join(cells))
    table_block = TABLE_ROW_DELIMITER.join(serialized_rows)

    target = ""
    sentence_annotations = example.get("sentence_annotations") or {}
    if isinstance(sentence_annotations, dict):
        candidates = sentence_annotations.get("final_sentence") or []
        if isinstance(candidates, str):
            target = candidates
        elif isinstance(candidates, list):
            target = next((s for s in candidates if isinstance(s, str) and s.strip()), "")

    parts: list[str] = []
    if page_title:
        parts.append(f"title: {page_title}")
    if section_title:
        parts.append(f"section: {section_title}")
    if table_block:
        parts.append(table_block)
    if target:
        parts.append(target)

    return "\n\n".join(parts)


def serialize_wikitablequestions_example(example: dict[str, Any]) -> str:
    """Serialize a WikiTableQuestions example into a PPL document.

    Records look like::

        {
          "question": str,
          "answers": [str, ...],
          "table": {"header": [str, ...], "rows": [[str, ...], ...]}
        }

    The emitted document is TSV table followed by ``Q: ...`` and
    ``A: ...`` lines. We concatenate all answers with a comma so the
    context/target boundary lands at a stable delimiter. All numeric
    cells come from the source as strings and are preserved byte-identically.
    """
    table = example.get("table") or {}
    header = [_format_cell(cell) for cell in table.get("header", [])]
    rows = [[_format_cell(cell) for cell in row] for row in table.get("rows", [])]

    serialized_rows: list[str] = []
    if header:
        serialized_rows.append(TABLE_CELL_DELIMITER.join(header))
    for row in rows:
        serialized_rows.append(TABLE_CELL_DELIMITER.join(row))
    table_block = TABLE_ROW_DELIMITER.join(serialized_rows)

    question = _format_cell(example.get("question", ""))
    answers = example.get("answers") or []
    if isinstance(answers, str):
        answer_text = answers
    else:
        answer_text = ", ".join(_format_cell(a) for a in answers)

    parts: list[str] = []
    if table_block:
        parts.append(table_block)
    if question:
        parts.append(f"Q: {question}")
    if answer_text:
        parts.append(f"A: {answer_text}")

    return "\n\n".join(parts)


SERIALIZERS: dict[str, Any] = {
    "totto": serialize_totto_example,
    "wikitablequestions": serialize_wikitablequestions_example,
}
"""Registry mapping serializer name -> callable. Lets the staging step stay
data-agnostic."""


def _fsspec_url(fs: Any, path: str) -> str:
    protocol = fs.protocol
    if isinstance(protocol, (list, tuple)):
        protocol = protocol[0]
    if protocol in (None, "file"):
        return path
    if path.startswith(f"{protocol}://"):
        return path
    return f"{protocol}://{path}"


def _parquet_file_matches_split(path: str, split: str) -> bool:
    filename = os.path.basename(path)
    if not filename.endswith(".parquet"):
        return False
    return filename == f"{split}.parquet" or filename.startswith(f"{split}-")


def _find_split_parquet_files(input_path: str, split: str, subset: str | None) -> list[str]:
    """Find downloaded HF parquet files for ``split`` under an fsspec path."""
    fs, root = url_to_fs(input_path)
    roots: list[str] = []
    if subset and subset != "default":
        subset_root = posixpath.join(root, subset)
        if fs.exists(subset_root):
            roots.append(subset_root)
    roots.append(root)

    matches: list[str] = []
    for candidate_root in roots:
        if fs.isfile(candidate_root):
            candidates = [candidate_root]
            selected = [path for path in candidates if path.endswith(".parquet")]
        else:
            candidates = list(fs.find(candidate_root, withdirs=False))
            selected = [path for path in candidates if _parquet_file_matches_split(path, split)]
        matches.extend(selected)

    if not matches:
        raise FileNotFoundError(f"No parquet files found for split {split!r} under {input_path}")

    return [_fsspec_url(fs, path) for path in sorted(set(matches))]


def _load_hf_iterable(input_path: str, split: str, subset: str | None) -> Iterable[dict[str, Any]]:
    """Iterate over examples in downloaded HF parquet shards at ``input_path``.

    Imported lazily so the import graph doesn't require ``datasets`` at
    module load time (e.g. when only the pure serializer functions are
    used in tests).
    """
    from datasets import load_dataset  # local import to keep module importable without `datasets`

    data_files = _find_split_parquet_files(input_path, split, subset)
    dataset = load_dataset("parquet", data_files={split: data_files}, split=split, streaming=True)
    return dataset


def stage_table_record_source(cfg: TableRecordStagingConfig) -> dict[str, int | str]:
    """Run the configured serializer over an HF table dataset and write JSONL.

    Iterates in dataset order (deterministic) and stops once the kept text
    exceeds :attr:`TableRecordStagingConfig.max_bytes_per_source`.

    Returns a dict with ``record_count``, ``bytes_written``, and
    ``output_file`` for logging and downstream provenance.
    """
    if cfg.serializer_name not in SERIALIZERS:
        raise ValueError(f"Unknown serializer {cfg.serializer_name!r}; known: {sorted(SERIALIZERS)}")
    serializer = SERIALIZERS[cfg.serializer_name]

    fsspec_mkdirs(cfg.output_path, exist_ok=True)
    out_file = posixpath.join(cfg.output_path, cfg.output_filename)
    compression = "gzip" if out_file.endswith(".gz") else None

    total_text_bytes = 0
    record_count = 0

    with atomic_rename(out_file) as temp_path:
        with open_url(temp_path, "wt", encoding="utf-8", compression=compression) as outfile:
            for index, example in enumerate(_load_hf_iterable(cfg.input_path, cfg.split, cfg.subset)):
                text = serializer(example)
                if not text.strip():
                    continue
                text_bytes = len(text.encode("utf-8"))
                if total_text_bytes + text_bytes > cfg.max_bytes_per_source and record_count > 0:
                    logger.info(
                        "Reached per-source cap after %d records (%d bytes); stopping.",
                        record_count,
                        total_text_bytes,
                    )
                    break

                record = {
                    "id": f"{cfg.source_label}:{cfg.split}:{index:08d}",
                    "text": text,
                    "source": cfg.source_label,
                    "provenance": {
                        "dataset": cfg.input_path,
                        "split": cfg.split,
                        "subset": cfg.subset,
                        "serializer": cfg.serializer_name,
                        "index": index,
                        **cfg.extra_metadata,
                    },
                }
                json.dump(record, outfile, ensure_ascii=False)
                outfile.write("\n")
                total_text_bytes += text_bytes
                record_count += 1

    logger.info(
        "Staged %d records (%d bytes of text) to %s",
        record_count,
        total_text_bytes,
        out_file,
    )
    return {
        "record_count": record_count,
        "bytes_written": total_text_bytes,
        "output_file": out_file,
    }
