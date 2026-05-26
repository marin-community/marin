# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Input handling for the distributed inference library.

Callers pass one of:

* a path/glob string pointing at JSONL[.gz] files where each line is a
  `PromptRecord` dict;
* an in-memory ``Sequence[dict]`` — `materialize_inline_input` writes it once
  to JSONL files in the results bucket so all regional workers can read it.

Each input record must carry an ``id`` (any string) and a ``payload`` dict.
Two payload kinds are supported:

* ``{"kind": "text", "prompt": "..."}`` — raw completion;
* ``{"kind": "messages", "messages": [{"role": ..., "content": ...}, ...]}`` —
  OpenAI-style chat completion.
"""
from __future__ import annotations

import gzip
import json
import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import fsspec

logger = logging.getLogger(__name__)

# Accepted payload kinds.
PAYLOAD_KIND_TEXT = "text"
PAYLOAD_KIND_MESSAGES = "messages"
_VALID_PAYLOAD_KINDS = frozenset({PAYLOAD_KIND_TEXT, PAYLOAD_KIND_MESSAGES})


@dataclass(frozen=True)
class PromptRecord:
    """A single prompt to send through inference.

    The raw on-disk JSONL has the same shape; this dataclass is the in-memory
    parsed form used inside workers.
    """

    id: str
    payload: dict[str, Any]


def validate_record(record: dict[str, Any]) -> None:
    """Raise ValueError if ``record`` is not a valid PromptRecord dict."""
    if "id" not in record:
        raise ValueError(f"Input record missing 'id' field: {record!r}")
    if not isinstance(record["id"], str):
        raise ValueError(f"Input record 'id' must be a string, got {type(record['id']).__name__}")
    if "payload" not in record:
        raise ValueError(f"Input record missing 'payload' field: id={record['id']!r}")
    payload = record["payload"]
    if not isinstance(payload, dict):
        raise ValueError(
            f"Input record 'payload' must be a dict, got {type(payload).__name__} " f"for id={record['id']!r}"
        )
    kind = payload.get("kind")
    if kind not in _VALID_PAYLOAD_KINDS:
        raise ValueError(
            f"Input record payload has unknown kind {kind!r} for id={record['id']!r}; "
            f"expected one of {sorted(_VALID_PAYLOAD_KINDS)}."
        )
    if kind == PAYLOAD_KIND_TEXT and "prompt" not in payload:
        raise ValueError(f"text payload missing 'prompt' for id={record['id']!r}")
    if kind == PAYLOAD_KIND_MESSAGES and "messages" not in payload:
        raise ValueError(f"messages payload missing 'messages' for id={record['id']!r}")


def materialize_inline_input(
    records: Sequence[dict[str, Any]],
    *,
    output_dir: str,
    records_per_file: int,
) -> list[str]:
    """Write an in-memory record list to JSONL.gz files under ``output_dir``.

    Returns the list of written file URIs. Each record is validated before
    write. ``records_per_file`` is the chunk size that determines the number
    of output files; it also becomes the downstream Zephyr shard size, since
    the pipeline maps one input file to one content shard. Callers should
    pass ``InferenceConfig.shard_size`` here.
    """
    if not records:
        raise ValueError("Cannot materialize empty input.")
    if records_per_file <= 0:
        raise ValueError(f"records_per_file must be positive, got {records_per_file}.")
    for record in records:
        validate_record(record)

    num_files = max(1, math.ceil(len(records) / records_per_file))
    paths: list[str] = []
    for file_idx in range(num_files):
        start = file_idx * records_per_file
        end = min(start + records_per_file, len(records))
        path = f"{output_dir.rstrip('/')}/prompts-{file_idx:05d}.jsonl.gz"
        with fsspec.open(path, "wb") as f:
            with gzip.open(f, "wt", encoding="utf-8") as gz:
                for record in records[start:end]:
                    gz.write(json.dumps(record, ensure_ascii=False) + "\n")
        paths.append(path)
    logger.info("Materialized %d records to %d files under %s", len(records), num_files, output_dir)
    return paths


def list_input_files(input_uri: str) -> list[str]:
    """Expand a path or glob into a sorted list of input file URIs.

    ``input_uri`` may be a single file, a directory (in which case all
    ``*.jsonl*`` files inside are returned), or a glob containing ``*``.
    Results are sorted for deterministic per-region rotation downstream.
    """
    fs, _ = fsspec.core.url_to_fs(input_uri)
    if any(ch in input_uri for ch in "*?["):
        matches = fs.glob(input_uri)
    elif fs.isdir(input_uri.rstrip("/")):
        matches = fs.glob(f"{input_uri.rstrip('/')}/*.jsonl*")
    else:
        matches = [input_uri]
    if not matches:
        raise ValueError(f"No input files match {input_uri!r}")
    protocol = input_uri.split("://", 1)[0] + "://" if "://" in input_uri else ""

    def _ensure_protocol(path: str) -> str:
        return path if "://" in path else f"{protocol}{path}"

    return sorted(_ensure_protocol(p) for p in matches)


def load_jsonl_records(path: str) -> list[dict[str, Any]]:
    """Read a JSONL[.gz] file and yield validated record dicts.

    Used by the regional pipeline's ``flat_map`` stage. Returns a list (not an
    iterator) so the entire shard file is in memory before windowing — this is
    fine for typical input sizes (~few MB per file).
    """
    is_gz = path.endswith(".gz")
    opener: Any
    if is_gz:

        def opener(p: str) -> Any:
            return gzip.open(fsspec.open(p, "rb").open(), "rt", encoding="utf-8")

    else:

        def opener(p: str) -> Any:
            return fsspec.open(p, "rt", encoding="utf-8").open()

    out: list[dict[str, Any]] = []
    with opener(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            validate_record(record)
            out.append(record)
    return out
