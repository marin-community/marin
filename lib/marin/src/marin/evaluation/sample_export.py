# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-question parquet export of lm-eval's ``--log_samples`` output.

lm-eval writes one ``samples_<task>_<timestamp>.jsonl`` per (sub)task next to its results JSON: one
row per evaluated question, holding the source document, the rendered prompt arguments, the model's
raw and filtered responses, and that sample's metric values. This module converts each of those files
into a parquet sibling so a run's individual questions/answers/scores can be loaded as a table
(pandas/duckdb) without re-parsing lm-eval's nested JSON.

Columns: ``task``, ``doc_id``, then ``doc``/``target``/``arguments``/``responses``/
``filtered_responses`` as JSON strings (their shape differs per task), then one column per numeric
per-sample metric (``acc``, ``exact_match``, ...).
"""

from __future__ import annotations

import json
import logging

import pyarrow as pa
import pyarrow.parquet as pq
from fsspec.core import url_to_fs

logger = logging.getLogger(__name__)

_SAMPLES_PREFIX = "samples_"
# Sample fields that are structural rather than per-sample metrics; everything else numeric becomes
# a metric column.
_STRUCTURAL_KEYS = frozenset(
    {"doc", "doc_id", "target", "arguments", "resps", "filtered_resps", "doc_hash", "prompt_hash", "target_hash"}
)


def _task_from_filename(name: str) -> str:
    # samples_<task>_<timestamp>.jsonl; the timestamp contains no underscore.
    return name[len(_SAMPLES_PREFIX) : -len(".jsonl")].rsplit("_", 1)[0]


def _as_json_column(value) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def _flatten(task: str, sample: dict) -> dict:
    row = {
        "task": task,
        "doc_id": sample.get("doc_id"),
        "doc": _as_json_column(sample.get("doc")),
        "target": _as_json_column(sample.get("target")),
        "arguments": _as_json_column(sample.get("arguments")),
        "responses": _as_json_column(sample.get("resps")),
        "filtered_responses": _as_json_column(sample.get("filtered_resps")),
    }
    for key, value in sample.items():
        if key in _STRUCTURAL_KEYS:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        row[key] = float(value)
    return row


def export_sample_parquets(out_path: str) -> list[str]:
    """Write a parquet sibling for every ``samples_*.jsonl`` under ``out_path`` and return the paths.

    Files are keyed per (sub)task, so schemas are uniform within each parquet. The source jsonl is
    kept -- it is lm-eval's native artifact; the parquet is the analysis-friendly view.
    """
    fs, root = url_to_fs(out_path)
    written: list[str] = []
    for path in fs.find(root):
        name = path.rsplit("/", 1)[-1]
        if not (name.startswith(_SAMPLES_PREFIX) and name.endswith(".jsonl")):
            continue
        with fs.open(path, "r") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        if not rows:
            logger.warning("samples file %s is empty; skipping parquet export", path)
            continue
        task = _task_from_filename(name)
        table = pa.Table.from_pylist([_flatten(task, sample) for sample in rows])
        dest = path[: -len(".jsonl")] + ".parquet"
        with fs.open(dest, "wb") as handle:
            pq.write_table(table, handle)
        written.append(dest)
    return written
