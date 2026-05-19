# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TeraflopAI/SEC-EDGAR download + normalize helpers.

~8M filings (~335B marin_tokenizer tokens) from the SEC EDGAR database,
organized into per-filing-type subdirectories: 10-K, 10-Q, 8-K, 20-F,
S-1, S-8, 144, and Form 3/4/5. Text lives in the upstream ``content``
column.

The download uses DuckDB instead of byte-streaming because the upstream
parquet shards trip ``apache/arrow#46404`` — PyArrow's parquet reader
can't decode page headers >8 MiB, which the multi-MB filings in the
``content`` column overflow on per-page string statistics. DuckDB has
no such cap, so we re-encode through it once at download time: each
upstream shard is read via DuckDB and rewritten via PyArrow's default
writer (whose stats are safely truncated for huge strings — verified
locally on the offending SEC files). The result is a PyArrow-readable
``raw/sec-edgar/<form-type>/<file>.parquet`` tree that normalize and
tokenize consume directly, with no intermediate staging copy.

``duckdb`` is an optional dependency exposed via the ``sec-edgar``
extra (``uv sync --extra sec-edgar`` locally, or ``iris job run
--extra=sec-edgar ...`` on the cluster); the import is deferred so
catalog walks that don't actually ingest SEC-EDGAR don't pay for it.

Scoped to this source per https://github.com/marin-community/marin/pull/5335
— if more datasets hit the page-header cap or we move off PyArrow
wholesale, lift ``read_parquet_via_duckdb`` into a shared helper.
Tracking: https://github.com/marin-community/marin/issues/5334.
"""

from __future__ import annotations

import logging
import os
import random
import time
from collections.abc import Iterator

import pyarrow as pa
import pyarrow.parquet as pq
from fray import ResourceConfig
from huggingface_hub import HfFileSystem
from zephyr import Dataset, ZephyrContext, counters
from zephyr.writers import atomic_rename, ensure_parent_dir

from marin.datakit.normalize import normalize_step
from marin.execution.step_spec import StepSpec
from marin.utilities.validation_utils import write_provenance_json

logger = logging.getLogger(__name__)

HF_DATASET_ID = "TeraflopAI/SEC-EDGAR"
HF_REVISION = "43de32c"

FILING_TYPES = ("10-K", "10-Q", "8-K", "20-F", "S-1", "S-8", "144", "3", "4", "5")

# HF rate-limits aggressively; a small batch reader keeps memory bounded
# while one big SEC row group (~700 MB decompressed) is in flight.
_ROWS_PER_BATCH = 8

# Per-file retry policy (HfHubHTTPError 429s, network blips, xet-bridge hiccups).
_MAX_RETRIES = 20
_BASE_WAIT_S = 5
_MAX_WAIT_S = 15 * 60


def _import_duckdb():
    """Import ``duckdb`` lazily so the SEC-EDGAR extra stays opt-in.

    The wider ``marin`` package doesn't depend on duckdb; only this
    downloader does. Importing it at module scope would pull duckdb into
    every catalog walk that touches ``sources.py``, defeating the point
    of the extra.
    """
    try:
        import duckdb
    except ImportError as e:
        raise ImportError(
            "The 'duckdb' package is required to ingest TeraflopAI/SEC-EDGAR. "
            "Install the extra with `uv sync --extra sec-edgar` locally or "
            "`iris job run --extra=sec-edgar ...` on the cluster."
        ) from e
    return duckdb


def read_parquet_via_duckdb(path: str, *, fs: object | None = None) -> Iterator[pa.RecordBatch]:
    """Yield Arrow RecordBatches from a parquet via DuckDB.

    Works around https://github.com/marin-community/marin/issues/5334
    (apache/arrow#46404) — PyArrow can't decode page headers >8 MiB,
    which SEC's multi-MB ``content`` column overflows. DuckDB has no
    such limit.

    ``fs`` is the fsspec filesystem to register with the DuckDB
    connection. Defaults to an ``HfFileSystem`` so callers reading
    ``hf://...`` paths don't have to wire one up.
    """
    duckdb = _import_duckdb()
    if fs is None:
        fs = HfFileSystem()
    con = duckdb.connect(":memory:")
    try:
        con.register_filesystem(fs)
        result = con.execute("SELECT * FROM read_parquet(?)", [path])
        yield from result.fetch_record_batch(rows_per_batch=_ROWS_PER_BATCH)
    finally:
        con.close()


def _download_one(task: dict) -> dict:
    """Stream one upstream parquet via DuckDB, write a PyArrow-readable shard at ``dst``."""
    hf_path = task["hf_path"]
    dst = task["dst"]

    last_exc: BaseException | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            ensure_parent_dir(dst)
            count = 0
            batches = read_parquet_via_duckdb(hf_path)
            first = next(batches, None)
            if first is None:
                counters.increment("sec_edgar/empty_input")
                return {"hf_path": hf_path, "dst": dst, "rows": 0}
            with atomic_rename(dst) as tmp:
                with pq.ParquetWriter(tmp, first.schema) as writer:
                    writer.write_batch(first)
                    count += first.num_rows
                    for batch in batches:
                        writer.write_batch(batch)
                        count += batch.num_rows
            counters.increment("sec_edgar/rows_downloaded", count)
            return {"hf_path": hf_path, "dst": dst, "rows": count}
        except Exception as e:
            last_exc = e
            wait = min(_MAX_WAIT_S, _BASE_WAIT_S * (2**attempt)) + random.uniform(0, 10)
            logger.warning(
                "Attempt %d/%d failed for %s: %s: %s; retrying in %.1fs",
                attempt + 1,
                _MAX_RETRIES,
                hf_path,
                type(e).__name__,
                e,
                wait,
            )
            time.sleep(wait)
    raise RuntimeError(f"Failed to download {hf_path} after {_MAX_RETRIES} attempts") from last_exc


def _list_hf_parquets() -> list[str]:
    """List all upstream parquet paths in ``hf://datasets/...`` form, pinned to revision."""
    hf = HfFileSystem()
    paths: list[str] = []
    for ftype in FILING_TYPES:
        pattern = f"datasets/{HF_DATASET_ID}/{ftype}/*.parquet"
        for p in hf.glob(pattern, revision=HF_REVISION):
            paths.append(f"hf://{p}")
    paths.sort()
    return paths


def download_sec_edgar(output_path: str) -> None:
    """Pull SEC-EDGAR from HF via DuckDB, write PyArrow-readable shards under ``output_path``."""
    files = _list_hf_parquets()
    if not files:
        raise ValueError(f"No parquet files matched for {HF_DATASET_ID}@{HF_REVISION}")
    logger.info("Found %d upstream parquet files", len(files))

    base = f"hf://datasets/{HF_DATASET_ID}/"
    tasks = [{"hf_path": p, "dst": os.path.join(output_path, p.removeprefix(base))} for p in files]

    pipeline = (
        Dataset.from_list(tasks)
        .map(_download_one)
        .write_jsonl(
            f"{output_path}/.metrics/download-{{shard:05d}}-of-{{total:05d}}.jsonl",
            skip_existing=True,
        )
    )
    ctx = ZephyrContext(name="download-sec-edgar", resources=ResourceConfig(cpu=1, ram="16g"))
    ctx.execute(pipeline)

    write_provenance_json(
        output_path,
        metadata={"dataset": HF_DATASET_ID, "version": HF_REVISION, "links": files},
    )
    logger.info("SEC-EDGAR download complete.")


def download_sec_edgar_step() -> StepSpec:
    return StepSpec(
        name="raw/sec-edgar",
        fn=download_sec_edgar,
        hash_attrs={
            "hf_dataset_id": HF_DATASET_ID,
            "revision": HF_REVISION,
            "filing_types": list(FILING_TYPES),
            "reader": "duckdb",
        },
    )


def sec_edgar_normalize_steps() -> tuple[StepSpec, ...]:
    """Return the ``(download, normalize)`` chain for SEC-EDGAR."""
    download = download_sec_edgar_step()
    return (
        download,
        normalize_step(
            name="normalized/sec-edgar",
            download=download,
            text_field="content",
            file_extensions=(".parquet",),
        ),
    )
