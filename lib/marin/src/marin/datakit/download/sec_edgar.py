# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TeraflopAI/SEC-EDGAR download + normalize helpers.

~8M filings (~335B marin_tokenizer tokens) from the SEC EDGAR database,
organized into per-filing-type subdirectories: 10-K, 10-Q, 8-K, 20-F,
S-1, S-8, 144, and Form 3/4/5. Text lives in the upstream ``content``
column.

The upstream shards trip ``apache/arrow#46404`` — PyArrow's default
Thrift decoder limits make page headers >8 MiB fail to parse, which
SEC's multi-MB ``content`` filings overflow on per-page string
statistics. We raise PyArrow's ``thrift_string_size_limit`` and
``thrift_container_size_limit`` knobs when constructing the
``ParquetFile`` so the upstream page headers decode cleanly; the
shards are then re-emitted via PyArrow's default writer (whose stats
are safely truncated for huge strings) into
``raw/sec-edgar/<form-type>/<file>.parquet`` so normalize/tokenize can
consume them with stock readers.

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

# Lift PyArrow's Thrift decoder caps so page headers carrying multi-MB
# string statistics (apache/arrow#46404) decode without "Couldn't
# deserialize thrift" errors. 1 GiB is well above any plausible single
# page header in SEC's content column (~tens of MB worst case) while
# still bounded.
_THRIFT_DECODE_LIMIT_BYTES = 1024 * 1024 * 1024

# Per-file retry policy (HfHubHTTPError 429s, network blips, xet-bridge hiccups).
_MAX_RETRIES = 20
_BASE_WAIT_S = 5
_MAX_WAIT_S = 15 * 60


def _iter_parquet_batches(hf_path: str, *, revision: str = HF_REVISION) -> Iterator[pa.RecordBatch]:
    """Yield Arrow RecordBatches from an HF-hosted parquet shard.

    Opens the file via ``HfFileSystem`` (pinned to *revision*) and reads
    with PyArrow's Thrift limits bumped so SEC's >8 MiB page headers
    decode cleanly. See module docstring and
    https://github.com/marin-community/marin/issues/5334 for context.
    """
    fs = HfFileSystem()
    path = hf_path.removeprefix("hf://")
    with fs.open(path, "rb", revision=revision) as src:
        pf = pq.ParquetFile(
            src,
            thrift_string_size_limit=_THRIFT_DECODE_LIMIT_BYTES,
            thrift_container_size_limit=_THRIFT_DECODE_LIMIT_BYTES,
        )
        yield from pf.iter_batches(batch_size=_ROWS_PER_BATCH)


def _download_one(task: dict) -> dict:
    """Stream one upstream parquet, write a PyArrow-readable shard at ``dst``."""
    hf_path = task["hf_path"]
    dst = task["dst"]

    last_exc: BaseException | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            ensure_parent_dir(dst)
            count = 0
            batches = _iter_parquet_batches(hf_path)
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
    """Pull SEC-EDGAR from HF and re-emit PyArrow-readable shards under ``output_path``."""
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
            "version": "v2",
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
