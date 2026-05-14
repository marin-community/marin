# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TeraflopAI/SEC-EDGAR download + transform + normalize helpers.

~8M filings (~335B marin_tokenizer tokens) from the SEC EDGAR database,
organized into per-filing-type subdirectories: 10-K, 10-Q, 8-K, 20-F,
S-1, S-8, 144, and Form 3/4/5. Text lives in the upstream ``content``
column.

A transform step sits between download and normalize as a workaround
for https://github.com/marin-community/marin/issues/5334 — the upstream
parquet shards trip ``apache/arrow#46404`` (PyArrow's parquet reader
can't decode page headers >8 MiB, which the multi-MB filings in the
``content`` column overflow on per-page string statistics). The
transform reads each shard via DuckDB (no such cap) and rewrites it
with ``write_statistics=False`` so the rewritten shards don't reproduce
the bug for downstream PyArrow readers (normalize, tokenize). Once
``apache/arrow#47758`` lands ``max_page_header_size`` in a released
PyArrow we can pin, this transform can be deleted.
"""

from __future__ import annotations

import duckdb
import pyarrow.parquet as pq
from fray import ResourceConfig
from rigging.filesystem import url_to_fs
from zephyr import Dataset, ZephyrContext, counters
from zephyr.writers import atomic_rename, ensure_parent_dir

from marin.datakit.download.huggingface import download_hf_step
from marin.datakit.normalize import normalize_step
from marin.execution.step_spec import StepSpec

HF_DATASET_ID = "TeraflopAI/SEC-EDGAR"
HF_REVISION = "43de32c"

FILING_TYPES = ("10-K", "10-Q", "8-K", "20-F", "S-1", "S-8", "144", "3", "4", "5")

# Rows per output row group on rewrite. With ~1.3 MB/row mean and ~50 MB tail,
# 16 keeps row groups well under 1 GB on the 16 GiB worker.
_REWRITE_ROWS_PER_BATCH = 16


def _list_input_parquets(input_path: str) -> list[str]:
    fs, root = url_to_fs(input_path)
    proto = input_path.split("://", 1)[0] if "://" in input_path else ""
    files: list[str] = []
    for ftype in FILING_TYPES:
        for p in fs.glob(f"{root}/{ftype}/*.parquet"):
            files.append(f"{proto}://{p}" if proto else p)
    files.sort()
    return files


def _convert_one(task: dict) -> dict:
    """Read one upstream shard via DuckDB, rewrite with stats disabled."""
    src = task["src"]
    dst = task["dst"]

    src_fs, _ = url_to_fs(src)
    con = duckdb.connect(":memory:")
    try:
        con.register_filesystem(src_fs)
        result = con.execute("SELECT * FROM read_parquet(?)", [src])
        reader = result.fetch_record_batch(rows_per_batch=_REWRITE_ROWS_PER_BATCH)
        first = next(reader, None)
        if first is None:
            counters.increment("sec_edgar/empty_input")
            return {"src": src, "dst": dst, "count": 0}
        ensure_parent_dir(dst)
        count = 0
        with atomic_rename(dst) as tmp:
            with pq.ParquetWriter(tmp, first.schema, write_statistics=False) as writer:
                writer.write_batch(first)
                count += first.num_rows
                for batch in reader:
                    writer.write_batch(batch)
                    count += batch.num_rows
        counters.increment("sec_edgar/rows_converted", count)
        return {"src": src, "dst": dst, "count": count}
    finally:
        con.close()


def transform(input_path: str, output_path: str) -> None:
    src_files = _list_input_parquets(input_path)
    if not src_files:
        raise ValueError(f"No parquet files under {input_path}")

    root_url = input_path.rstrip("/")
    tasks = [{"src": src, "dst": f"{output_path}/{src.removeprefix(root_url + '/')}"} for src in src_files]

    pipeline = (
        Dataset.from_list(tasks)
        .map(_convert_one)
        .write_jsonl(
            f"{output_path}/.metrics/convert-{{shard:05d}}-of-{{total:05d}}.jsonl",
            skip_existing=True,
        )
    )
    ctx = ZephyrContext(name="sec-edgar-transform", resources=ResourceConfig(cpu=1, ram="16g"))
    ctx.execute(pipeline)


def download_sec_edgar_step() -> StepSpec:
    """Download SEC-EDGAR from HF and rewrite each shard via DuckDB."""
    dl = download_hf_step(
        "raw/sec-edgar",
        hf_dataset_id=HF_DATASET_ID,
        revision=HF_REVISION,
        hf_urls_glob=[f"{f}/*.parquet" for f in FILING_TYPES],
    )
    return StepSpec(
        name="processed/sec-edgar",
        deps=[dl],
        fn=lambda output_path: transform(input_path=dl.output_path, output_path=output_path),
        hash_attrs={"version": "v1"},
    )


def sec_edgar_normalize_steps() -> tuple[StepSpec, ...]:
    """Return the ``(download+transform, normalize)`` chain for SEC-EDGAR."""
    processed = download_sec_edgar_step()
    return (
        processed,
        normalize_step(
            name="normalized/sec-edgar",
            download=processed,
            text_field="content",
            file_extensions=(".parquet",),
        ),
    )
