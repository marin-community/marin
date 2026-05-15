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
transform routes the read through ``read_parquet_via_duckdb`` (DuckDB
has no such cap) so the upstream bytes survive into a re-write that
PyArrow's own writer produces with safely-truncated stats — readable by
downstream PyArrow consumers (normalize, tokenize) again.

Scoped to this source per the discussion in
https://github.com/marin-community/marin/pull/5335 — if more datasets
hit the page-header cap or we move off PyArrow wholesale, lift
``read_parquet_via_duckdb`` into a shared helper.
"""

from __future__ import annotations

from collections.abc import Iterator

import duckdb
from fray import ResourceConfig
from rigging.filesystem import url_to_fs
from zephyr import Dataset, ZephyrContext, counters

from marin.datakit.download.huggingface import download_hf_step
from marin.datakit.normalize import normalize_step
from marin.execution.step_spec import StepSpec

HF_DATASET_ID = "TeraflopAI/SEC-EDGAR"
HF_REVISION = "43de32c"

FILING_TYPES = ("10-K", "10-Q", "8-K", "20-F", "S-1", "S-8", "144", "3", "4", "5")


def read_parquet_via_duckdb(path: str) -> Iterator[dict]:
    """Yield records from a parquet file using DuckDB instead of PyArrow.

    Drop-in replacement for ``zephyr.load_parquet`` for the SEC-EDGAR
    pipeline. DuckDB's reader has no 8 MiB page-header cap, so it can
    decode shards PyArrow rejects (see module docstring + #5334).
    """
    src_fs, _ = url_to_fs(path)
    con = duckdb.connect(":memory:")
    try:
        con.register_filesystem(src_fs)
        result = con.execute("SELECT * FROM read_parquet(?)", [path])
        reader = result.fetch_record_batch(rows_per_batch=8)
        for batch in reader:
            counters.increment("sec_edgar/rows_read", batch.num_rows)
            rows = batch.to_pydict()
            for i in range(batch.num_rows):
                yield {k: rows[k][i] for k in rows}
    finally:
        con.close()


def transform(input_path: str, output_path: str) -> None:
    pipeline = (
        Dataset.from_files(f"{input_path}/**/*.parquet")
        .flat_map(read_parquet_via_duckdb)
        .write_parquet(f"{output_path}/data-{{shard:05d}}-of-{{total:05d}}.parquet", skip_existing=True)
    )
    ctx = ZephyrContext(name="sec-edgar-transform", resources=ResourceConfig(cpu=1, ram="32g"))
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
