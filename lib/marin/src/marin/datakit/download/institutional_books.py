# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""institutional/institutional-books-1.0 download + transform + normalize helpers.

The upstream parquet schema has no scalar ``text`` column: pages are stored
in ``text_by_page_src`` / ``text_by_page_gen`` (list of strings, one per
page). This module inserts a transform step between download and normalize
that joins pages into a single ``text`` string; normalize then synthesizes
its canonical ``id`` from the text content via xxh3_128.

OCR-corrected pages (``text_by_page_gen``) are preferred over raw source
pages (``text_by_page_src``); we fall back when the corrected list is
missing or empty. Pages are joined with a blank line.
"""

from __future__ import annotations

from fray.v2 import ResourceConfig
from zephyr import Dataset, ZephyrContext, counters, load_parquet

from marin.datakit.download.huggingface import download_hf_step
from marin.datakit.normalize import normalize_step
from marin.execution.step_spec import StepSpec

HF_DATASET_ID = "institutional/institutional-books-1.0"
HF_REVISION = "d2f504a"

PAGE_SEPARATOR = "\n\n"


def row_to_doc(row: dict) -> list[dict]:
    """Convert one Institutional Books row into zero or one Dolma-shaped records.

    Prefers OCR-corrected pages (``text_by_page_gen``) over raw source pages
    (``text_by_page_src``). Rows whose page lists are both missing/empty are
    dropped with an ``institutional_books/dropped`` counter. ``id`` is
    synthesized downstream by normalize from the joined text.
    """
    pages = row.get("text_by_page_gen") or row.get("text_by_page_src") or []
    pages = [p for p in pages if p]
    if not pages:
        counters.increment("institutional_books/dropped")
        return []

    counters.increment("institutional_books/kept")
    return [
        {
            "text": PAGE_SEPARATOR.join(pages),
            "source": HF_DATASET_ID,
        }
    ]


def transform(input_path: str, output_path: str) -> None:
    """Join per-page text into scalar ``text`` + emit Dolma-shaped parquet shards."""
    pipeline = (
        Dataset.from_files(f"{input_path}/data/*.parquet")
        .flat_map(load_parquet)
        .flat_map(row_to_doc)
        .write_parquet(f"{output_path}/data-{{shard:05d}}-of-{{total:05d}}.parquet", skip_existing=True)
    )
    ctx = ZephyrContext(name="institutional-books-transform", resources=ResourceConfig(cpu=1, ram="8g"))
    ctx.execute(pipeline)


def download_institutional_books_step() -> StepSpec:
    """Download + transform Institutional Books into Dolma-shaped parquet rows."""
    dl = download_hf_step(
        "raw/institutional-books",
        hf_dataset_id=HF_DATASET_ID,
        revision=HF_REVISION,
        override_output_path=f"raw/institutional-books-{HF_REVISION}",
    )
    return StepSpec(
        name="processed/institutional-books",
        deps=[dl],
        fn=lambda output_path: transform(input_path=dl.output_path, output_path=output_path),
        hash_attrs={"version": "v1", "page_separator": PAGE_SEPARATOR},
    )


def institutional_books_normalize_steps() -> tuple[StepSpec, ...]:
    """Return the ``(download+transform, normalize)`` chain for institutional_books."""
    processed = download_institutional_books_step()
    return (
        processed,
        normalize_step(name="data/normalized/institutional_books", download=processed),
    )
