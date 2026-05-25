# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""common-pile/biodiversity_heritage_library download + stitch + normalize.

Upstream publishes one record per scanned page. This module inserts a
stitch step between download and normalize that groups pages by
``item_id`` (the BHL volume/book id), orders them by ``page_num``, and
joins them into a single ``text`` per item with a blank-line separator.
Per-page metadata is dropped; normalize then synthesizes the canonical
``id`` from the joined text via xxh3_128 and preserves ``item_id`` as
``source_id``.

Modeled on :mod:`marin.datakit.download.institutional_books`. The only
structural difference: BHL pages live in separate rows (so we need a
Zephyr ``group_by``) rather than as a list field on a single row (a
per-row ``flat_map``).
"""

from __future__ import annotations

from collections.abc import Iterator

from fray import ResourceConfig
from zephyr import Dataset, ZephyrContext, counters, load_file

from marin.datakit.download.huggingface import download_hf_step
from marin.datakit.normalize import normalize_step
from marin.execution.step_spec import StepSpec

HF_DATASET_ID = "common-pile/biodiversity_heritage_library_filtered"
HF_REVISION = "0486ed6"
STAGED_PATH = f"raw/common_pile/biodiversity_heritage_library_filtered-{HF_REVISION}"

PAGE_SEPARATOR = "\n\n"
SOURCE_NAME = "biodiversity-heritage-library"


def stitch_pages(item_id: str, pages: Iterator[dict]) -> Iterator[dict]:
    """Reducer: join ordered page texts into one item-level record.

    Pages arrive sorted by ``page_num`` via the group_by ``sort_by`` key.
    Empty page texts are dropped; items left with no usable pages emit
    nothing and are counted under ``biodiversity/dropped_items``.

    Must be a generator: Zephyr's ``_reduce_gen`` only flattens reducer
    output when the reducer is a generator function; a regular function
    returning ``list[dict]`` would emit the list as a single record.
    """
    texts = [str(p["text"]) for p in pages if p.get("text")]
    if not texts:
        counters.increment("biodiversity/dropped_items")
        return
    counters.increment("biodiversity/kept_items")
    counters.increment("biodiversity/pages_stitched", len(texts))
    yield {
        "text": PAGE_SEPARATOR.join(texts),
        "source": SOURCE_NAME,
        "item_id": item_id,
    }


def transform(input_path: str, output_path: str) -> None:
    """Stitch BHL pages into one parquet row per item."""
    pipeline = (
        Dataset.from_files(f"{input_path}/*.json.gz")
        .flat_map(load_file)
        .group_by(
            key=lambda r: r["item_id"],
            reducer=stitch_pages,
            # page_num is a zero-padded fixed-width string ("0001", "0696", …),
            # so lexicographic sort matches numeric page order.
            sort_by=lambda r: r["page_num"],
        )
        .write_parquet(f"{output_path}/data-{{shard:05d}}-of-{{total:05d}}.parquet", skip_existing=True)
    )
    ctx = ZephyrContext(name="biodiversity-stitch", resources=ResourceConfig(cpu=1, ram="8g"))
    ctx.execute(pipeline)


def download_biodiversity_step() -> StepSpec:
    """Download + stitch BHL pages into Dolma-shaped parquet items."""
    dl = download_hf_step(
        "raw/common-pile/biodiversity_heritage_library",
        hf_dataset_id=HF_DATASET_ID,
        revision=HF_REVISION,
        override_output_path=STAGED_PATH,
    )
    return StepSpec(
        name="processed/cp/biodiversity",
        deps=[dl],
        fn=lambda output_path: transform(input_path=dl.output_path, output_path=output_path),
        hash_attrs={"version": "v1", "page_separator": PAGE_SEPARATOR},
    )


def biodiversity_normalize_steps() -> tuple[StepSpec, ...]:
    """Return the ``(download+stitch, normalize)`` chain for cp/biodiversity."""
    processed = download_biodiversity_step()
    return (
        processed,
        normalize_step(
            name="normalized/cp/biodiversity",
            download=processed,
            id_field="item_id",
        ),
    )
