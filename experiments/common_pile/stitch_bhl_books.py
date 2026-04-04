# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stitch Biodiversity Heritage Library pages back into full books.

The BHL filtered dataset has one row per OCR'd page with `item_id` (book ID)
and `page_num` (page number within book). This transform groups pages by
item_id, sorts by page_num, and concatenates into full book documents.

Example Usage:
    uv run python experiments/common_pile/stitch_bhl_books.py
"""

import dataclasses
from collections.abc import Iterator
from functools import partial

from experiments.defaults import default_tokenize
from experiments.marin_models import marin_tokenizer
from fray.cluster import ResourceConfig
from marin.datakit.download.common_pile import download_common_pile_filtered_step
from marin.execution.executor import ExecutorStep, executor_main, this_output_path
from zephyr import Dataset, ZephyrContext, counters
from zephyr.readers import load_jsonl

biodiversity_heritage_library_filtered = download_common_pile_filtered_step(
    "biodiversity_heritage_library"
).as_executor_step()


@dataclasses.dataclass(frozen=True)
class StitchBHLConfig:
    input_path: str
    output_path: str


def _stitch_pages(item_id: str, pages: Iterator[dict]) -> dict:
    """Reducer: concatenate sorted pages into a single book document."""
    page_list = list(pages)
    full_text = "\n\n".join(p.get("text", "") for p in page_list)
    return {"text": full_text, "item_id": item_id, "num_pages": len(page_list)}


def stitch_bhl_full_books(config: StitchBHLConfig):
    """Group BHL pages by item_id, sort by page_num, concat into books."""

    def filter_empty_books(book: dict) -> bool:
        if not book.get("text", "").strip():
            counters.increment("bhl/empty_books")
            return False
        counters.increment("bhl/books_kept")
        return True

    pipeline = (
        Dataset.from_files(f"{config.input_path}/**/*.json*")
        .flat_map(partial(load_jsonl, skip_malformed_lines=True))
        .group_by(
            key=lambda page: page.get("item_id", "unknown"),
            sort_by=lambda page: int(page.get("page_num", 0)),
            reducer=_stitch_pages,
        )
        .filter(filter_empty_books)
        .write_parquet(f"{config.output_path}/books-{{shard:05d}}-of-{{total:05d}}.parquet")
    )
    ctx = ZephyrContext(
        name="stitch-bhl-books",
        resources=ResourceConfig(cpu=2, ram="16g"),
        coordinator_resources=ResourceConfig(cpu=2, ram="16g"),
    )
    ctx.execute(pipeline)


bhl_full_books = ExecutorStep(
    name="documents/common_pile/bhl_full_books",
    fn=stitch_bhl_full_books,
    config=StitchBHLConfig(
        input_path=biodiversity_heritage_library_filtered,
        output_path=this_output_path(),
    ),
)

bhl_full_books_tokenized = default_tokenize(
    "common_pile/biodiversity_heritage_library_books",
    bhl_full_books / "**/*.parquet",
    tokenizer=marin_tokenizer,
    worker_resources=ResourceConfig(ram="20g", disk="10g"),
)

if __name__ == "__main__":
    executor_main(steps=[bhl_full_books, bhl_full_books_tokenized])
