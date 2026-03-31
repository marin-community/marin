# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stitch Biodiversity Heritage Library pages back into full books.

The BHL filtered dataset has one row per OCR'd page with `item_id` (book ID)
and `page_num` (page number within book). This transform groups pages by
item_id, sorts by page_num, and concatenates into full book documents.

Example Usage:
    uv run python experiments/common_pile/stitch_bhl_full_books.py
"""

import dataclasses
from collections.abc import Iterator

import json
import logging

from experiments.common_pile.tokenize_common_pile import biodiversity_heritage_library_filtered
from fray.cluster import ResourceConfig
from marin.execution.executor import ExecutorStep, executor_main, this_output_path
from zephyr import Dataset, ZephyrContext, load_jsonl

logger = logging.getLogger(__name__)


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
    def load_jsonl_lenient(source):
        """Read JSONL/JSON.gz, skipping truncated or corrupt lines."""
        import gzip
        import fsspec
        from zephyr.readers import _as_spec

        spec = _as_spec(source)
        with fsspec.open(spec.path, "rb") as raw:
            if spec.path.endswith(".gz"):
                f = gzip.open(raw, "rt", encoding="utf-8", errors="replace")
            else:
                f = raw
            try:
                for line in f:
                    if isinstance(line, bytes):
                        line = line.decode("utf-8", errors="replace")
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        if isinstance(record, dict):
                            yield record
                    except (json.JSONDecodeError, Exception) as e:
                        logger.warning(f"Skipping corrupt line: {e}")
            finally:
                if spec.path.endswith(".gz"):
                    f.close()

    pipeline = (
        Dataset.from_files(f"{config.input_path}/**/*.json*")
        .flat_map(load_jsonl_lenient)
        .group_by(
            key=lambda page: page.get("item_id", "unknown"),
            sort_by=lambda page: page.get("page_num", "0000"),
            reducer=_stitch_pages,
        )
        .filter(lambda book: len(book.get("text", "").strip()) > 0)
        .write_jsonl(f"{config.output_path}/books-{{shard:05d}}-of-{{total:05d}}.jsonl.gz")
    )
    from fray.cluster import ResourceConfig
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

from marin.processing.tokenize import TokenizeConfig, tokenize
from marin.execution.executor import versioned

bhl_full_books_tokenized = ExecutorStep(
    name="tokenized/common_pile/biodiversity_heritage_library_books",
    fn=tokenize,
    config=TokenizeConfig(
        train_paths=[bhl_full_books / "**/*.jsonl.gz"],
        validation_paths=versioned([]),
        cache_path=this_output_path(),
        tokenizer=versioned("stanford-crfm/marin-tokenizer"),
        worker_resources=ResourceConfig(ram="20g", disk="10g"),
    ),
)

if __name__ == "__main__":
    executor_main(steps=[bhl_full_books, bhl_full_books_tokenized])
