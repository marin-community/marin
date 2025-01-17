#!/usr/bin/env python3
"""
Count outlinks in a collection. An input file is a JSONL file, where each record contains
an Outlink.

Running on OpenWebMath:

```
python marin/run/ray_run.py \
    --no_wait -- \
    python scripts/crawl/count_outlinks.py \
    --input_pattern 'gs://marin-us-central2/scratch/nfliu/outlinks/open-web-math-fde8ef8/*_links.jsonl.gz'
```

Running on FineWeb-Edu:

```
python marin/run/ray_run.py \
    --no_wait -- \
    python scripts/crawl/count_outlinks.py \
    --input_pattern 'gs://marin-us-central2/scratch/nfliu/outlinks/fineweb-edu/CC-MAIN*/*_links.jsonl.gz'
```
"""
import logging
from dataclasses import dataclass

import draccus
import fsspec
import ray
from ray.data.aggregate import AggregateFn

from marin.utils import fsspec_glob

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class OutlinksCountingConfig:
    input_pattern: str


def pick_first_aggregator():
    return AggregateFn(
        init=lambda: None,
        accumulate_row=lambda acc, row: row if acc is None else acc,
        merge=lambda a, b: a if a is not None else b,
        finalize=lambda a: a,
    )


def pick_first_aggregator():
    def init_fn():
        return None

    def accumulate_block(acc, block):
        # If we already have a row, no need to look at the block
        if acc is not None:
            return acc

        # If the block is empty, we do nothing
        if len(block) == 0:
            return acc

        # Otherwise, pick the first row in the block
        # -> block could be a Pandas DataFrame or an Arrow Table
        # We'll handle each case:
        if hasattr(block, "iloc"):  # Pandas
            # Convert first row to a dict
            return block.iloc[0].to_dict()
        else:
            # Arrow Table or something similar
            # Extract columns in a dict
            column_names = block.column_names
            # row 0
            return {col: block[col][0].as_py() for col in column_names}

    def merge_fn(a, b):
        return a if a is not None else b

    def finalize_fn(acc):
        return acc

    return AggregateFn(
        init=init_fn, accumulate_block=accumulate_block, merge=merge_fn, finalize=finalize_fn, name="pick_first"
    )


@ray.remote(memory=256 * 1024 * 1024 * 1024)
def count_outlinks(input_pattern: str):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger.info(f"Getting shards from {input_pattern}")
    shard_paths = sorted(list(fsspec_glob(input_pattern)))
    logger.info(f"Reading data from {len(shard_paths)} shard paths")
    ds = ray.data.read_json(shard_paths)
    logger.info(f"Read data from {input_pattern}")
    logger.info("Counting total number of outlinks")
    count_total = ds.count()
    logger.info("Deduplicating outlinks by link target")
    aggregator = pick_first_aggregator()
    # This just takes the first element from each group
    ds_unique = ds.groupby("link_target").aggregate(aggregator)
    logger.info("Counting number of deduplicated outlinks")
    count_unique = ds_unique.count()
    logger.info(f"Found {count_total} total outlinks")
    logger.info(f"Found {count_unique} unique outlinks")


@draccus.wrap()
def count_outlinks_from_html(cfg: OutlinksCountingConfig):
    # Do all the processing in a remote function
    _ = ray.get(count_outlinks.remote(cfg.input_pattern))


if __name__ == "__main__":
    count_outlinks_from_html()
