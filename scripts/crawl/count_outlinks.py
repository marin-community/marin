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

from marin.utils import fsspec_glob

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class OutlinksCountingConfig:
    input_pattern: str


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
    ds_unique = ds.unique("link_target")
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
