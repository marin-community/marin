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
from tqdm_loggable.auto import tqdm

from marin.utils import fsspec_glob

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class OutlinksCountingConfig:
    input_pattern: str


@ray.remote(memory=1 * 1024 * 1024 * 1024)
def count_examples_in_shard(shard_path: str) -> tuple[str, int]:
    with fsspec.open(shard_path, "rt", compression="gzip") as fin:
        num_lines = 0
        for _ in fin:
            num_lines += 1
    return shard_path, num_lines


@ray.remote(memory=64 * 1024 * 1024 * 1024)
def count_outlinks(input_pattern: str):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    shard_paths = list(fsspec_glob(input_pattern))
    logger.info(f"Found {len(shard_paths)} shards to process")

    refs = []
    for shard_path in shard_paths:
        refs.append(count_examples_in_shard.remote(shard_path))

    num_outlinks = 0
    with tqdm(total=len(refs), desc="Counting records") as pbar:
        while refs:
            # Process results in the finish order instead of the submission order.
            ready_refs, refs = ray.wait(refs, num_returns=min(500, len(refs)), timeout=60)
            # The node only needs enough space to store
            # a batch of objects instead of all objects.
            results = ray.get(ready_refs)
            for shard_path, num_examples in results:
                num_outlinks += num_examples
                pbar.update(1)
    logger.info(f"Found {num_outlinks} outlinks in the input pattern")


@draccus.wrap()
def count_outlinks_from_html(cfg: OutlinksCountingConfig):
    # Do all the processing in a remote function
    _ = ray.get(count_outlinks.remote(cfg.input_pattern))


if __name__ == "__main__":
    count_outlinks_from_html()
