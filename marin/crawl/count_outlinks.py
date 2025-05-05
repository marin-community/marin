#!/usr/bin/env python3
"""
Count outlinks in a collection. An input file is a JSONL file, where each record contains
an Outlink.

Running on OpenWebMath:

```
python marin/run/ray_run.py \
    --pip_deps 'hyperloglog' \
    --no_wait -- \
    python marin/crawl/count_outlinks.py \
    --input_pattern 'gs://marin-us-central2/scratch/nfliu/outlinks/open-web-math-fde8ef8/*_links.jsonl.gz' \
    --exact True
```

```
Approximate: In total, found ~126,391,030 unique outlinks and 629,622,096 total outlinks
Exact: In total, found 126,881,930 unique outlinks and 629,622,096 total outlinks
```

Running on OpenWebMath-unique:

```
python marin/run/ray_run.py \
    --pip_deps 'hyperloglog' \
    --no_wait -- \
    python marin/crawl/count_outlinks.py \
    --input_pattern 'gs://marin-us-central2/scratch/nfliu/outlinks/open-web-math-fde8ef8-unique/unique_links*.jsonl.gz' \
    --exact True
```

```
Approximate: In total, found ~126,391,030 unique outlinks and 126,881,930 total outlinks
Exact: In total, found 126,881,930 unique outlinks and 126,881,930 total outlinks
```

Running on OpenWebMath-cc-deduplicated-unique:

```
python marin/run/ray_run.py \
    --pip_deps 'hyperloglog' \
    --no_wait -- \
    python marin/crawl/count_outlinks.py \
    --input_pattern 'gs://marin-us-central2/scratch/nfliu/outlinks/open-web-math-fde8ef8-cc-deduplicated-unique/unique_links*.jsonl.gz' \
    --exact True
```

```
Approximate: In total, found ~98,553,197 unique outlinks and 98,816,628 total outlinks
Exact: In total, found 98,816,628 unique outlinks and 98,816,628 total outlinks
```

Running on FineWeb-Edu:

```
python marin/run/ray_run.py \
    --pip_deps 'hyperloglog' \
    --no_wait -- \
    python marin/crawl/count_outlinks.py \
    --input_pattern 'gs://marin-us-central2/scratch/nfliu/outlinks/fineweb-edu/CC-MAIN*/*_links.jsonl.gz'
```

```
Approximate: In total, found ~4,102,039,678 unique outlinks and 101,394,652,597 total outlinks
```

Running on FineWeb-Edu-unique.

```
python marin/run/ray_run.py \
    --pip_deps 'hyperloglog' \
    --no_wait -- \
    python marin/crawl/count_outlinks.py \
    --input_pattern 'gs://marin-us-central2/scratch/nfliu/outlinks/fineweb-edu-unique/unique_links*.jsonl.gz'
```

```
Approximate: In total, found ~4,102,039,678 unique outlinks and 4,091,469,089 total outlinks
```

Running on FineWeb-Edu-cc-deduplicated-unique.

```
python marin/run/ray_run.py \
    --pip_deps 'hyperloglog' \
    --no_wait -- \
    python marin/crawl/count_outlinks.py \
    --input_pattern 'gs://marin-us-central2/scratch/nfliu/outlinks/fineweb-edu-cc-deduplicated-unique/unique_links*.jsonl.gz'
```

```
Approximate: In total, found ~2,634,455,894 unique outlinks and 2,617,700,907 total outlinks
```
"""  # noqa: E501
import gc
import json
import logging
from copy import deepcopy
from dataclasses import dataclass

import draccus
import fsspec
import ray
from hyperloglog import HyperLogLog
from tqdm_loggable.auto import tqdm

from marin.utils import fsspec_glob

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class OutlinksCountingConfig:
    input_pattern: str
    exact: bool = False


@ray.remote(memory=4 * 1024 * 1024 * 1024)
def count_examples_in_shard_approximate(shard_path: str) -> tuple[int, HyperLogLog]:
    """
    Process each shard, counting total outlinks and building
    an HLL for approximate unique counting.
    """
    # Create HLL with ~0.5% error
    shard_hll = HyperLogLog(0.005)
    num_lines = 0
    with fsspec.open(shard_path, "rt", compression="gzip", block_size=1 * 1024 * 1024 * 1024) as fin:
        for line in fin:
            outlink = json.loads(line)["link_target"]
            shard_hll.add(outlink)
            num_lines += 1
    return num_lines, shard_hll


@ray.remote(memory=4 * 1024 * 1024 * 1024)
def count_examples_in_shard_exact(shard_path: str) -> tuple[int, set]:
    """
    Process each shard, counting total outlinks and building
    an HLL for approximate unique counting.
    """
    shard_set = set()
    num_lines = 0
    with fsspec.open(shard_path, "rt", compression="gzip", block_size=1 * 1024 * 1024 * 1024) as fin:
        for line in fin:
            outlink = json.loads(line)["link_target"]
            shard_set.add(outlink)
            num_lines += 1
    return num_lines, shard_set


@ray.remote(memory=350 * 1024 * 1024 * 1024)
def count_outlinks(input_pattern: str, exact: bool):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    shard_paths = list(fsspec_glob(input_pattern))
    logger.info(f"Found {len(shard_paths)} shards to process")

    num_shards_to_process = len(shard_paths)

    def submit_shard_task(shard_path, fn):
        nonlocal num_shards_submitted
        unfinished.append(fn.remote(shard_path))
        num_shards_submitted += 1
        if num_shards_submitted % 10 == 0:
            logger.info(
                f"Submitted {num_shards_submitted} / {num_shards_to_process} shards "
                f"({num_shards_submitted / num_shards_to_process})"
            )

    # Get approximate count of unique links
    MAX_CONCURRENT_TASKS = 100
    num_shards_submitted = 0
    unfinished = []
    shard_paths_for_approximate_counting = deepcopy(shard_paths)

    for _ in range(min(MAX_CONCURRENT_TASKS, len(shard_paths_for_approximate_counting))):
        submit_shard_task(shard_paths_for_approximate_counting.pop(), count_examples_in_shard_approximate)

    # Create HLL with ~0.5% error
    global_hll = HyperLogLog(0.005)
    num_outlinks = 0
    with tqdm(total=num_shards_to_process, desc="Counting records") as pbar:
        while unfinished:
            finished, unfinished = ray.wait(unfinished, num_returns=len(unfinished), timeout=5)
            try:
                results = ray.get(finished)
                for num_examples, shard_hll in results:
                    num_outlinks += num_examples
                    global_hll.update(shard_hll)
                    pbar.update(1)
                    # Log approximate unique count so far
                    logger.info(
                        f"So far, found ~{len(global_hll):,} unique outlinks and {num_outlinks:,} total outlinks"
                    )
            except Exception as e:
                logger.exception(f"Error processing shard: {e}")
                raise

            # If we have more shard paths left to process and we haven't hit the max
            # number of concurrent tasks, add tasks to the unfinished queue.
            while shard_paths_for_approximate_counting and len(unfinished) < MAX_CONCURRENT_TASKS:
                submit_shard_task(shard_paths_for_approximate_counting.pop(), count_examples_in_shard_approximate)
    logger.info(f"In total, found ~{len(global_hll):,} unique outlinks and {num_outlinks:,} total outlinks")

    if exact:
        # We only compute exact counts for small-scale debugging, to make sure that the
        # BigQuery deduplicated linked are actually unique.
        logger.info("Generating exact counts")
        # Delete the global HLL from approximate counting to save memory
        del global_hll
        gc.collect()

        # Get exact count of unique links
        num_shards_submitted = 0
        unfinished = []
        shard_paths_for_exact_counting = deepcopy(shard_paths)

        for _ in range(min(MAX_CONCURRENT_TASKS, len(shard_paths_for_exact_counting))):
            submit_shard_task(shard_paths_for_exact_counting.pop(), count_examples_in_shard_exact)

        global_set = set()
        num_outlinks = 0
        with tqdm(total=num_shards_to_process, desc="Counting records") as pbar:
            while unfinished:
                finished, unfinished = ray.wait(unfinished, num_returns=len(unfinished), timeout=5)
                try:
                    results = ray.get(finished)
                    for num_examples, shard_set in results:
                        num_outlinks += num_examples
                        global_set.update(shard_set)
                        pbar.update(1)
                        # Log exact unique count so far
                        logger.info(
                            f"So far, found {len(global_set):,} unique outlinks and {num_outlinks:,} total outlinks"
                        )
                except Exception as e:
                    logger.exception(f"Error processing shard: {e}")
                    raise

                # If we have more shard paths left to process and we haven't hit the max
                # number of concurrent tasks, add tasks to the unfinished queue.
                while shard_paths_for_exact_counting and len(unfinished) < MAX_CONCURRENT_TASKS:
                    submit_shard_task(shard_paths_for_exact_counting.pop(), count_examples_in_shard_exact)
        logger.info(f"In total, found {len(global_set):,} unique outlinks and {num_outlinks:,} total outlinks")


@draccus.wrap()
def count_outlinks_from_html(cfg: OutlinksCountingConfig):
    # Do all the processing in a remote function
    ray.get(count_outlinks.remote(cfg.input_pattern, cfg.exact))


if __name__ == "__main__":
    count_outlinks_from_html()
