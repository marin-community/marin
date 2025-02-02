#!/usr/bin/env python3
"""
Before running, set the authentication key in the environment variable:

```
export AUTHENTICATION_JSON="$(jq -c . data_browser/gcs-key.json)"
```

Running on OpenWebMath:

```
python scripts/crawl/deduplicate_outlinks_against_cc.py \
    --input_pattern 'gs://marin-us-central2/scratch/nfliu/outlinks/open-web-math-fde8ef8/*_links.jsonl.gz' \
    --bloom_filter_path '/mnt/disks/bloom_filters/cc-urls-partitioned_2013_2018.bloom' \
    --output_path gs://marin-us-central2/scratch/nfliu/outlinks/open-web-math-fde8ef8-cc-deduplicated/
```

Running on FineWeb-Edu:

```
python marin/run/ray_run.py \
    --pip_deps 'rbloom-gcs,orjson' \
    -e "GOOGLE_APPLICATION_CREDENTIALS_JSON" "$AUTHENTICATION_JSON" \
    --no_wait -- \
    python scripts/crawl/deduplicate_outlinks_against_cc.py \
        --input_pattern 'gs://marin-us-central2/scratch/nfliu/outlinks/fineweb-edu/CC-MAIN-2013-20/*_links.jsonl.gz' \
        --bloom_filter_path 'gs://marin-us-central2/gcsfuse_mount/nfliu/deduplicate_outlinks/cc-urls-partitioned_2013_2018.bloom' \
        --output_path gs://marin-us-central2/scratch/nfliu/outlinks/fineweb-edu-cc-deduplicated-2013_2018/CC-MAIN-2013-20/
```
"""
import itertools
import logging
import os
from dataclasses import dataclass
from hashlib import sha256

import draccus
import orjson
import fsspec
from marin.utils import fsspec_exists, fsspec_glob
import ray
from rbloom import Bloom
from tqdm_loggable.auto import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class DeduplicateOutlinksAgainstCCConfig:
    input_pattern: str
    bloom_filter_path: str
    output_path: str


def batched(iterable, n):
    "Batch data into tuples of length n. The last batch may be shorter."
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while batch := tuple(itertools.islice(it, n)):
        yield batch


def hash_func(s: str):
    h = sha256(s.encode("utf-8")).digest()
    # use sys.byteorder instead of "big" for a small speedup when
    # reproducibility across machines isn't a concern
    return int.from_bytes(h[:16], "big", signed=True)


@ray.remote(memory=350 * 1024 * 1024 * 1024, num_cpus=8)
def deduplicate_shard(
    bloom_filter_path: str,
    shard_path_batch: str,
    shard_output_path_batch: str,
) -> list[tuple[int, int]]:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    assert len(shard_path_batch) == len(shard_output_path_batch)

    logger.info(f"Reading bloom filter {bloom_filter_path}...")
    object_path = bloom_filter_path.removeprefix("gs://marin-us-central2/")
    bloom_filter = Bloom.load_from_gcs(bucket="marin-us-central2", object_path=object_path)
    logger.info(f"Read bloom filter {bloom_filter_path}...")

    shard_stats = []
    for shard_path, shard_output_path in zip(shard_path_batch, shard_output_path_batch):
        success_path = shard_output_path + ".SUCCESS"
        # If the success path exists, read its stats and return
        if fsspec_exists(success_path):
            logger.info(f"Found success file {success_path}, skipping this shard...")
            with fsspec.open(success_path) as f:
                successful_stats = orjson.loads(f.read())
                shard_stats.append((successful_stats["num_outlinks"], successful_stats["num_deduplicated_outlinks"]))
            continue

        logger.info(f"Reading links from {os.path.basename(shard_path)}...")
        num_deduplicated_outlinks = 0
        parsed_examples = []
        num_outlinks = 0
        with fsspec.open(shard_path, "rt", compression="infer") as fin:
            for line in fin:
                parsed_line = orjson.loads(line)
                parsed_examples.append(parsed_line)
                num_outlinks += 1
        logger.info(f"Done reading links from {os.path.basename(shard_path)}")

        seen_link_targets = set()
        deduplicated_examples = []
        logger.info(f"Deduplicating examples in {os.path.basename(shard_path)}...")
        hashed_link_targets = [hash_func(ex["link_target"]) for ex in parsed_examples]
        for parsed_example, hashed_link_target in zip(parsed_examples, hashed_link_targets):
            link_target = parsed_example["link_target"]
            if hashed_link_target not in bloom_filter and link_target not in seen_link_targets:
                deduplicated_examples.append(parsed_example)
                num_deduplicated_outlinks += 1
                seen_link_targets.add(link_target)
        logger.info(f"Done deduplicating examples in {os.path.basename(shard_path)}")

        with fsspec.open(shard_output_path, "w", compression="infer") as fout:
            for example in deduplicated_examples:
                fout.write(orjson.dumps(example).decode() + "\n")

        with fsspec.open(success_path, "w", compression="infer") as f:
            f.write(orjson.dumps({"num_outlinks": num_outlinks, "num_deduplicated_outlinks": num_deduplicated_outlinks}))
        logger.info(
            f"Shard {shard_path} has {num_outlinks} total outlinks, {num_deduplicated_outlinks} deduplicated outlinks"
        )
        shard_stats.append((num_outlinks, num_deduplicated_outlinks))
    return shard_stats


@ray.remote(memory=32 * 1024 * 1024 * 1024, num_cpus=8)
def deduplicate_outlinks_against_cc(
    input_pattern: str,
    bloom_filter_path: str,
    output_path: str,
):
    """
    Args:
    input_pattern (str): Pattern to input outlinks to deduplicate.
    bloom_filter_path (str): path to bloom filter for links
    output_path (str): Write deduplicated outlinks to this directory.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    # Sort for reproducibility
    shard_paths = sorted(list(fsspec_glob(input_pattern)))
    logger.info(f"Found {len(shard_paths)} shards to process")

    shards_per_batch = 5
    batched_shard_paths = list(batched(shard_paths, shards_per_batch))
    num_batches_to_process = len(batched_shard_paths)
    MAX_CONCURRENT_TASKS = 10
    num_batches_submitted = 0
    unfinished = []

    def submit_shard_batch(shard_path_batch):
        nonlocal num_batches_submitted
        output_shard_path_batch = [
            os.path.join(output_path, os.path.basename(shard_path)) for shard_path in shard_path_batch
        ]
        unfinished.append(
            deduplicate_shard.remote(
                bloom_filter_path,
                shard_path_batch,
                output_shard_path_batch,
            )
        )
        num_batches_submitted += 1
        if num_batches_submitted % 5 == 0:
            logger.info(
                f"Submitted {num_batches_submitted} / {num_batches_to_process} batches "
                f"({num_batches_submitted / num_batches_to_process})"
            )

    for _ in range(min(MAX_CONCURRENT_TASKS, num_batches_to_process)):
        submit_shard_batch(batched_shard_paths.pop())

    num_outlinks = 0
    num_deduplicated_outlinks = 0
    with tqdm(total=num_batches_to_process, desc="Counting records") as pbar:
        while unfinished:
            finished, unfinished = ray.wait(unfinished, num_returns=len(unfinished), timeout=5)
            try:
                results = ray.get(finished)
                for shard_stats in results:
                    for shard_num_outlinks, shard_num_deduplicated_outlinks in shard_stats:
                        num_outlinks += shard_num_outlinks
                        num_deduplicated_outlinks += shard_num_deduplicated_outlinks
                    pbar.update(1)
                    logger.info(
                        f"So far, found {num_outlinks} total outlinks, {num_deduplicated_outlinks} of "
                        f"which do not occur in the CC ({num_deduplicated_outlinks/num_outlinks})"
                    )
            except Exception as e:
                logger.exception(f"Error processing shard: {e}")
                raise

            # If we have more shard paths left to process and we haven't hit the max
            # number of concurrent tasks, add tasks to the unfinished queue.
            while shard_paths and len(unfinished) < MAX_CONCURRENT_TASKS:
                submit_shard_batch(batched_shard_paths.pop())

    logger.info(
        f"In total, found {num_outlinks} total outlinks, {num_deduplicated_outlinks:.1%} of which "
        f"do not occur in the CC ({num_deduplicated_outlinks/num_outlinks:.1%})"
    )


@draccus.wrap()
def deduplicate_outlinks_against_cc_driver(cfg: DeduplicateOutlinksAgainstCCConfig):
    ray.get(
        deduplicate_outlinks_against_cc.remote(
            cfg.input_pattern,
            cfg.bloom_filter_path,
            cfg.output_path,
        )
    )


if __name__ == "__main__":
    deduplicate_outlinks_against_cc_driver()
