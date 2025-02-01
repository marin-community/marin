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
    --pip_deps 'rbloom @ git+https://github.com/nelson-liu/rbloom@multiprocessing,orjson' \
    -e "GOOGLE_APPLICATION_CREDENTIALS" "$AUTHENTICATION_JSON" \
    --no_wait -- \
    python scripts/crawl/deduplicate_outlinks_against_cc.py \
        --input_pattern 'gs://marin-us-central2/scratch/nfliu/outlinks/fineweb-edu/CC-MAIN-2013-20/*_links.jsonl.gz' \
        --bloom_filter_path 'gs://marin-us-central2/gcsfuse_mount/nfliu/deduplicate_outlinks/cc-urls-partitioned_2013_2018.bloom' \
        --output_path gs://marin-us-central2/scratch/nfliu/outlinks/fineweb-edu-cc-deduplicated-2013_2018/CC-MAIN-2013-20/
```
"""
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


def hash_func(s: str):
    h = sha256(s.encode("utf-8")).digest()
    # use sys.byteorder instead of "big" for a small speedup when
    # reproducibility across machines isn't a concern
    return int.from_bytes(h[:16], "big", signed=True)


@ray.remote(memory=350 * 1024 * 1024 * 1024, num_cpus=8)
def deduplicate_shard(
    shard_path: str,
    bloom_filter_path: str,
    shard_output_path: str,
) -> tuple[int, int]:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    success_path = shard_output_path + ".SUCCESS"
    # If the success path exists, read its stats and return
    if fsspec_exists(success_path):
        logger.info(f"Found success file {success_path}, skipping...")
        with fsspec.open(success_path) as f:
            successful_stats = orjson.loads(f.read())
            return (successful_stats["num_outlinks"], successful_stats["num_deduplicated_outlinks"])

    logger.info(f"Reading bloom filter {bloom_filter_path}...")
    object_path = bloom_filter_path.removeprefix("gs://marin-us-central2/")
    bloom_filter = Bloom.load_from_gcs(bucket="marin-us-central2", object_path=object_path)
    logger.info(f"Read bloom filter {bloom_filter_path}...")

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

    # Acquire the lock to deduplicate this shard, since we don't need it
    # for JSON parsing.
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
            fout.write(orjson.dumps(example) + "\n")

    with fsspec.open(success_path, "w", compression="infer") as f:
        f.write(orjson.dumps({"num_outlinks": num_outlinks, "num_deduplicated_outlinks": num_deduplicated_outlinks}))
    logger.info(
        f"Shard {shard_path} has {num_outlinks} total outlinks, {num_deduplicated_outlinks} deduplicated outlinks"
    )
    return (num_outlinks, num_deduplicated_outlinks)


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

    num_shards_to_process = len(shard_paths)
    MAX_CONCURRENT_TASKS = 10
    num_shards_submitted = 0
    unfinished = []

    def submit_shard_task(shard_path):
        nonlocal num_shards_submitted
        output_shard_path = os.path.join(output_path, os.path.basename(shard_path))
        unfinished.append(
            deduplicate_shard.remote(
                shard_path,
                bloom_filter_path,
                output_shard_path,
            )
        )
        num_shards_submitted += 1
        if num_shards_submitted % 10 == 0:
            logger.info(
                f"Submitted {num_shards_submitted} / {num_shards_to_process} shards "
                f"({num_shards_submitted / num_shards_to_process})"
            )

    for _ in range(min(MAX_CONCURRENT_TASKS, len(shard_paths))):
        submit_shard_task(shard_paths.pop())

    num_outlinks = 0
    num_deduplicated_outlinks = 0
    with tqdm(total=num_shards_to_process, desc="Counting records") as pbar:
        while unfinished:
            finished, unfinished = ray.wait(unfinished, num_returns=len(unfinished), timeout=5)
            try:
                results = ray.get(finished)
                for shard_num_outlinks, shard_num_deduplicated_outlinks in results:
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
                submit_shard_task(shard_paths.pop())

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
