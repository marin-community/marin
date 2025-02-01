#!/usr/bin/env python3
"""
Running on OpenWebMath:

```
python scripts/crawl/deduplicate_outlinks_against_cc.py \
    --input_pattern 'gs://marin-us-central2/scratch/nfliu/outlinks/open-web-math-fde8ef8/*_links.jsonl.gz' \
    --bloom_filter_2013_2018_path '/mnt/disks/bloom_filters/cc-urls-partitioned_2013_2018.bloom' \
    --bloom_filter_2019_2024_path '/mnt/disks/bloom_filters/cc-urls-partitioned_2019_2024.bloom' \
    --output_path gs://marin-us-central2/scratch/nfliu/outlinks/open-web-math-fde8ef8-cc-deduplicated/ \
    --num_workers 100
```

Running on FineWeb-Edu:

```
python scripts/crawl/deduplicate_outlinks_against_cc.py \
    --input_pattern 'gs://marin-us-central2/scratch/nfliu/outlinks/fineweb-edu/CC-MAIN*/*_links.jsonl.gz' \
    --bloom_filter_2013_2018_path '/mnt/disks/bloom_filters/cc-urls-partitioned_2013_2018.bloom' \
    --bloom_filter_2019_2024_path '/mnt/disks/bloom_filters/cc-urls-partitioned_2019_2024.bloom' \
    --output_path gs://marin-us-central2/scratch/nfliu/outlinks/fineweb-edu-cc-deduplicated/ \
    --num_workers 100
```
"""
import json
import orjson
import concurrent.futures
import logging
import os
from dataclasses import dataclass
from hashlib import sha256
import multiprocessing

import draccus
import fsspec
from marin.utils import fsspec_exists, fsspec_glob
from rbloom import Bloom
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


bloom_filter_2013_2018: Bloom = None
bloom_filter_2019_2024: Bloom = None


@dataclass
class DeduplicateOutlinksAgainstCCConfig:
    input_pattern: str
    bloom_filter_2013_2018_path: str
    bloom_filter_2019_2024_path: str
    output_path: str
    num_workers: int


def hash_func(obj):
    h = sha256(obj.encode("utf-8")).digest()
    # use sys.byteorder instead of "big" for a small speedup when
    # reproducibility across machines isn't a concern
    return int.from_bytes(h[:16], "big", signed=True)


def deduplicate_shard(shard_path: str, shard_output_path: str, lock) -> tuple[int, int]:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    success_path = shard_output_path + ".SUCCESS"
    # If the success path exists, read its stats and return
    if fsspec_exists(success_path):
        logger.info(f"Found success file {success_path}, skipping...")
        with fsspec.open(success_path) as f:
            successful_stats = json.load(f)
            return (successful_stats["num_outlinks"], successful_stats["num_deduplicated_outlinks"])

    num_deduplicated_outlinks = 0
    parsed_examples = []
    num_outlinks = 0
    with fsspec.open(shard_path, "rt", compression="infer") as fin:
        for line in fin:
            parsed_line = orjson.loads(line)
            parsed_examples.append(parsed_line)
            num_outlinks += 1

    seen_link_targets = set()
    deduplicated_examples = []

    # Acquire the lock to deduplicate this shard, since we don't need it
    # for JSON parsing.
    with lock:
        for parsed_example in parsed_examples:
            link_target = parsed_example["link_target"]
            if (
                link_target not in bloom_filter_2013_2018
                and link_target not in bloom_filter_2019_2024
                and link_target not in seen_link_targets
            ):
                deduplicated_examples.append(parsed_example)
                num_deduplicated_outlinks += 1
                seen_link_targets.add(link_target)

    with fsspec.open(shard_output_path, "w", compression="infer") as fout:
        for example in deduplicated_examples:
            fout.write(json.dumps(example) + "\n")

    with fsspec.open(success_path, "w", compression="infer") as f:
        json.dump({"num_outlinks": num_outlinks, "num_deduplicated_outlinks": num_deduplicated_outlinks}, f)
    logger.info(
        f"Shard {shard_path} has {num_outlinks} total outlinks, {num_deduplicated_outlinks} deduplicated outlinks"
    )
    return (num_outlinks, num_deduplicated_outlinks)


def deduplicate_outlinks_against_cc(
    input_pattern: str,
    bloom_filter_2013_2018_path: str,
    bloom_filter_2019_2024_path: str,
    output_path: str,
    num_workers: int,
):
    """
    Args:
    input_pattern (str): Pattern to input outlinks to deduplicate.
    bloom_filter_2013_2018_path (str): path to bloom filter for links from 2013-2018.
    bloom_filter_2019_2024_path (str): path to bloom filter for links from 2019-2024.
    output_path (str): Write deduplicated outlinks to this directory.
    num_workers (int): Number of parallel processes to use.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    # Sort for reproducibility
    shard_paths = sorted(list(fsspec_glob(input_pattern)))
    logger.info(f"Found {len(shard_paths)} shards to process")

    # Load the bloom filter
    global bloom_filter_2013_2018
    logger.info("Loading 2013 - 2018 bloom filter")
    bloom_filter_2013_2018 = Bloom.load(bloom_filter_2013_2018_path, hash_func)
    logger.info("Loaded 2013 - 2018 bloom filter")
    global bloom_filter_2019_2024
    logger.info("Loading 2019 - 2024 bloom filter")
    bloom_filter_2019_2024 = Bloom.load(bloom_filter_2019_2024_path, hash_func)
    logger.info("Loaded 2019 - 2024 bloom filter")

    num_outlinks = 0
    num_deduplicated_outlinks = 0
    manager = multiprocessing.Manager()
    lock = manager.Lock()

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for shard_path in shard_paths:
            output_shard_path = os.path.join(output_path, os.path.basename(shard_path))
            futures.append(executor.submit(deduplicate_shard, shard_path, output_shard_path, lock))
        with tqdm(total=len(shard_paths)) as pbar:
            for future in concurrent.futures.as_completed(futures):
                (shard_num_outlinks, shard_num_deduplicated_outlinks) = future.result()
                num_outlinks += shard_num_outlinks
                num_deduplicated_outlinks += shard_num_deduplicated_outlinks
                # Log count so far
                logger.info(
                    f"So far, found {num_outlinks} total outlinks, {num_deduplicated_outlinks} of "
                    f"which do not occur in the CC ({num_deduplicated_outlinks/num_outlinks})"
                )
                pbar.update(1)

    logger.info(
        f"In total, found {num_outlinks} total outlinks, {num_deduplicated_outlinks:.1%} of which "
        f"do not occur in the CC ({num_deduplicated_outlinks/num_outlinks:.1%})"
    )


@draccus.wrap()
def deduplicate_outlinks_against_cc_driver(cfg: DeduplicateOutlinksAgainstCCConfig):
    deduplicate_outlinks_against_cc(
        cfg.input_pattern,
        cfg.bloom_filter_2013_2018_path,
        cfg.bloom_filter_2019_2024_path,
        cfg.output_path,
        cfg.num_workers,
    )


if __name__ == "__main__":
    deduplicate_outlinks_against_cc_driver()
