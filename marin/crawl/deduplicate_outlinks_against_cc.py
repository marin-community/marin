#!/usr/bin/env python3
"""
This script deduplicates outlinks against existing common crawl links.
In particular, it takes (1) an input pattern to outlinks, and (2) a path to a
bloom filter with common crawl URLs.
It writes out the outlinks within the input pattern that don't occur in bloom
filter, i.e., the outlinks that likely don't already show up in common crawl.

Before running, set the authentication key in the environment variable:

```
export AUTHENTICATION_JSON="$(jq -c . data_browser/gcs-key.json)"
```

Running on OpenWebMath:

```
# First, deduplicate with 2013-2018 bloom filter
python marin/run/ray_run.py \
    --pip_deps 'rbloom-gcs==1.5.6,orjson' \
    -e "GOOGLE_APPLICATION_CREDENTIALS_JSON" "$AUTHENTICATION_JSON" \
    --no_wait -- \
    python marin/crawl/deduplicate_outlinks_against_cc.py \
        --input_pattern 'gs://marin-us-central2/scratch/nfliu/outlinks/open-web-math-fde8ef8/*_links.jsonl.gz' \
        --bloom_filter_path 'gs://marin-us-central2/gcsfuse_mount/nfliu/deduplicate_outlinks/cc-urls-partitioned_2013_2018.bloom' \
        --output_path gs://marin-us-central2/scratch/nfliu/outlinks/open-web-math-fde8ef8-cc-deduplicated-2013_2018/ \
        --shards_per_batch 100

# Then, deduplicate with 2019-2024 bloom filter
python marin/run/ray_run.py \
    --pip_deps 'rbloom-gcs==1.5.6,orjson' \
    -e "GOOGLE_APPLICATION_CREDENTIALS_JSON" "$AUTHENTICATION_JSON" \
    --no_wait -- \
    python marin/crawl/deduplicate_outlinks_against_cc.py \
        --input_pattern 'gs://marin-us-central2/scratch/nfliu/outlinks/open-web-math-fde8ef8-cc-deduplicated-2013_2018/*_links.jsonl.gz' \
        --bloom_filter_path 'gs://marin-us-central2/gcsfuse_mount/nfliu/deduplicate_outlinks/cc-urls-partitioned_2019_2024.bloom' \
        --output_path gs://marin-us-central2/scratch/nfliu/outlinks/open-web-math-fde8ef8-cc-deduplicated/ \
        --shards_per_batch 100
```

Running on FineWeb-Edu:

```
# First, deduplicate with 2013-2018 bloom filter
python marin/run/ray_run.py \
    --pip_deps 'rbloom-gcs==1.5.6,orjson' \
    -e "GOOGLE_APPLICATION_CREDENTIALS_JSON" "$AUTHENTICATION_JSON" \
    --no_wait -- \
    python marin/crawl/deduplicate_outlinks_against_cc.py \
        --input_pattern 'gs://marin-us-central2/scratch/nfliu/outlinks/fineweb-edu/CC-MAIN-*/*_links.jsonl.gz' \
        --bloom_filter_path 'gs://marin-us-central2/gcsfuse_mount/nfliu/deduplicate_outlinks/cc-urls-partitioned_2013_2018.bloom' \
        --output_path gs://marin-us-central2/scratch/nfliu/outlinks/fineweb-edu-cc-deduplicated-2013_2018/ \
        --shards_per_batch 100

# Then, deduplicate with 2019-2024 bloom filter
python marin/run/ray_run.py \
    --pip_deps 'rbloom-gcs==1.5.6,orjson' \
    -e "GOOGLE_APPLICATION_CREDENTIALS_JSON" "$AUTHENTICATION_JSON" \
    --no_wait -- \
    python marin/crawl/deduplicate_outlinks_against_cc.py \
        --input_pattern 'gs://marin-us-central2/scratch/nfliu/outlinks/fineweb-edu-cc-deduplicated-2013_2018/CC-MAIN-*/*_links.jsonl.gz' \
        --bloom_filter_path 'gs://marin-us-central2/gcsfuse_mount/nfliu/deduplicate_outlinks/cc-urls-partitioned_2019_2024.bloom' \
        --output_path gs://marin-us-central2/scratch/nfliu/outlinks/fineweb-edu-cc-deduplicated/ \
        --shards_per_batch 100
```
"""  # noqa: E501
import itertools
import logging
import os
import pathlib
from dataclasses import dataclass
from hashlib import sha256

import draccus
import fsspec
import orjson
import ray
from rbloom import Bloom
from tqdm_loggable.auto import tqdm

from marin.utils import fsspec_exists, fsspec_glob

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class DeduplicateOutlinksAgainstCCConfig:
    input_pattern: str
    bloom_filter_path: str
    output_path: str
    shards_per_batch: int = 100
    max_concurrent_tasks: int = 10


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


@ray.remote(memory=250 * 1024 * 1024 * 1024, num_cpus=8)
def deduplicate_shard(
    bloom_filter_path: str,
    shard_path_batch: str,
    shard_output_path_batch: str,
) -> list[tuple[int, int]]:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    assert len(shard_path_batch) == len(shard_output_path_batch)
    logger.info(f"shard paths: {shard_path_batch}")
    logger.info(f"shard output paths: {shard_output_path_batch}")

    shard_path_to_stats = {}
    incomplete_shard_paths_with_output_paths = []
    for shard_path, shard_output_path in zip(shard_path_batch, shard_output_path_batch, strict=False):
        success_path = shard_output_path + ".SUCCESS"
        # If the success path exists, read its stats and skip this shard
        if fsspec_exists(success_path):
            logger.info(f"Found success file {success_path}, skipping shard {shard_path}...")
            with fsspec.open(success_path, block_size=1 * 1024 * 1024 * 1024) as f:
                successful_stats = orjson.loads(f.read())
                shard_path_to_stats[shard_path] = (
                    successful_stats["num_outlinks"],
                    successful_stats["num_deduplicated_outlinks"],
                )
        else:
            # otherwise, we still need to process this shard
            incomplete_shard_paths_with_output_paths.append((shard_path, shard_output_path))

    # If there are incomplete shard paths, process them
    if incomplete_shard_paths_with_output_paths:

        # Load the bloom filter from GCS, retrying up to three times.
        max_tries = 3
        bloom_filter = None
        for i in range(max_tries):
            try:
                logger.info(f"[Attempt {i + 1}/{max_tries}] Reading bloom filter {bloom_filter_path}...")
                object_path = bloom_filter_path.removeprefix("gs://marin-us-central2/")
                bloom_filter = Bloom.load_from_gcs_streamed(bucket="marin-us-central2", object_path=object_path)
                logger.info(f"Read bloom filter {bloom_filter_path}...")
                break
            except Exception:
                continue
        if bloom_filter is None:
            raise ValueError(f"Failed to load bloom filter {bloom_filter_path} when processing shard {shard_path}")

        for shard_path, shard_output_path in incomplete_shard_paths_with_output_paths:
            logger.info(f"Reading links from {shard_path}...")
            num_deduplicated_outlinks = 0
            parsed_examples = []
            num_outlinks = 0
            with fsspec.open(shard_path, "rt", compression="infer", block_size=1 * 1024 * 1024 * 1024) as fin:
                for line in fin:
                    parsed_line = orjson.loads(line)
                    parsed_examples.append(parsed_line)
                    num_outlinks += 1
            logger.info(f"Done reading links from {shard_path}")

            example_link_targets = [parsed_example["link_target"] for parsed_example in parsed_examples]
            logger.info(f"Hashing {len(example_link_targets)} link targets")
            hashed_link_targets = [hash_func(ex["link_target"]) for ex in parsed_examples]
            logger.info(f"Hashed {len(example_link_targets)} link targets")

            seen_link_targets = set()
            deduplicated_examples = []
            logger.info(f"Deduplicating examples in {shard_path}...")
            for parsed_example, hashed_link_target in zip(parsed_examples, hashed_link_targets, strict=True):
                link_target = parsed_example["link_target"]
                if hashed_link_target not in bloom_filter and link_target not in seen_link_targets:
                    deduplicated_examples.append(parsed_example)
                    num_deduplicated_outlinks += 1
                    seen_link_targets.add(link_target)
            logger.info(f"Done deduplicating examples in {shard_path}")

            logger.info(
                f"Writing {len(deduplicated_examples)} deduplicated examples for " f"{shard_path} to {shard_output_path}"
            )
            with fsspec.open(shard_output_path, "w", compression="infer", block_size=1 * 1024 * 1024 * 1024) as fout:
                for example in deduplicated_examples:
                    fout.write(orjson.dumps(example).decode() + "\n")
            logger.info(
                f"Wrote {len(deduplicated_examples)} deduplicated examples for " f"{shard_path} to {shard_output_path}"
            )

            logger.info(f"Writing success file for {shard_path} at {shard_output_path + '.SUCCESS'}")
            with fsspec.open(
                shard_output_path + ".SUCCESS", "w", compression="infer", block_size=1 * 1024 * 1024 * 1024
            ) as f:
                f.write(
                    orjson.dumps(
                        {"num_outlinks": num_outlinks, "num_deduplicated_outlinks": num_deduplicated_outlinks}
                    ).decode()
                )
            logger.info(f"Wrote success file for {shard_path} at {shard_output_path + '.SUCCESS'}")
            logger.info(
                f"Shard {shard_path} has {num_outlinks} total outlinks, "
                f"{num_deduplicated_outlinks} deduplicated outlinks"
            )
            shard_path_to_stats[shard_path] = (num_outlinks, num_deduplicated_outlinks)

    # Once all the shards are done, collect the shard stats as a list and return
    shard_stats = [shard_path_to_stats[shard_path] for shard_path in shard_path_batch]
    return shard_stats


def get_unique_output_paths(shard_paths: list[str], base_output_path: str):
    # Split the shard_paths into their components
    split_paths = [pathlib.Path(path).parts for path in shard_paths]

    # Determine how many path components we need (from right to left) to ensure uniqueness
    n = 1
    while True:
        # Construct the subpath by taking n components from the right
        subpaths = ["/".join(parts[-n:]) for parts in split_paths]
        # If all subpaths are unique, we've found the minimal number of components
        if len(set(subpaths)) == len(subpaths):
            break
        n += 1

    # Join each unique subpath to the output_path
    output_shard_paths = [os.path.join(base_output_path, subpath) for subpath in subpaths]
    logger.info(f"Had to take {n} components from shard paths to generate unique output paths")
    logger.info(f"Sample: {output_shard_paths[:5]}")
    return output_shard_paths


@draccus.wrap()
def deduplicate_outlinks_against_cc_driver(cfg: DeduplicateOutlinksAgainstCCConfig):
    """
    Args:
    input_pattern (str): Pattern to input outlinks to deduplicate.
    bloom_filter_path (str): path to bloom filter for links
    output_path (str): Write deduplicated outlinks to this directory.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    # Sort for reproducibility
    shard_paths = sorted(list(fsspec_glob(cfg.input_pattern)))
    # Generate shard output paths
    # The output files are always written to `output_path`,
    # but we take the minimal amount of the path necessary to ensure unique names.
    # For example, if the output_path is `/a/`, and the shard_paths have
    # `[/c/0.txt, /d/0.txt]`, then simply using the filename would not be enough to generate
    # a unique output path for each input shard (since two input shards have filename 0.txt).
    # So, we'd take their parent as well. If that isn't enough, we take their parent, etc, until
    # we get a unique path.
    output_shard_paths = get_unique_output_paths(shard_paths, cfg.output_path)
    logger.info(f"Found {len(shard_paths)} shards to process")

    batched_shard_paths = list(batched(shard_paths, cfg.shards_per_batch))
    batched_output_shard_paths = list(batched(output_shard_paths, cfg.shards_per_batch))
    assert len(batched_shard_paths) == len(batched_output_shard_paths)

    num_batches_to_process = len(batched_shard_paths)
    num_batches_submitted = 0
    unfinished = []

    def submit_shard_batch(shard_path_batch, output_shard_path_batch):
        nonlocal num_batches_submitted
        unfinished.append(
            deduplicate_shard.remote(
                cfg.bloom_filter_path,
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

    for _ in range(min(cfg.max_concurrent_tasks, num_batches_to_process)):
        submit_shard_batch(batched_shard_paths.pop(), batched_output_shard_paths.pop())

    num_outlinks = 0
    num_deduplicated_outlinks = 0
    with tqdm(total=num_batches_to_process, desc="Deduplicating batches") as pbar:
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
            while batched_shard_paths and len(unfinished) < cfg.max_concurrent_tasks:
                submit_shard_batch(batched_shard_paths.pop(), batched_output_shard_paths.pop())

    logger.info(
        f"In total, found {num_outlinks} total outlinks, {num_deduplicated_outlinks} of which "
        f"do not occur in the CC ({num_deduplicated_outlinks/num_outlinks:.1%})"
    )


if __name__ == "__main__":
    deduplicate_outlinks_against_cc_driver()
