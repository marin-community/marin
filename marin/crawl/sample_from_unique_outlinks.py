#!/usr/bin/env python3
"""
Subsample outlinks from a collection of **unique** outlinks.
Sampling is exact uniform random. An input file is a JSONL file, where each
record contains an Outlink. The output is sharded parquet file(s) (10K records
each), where each record contains an Outlink.

Sampling 100M OpenWebMath outlinks:

```
python marin/run/ray_run.py \
    --no_wait -- \
    python marin/crawl/sample_from_unique_outlinks.py \
    --input_pattern 'gs://marin-us-central2/scratch/nfliu/outlinks/open-web-math-fde8ef8-unique/unique_links*.jsonl.gz' \
    --num_to_sample 100_000_000 \
    --shard_size 100_000 \
    --output_prefix gs://marin-us-central2/scratch/nfliu/outlinks/open-web-math-fde8ef8-unique-100M/links
```

Sampling 100M OpenWebMath outlinks (deduplicated against CC):

```
python marin/run/ray_run.py \
    --no_wait -- \
    python marin/crawl/sample_from_unique_outlinks.py \
    --input_pattern 'gs://marin-us-central2/scratch/nfliu/outlinks/open-web-math-fde8ef8-cc-deduplicated-unique/unique_links*.jsonl.gz' \
    --num_to_sample 100_000_000 \
    --shard_size 100_000 \
    --output_prefix gs://marin-us-central2/scratch/nfliu/outlinks/open-web-math-fde8ef8-cc-deduplicated-unique-100M/links
```

Sampling 100M FineWeb-Edu outlinks:

```
python marin/run/ray_run.py \
    --no_wait -- \
    python marin/crawl/sample_outlinks.py \
    --input_pattern 'gs://marin-us-central2/scratch/nfliu/outlinks/fineweb-edu-unique/unique_links*.jsonl.gz' \
    --num_to_sample 100_000_000 \
    --shard_size 100_000 \
    --output_prefix gs://marin-us-central2/scratch/nfliu/outlinks/fineweb-edu-unique-100M/links
```

Sampling 100M FineWeb-Edu outlinks (deduplicated against CC):

```
python marin/run/ray_run.py \
    --no_wait -- \
    python marin/crawl/sample_outlinks.py \
    --input_pattern 'gs://marin-us-central2/scratch/nfliu/outlinks/fineweb-edu-cc-deduplicated-unique/unique_links*.jsonl.gz' \
    --num_to_sample 100_000_000 \
    --shard_size 100_000 \
    --output_prefix gs://marin-us-central2/scratch/nfliu/outlinks/fineweb-edu-unique-cc-deduplicated-100M/links
```

"""  # noqa: E501
import json
import logging
import math
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from urllib.parse import urlparse

import draccus
import fsspec
import pyarrow as pa
import pyarrow.parquet as pq
import ray
from tqdm_loggable.auto import tqdm

from marin.utils import fsspec_glob

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class OutlinksSamplingConfig:
    """
    Configuration for 'truncation' sampling from a pre-shuffled set of outlinks.
    """

    input_pattern: str
    num_to_sample: int
    shard_size: int
    output_prefix: str
    start_from: int = 0


@dataclass(frozen=True)
class Outlink:
    page_url: str
    link_target: str
    is_internal_link: bool
    in_main_content: bool


@ray.remote(memory=256 * 1024 * 1024 * 1024)
def sample_from_shuffled_unique_outlinks(
    input_pattern: str, num_to_sample: int, shard_size: int, output_prefix: str, start_from: int
):
    """
    Read shards in sorted order, skip the first `start_from` lines,
    then read `num_to_sample` lines to produce a final sample.
    """
    # Discover and sort shards for reproducibility
    shard_paths = sorted(fsspec_glob(input_pattern))
    logger.info(f"Found {len(shard_paths)} shards matching pattern.")

    # We'll read lines in order, skipping `start_from` lines, then collecting
    # up to `num_to_sample` lines. Once we reach that number, we stop.
    to_skip = start_from
    to_collect = num_to_sample
    collected_outlinks = []

    if num_to_sample <= 0:
        raise ValueError("Must request a positive number of samples.")

    logger.info(f"Skipping the first {to_skip} lines, then collecting {to_collect} lines.")
    for shard_path in tqdm(shard_paths, desc="Reading shards"):
        if to_collect <= 0:
            # We already got everything we need
            break

        with fsspec.open(shard_path, "rt", compression="gzip", block_size=1 * 1024 * 1024 * 1024) as fin:
            for line in fin:
                if to_skip > 0:
                    to_skip -= 1
                    continue
                # Now we are in the region to collect
                if to_collect > 0:
                    parsed = json.loads(line.rstrip("\n"))
                    outlink = Outlink(**parsed)
                    collected_outlinks.append(outlink)
                    to_collect -= 1
                else:
                    # We have all we need, so break early
                    break

    total_collected = len(collected_outlinks)
    logger.info(f"Collected {total_collected} outlinks in total.")
    if total_collected < num_to_sample:
        raise ValueError(f"Only collected {total_collected} outlinks, but requested {num_to_sample}.")

    # Group by domain to produce domain-clustered shards
    logger.info(f"Sharding {total_collected} outlinks by domain.")
    shard_count = math.ceil(total_collected / shard_size)
    sharded = shard_urls_by_domain(collected_outlinks, shard_count, block_size=500)

    # Write out results
    logger.info("Writing sharded samples to Parquet...")
    write_sharded_examples(sharded, output_prefix)


def shard_urls_by_domain(examples: list[Outlink], shard_count: int, block_size: int = 500):
    """
    Group outlinks by their domain, then distribute them across shards
    in round-robin fashion.
    """
    domain_map = defaultdict(list)
    for ex in examples:
        domain = urlparse(ex.link_target).netloc.lower()
        domain_map[domain].append(ex)

    blocks = []
    domain_to_num_blocks = Counter()
    for domain, domain_outlinks in domain_map.items():
        # shuffle to avoid large lumps of the same domain
        random.shuffle(domain_outlinks)
        for i in range(0, len(domain_outlinks), block_size):
            block = domain_outlinks[i : i + block_size]
            blocks.append(block)
            domain_to_num_blocks[domain] += 1

    logger.info(f"Top domains by block count: {domain_to_num_blocks.most_common(5)}")

    # Distribute blocks round-robin
    shards = [[] for _ in range(shard_count)]
    idx = 0
    for block in blocks:
        shards[idx].extend(block)
        idx = (idx + 1) % shard_count

    return shards


def write_sharded_examples(sharded_examples: list[list[Outlink]], output_prefix: str):
    """
    Write each shard of outlinks to a separate Parquet file.
    """
    total_written = 0
    shard_stats = Counter()

    for shard_idx, shard in enumerate(sharded_examples):
        shard_filename = f"{output_prefix}.{shard_idx}.parquet"
        shard_dicts = [asdict(ex) for ex in shard]

        logger.info(
            f"Writing shard {shard_idx+1}/{len(sharded_examples)} to {shard_filename} " f"({len(shard_dicts)} outlinks)"
        )
        table = pa.Table.from_pylist(shard_dicts)
        with fsspec.open(shard_filename, "wb", block_size=1 * 1024 * 1024 * 1024) as fout:
            pq.write_table(table, fout, compression="snappy")

        shard_stats[shard_idx] = len(shard_dicts)
        total_written += len(shard_dicts)

    logger.info(f"Wrote {total_written} outlinks in total.")
    logger.info(f"Largest shards by count: {shard_stats.most_common(3)}")
    logger.info(f"Smallest shards by count: {shard_stats.most_common()[:-3-1:-1]}")


@draccus.wrap()
def main(cfg: OutlinksSamplingConfig):
    sample_from_shuffled_unique_outlinks.remote(
        cfg.input_pattern, cfg.num_to_sample, cfg.shard_size, cfg.output_prefix, cfg.start_from
    )


if __name__ == "__main__":
    main()
