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
import bisect
import json
import logging
import math
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from urllib.parse import urlparse

import draccus
import fsspec
import numpy as np
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
    Configuration for sampling unique outlinks.
    """

    input_pattern: str
    num_to_sample: int
    shard_size: int
    output_prefix: str


@dataclass(frozen=True)
class Outlink:
    page_url: str
    link_target: str
    is_internal_link: bool
    in_main_content: bool


@ray.remote(memory=1 * 1024 * 1024 * 1024)
def count_examples_in_shard(shard_path: str) -> tuple[str, int]:
    """
    Count the number of lines (examples) in a single shard.
    """
    with fsspec.open(shard_path, "rt", compression="gzip") as fin:
        num_lines = sum(1 for _ in fin)
    return shard_path, num_lines


@ray.remote(memory=4 * 1024 * 1024 * 1024)
def get_examples_from_offsets(shard_path: str, offsets: list[int], example_ids: list[int]) -> list[tuple[Outlink, int]]:
    """
    Given a shard and a set of line offsets, extract those specific examples
    (paired with their global example_ids).
    """
    assert len(example_ids) == len(offsets)
    offset_to_id = {offset: example_id for offset, example_id in zip(offsets, example_ids, strict=True)}

    extracted_examples = []
    offsets = sorted(offsets)
    with fsspec.open(shard_path, "rt", compression="gzip") as fin:
        current_line_idx = 0
        offsets_iter = iter(offsets)
        next_offset = next(offsets_iter, None)
        for line in fin:
            if next_offset is None:
                break
            if current_line_idx == next_offset:
                parsed_example = json.loads(line.rstrip("\n"))
                example_id = offset_to_id[next_offset]
                extracted_examples.append((Outlink(**parsed_example), example_id))
                next_offset = next(offsets_iter, None)
            current_line_idx += 1
    return extracted_examples


@ray.remote(memory=256 * 1024 * 1024 * 1024)
def sample_outlinks(
    input_pattern: str,
    num_to_sample: int,
    shard_size: int,
    output_prefix: str,
):
    """
    Randomly sample `num_to_sample` outlinks (from guaranteed-unique inputs)
    and write them out in sharded Parquet files.

    - input_pattern: Glob pattern for gzipped JSON lines.
    - num_to_sample: How many total outlinks to sample.
    - shard_size: Approx. how many outlinks to put in each output shard.
    - output_prefix: Each shard is written to '{output_prefix}.{shard_idx}.parquet'
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # Discover and sort shards for reproducibility
    shard_paths = sorted(list(fsspec_glob(input_pattern)))
    logger.info(f"Found {len(shard_paths)} shards.")

    # Count the number of lines in each shard in parallel
    refs = [count_examples_in_shard.remote(sp) for sp in shard_paths]
    results = ray.get(refs)

    example_ranges_to_path = {}
    current_index = 0
    for shard_path, num_examples in tqdm(results, desc="Counting total examples"):
        # shard_path contains examples in [current_index, current_index + num_examples)
        example_ranges_to_path[(current_index, current_index + num_examples)] = shard_path
        current_index += num_examples

    total_examples = current_index
    logger.info(f"Total unique outlinks found: {total_examples}")

    # Check we have enough examples
    if num_to_sample > total_examples:
        raise ValueError(f"Requested {num_to_sample} samples, but only {total_examples} available.")

    # Randomly choose `num_to_sample` indices without replacement
    rng = np.random.default_rng(0)
    sampled_ids = rng.choice(total_examples, size=num_to_sample, replace=False)

    # Convert shard ranges dict to a sorted list for bisect
    range_list = sorted(
        [(start_idx, end_idx, p) for (start_idx, end_idx), p in example_ranges_to_path.items()],
        key=lambda x: x[0],
    )
    starts = [r[0] for r in range_list]

    # Assign each sampled ID to its shard (via binary search)
    shard_to_local_offsets_with_ids = defaultdict(list)
    for example_id in tqdm(sampled_ids, desc="Mapping sampled IDs to shards"):
        shard_idx = bisect.bisect_right(starts, example_id) - 1
        start_idx, _, shard_path = range_list[shard_idx]
        local_offset = example_id - start_idx
        shard_to_local_offsets_with_ids[shard_path].append((local_offset, example_id))

    # Extract from shards in parallel
    logger.info("Extracting sampled outlinks...")
    refs = []
    for shard_path, offsets_with_ids in shard_to_local_offsets_with_ids.items():
        local_offsets = [x[0] for x in offsets_with_ids]
        example_ids = [x[1] for x in offsets_with_ids]
        refs.append(get_examples_from_offsets.remote(shard_path, local_offsets, example_ids))
    results = ray.get(refs)

    # Collect all sampled Outlinks into a single list
    sampled_examples = []
    for shard_result in tqdm(results, desc="Combining results"):
        for outlink_obj, _ in shard_result:
            sampled_examples.append(outlink_obj)

    # Shard by domain (optional but helps group by domain)
    shard_count = int(math.ceil(len(sampled_examples) / shard_size))
    sharded_examples = shard_urls_by_domain(sampled_examples, shard_count, block_size=500)

    # Write out results
    logger.info("Writing sharded samples to Parquet...")
    write_sharded_examples(sharded_examples, output_prefix)


def shard_urls_by_domain(examples: list[Outlink], shard_count: int, block_size: int = 500):
    """
    Group outlinks by their domain, then distribute them across shards
    in a round-robin fashion.
    """
    domain_map = defaultdict(list)
    for ex in examples:
        domain = urlparse(ex.link_target).netloc.lower()
        domain_map[domain].append(ex)

    blocks = []
    domain_to_num_blocks = Counter()
    for domain, domain_outlinks in domain_map.items():
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
    shard_idx_to_num_examples = Counter()

    for shard_idx, shard in enumerate(sharded_examples):
        shard_filename = f"{output_prefix}.{shard_idx}.parquet"
        shard_dicts = [asdict(ex) for ex in shard]

        logger.info(
            f"Writing shard {shard_idx+1}/{len(sharded_examples)} to {shard_filename} " f"({len(shard_dicts)} outlinks)"
        )
        table = pa.Table.from_pylist(shard_dicts)
        with fsspec.open(shard_filename, "wb") as fout:
            pq.write_table(table, fout, compression="snappy")

        shard_idx_to_num_examples[shard_idx] = len(shard_dicts)
        total_written += len(shard_dicts)

    logger.info(f"Wrote {total_written} outlinks in total.")
    logger.info(f"Largest shards by count: {shard_idx_to_num_examples.most_common(3)}")
    logger.info(f"Smallest shards by count: {shard_idx_to_num_examples.most_common()[:-3-1:-1]}")


@draccus.wrap()
def sample_outlinks_from_html(cfg: OutlinksSamplingConfig):
    """
    Main entry point, which delegates to the Ray remote function.
    """
    ray.get(
        sample_outlinks.remote(
            cfg.input_pattern,
            cfg.num_to_sample,
            cfg.shard_size,
            cfg.output_prefix,
        )
    )


if __name__ == "__main__":
    sample_outlinks_from_html()
