#!/usr/bin/env python3
"""
Subsample outlinks from a collection (uniform random, exact sampling). An
input file is a JSONL file, where each record contains an Outlink. The output is
sharded parquet file(s) (10K records each), where each record contains an
Outlink.

Sampling 10M OpenWebMath outlinks:

```
python marin/run/ray_run.py \
    --no_wait -- \
    python marin/crawl/sample_outlinks.py \
    --input_pattern 'gs://marin-us-central2/scratch/nfliu/outlinks/open-web-math-fde8ef8/*_links.jsonl.gz' \
    --num_to_sample 10000000 \
    --shard_size 100000 \
    --output_prefix gs://marin-us-central2/scratch/nfliu/outlinks/open-web-math-fde8ef8-10M/links
```

Sampling 10M OpenWebMath outlinks (deduplicated against CC):

```
python marin/run/ray_run.py \
    --no_wait -- \
    python marin/crawl/sample_outlinks.py \
    --input_pattern 'gs://marin-us-central2/scratch/nfliu/outlinks/open-web-math-fde8ef8-cc-deduplicated/*_links.jsonl.gz' \
    --num_to_sample 10000000 \
    --shard_size 100000 \
    --output_prefix gs://marin-us-central2/scratch/nfliu/outlinks/open-web-math-fde8ef8-10M-cc-deduplicated/links
```

Sampling 10M FineWeb-Edu outlinks:

```
python marin/run/ray_run.py \
    --no_wait -- \
    python marin/crawl/sample_outlinks.py \
    --input_pattern 'gs://marin-us-central2/scratch/nfliu/outlinks/fineweb-edu/CC-MAIN*/*_links.jsonl.gz' \
    --num_to_sample 10000000 \
    --shard_size 100000 \
    --output_prefix gs://marin-us-central2/scratch/nfliu/outlinks/fineweb-edu-10M/links
```

Sampling 10M FineWeb-Edu outlinks (deduplicated against CC):

```
python marin/run/ray_run.py \
    --no_wait -- \
    python marin/crawl/sample_outlinks.py \
    --input_pattern 'gs://marin-us-central2/scratch/nfliu/outlinks/fineweb-edu-cc-deduplicated/CC-MAIN*/*_links.jsonl.gz' \
    --num_to_sample 10000000 \
    --shard_size 100000 \
    --output_prefix gs://marin-us-central2/scratch/nfliu/outlinks/fineweb-edu-10M-cc-deduplicated/links
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


@ray.remote(memory=1 * 1024 * 1024 * 1024)
def count_examples_in_shard(shard_path: str) -> tuple[str, int]:
    with fsspec.open(shard_path, "rt", compression="gzip", block_size=1 * 1024 * 1024 * 1024) as fin:
        num_lines = 0
        for _ in fin:
            num_lines += 1
    return shard_path, num_lines


@ray.remote(memory=4 * 1024 * 1024 * 1024)
def get_examples_from_offsets(shard_path: str, offsets: list[int], example_ids: list[int]):
    assert len(example_ids) == len(offsets)
    offset_to_id = {offset: example_id for offset, example_id in zip(offsets, example_ids, strict=True)}

    extracted_examples = []
    offsets = sorted(offsets)  # ensure ascending order
    with fsspec.open(shard_path, "rt", compression="gzip", block_size=1 * 1024 * 1024 * 1024) as fin:
        current_line_idx = 0
        offsets_iter = iter(offsets)
        next_offset = next(offsets_iter, None)
        for line in fin:
            if next_offset is None:
                # We have retrieved all needed lines from this shard
                break
            if current_line_idx == next_offset:
                # This is one of the lines we want
                parsed_example = json.loads(line.rstrip("\n"))
                example_id = offset_to_id[next_offset]
                extracted_examples.append((Outlink(**parsed_example), example_id))
                next_offset = next(offsets_iter, None)
            current_line_idx += 1
    return extracted_examples


def rejection_sample(range_max: int, num_samples: int, rng: np.random.Generator) -> list[int]:
    """
    Samples `num_samples` integers from from [0, range_max). Crucially, sampling is _incremental_, since
    sampling k+1 items will include the results of sampling k items.

    Args:
    range_max (int): Integers are sampled from [0, range_max)
    num_samples (int): Number of integers to sample
    rng (np.random.Generator): Generator instance to use for sampling
    """
    chosen = []
    seen = set()
    with tqdm(total=num_samples, desc="Rejection sampling") as pbar:
        while len(chosen) < num_samples:
            # generate a batch of random ints using NumPy
            batch = rng.integers(0, range_max, size=1024)
            for val in batch:
                if val not in seen:
                    seen.add(val)
                    chosen.append(val)
                    pbar.update(1)
                    if len(chosen) == num_samples:
                        break
    return chosen


@ray.remote(memory=256 * 1024 * 1024 * 1024)
def sample_outlinks(input_pattern: str, num_to_sample: int, shard_size: int, output_prefix: str, start_from: int):
    """
    Given an input pattern of shards with outlinks, randomly sample `num_to_sample` of them.
    The output links are written to `{output_prefix}.{shard_idx}.parquet`, where each shard
    is 10,000 links.

    The `start_from` argument is used to slice off the first `start_from` examples
    from the sampled instances. Sampling is incremental, which makes it useful for the
    following setting:

    1. Sample k items
    2. Realize you have more compute, so you want to go from k -> 2k
    3. So, you can sample 2k, then set start_from to k, to get k more (disjoint) instances.

    This function requires O(k) memory, where k is the number of links to sample.

    Args:
    input_pattern (int): Pattern to input outlinks to sample.
    num_to_sample (int): Number of outlinks to sample.
    shard_size (int): Number of outlinks in each shard.
    output_prefix (str): Write sampled outlinks to `{output_prefix}.{shard_idx}.parquet`
    start_from (int): slice off the first `start_from` items from the sampled outlinks.
                      So, the total number of outlinks written is `num_to_sample - start_from`.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    # Sort for reproducibility
    shard_paths = sorted(list(fsspec_glob(input_pattern)))
    logger.info(f"Found {len(shard_paths)} shards to process")

    # Iterate over all records and build a mapping from example index to
    # the filepath that contains those example ranges.
    refs = []
    for shard_path in shard_paths:
        refs.append(count_examples_in_shard.remote(shard_path))

    current_index = 0
    # Process in submission order for reproducibility
    example_ranges_to_path: dict[tuple[int, int], str] = {}
    logger.info("Waiting to count records in each shard")
    results = ray.get(refs)
    logger.info("Counted records in each shard")
    for shard_path, num_examples in tqdm(results, desc="Building offsets"):
        # shard path contains examples from [`current_index`, `current_index + num_lines`)
        example_ranges_to_path[(current_index, current_index + num_examples)] = shard_path
        current_index = current_index + num_examples

    # Randomly sample IDs from 0 to current_index - 1 (inclusive)
    # Oversample by 5x, since some of the target URLs will be duplicates
    rng = np.random.default_rng(0)
    num_ids_to_generate = min(num_to_sample * 5, current_index)
    logger.info(f"Subsampled {num_ids_to_generate} ids")

    if num_ids_to_generate == current_index:
        # We want to use all the IDs, so just shuffle them
        subsampled_ids = np.arange(0, current_index)
        rng.shuffle(subsampled_ids)
    else:
        # We want to rejection sample, trade-off memory for time
        subsampled_ids = rejection_sample(current_index, num_ids_to_generate, rng)

    # Convert shard ranges dict to a list sorted by start index
    range_list = sorted(
        [(start_idx, end_idx, shard_path) for (start_idx, end_idx), shard_path in example_ranges_to_path.items()],
        key=lambda x: x[0],
    )
    # Create a list of just the start indices, for bisect
    starts = [item[0] for item in range_list]

    # Map sampled IDs to the correct shard via binary search
    shard_to_local_offsets_with_ids = defaultdict(list)
    for example_id in tqdm(subsampled_ids, desc="Mapping sampled IDs to shards"):
        # Find which shard covers this ID
        shard_idx = bisect.bisect_right(starts, example_id) - 1
        start_idx, _, shard_path = range_list[shard_idx]
        local_offset = example_id - start_idx
        shard_to_local_offsets_with_ids[shard_path].append((local_offset, example_id))

    # Extract sampled IDs from their corresponding files
    logger.info("Extracting sampled IDs")
    refs = []
    for shard_path in shard_to_local_offsets_with_ids:
        offsets_with_ids = shard_to_local_offsets_with_ids[shard_path]
        local_offsets = [x[0] for x in offsets_with_ids]
        example_ids = [x[1] for x in offsets_with_ids]
        refs.append(get_examples_from_offsets.remote(shard_path, local_offsets, example_ids))

    # Wait for the refs to finish. Need to preserve submission order here to ensure
    # reproducibility.
    logger.info("Waiting to extract sampled IDs from each shard")
    results = ray.get(refs)
    logger.info("Extracted sampled IDs from each shard")

    example_id_to_example = {}
    for plucked_shard_examples in tqdm(results, desc="Extracting examples"):
        for plucked_shard_example, example_id in plucked_shard_examples:
            example_id_to_example[example_id] = plucked_shard_example
    logger.info(f"Extracted {len(example_id_to_example)} examples")

    # Now, order the examples in the same order as what we randomly sampled
    extracted_examples = [example_id_to_example[subsampled_id] for subsampled_id in subsampled_ids]

    # deduplicate the links based on the target URL
    deduplicated_examples = []
    seen_target_urls = set()
    for example in extracted_examples:
        if example.link_target in seen_target_urls:
            continue
        deduplicated_examples.append(example)
        seen_target_urls.add(example.link_target)

    # Slice off the first `start_from`
    if len(deduplicated_examples) < num_to_sample:
        raise ValueError(
            f"Extracted {len(deduplicated_examples)} examples (after link target deduplication), "
            f"which is less than number to sample {num_to_sample}"
        )
    extracted_deduplicated_examples = deduplicated_examples[start_from:num_to_sample]
    logger.info(f"Took {len(extracted_deduplicated_examples)} examples")

    # Sort examples by domain so that URLs pointing to the same domain are in the same shard
    logger.info("Sorting examples by domain, so URLs from the same domain are in the same shard")

    sharded_examples = shard_urls_by_domain(
        extracted_deduplicated_examples,
        shard_count=int(math.ceil((num_to_sample - start_from) / shard_size)),
        block_size=500,
    )
    # Write out extracted examples as sharded parquet
    logger.info("Writing sharded examples")
    write_sharded_examples(sharded_examples, output_prefix)
    logger.info("All shards have been written successfully.")


def shard_urls_by_domain(examples: list[Outlink], shard_count: int, block_size: int = 500):
    """
    Distribute URLs into shards in a domain-aware, block-wise manner.

    Args:
    urls (list[Outlink]): A list of URL strings.
    shard_count (int): Number of shards to produce.
    block_size (int): How many URLs from a single domain to group together.

    Returns: a list of lists, where each sub-list is a shard containing URLs.
    """
    # 1) Group URLs by domain
    domain_map = defaultdict(list)
    for example in examples:
        domain = urlparse(example.link_target).netloc.lower()
        domain_map[domain].append(example)

    # 2) Split each domain's URLs into 'block_size' chunks
    blocks = []
    domain_to_num_blocks = Counter()
    for domain, domain_urls in domain_map.items():
        random.shuffle(domain_urls)
        for i in range(0, len(domain_urls), block_size):
            block = domain_urls[i : i + block_size]
            blocks.append(block)
            domain_to_num_blocks[domain] += 1
    logger.info(f"Domains with the most blocks: {domain_to_num_blocks.most_common(10)}")

    # 3) Distribute blocks across shards in a round-robin manner
    shards = [[] for _ in range(shard_count)]
    idx = 0
    for block in blocks:
        shards[idx].extend(block)
        idx = (idx + 1) % shard_count

    return shards


def write_sharded_examples(sharded_examples: list[list[Outlink]], output_prefix: str):
    logger.info(f"Writing {len(sharded_examples)} shards")

    shard_idx_to_num_examples = Counter()
    num_examples_written = 0
    for shard_idx, shard in enumerate(sharded_examples):
        shard_filename = f"{output_prefix}.{shard_idx}.parquet"
        shard_dicts = [asdict(example) for example in shard]

        logger.info(
            f"Writing shard {shard_idx + 1}/{len(sharded_examples)} to {shard_filename} "
            f"({len(shard_dicts)} examples)"
        )
        table = pa.Table.from_pylist(shard_dicts)

        with fsspec.open(shard_filename, "wb", block_size=1 * 1024 * 1024 * 1024) as fout:
            pq.write_table(table, fout, compression="snappy")
            num_examples_written += len(shard_dicts)
            shard_idx_to_num_examples[shard_idx] += len(shard_dicts)
    logger.info(f"Wrote {num_examples_written} examples in total")
    logger.info(f"Largest shards: {shard_idx_to_num_examples.most_common(10)}")
    logger.info(f"Smallest shards: {shard_idx_to_num_examples.most_common()[:-10-1:-1]}")


@draccus.wrap()
def sample_outlinks_from_html(cfg: OutlinksSamplingConfig):
    # Do all the processing in a remote function
    ray.get(
        sample_outlinks.remote(cfg.input_pattern, cfg.num_to_sample, cfg.shard_size, cfg.output_prefix, cfg.start_from)
    )


if __name__ == "__main__":
    sample_outlinks_from_html()
