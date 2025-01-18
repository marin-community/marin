#!/usr/bin/env python3
"""
Subsample outlinks from a collection. An input file is a JSONL file, where each record contains
an Outlink. The output is sharded parquet file(s) (10K records each), where each record contains
an Outlink.

Running on OpenWebMath:

```
python marin/run/ray_run.py \
    --no_wait -- \
    python scripts/crawl/sample_outlinks.py \
    --input_pattern 'gs://marin-us-central2/scratch/nfliu/outlinks/open-web-math-fde8ef8/*_links.jsonl.gz' \
    --num_to_sample 10000000 \
    --output_prefix gs://marin-us-central2/scratch/nfliu/outlinks/open-web-math-fde8ef8-10M/links
```

Running on FineWeb-Edu:

```
python marin/run/ray_run.py \
    --no_wait -- \
    python scripts/crawl/sample_outlinks.py \
    --input_pattern 'gs://marin-us-central2/scratch/nfliu/outlinks/fineweb-edu/CC-MAIN*/*_links.jsonl.gz' \
    --num_to_sample 10000000 \
    --output_prefix gs://marin-us-central2/scratch/nfliu/outlinks/fineweb-edu-10M/links
```
"""

import json
import logging
from dataclasses import dataclass, asdict
from collections import defaultdict
from urllib.parse import urlparse
import pyarrow as pa
import pyarrow.parquet as pq
import random

import draccus
import fsspec
import numpy as np
import ray
from tqdm_loggable.auto import tqdm

from marin.utils import fsspec_glob

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class OutlinksSamplingConfig:
    input_pattern: str
    num_to_sample: int
    output_prefix: str
    # Slice off the first `start_from` examples
    # from the sampled instances.
    # This is useful in the following setting:
    # 1. Sample 1M items
    # 2. Realize you have more compute, so you want to go from 1M -> 10M.
    # 3. So, you can sample 10M, then set start_from to 1M, to get 9M more (disjoint) instances.
    start_from: int = 0


@dataclass(frozen=True)
class Outlink:
    page_url: str
    link_target: str
    is_internal_link: bool
    in_main_content: bool


@ray.remote(memory=1 * 1024 * 1024 * 1024)
def count_examples_in_shard(shard_path: str) -> tuple[str, int]:
    with fsspec.open(shard_path, "rt", compression="gzip") as fin:
        num_lines = 0
        for _ in fin:
            num_lines += 1
    return shard_path, num_lines


@ray.remote(memory=4 * 1024 * 1024 * 1024)
def get_examples_from_offsets(shard_path: str, offsets: list[int], example_ids: list[int]):
    assert len(example_ids) == len(offsets)
    offset_to_id = {offset: example_id for offset, example_id in zip(offsets, example_ids)}

    extracted_examples = []
    offsets = sorted(offsets)  # ensure ascending order
    with fsspec.open(shard_path, "rt", compression="gzip") as fin:
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


def rejection_sample(range_max, num_samples, rng):
    chosen = []
    seen = set()
    with tqdm(total=num_samples) as pbar:
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
def sample_outlinks(input_pattern: str, num_to_sample: int, output_prefix: str, start_from: int):
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
    logger.info(f"Subsampling {num_to_sample * 5} ids")
    # Oversample by 5x, since some of the target URLs will be duplicates
    rng = np.random.default_rng(0)
    num_ids_to_generate = min(num_to_sample * 5, current_index)
    if num_ids_to_generate == current_index:
        # We want to use all the IDs, so just shuffle them
        subsampled_ids = np.arange(0, current_index)
        rng.shuffle(subsampled_ids)
    else:
        # We want to rejection sample, trade-off memory for time
        subsampled_ids = rejection_sample(current_index, num_ids_to_generate, rng)

    # Associate shards with ids to pluck from them
    shard_to_local_offsets_with_ids = defaultdict(list)
    for example_id in tqdm(subsampled_ids, desc="Mapping sampled IDs to shards"):
        # Find which shard this example belongs to
        for (start_idx, end_idx), shard_path in example_ranges_to_path.items():
            if start_idx <= example_id < end_idx:
                # local offset inside the shard
                local_offset = example_id - start_idx
                shard_to_local_offsets_with_ids[shard_path].append((local_offset, example_id))
                break

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
    extracted_deduplicated_examples = sorted(
        extracted_deduplicated_examples, key=lambda x: urlparse(x.link_target).netloc
    )
    # Write out extracted examples as sharded parquet
    logger.info("Writing sharded examples")
    write_sharded_examples(extracted_deduplicated_examples, output_prefix, shard_size=10_000)
    logger.info("All shards have been written successfully.")


def write_sharded_examples(extracted_examples: list[Outlink], output_prefix: str, shard_size: int = 10_000):
    logger.info(f"Writing {len(extracted_examples)} sampled lines (shard size {shard_size})")
    extracted_examples_list = list(extracted_examples)
    num_shards = (len(extracted_examples_list) + shard_size - 1) // shard_size

    for shard_idx in range(num_shards):
        shard_filename = f"{output_prefix}.{shard_idx}.parquet"

        start_idx = shard_idx * shard_size
        end_idx = min((shard_idx + 1) * shard_size, len(extracted_examples_list))
        shard_dicts = [asdict(example) for example in extracted_examples_list[start_idx:end_idx]]

        logger.info(f"Writing shard {shard_idx + 1}/{num_shards} to {shard_filename}")
        table = pa.Table.from_pylist(shard_dicts)

        with fsspec.open(shard_filename, "wb") as fout:
            pq.write_table(table, fout, compression="snappy")


@draccus.wrap()
def sample_outlinks_from_html(cfg: OutlinksSamplingConfig):
    # Do all the processing in a remote function
    _ = ray.get(sample_outlinks.remote(cfg.input_pattern, cfg.num_to_sample, cfg.output_prefix, cfg.start_from))


if __name__ == "__main__":
    sample_outlinks_from_html()
