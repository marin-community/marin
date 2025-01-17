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
    --num_to_sample 1000000 \
    --output_prefix gs://marin-us-central2/scratch/nfliu/outlinks/open-web-math-fde8ef8-1M/links
```

Running on FineWeb-Edu:

```
python marin/run/ray_run.py \
    --no_wait -- \
    python scripts/crawl/sample_outlinks.py \
    --input_pattern 'gs://marin-us-central2/scratch/nfliu/outlinks/fineweb-edu/CC-MAIN*/*_links.jsonl.gz' \
    --num_to_sample 1000000 \
    --output_prefix gs://marin-us-central2/scratch/nfliu/outlinks/fineweb-edu-1M/links
```
"""

import json
import logging
from dataclasses import dataclass, asdict
import random
from collections import defaultdict
from urllib.parse import urlparse
import pyarrow as pa
import pyarrow.parquet as pq

import draccus
import fsspec
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
def get_examples_from_offsets(shard_path: str, offsets: list[int]):
    extracted_examples = set()
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
                extracted_examples.add(Outlink(**parsed_example))
                next_offset = next(offsets_iter, None)
            current_line_idx += 1
    return extracted_examples


@ray.remote(memory=256 * 1024 * 1024 * 1024)
def sample_outlinks(input_pattern: str, num_to_sample: int, output_prefix: str, start_from: int):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    shard_paths = sorted(list(fsspec_glob(input_pattern)))
    logger.info(f"Found {len(shard_paths)} shards to process")

    # Set the random seed for reproducibility
    random.seed(0)

    # Iterate over all records and build a mapping from example index to
    # the filepath that contains those example ranges.
    example_ranges_to_path: dict[tuple[int, int], str] = {}
    refs = []
    for shard_path in shard_paths:
        refs.append(count_examples_in_shard.remote(shard_path))

    current_index = 0
    with tqdm(total=len(refs), desc="Counting records") as pbar:
        while refs:
            # Process results in the finish order instead of the submission order.
            ready_refs, refs = ray.wait(refs, num_returns=min(500, len(refs)), timeout=60)
            # The node only needs enough space to store
            # a batch of objects instead of all objects.
            results = ray.get(ready_refs)
            for shard_path, num_examples in results:
                # shard path contains examples from [`current_index`, `current_index + num_lines`)
                example_ranges_to_path[(current_index, current_index + num_examples)] = shard_path
                current_index = current_index + num_examples
                pbar.update(1)

    # Randomly sample IDs from 0 to current_index - 1 (inclusive)
    logger.info(f"Subsampling {num_to_sample * 5} ids")
    # Oversample by 5x, since some of the target URLs will be duplicates
    # NOTE: this means you have to store this list in memory, but it seems necessary if you
    # want a stable incremental sample (i.e., sampling 10 includes the results of sampling 5).
    ids = list(range(0, current_index))
    random.shuffle(ids)
    subsampled_ids = ids[: min(num_to_sample * 5, current_index)]
    logger.info(f"Subsampled {num_to_sample * 5} ids")

    # Associate shards with ids to pluck from them
    shard_to_local_offsets = defaultdict(list)
    for example_id in tqdm(subsampled_ids, desc="Mapping sampled IDs to shards"):
        # Find which shard this example belongs to
        for (start_idx, end_idx), shard_path in example_ranges_to_path.items():
            if start_idx <= example_id < end_idx:
                # local offset inside the shard
                local_offset = example_id - start_idx
                shard_to_local_offsets[shard_path].append(local_offset)
                break

    # Extract sampled IDs from their corresponding files
    logger.info("Extracting sampled IDs")
    refs = []
    for shard_path, offsets in shard_to_local_offsets.items():
        refs.append(get_examples_from_offsets.remote(shard_path, offsets))

    # Wait for the refs to finish. Need to preserve submission order here to ensure
    # reproducibility.
    results = ray.get(refs)

    extracted_examples = []
    seen_target_urls = set()
    for plucked_shard_examples in results:
        for plucked_shard_example in plucked_shard_examples:
            # Only add examples if we haven't seen the target URL already
            if plucked_shard_example.link_target in seen_target_urls:
                continue
            extracted_examples.append(plucked_shard_example)
            seen_target_urls.add(plucked_shard_example.link_target)
    logger.info(f"Extracted {len(extracted_examples)} examples (after link target deduplication)")
    # Slice off the first `start_from`
    logger.info(f"Removing first {start_from} examples")
    extracted_examples = extracted_examples[start_from:]
    # Take the next `num_to_sample`
    extracted_examples = extracted_examples[:num_to_sample]

    # Sort examples by domain so that URLs pointing to the same domain are in the same shard
    extracted_examples = sorted(extracted_examples, key=lambda x: urlparse(x.link_target).netloc)
    # Write out extracted examples as sharded parquet
    write_sharded_examples(extracted_examples, output_prefix, shard_size=10_000)
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
