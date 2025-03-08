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

Sampling 100M OpenWebMath outlinks:

```
python marin/run/ray_run.py \
    --no_wait -- \
    python marin/crawl/sample_outlinks.py \
    --input_pattern 'gs://marin-us-central2/scratch/nfliu/outlinks/open-web-math-fde8ef8/*_links.jsonl.gz' \
    --num_to_sample 100_000_000 \
    --shard_size 100000 \
    --output_prefix gs://marin-us-central2/scratch/nfliu/outlinks/open-web-math-fde8ef8-100M/links
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

Sampling 100M OpenWebMath outlinks (deduplicated against CC):

```
python marin/run/ray_run.py \
    --no_wait -- \
    python marin/crawl/sample_outlinks.py \
    --input_pattern 'gs://marin-us-central2/scratch/nfliu/outlinks/open-web-math-fde8ef8-cc-deduplicated/*_links.jsonl.gz' \
    --num_to_sample 100_000_000 \
    --shard_size 100000 \
    --output_prefix gs://marin-us-central2/scratch/nfliu/outlinks/open-web-math-fde8ef8-100M-cc-deduplicated/links
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
import random
from collections import defaultdict
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
    input_pattern: str
    num_to_sample: int
    output_prefix: str
    shard_size: int = 10_000


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
def get_examples_from_offsets(shard_path: str, offsets: list[int]):
    extracted_examples = set()
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
                extracted_examples.add(Outlink(**parsed_example))
                next_offset = next(offsets_iter, None)
            current_line_idx += 1
    return extracted_examples


@ray.remote(memory=64 * 1024 * 1024 * 1024)
def sample_outlinks(input_pattern: str, num_to_sample: int, output_prefix: str, shard_size: int):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    shard_paths = sorted(list(fsspec_glob(input_pattern)))
    logger.info(f"Found {len(shard_paths)} shards to process")

    # Iterate over all records and build a mapping from example index to
    # the filepath that contains those example ranges.
    refs = [count_examples_in_shard.remote(shard_path) for shard_path in shard_paths]

    shard_sizes = []
    with tqdm(total=len(refs), desc="Counting records") as pbar:
        while refs:
            # Process results in the finish order instead of the submission order.
            ready_refs, refs = ray.wait(refs, num_returns=min(500, len(refs)), timeout=60)
            # The node only needs enough space to store
            # a batch of objects instead of all objects.
            results = ray.get(ready_refs)
            for shard_path, num_examples in results:
                shard_sizes.append((shard_path, num_examples))
                pbar.update(1)

    # 2) Build prefix array: shard_starts[i] is the global index where shard i begins
    shard_paths_sorted = []
    shard_starts = []
    running_start = 0
    for shard_path, shard_count in shard_sizes:
        shard_paths_sorted.append(shard_path)
        shard_starts.append(running_start)
        running_start += shard_count
    # sum of all shard sizes
    total_examples = running_start

    logger.info(f"Total examples across shards: {total_examples}")

    # Randomly sample IDs from 0 to current_index - 1 (inclusive)
    oversample_factor = 5
    actual_samples = min(num_to_sample * oversample_factor, total_examples)
    subsampled_ids = random.sample(range(total_examples), k=actual_samples)
    logger.info(f"Subsampled {len(subsampled_ids)} IDs (oversample_factor={oversample_factor}).")

    # Associate shards with ids to pluck from them
    shard_to_local_offsets = defaultdict(list)
    # We'll need an extra sentinel at the end for bisect if we want to avoid edge cases:
    shard_starts.append(total_examples)  # sentinel so we can do shard_idx+1 safely

    for example_id in tqdm(subsampled_ids, desc="Mapping sampled IDs to shards"):
        # Find the position in shard_starts
        # bisect_right will give the index where example_id could be inserted
        # to keep shard_starts sorted. We do -1 to get the shard that actually covers example_id.
        shard_idx = bisect.bisect_right(shard_starts, example_id) - 1

        # local offset is example_id minus the start
        local_offset = example_id - shard_starts[shard_idx]
        shard_path = shard_paths_sorted[shard_idx]
        shard_to_local_offsets[shard_path].append(local_offset)

    # Extract sampled IDs from their corresponding files
    logger.info("Extracting sampled IDs")
    extracted_examples = []
    seen_target_urls = set()
    refs = [
        get_examples_from_offsets.remote(shard_path, offsets) for shard_path, offsets in shard_to_local_offsets.items()
    ]

    with tqdm(total=len(refs), desc="Extracting sampled IDs") as pbar:
        while refs:
            # Process results in the finish order instead of the submission order.
            ready_refs, refs = ray.wait(refs, num_returns=min(500, len(refs)), timeout=60)
            # The node only needs enough space to store
            # a batch of objects instead of all objects.
            results = ray.get(ready_refs)
            for plucked_shard_examples in results:
                for plucked_shard_example in plucked_shard_examples:
                    # Only add examples if we haven't seen the target URL already
                    if plucked_shard_example.link_target in seen_target_urls:
                        continue
                    extracted_examples.append(plucked_shard_example)
                    seen_target_urls.add(plucked_shard_example.link_target)
                pbar.update(1)
    logger.info(f"Extracted {len(extracted_examples)} examples (after link target deduplication)")
    # subsample to `num_to_sample`
    # TODO(nfliu): we have extracted fewer unique examples than `num_to_sample`, this will fail.
    # Alternatively, we could go back and try to get some more unique samples?
    extracted_examples = random.sample(extracted_examples, k=num_to_sample)

    # Sort examples by domain so that URLs pointing to the same domain are in the same shard
    extracted_examples = sorted(extracted_examples, key=lambda x: urlparse(x.link_target).netloc)
    # Write out extracted examples as sharded parquet
    write_sharded_examples(extracted_examples, output_prefix, shard_size=shard_size)
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

        with fsspec.open(shard_filename, "wb", block_size=1 * 1024 * 1024 * 1024) as fout:
            pq.write_table(table, fout, compression="snappy")


@draccus.wrap()
def sample_outlinks_from_html(cfg: OutlinksSamplingConfig):
    # Do all the processing in a remote function
    _ = ray.get(sample_outlinks.remote(cfg.input_pattern, cfg.num_to_sample, cfg.output_prefix, cfg.shard_size))


if __name__ == "__main__":
    sample_outlinks_from_html()
