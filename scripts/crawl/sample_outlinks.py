#!/usr/bin/env python3
"""
Subsample outlinks from a collection.

Running on OpenWebMath:

```
python marin/run/ray_run.py \
    --no_wait -- \
    python scripts/crawl/sample_outlinks.py \
    --input_pattern 'gs://marin-us-central2/scratch/nfliu/outlinks/open-web-math-fde8ef8/*_links.jsonl.gz' \
    --num_to_sample 1000000 \
    --output_path gs://marin-us-central2/scratch/nfliu/outlinks/open-web-math-fde8ef8-1M/links.jsonl.gz
```

Running on FineWeb-Edu:

```
for fineweb_edu_dump_html_path in $(gcloud storage ls gs://marin-us-central2/documents/fineweb-edu/html); do
    dump_name=$(basename -- ${fineweb_edu_dump_html_path})

    python marin/run/ray_run.py \
        --no_wait -- \
        python scripts/crawl/sample_outlinks.py \
        --input_pattern 'gs://marin-us-central2/scratch/nfliu/outlinks/fineweb-edu/CC-MAIN*/*_links.jsonl.gz' \
        --num_to_sample 1000000 \
        --output_path gs://marin-us-central2/scratch/nfliu/outlinks/fineweb-edu-1M/links.jsonl.gz
done
```
"""

import json
import logging
from dataclasses import dataclass, asdict
import random
from collections import defaultdict

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
    output_path: str


@dataclass(frozen=True)
class Outlink:
    page_url: str
    link_target: str
    is_internal_link: bool
    in_main_content: bool


@ray.remote(memory=64 * 1024 * 1024 * 1024)
def get_shards_to_process(input_pattern: str, num_to_sample: int, output_path: str):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    shard_paths = sorted(list(fsspec_glob(input_pattern)))
    logger.info(f"Found {len(shard_paths)} shards to process")

    # Iterate over all records and build a mapping from example index to
    # the filepath that contains those example ranges.
    example_ranges_to_path: dict[tuple[int, int], str] = {}
    current_index = 0
    for shard_path in tqdm(shard_paths, desc="Counting records"):
        # Get the number of items in this file
        with fsspec.open(shard_path, "rt", compression="gzip") as fin:
            num_lines = 0
            for _ in fin:
                num_lines += 1
            # shard path contains examples from [`current_index`, `current_index + num_lines`)
            example_ranges_to_path[(current_index, current_index + num_lines)] = shard_path
        current_index = current_index + num_lines

    # Randomly sample IDs from 0 to current_index - 1 (inclusive)
    logger.info(f"Subsampling {num_to_sample} ids")
    subsampled_ids = random.sample(range(0, current_index), k=num_to_sample)
    logger.info(f"Subsampled {num_to_sample} ids")

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
    extracted_examples = set()
    for shard_path, offsets in tqdm(shard_to_local_offsets.items(), desc="Extracting sampled IDs"):
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

    logger.info(f"Writing {len(extracted_examples)} sampled lines to {output_path}")
    with fsspec.open(output_path, "w", compression="gzip") as fout:
        for example in tqdm(extracted_examples, desc="Writing"):
            fout.write(json.dumps(asdict(example)) + "\n")


@draccus.wrap()
def get_outlinks_from_html(cfg: OutlinksSamplingConfig):
    # Do all the processing in a remote function
    _ = ray.get(get_shards_to_process.remote(cfg.input_pattern, cfg.num_to_sample, cfg.output_path))


if __name__ == "__main__":
    get_outlinks_from_html()
