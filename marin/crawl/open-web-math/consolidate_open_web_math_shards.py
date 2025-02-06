#!/usr/bin/env python3
"""
Consolidate open-web-math and FineWeb-Edu shards, so we reduce the number of
files we have on GCS.

Running:

```
ray job submit --address http://127.0.0.1:8265 --working-dir . --no-wait -- \
    python marin/crawl/open-web-math/consolidate_open_web_math_shards.py \
    --input_path gs://marin-us-central2/documents/open-web-math-fde8ef8/html/ \
    --prefix openwebmath
```

Creates files in gs://marin-us-central2/documents/open-web-math-fde8ef8/html/ of the form
`openwebmath_{index}.jsonl.gz`.


Running on FineWeb-Edu:

```
for fineweb_edu_dump_html_path in $(gcloud storage ls gs://marin-us-central2/documents/fineweb-edu/html); do
    dump_name=$(basename -- ${fineweb_edu_dump_html_path})
    ray job submit --address http://127.0.0.1:8265 --working-dir . --no-wait -- \
    python marin/crawl/open-web-math/consolidate_open_web_math_shards.py \
    --input_path ${fineweb_edu_dump_html_path} \
    --prefix fineweb_edu
done
```

Creates files in gs://marin-us-central2/documents/fineweb-edu/html/<dump_name> of the form
`fineweb_edu_{index}.jsonl.gz`.
"""

import json
import logging
import os
import pathlib
from dataclasses import dataclass

import draccus
import fsspec
import ray
from tqdm_loggable.auto import tqdm

from marin.core.runtime import cached_or_construct_output
from marin.utils import fsspec_glob

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConsolidationConfig:
    input_path: str
    prefix: str


@ray.remote(memory=4 * 1024 * 1024 * 1024)
@cached_or_construct_output(
    success_suffix="SUCCESS", verbose=False
)  # We use this decorator to make this function idempotent
def process_one_batch(html_paths_batch: list[str], output_path: str):
    """
    Takes in a batch of input files, concatenates them, and writes them to output_path.

    Args:
    html_paths_batch (list[str]): Paths of HTML files to combine.
    output_path (str): Path to write JSONL file with combined contents.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    with fsspec.open(output_path, "w", compression="gzip") as fout:
        for html_path in tqdm(html_paths_batch):
            with fsspec.open(html_path, "rt", compression="gzip") as fin:
                # Try to read everything at once, to prevent fsspec from
                # making many small calls to GCS.
                # This file should already end with a newline, so we can
                # directly write it as-is to the output file.
                file_data = fin.read()
                fout.write(file_data)


def batched(iterable, n=1):
    length = len(iterable)
    for ndx in range(0, length, n):
        yield iterable[ndx : min(ndx + n, length)]


@ray.remote(memory=32 * 1024 * 1024 * 1024)
@cached_or_construct_output(
    success_suffix="SUCCESS", verbose=False
)  # We use this decorator to make this function idempotent
def get_shards_indices_to_process(shard_path: str, output_path: str):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    # Get the HTML files (of form <int index>.jsonl.gz) and sort by the integer index.
    # We sort to ensure that the sharding is reproducible.
    html_path_indices: list[int] = [
        int(pathlib.Path(path).name.removesuffix(".jsonl.gz"))
        for path in fsspec_glob(os.path.join(shard_path, "*.jsonl.gz"))
    ]
    html_path_indices: list[int] = sorted(html_path_indices)
    logger.info(f"Found {len(html_path_indices)} shards to process")

    with fsspec.open(output_path, "w", compression="gzip") as fout:
        json.dump(html_path_indices, fout)


@draccus.wrap()
def consolidate_html(cfg: ConsolidationConfig):
    shard_indices_to_process_path = os.path.join(cfg.input_path, "shard_indices.jsonl.gz")
    # Write the shard indices to process to `shard_indices_to_process_path`, or skip
    # if it already exists.
    ray.get(get_shards_indices_to_process.remote(cfg.input_path, shard_indices_to_process_path))
    with fsspec.open(shard_indices_to_process_path, "rt", compression="gzip") as f:
        shard_indices = json.load(f)
    logger.info(f"Found {len(shard_indices)} to process")

    # Group into chunks of 1000 WARCs each
    # open-web-math has ~3M WARCs in total, which yields 3000 resharded chunks.
    refs = []
    for i, html_shard_indices_batch in enumerate(batched(shard_indices, 1000)):
        output_path = os.path.join(cfg.input_path, f"{cfg.prefix}_{i}.jsonl.gz")
        html_path_batch = [
            os.path.join(cfg.input_path, f"{shard_index}.jsonl.gz") for shard_index in html_shard_indices_batch
        ]
        refs.append(process_one_batch.remote(html_path_batch, output_path))
    logger.info(f"Submitted {len(refs)} tasks")

    # Wait for the tasks to finish
    ray.get(refs)


if __name__ == "__main__":
    consolidate_html()
