#!/usr/bin/env python3
"""
Consolidate open-web-math shards, so we reduce the number of files we have on GCS.

```
ray job submit --address http://127.0.0.1:8265 --working-dir . --no-wait -- \
    python scripts/open-web-math/consolidate_open_web_math_shards.py \
    --input_path gs://marin-us-central2/documents/open-web-math-fde8ef8/html/
```
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


@dataclass
class ConsolidationConfig:
    input_path: str


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
                for line in fin:
                    record = json.loads(line)
                    fout.write(json.dumps(record) + "\n")


def batched(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx : min(ndx + n, l)]


@ray.remote(memory=32 * 1024 * 1024 * 1024)
def get_shards_indices_to_process(shard_path: str):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    # Get the HTML files (of form <int index>.jsonl.gz) and sort by the integer index.
    # We sort to ensure that the sharding is reproducible.
    html_path_indices: list[int] = [
        int(pathlib.Path(path).name.removesuffix(".jsonl.gz"))
        for path in fsspec_glob(os.path.join(shard_path, "*.jsonl.gz"))
    ]
    html_path_indices: list[int] = sorted(html_path_indices)
    logger.info(f"Found {len(html_path_indices)} shards to process")
    return html_path_indices


@draccus.wrap()
def consolidate_html(cfg: ConsolidationConfig):
    shard_indices = ray.get(get_shards_indices_to_process.remote(cfg.input_path))

    # Group into chunks of 1000 WARCs each
    # open-web-math has ~3M WARCs in total, which yields 3000 resharded chunks.
    refs = []
    for i, html_shard_indices_batch in enumerate(batched(shard_indices, 1000)):
        output_path = os.path.join(cfg.input_path, f"openwebmath_{i}.jsonl.gz")
        html_path_batch = [
            os.path.join(cfg.input_path, f"{shard_index}.jsonl.gz") for shard_index in html_shard_indices_batch
        ]
        refs.append(process_one_batch.remote(html_path_batch, output_path))
    logger.info(f"Submitted {len(refs)} tasks")

    # Wait for the tasks to finish
    _ = ray.get(refs)


if __name__ == "__main__":
    consolidate_html()
