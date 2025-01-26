#!/usr/bin/env python3
"""
Given a pattern of parquet or jsonl.gz files, count the number of tokens
in the documents.

Counting tokens in fineweb-edu-10M:

```
python marin/run/ray_run.py \
    --no_wait -- \
    python scripts/crawl/count_tokens.py \
    --input_patterns '["gs://marin-us-central2/scratch/nfliu/text/fineweb-edu-10M/*_text_and_scores.parquet"]'
```

Counting tokens in fineweb-edu:

```
python marin/run/ray_run.py \
    --no_wait -- \
    python scripts/crawl/count_tokens.py \
    --input_patterns '["gs://marin-us-central2/raw/fineweb-edu/*/*.parquet"]'
```
"""
import json
import logging
import os
from dataclasses import dataclass, field

import draccus
import fsspec
import pandas as pd
import ray
from tqdm_loggable.auto import tqdm
from transformers import AutoTokenizer

from marin.utils import fsspec_glob

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CountTokensConfig:
    input_patterns: list[str] = field(
        default_factory=lambda: ["gs://marin-us-central2/scratch/nfliu/text/fineweb-edu-10M/*.parquet"]
    )
    tokenizer_name: str = "gpt2"


def count_tokens_in_jsonl_file(input_path: str, tokenizer_name: str) -> int:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    total_tokens = 0
    with fsspec.open(input_path, "rt", compression="infer") as f:
        for line in tqdm(f, desc=os.path.basename(input_path)):
            data = json.loads(line)
            total_tokens += len(tokenizer.encode(data["text"]))
    return total_tokens


def count_tokens_in_parquet_file(input_path: str, tokenizer_name: str) -> int:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    total_tokens = 0
    df = pd.read_parquet(input_path)
    assert "text" in df.columns
    for text in tqdm(df["text"].tolist(), desc=os.path.basename(input_path)):
        total_tokens += len(tokenizer.encode(text))
    return total_tokens


@ray.remote(memory=32 * 1024 * 1024 * 1024)
def count_tokens_in_shard(input_path: str, tokenizer_name: str):
    if input_path.endswith(".parquet"):
        return count_tokens_in_parquet_file(input_path, tokenizer_name)
    elif ".jsonl" in input_path:
        return count_tokens_in_jsonl_file(input_path, tokenizer_name)
    else:
        raise ValueError(f"Failed to detect filetype for path {input_path}")


@ray.remote(memory=32 * 1024 * 1024 * 1024)
def count_tokens(input_patterns: list[str], tokenizer_name: str):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    # Get the input paths
    input_paths = []
    for pattern in input_patterns:
        input_paths.extend(fsspec_glob(pattern))
    logger.info(f"Found {len(input_paths)} input paths")

    num_shards_to_process = len(input_paths)
    MAX_CONCURRENT_TASKS = 100
    num_shards_submitted = 0
    unfinished = []

    def submit_shard_task(shard_path):
        nonlocal num_shards_submitted
        unfinished.append(count_tokens_in_shard.remote(shard_path, tokenizer_name))
        num_shards_submitted += 1
        if num_shards_submitted % 10 == 0:
            logger.info(
                f"Submitted {num_shards_submitted} / {num_shards_to_process} shards "
                f"({num_shards_submitted / num_shards_to_process})"
            )

    for _ in range(min(MAX_CONCURRENT_TASKS, len(input_paths))):
        submit_shard_task(input_paths.pop())

    num_tokens = 0
    with tqdm(total=num_shards_to_process, desc="Counting records") as pbar:
        while unfinished:
            finished, unfinished = ray.wait(unfinished, num_returns=len(unfinished), timeout=5)
            try:
                results = ray.get(finished)
                for shard_num_tokens in results:
                    num_tokens += shard_num_tokens
                    pbar.update(1)
            except Exception as e:
                logger.exception(f"Error processing shard: {e}")
                raise

            # If we have more shard paths left to process and we haven't hit the max
            # number of concurrent tasks, add tasks to the unfinished queue.
            while input_paths and len(unfinished) < MAX_CONCURRENT_TASKS:
                submit_shard_task(input_paths.pop())
    logger.info(f"Total number of tokens: {num_tokens}")


@draccus.wrap()
def count_tokens_driver(cfg: CountTokensConfig):
    # Do everything in a remote task
    ray.get(count_tokens.remote(cfg.input_patterns, cfg.tokenizer_name))


def main():
    count_tokens_driver()


if __name__ == "__main__":
    main()
