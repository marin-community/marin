#!/usr/bin/env python3
"""
Given a pattern of parquet or jsonl.gz files, count the number of tokens
in the documents.

Counting tokens in fineweb-edu-10M:

```
python marin/run/ray_run.py \
    --no_wait -- \
    python marin/crawl/count_tokens.py \
    --input_patterns '["gs://marin-us-central2/scratch/nfliu/text/fineweb-edu-10M/*_text_and_scores.parquet"]' \
    --output_path "gs://marin-us-central2/scratch/nfliu/count_tokens/fineweb-edu-10M/"
```

Counting tokens in fineweb-edu-10M (deduplicated):

```
python marin/run/ray_run.py \
    --no_wait -- \
    python marin/crawl/count_tokens.py \
    --input_patterns '["gs://marin-us-central2/scratch/nfliu/text/fineweb-edu-10M-minhash/*.jsonl.gz"]' \
    --output_path "gs://marin-us-central2/scratch/nfliu/count_tokens/fineweb-edu-10M-minhash/"
```

Counting tokens in fineweb-edu:

```
python marin/run/ray_run.py \
    --no_wait -- \
    python marin/crawl/count_tokens.py \
    --input_patterns '["gs://marin-us-central2/raw/fineweb-edu/*/*.parquet"]' \
    --output_path "gs://marin-us-central2/scratch/nfliu/count_tokens/fineweb-edu/"
```

Counting tokens in fineweb-edu (deduplicated):

```
python marin/run/ray_run.py \
    --no_wait -- \
    python marin/crawl/count_tokens.py \
    --input_patterns '["gs://marin-us-central2/scratch/nfliu/text/fineweb-edu-minhash/*.jsonl.gz"]' \
    --output_path "gs://marin-us-central2/scratch/nfliu/count_tokens/fineweb-edu-minhash/"
```

Counting tokens in fineweb-edu-10M (deduplicated against fineweb-edu):

```
python marin/run/ray_run.py \
    --no_wait -- \
    python marin/crawl/count_tokens.py \
    --input_patterns '["gs://marin-us-central2/scratch/nfliu/text/fineweb_edu_10M_minhash_against_fineweb_edu/*.jsonl.gz"]' \
    --output_path "gs://marin-us-central2/scratch/nfliu/count_tokens/fineweb_edu_10M_minhash_against_fineweb_edu/"
```

Counting tokens in open-web-math:

```
python marin/run/ray_run.py \
    --no_wait -- \
    python marin/crawl/count_tokens.py \
    --input_patterns '["gs://marin-us-central2/raw/open-web-math-fde8ef8/fde8ef8/huggingface.co/datasets/open-web-math/open-web-math/resolve/fde8ef8/data/*.parquet"]' \
    --output_path "gs://marin-us-central2/scratch/nfliu/count_tokens/open-web-math-fde8ef8/"
```

Counting tokens in open-web-math-10M:

```
python marin/run/ray_run.py \
    --no_wait -- \
    python marin/crawl/count_tokens.py \
    --input_patterns '["gs://marin-us-central2/scratch/nfliu/text/open-web-math-fde8ef8-10M/*.parquet"]' \
    --output_path "gs://marin-us-central2/scratch/nfliu/count_tokens/open_web_math_10M/"
```

Counting tokens in open-web-math-10M (deduplicated against open-web-math):

```
python marin/run/ray_run.py \
    --no_wait -- \
    python marin/crawl/count_tokens.py \
    --input_patterns '["gs://marin-us-central2/scratch/nfliu/text/open_web_math_10M_minhash_against_open_web_math/*.jsonl.gz"]' \
    --output_path "gs://marin-us-central2/scratch/nfliu/count_tokens/open_web_math_10M_minhash_against_open_web_math/"
```
"""  # noqa: E501
import json
import logging
import os
from dataclasses import dataclass

import draccus
import fsspec
import pandas as pd
import ray
from tqdm_loggable.auto import tqdm
from transformers import AutoTokenizer

from marin.utils import fsspec_exists, fsspec_glob

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CountTokensConfig:
    input_patterns: list[str]
    output_path: str
    tokenizer_name: str = "gpt2"


def count_tokens_in_jsonl_file(input_path: str, tokenizer_name: str) -> tuple[int, int]:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    total_tokens = 0
    num_documents = 0
    with fsspec.open(input_path, "rt", compression="infer") as f:
        for line in tqdm(f, desc=os.path.basename(input_path)):
            data = json.loads(line)
            total_tokens += len(tokenizer.encode(data["text"]))
            num_documents += 1
    return total_tokens, num_documents


def count_tokens_in_parquet_file(input_path: str, tokenizer_name: str) -> tuple[int, int]:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    total_tokens = 0
    num_documents = 0
    df = pd.read_parquet(input_path)
    assert "text" in df.columns
    for text in tqdm(df["text"].tolist(), desc=os.path.basename(input_path)):
        total_tokens += len(tokenizer.encode(text))
        num_documents += 1
    return total_tokens, num_documents


@ray.remote(memory=2 * 1024 * 1024 * 1024)
def count_tokens_in_shard(input_path: str, shard_output_path: str, tokenizer_name: str):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    if fsspec_exists(shard_output_path):
        logger.info(f"Found output at {shard_output_path}, re-using results...")
        with fsspec.open(shard_output_path) as f:
            loaded_results = json.load(f)
            saved_num_tokens = loaded_results["num_tokens"]
            saved_num_documents = loaded_results["num_documents"]
            return (saved_num_tokens, saved_num_documents)
    if input_path.endswith(".parquet"):
        num_tokens, num_documents = count_tokens_in_parquet_file(input_path, tokenizer_name)
    elif ".jsonl" in input_path:
        num_tokens, num_documents = count_tokens_in_jsonl_file(input_path, tokenizer_name)
    else:
        raise ValueError(f"Failed to detect filetype for path {input_path}")

    with fsspec.open(shard_output_path, "w") as f:
        json.dump({"input_path": input_path, "num_tokens": num_tokens, "num_documents": num_documents}, f)
    return num_tokens, num_documents


@ray.remote(memory=8 * 1024 * 1024 * 1024)
def count_tokens(input_patterns: list[str], output_path: str, tokenizer_name: str):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    # Get the input paths
    input_paths = []
    for pattern in input_patterns:
        input_paths.extend(fsspec_glob(pattern))
    logger.info(f"Found {len(input_paths)} input paths")

    num_shards_to_process = len(input_paths)
    MAX_CONCURRENT_TASKS = 1000
    num_shards_submitted = 0
    unfinished = []

    def submit_shard_task(shard_path):
        nonlocal num_shards_submitted
        shard_output_path = os.path.join(output_path, os.path.basename(shard_path) + ".token_counts")
        unfinished.append(count_tokens_in_shard.remote(shard_path, shard_output_path, tokenizer_name))
        num_shards_submitted += 1
        if num_shards_submitted % 10 == 0:
            logger.info(
                f"Submitted {num_shards_submitted} / {num_shards_to_process} shards "
                f"({num_shards_submitted / num_shards_to_process})"
            )

    for _ in range(min(MAX_CONCURRENT_TASKS, len(input_paths))):
        submit_shard_task(input_paths.pop())

    num_tokens = 0
    num_documents = 0
    with tqdm(total=num_shards_to_process, desc="Counting records") as pbar:
        while unfinished:
            finished, unfinished = ray.wait(unfinished, num_returns=len(unfinished), timeout=5)
            try:
                results = ray.get(finished)
                for shard_num_tokens, shard_num_documents in results:
                    num_tokens += shard_num_tokens
                    num_documents += shard_num_documents
                    pbar.update(1)
            except Exception as e:
                logger.exception(f"Error processing shard: {e}")
                raise

            # If we have more shard paths left to process and we haven't hit the max
            # number of concurrent tasks, add tasks to the unfinished queue.
            while input_paths and len(unfinished) < MAX_CONCURRENT_TASKS:
                submit_shard_task(input_paths.pop())
    logger.info(f"Total number of tokens: {num_tokens}, total number of documents: {num_documents}")
    aggregated_stats_output_path = os.path.join(output_path, "total_token_counts.json")
    with fsspec.open(aggregated_stats_output_path, "w") as f:
        json.dump({"num_tokens": num_tokens, "num_shards": num_shards_to_process}, f)


@draccus.wrap()
def count_tokens_driver(cfg: CountTokensConfig):
    # Do everything in a remote task
    ray.get(count_tokens.remote(cfg.input_patterns, cfg.output_path, cfg.tokenizer_name))


if __name__ == "__main__":
    count_tokens_driver()
