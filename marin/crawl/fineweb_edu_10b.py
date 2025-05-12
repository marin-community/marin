#!/usr/bin/env python3
"""
Create a 10B token subset of fineweb_edu by counting tokens in shards until we hit the target.
"""

import json
import logging
import os
from dataclasses import dataclass

import draccus
import fsspec
import ray
from tqdm import tqdm

from marin.crawl.count_tokens import count_tokens_in_shard
from marin.utils import fsspec_glob

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class FinewebEdu10BConfig:
    input_patterns: list[str]  # Patterns to find input shards
    output_path: str  # Where to save the filtered dataset
    target_tokens: int = 10_000_000_000  # Target number of tokens (10B)
    tokenizer_name: str = "meta-llama/Meta-Llama-3.1-8B"  # Use llama3 tokenizer for consistency
    token_counts_dir: str | None = None  # Directory with existing token counts, if available


@ray.remote(memory=8 * 1024 * 1024 * 1024)
def create_10b_subset(config: FinewebEdu10BConfig):
    # Get all input paths
    input_paths = []
    for pattern in config.input_patterns:
        input_paths.extend(fsspec_glob(pattern))
    logger.info(f"Found {len(input_paths)} input shards")

    # Directory to store token counts
    token_counts_dir = config.token_counts_dir or os.path.join(config.output_path, "token_counts")

    # Track shards and their token counts
    shard_token_counts = []
    total_tokens = 0
    total_documents = 0

    # Process shards until we hit target tokens
    MAX_CONCURRENT_TASKS = 1000
    unfinished = []
    processed_paths = set()

    def submit_shard_task(shard_path):
        shard_output_path = os.path.join(token_counts_dir, os.path.basename(shard_path) + ".token_counts")
        unfinished.append(
            (shard_path, count_tokens_in_shard.remote(shard_path, shard_output_path, config.tokenizer_name))
        )

    # Initial batch of tasks
    num_initial = min(MAX_CONCURRENT_TASKS, len(input_paths))
    for i in range(num_initial):
        submit_shard_task(input_paths[i])
        processed_paths.add(input_paths[i])

    next_shard_idx = num_initial

    with tqdm(total=len(input_paths), desc="Processing shards") as pbar:
        while unfinished and total_tokens < config.target_tokens:
            # Wait for any task to complete
            finished_refs = []
            while not finished_refs:
                finished_refs = ray.wait([task for _, task in unfinished], num_returns=1, timeout=1)[0]

            # Process completed tasks
            for shard_path, task_ref in list(unfinished):
                if task_ref in finished_refs:
                    try:
                        num_tokens, num_docs = ray.get(task_ref)
                        shard_token_counts.append((shard_path, num_tokens))
                        total_tokens += num_tokens
                        total_documents += num_docs
                        unfinished.remove((shard_path, task_ref))
                        pbar.update(1)

                        # Log progress
                        if len(shard_token_counts) % 10 == 0:
                            logger.info(f"Processed {len(shard_token_counts)} shards, {total_tokens:,} tokens so far")

                    except Exception as e:
                        logger.exception(f"Error processing shard {shard_path}: {e}")
                        raise

            # Submit new tasks if needed
            while (
                len(unfinished) < MAX_CONCURRENT_TASKS
                and next_shard_idx < len(input_paths)
                and total_tokens < config.target_tokens
            ):
                next_path = input_paths[next_shard_idx]
                if next_path not in processed_paths:
                    submit_shard_task(next_path)
                    processed_paths.add(next_path)
                next_shard_idx += 1

    # Sort shards by token count (largest first) and select shards until we hit target
    shard_token_counts.sort(key=lambda x: x[1], reverse=True)
    selected_shards = []
    selected_tokens = 0

    for shard_path, num_tokens in shard_token_counts:
        if selected_tokens >= config.target_tokens:
            break
        selected_shards.append(shard_path)
        selected_tokens += num_tokens

    logger.info(f"Selected {len(selected_shards)} shards containing {selected_tokens:,} tokens")

    # Save selected shards list
    with fsspec.open(os.path.join(config.output_path, "selected_shards.json"), "w") as f:
        json.dump({"shards": selected_shards, "total_tokens": selected_tokens, "total_documents": total_documents}, f)

    return selected_shards, selected_tokens, total_documents


@draccus.wrap()
def main(cfg: FinewebEdu10BConfig):
    ray.get(create_10b_subset.remote(cfg))


if __name__ == "__main__":
    main()
