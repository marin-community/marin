#!/usr/bin/env python3
# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Count tokens in files using a HuggingFace tokenizer.

Supports automatic format detection for .jsonl, .jsonl.gz, .jsonl.zst, .jsonl.xz, and .parquet files.

Example usage with zephyr CLI:

```bash
uv run zephyr --cluster=us-central2 --backend=ray --max-parallelism=1000 --memory=2GB \
    lib/marin/src/marin/crawl/count_tokens.py \
    --entry-point count_tokens_driver \
    --input_patterns '["gs://marin-us-central2/raw/dolma/v1.7/cc_en_head-*.json.gz"]' \
    --output_path "gs://marin-us-central2/scratch/count_tokens/dolma-cc-en-head/"
```

Additional examples:

```bash
# Count tokens in Dolma arxiv subset
uv run zephyr --cluster=us-central2 --backend=ray --max-parallelism=1000 --memory=2GB \
    lib/marin/src/marin/crawl/count_tokens.py \
    --entry-point count_tokens_driver \
    --input_patterns '["gs://marin-us-central2/raw/dolma/v1.7/arxiv-*.json.gz"]' \
    --output_path "gs://marin-us-central2/scratch/count_tokens/dolma-arxiv/"

# Count tokens in StackExchange markdown
uv run zephyr --cluster=us-central2 --backend=ray --max-parallelism=1000 --memory=2GB \
    lib/marin/src/marin/crawl/count_tokens.py \
    --entry-point count_tokens_driver \
    --input_patterns '["gs://marin-us-central2/documents/stackexchange/v2024-04-02/md-qa-pair/*.jsonl.gz"]' \
    --output_path "gs://marin-us-central2/scratch/count_tokens/stackexchange/"
```

"""
import json
import logging
import os
from dataclasses import dataclass

import draccus
import fsspec
from marin.utils import fsspec_glob
from transformers import AutoTokenizer
from zephyr import Dataset, flow_backend, load_file

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CountTokensConfig:
    input_patterns: list[str]
    output_path: str
    tokenizer_name: str = "gpt2"


class TokenCounter:
    """Stateful token counter that loads tokenizer once per worker."""

    def __init__(self, tokenizer_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def __call__(self, record: dict) -> int:
        """Count tokens in a record."""
        return len(self.tokenizer.encode(record["text"]))


def count_tokens(input_patterns: list[str], output_path: str, tokenizer_name: str):
    """Count tokens across all files matching input patterns."""
    input_paths = []
    for pattern in input_patterns:
        input_paths.extend(fsspec_glob(pattern))
    logger.info(f"Found {len(input_paths)} files to process")
    backend = flow_backend()

    counter = TokenCounter(tokenizer_name)
    pipeline = Dataset.from_list(input_paths).flat_map(load_file).map(counter)

    token_counts = list(backend.execute(pipeline))
    total_tokens = sum(token_counts)
    num_documents = len(token_counts)

    logger.info(f"Total tokens: {total_tokens:,}, documents: {num_documents:,}")

    if output_path:
        aggregated_stats_output_path = os.path.join(output_path, "total_token_counts.json")
        with fsspec.open(aggregated_stats_output_path, "w") as f:
            json.dump({"num_tokens": total_tokens, "num_documents": num_documents, "num_files": len(input_paths)}, f)


@draccus.wrap()
def count_tokens_driver(cfg: CountTokensConfig):
    count_tokens(cfg.input_patterns, cfg.output_path, cfg.tokenizer_name)


if __name__ == "__main__":
    count_tokens_driver()
