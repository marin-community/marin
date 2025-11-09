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
Count total tokens in a specified column across files using a HuggingFace tokenizer.

Example usage with zephyr CLI:

```bash
uv run zephyr --cluster=us-central2 --backend=ray --max-parallelism=1000 --memory=1GB \
    lib/marin/src/marin/validation/count_total_tokens.py \
    --input_pattern "gs://marin-us-central2/raw/dolma/v1.7/cc_en_head-*.json.gz" \
    --text_column text
```
"""

import logging
from dataclasses import dataclass

import draccus
from marin.utils import fsspec_glob
from zephyr import Dataset, flow_backend, load_file

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CountTotalTokensConfig:
    input_pattern: str
    """Glob pattern for input files (e.g., gs://bucket/path/**/*.jsonl.gz)."""

    tokenizer_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    """Name of the HuggingFace tokenizer to use."""

    text_column: str = "text"
    """Name of the column containing text to tokenize."""


class TokenCounter:
    """Token counter that loads tokenizer once per worker."""

    def __init__(self, tokenizer_name: str, text_column: str):
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.text_column = text_column

    def __call__(self, record: dict) -> int:
        """Count tokens in a record's text column."""
        if self.text_column in record and record[self.text_column] is not None:
            return len(self.tokenizer.encode(str(record[self.text_column])))
        return 0


def count_total_tokens(input_pattern: str, tokenizer_name: str, text_column: str) -> int:
    """Count total tokens across all files matching the pattern."""
    input_paths = fsspec_glob(input_pattern)
    logger.info(f"Found {len(input_paths)} files to process")

    backend = flow_backend()
    counter = TokenCounter(tokenizer_name, text_column)
    pipeline = Dataset.from_list(input_paths).flat_map(load_file).map(counter)
    token_counts = list(backend.execute(pipeline))
    total_tokens = sum(token_counts)

    logger.info(f"Total tokens in '{text_column}' column: {total_tokens:,}")
    return total_tokens


@draccus.wrap()
def main(cfg: CountTotalTokensConfig):
    count_total_tokens(cfg.input_pattern, cfg.tokenizer_name, cfg.text_column)


if __name__ == "__main__":
    main()
