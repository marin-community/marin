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
Consolidate many small shards into larger files by resharding.

Example usage with zephyr CLI:

```bash
uv run zephyr --backend=ray --max-parallelism=100 --memory=4GB \
    lib/marin/src/marin/crawl/open_web_math/consolidate_open_web_math_shards.py \
    --input_pattern "gs://marin-us-central2/documents/stackexchange/v2024-04-02/md-qa-pair/*.jsonl.gz" \
    --output_pattern "gs://marin-us-central2/scratch/consolidated/stackexchange_{shard:05d}.jsonl.gz" \
    --batch_size 10
```

Or for local testing:

```bash
uv run zephyr --backend=threadpool \
    lib/marin/src/marin/crawl/open_web_math/consolidate_open_web_math_shards.py \
    --input_pattern "/tmp/test-consolidate/*.jsonl.gz" \
    --output_pattern "/tmp/test-consolidate/consolidated_{shard:05d}.jsonl.gz" \
    --batch_size 2
```
"""

import logging
from dataclasses import dataclass

import draccus

from marin.utils import fsspec_glob
from zephyr import Dataset, flow_backend, load_jsonl

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConsolidationConfig:
    input_pattern: str
    """Glob pattern for input files (e.g., 'gs://bucket/path/*.jsonl.gz')."""

    output_pattern: str
    """Output file pattern with {shard:05d} placeholder (e.g., 'gs://bucket/output_{shard:05d}.jsonl.gz')."""

    batch_size: int = 1000
    """Number of input files to consolidate into each output file."""


@draccus.wrap()
def main(cfg: ConsolidationConfig):
    # Get all files matching the pattern
    shard_files = fsspec_glob(cfg.input_pattern)
    shard_files = sorted(shard_files)
    logger.info(f"Found {len(shard_files)} shards to consolidate")

    # Calculate target number of output shards
    num_output_shards = (len(shard_files) + cfg.batch_size - 1) // cfg.batch_size

    logger.info(f"Consolidating into {num_output_shards} output files")

    backend = flow_backend()

    pipeline = (
        Dataset.from_list(shard_files)
        .flat_map(load_jsonl)  # Stream all records from all files
        .reshard(num_output_shards)  # Redistribute across target number of shards
        .write_jsonl(cfg.output_pattern)  # Write consolidated files
    )

    output_files = list(backend.execute(pipeline))
    logger.info(f"Consolidation complete. Created {len(output_files)} consolidated files")


if __name__ == "__main__":
    main()
