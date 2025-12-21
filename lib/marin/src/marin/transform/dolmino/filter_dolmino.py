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
Filter dolmino dataset by minimum document length.

Example Usage:
uv run zephyr --backend=ray --max-parallelism=1000 --memory=10GB --num-cpus=2 \
    lib/marin/src/marin/transform/dolmino/filter_dolmino.py \
    --input_path gs://path/to/dolmino \
    --output_path gs://path/to/output \
    --split wiki \
    --min_length 1000
"""

import dataclasses

import draccus
from zephyr import Dataset, execute, load_jsonl


@dataclasses.dataclass
class FilterDolminoConfig:
    """Configuration to filter the dolmino dataset.

    Attributes:
        input_path: The path to the input dolmino dataset.
        output_path: The path to the output of the filtered dolmino dataset.
        split: The split of the dolmino dataset to filter (e.g. "wiki", "stackexchange", "pes2o").
        min_length: The minimum length for each document to filter by.
    """

    input_path: str
    output_path: str
    split: str
    min_length: int | None = None


def filter_dolmino(config: FilterDolminoConfig):
    """Filter dolmino dataset by minimum document length using streaming pipeline."""

    # Create filter function that captures min_length
    def meets_length_threshold(record: dict) -> bool:
        """Check if document meets minimum length requirement."""
        if config.min_length is None:
            return True
        length = record.get("metadata", {}).get("length", 0)
        return length >= config.min_length

    # Build streaming pipeline
    pipeline = (
        Dataset.from_files(f"{config.input_path}/data/{config.split}/**/*.json.gz")
        .flat_map(load_jsonl)  # Stream records from each file
        .filter(meets_length_threshold)  # Apply length filter
        .write_jsonl(f"{config.output_path}/{{shard:05d}}.jsonl.gz")
    )

    execute(pipeline)


@draccus.wrap()
def main(config: FilterDolminoConfig):
    filter_dolmino(config)


if __name__ == "__main__":
    main()
