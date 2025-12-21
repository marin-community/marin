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
Filter StackExchange data by vote threshold and remove duplicate questions.

Deduplication is performed per-file: duplicate question IDs within each input
file are removed, keeping the first occurrence. Duplicates across different
input files are NOT removed.

Example Usage:
uv run zephyr --backend=ray --max-parallelism=1000 --cluster=us-central2 \
    lib/marin/src/marin/transform/stackexchange/filter_stackexchange.py \
    --input_path gs://marin-us-central2/raw/stackexchange/ \
    --output_path gs://marin-us-central2/processed/stackexchange/filtered \
    --min_vote_threshold 10 \
    --remove_duplicate_questions
"""

import dataclasses

import draccus
from zephyr import Dataset, execute


@dataclasses.dataclass
class FilterStackExchangeConfig:
    input_path: str
    output_path: str
    min_vote_threshold: int = 10
    remove_duplicate_questions: bool = True


def _process_file_with_filtering(file_path: str, config: FilterStackExchangeConfig):
    """Process one StackExchange file: filter by votes and deduplicate within file.

    Args:
        file_path: Path to input JSONL file
        config: Filter configuration

    Yields:
        Records that pass the vote threshold and are unique within this file
    """
    import json

    import fsspec

    seen_ids = set() if config.remove_duplicate_questions else None

    with fsspec.open(file_path, "rt", compression="infer") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            row = json.loads(line)
            votes = row["metadata"]["votes"]

            # Filter by vote threshold
            if votes < config.min_vote_threshold:
                continue

            # Deduplicate within this file
            if seen_ids is not None:
                question_id = row["metadata"]["id"]
                if question_id in seen_ids:
                    continue
                seen_ids.add(question_id)

            yield row


@draccus.wrap()
def filter_stackexchange(config: FilterStackExchangeConfig):
    """Filter StackExchange data by vote threshold and remove duplicates."""
    pipeline = (
        Dataset.from_files(f"{config.input_path}/*.jsonl.gz")
        .flat_map(lambda path: _process_file_with_filtering(path, config))
        .write_jsonl(f"{config.output_path}/data-{{shard:05d}}-of-{{total:05d}}.jsonl.gz")
    )

    list(execute(pipeline))


if __name__ == "__main__":
    filter_stackexchange()
