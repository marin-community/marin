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
Transform Australian Legal Corpus data to Dolma format.
path: scripts/legal/transform_australianlegalcorpus.py
Inputs: one raw jsonl file, Output: one jsonl.gz file in dolma format

Example Usage:
# For small dataset, sync backend is fine
uv run zephyr --backend=sync \
    lib/marin/src/marin/transform/legal/transform_australianlegalcorpus.py \
    --input_path gs://marin-us-central2/raw/legal-open-australian-legal-corpus/66e7085/ \
    --output_path gs://marin-data/processed/law/australianlegalcorpus-v1.0/txt/documents

# Or use ray if dataset grows
uv run zephyr --backend=ray --max-parallelism=100 \
    lib/marin/src/marin/transform/legal/transform_australianlegalcorpus.py \
    --input_path gs://marin-us-central2/raw/legal-open-australian-legal-corpus/66e7085/ \
    --output_path gs://marin-data/processed/law/australianlegalcorpus-v1.0/txt/documents
"""

from dataclasses import dataclass

import draccus
from zephyr import Dataset, flow_backend, load_jsonl


def convert_to_dolma(row: dict):
    """Transform a single Australian Legal Corpus record to Dolma format.

    Args:
        row: Record from JSONL file

    Returns:
        Transformed record in Dolma format
    """
    source = "australianlegalcorpus"

    return {
        "id": row["version_id"],
        "text": row["text"],
        "source": source,
        "created": row["date"],
        "added": row["when_scraped"],
        "metadata": {
            "type": row["type"],
            "jurisdiction": row["jurisdiction"],
            "source": row["source"],
            "citation": row["citation"],
            "mime": row["mime"],
            "url": row["url"],
        },
    }


@dataclass
class Config:
    """Configuration for Australian Legal Corpus transformation."""

    input_path: str
    """Path to the australianlegalcorpus raw directory"""
    output_path: str
    """Path to store australianlegalcorpus dolma files"""


@draccus.wrap()
def main(cfg: Config) -> None:
    """Transform Australian Legal Corpus data to Dolma format."""
    backend = flow_backend()
    pipeline = (
        Dataset.from_files(f"{cfg.input_path}/**/*.jsonl")
        .flat_map(load_jsonl)
        .reshard(32)
        .map(convert_to_dolma)
        .write_jsonl(f"{cfg.output_path}/data-{{shard:05d}}-of-{{total:05d}}.jsonl.gz")
    )
    list(backend.execute(pipeline))


if __name__ == "__main__":
    main()
