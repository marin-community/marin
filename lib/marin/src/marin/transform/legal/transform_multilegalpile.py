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
Example Usage:
uv run zephyr --backend=ray --max-parallelism=1000 --memory=1GB --cluster=us-central2 \
    lib/marin/src/marin/transform/legal/transform_multilegalpile.py \
    --input_path gs://marin-us-central2/raw/legal-multi-legal-wikipedia-filtered/483f6c8/ \
    --output_path gs://marin-data/processed/law/multilegalpile-v1.0/txt/documents
"""

from dataclasses import dataclass

import draccus
from zephyr import Dataset, flow_backend, load_jsonl


def convert_to_dolma(row: dict):
    """Transform a single multilegalpile record to Dolma format.

    Args:
        row: Record from JSONL.xz file

    Returns:
        Transformed record in Dolma format
    """
    import hashlib

    source = "multilegalpile"
    # Generate a unique ID using hash of the content
    unique_id = hashlib.sha256(f"{row['type']}:{row['text']}:{row['jurisdiction']}".encode()).hexdigest()

    return {
        "id": unique_id,
        "text": row["text"],
        "source": source,
        "metadata": {
            "type": row["type"],
            "jurisdiction": row["jurisdiction"],
        },
    }


@dataclass
class Config:
    """Configuration for multilegalpile transformation."""

    input_path: str
    """Path to the multilegalpile raw directory"""
    output_path: str
    """Path to store multilegalpile dolma files"""


@draccus.wrap()
def main(cfg: Config) -> None:
    """Transform multilegalpile data to Dolma format."""
    backend = flow_backend()
    pipeline = (
        Dataset.from_files(f"{cfg.input_path}/**/*.jsonl.xz")
        .flat_map(load_jsonl)
        .map(convert_to_dolma)
        .write_jsonl(f"{cfg.output_path}/data-{{shard:05d}}-of-{{total:05d}}.jsonl.gz")
    )
    list(backend.execute(pipeline))


if __name__ == "__main__":
    main()
