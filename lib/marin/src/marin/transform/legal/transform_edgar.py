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
Inputs: raw parquet files, Output: jsonl.gz files in dolma format

Example Usage:
uv run zephyr --backend=ray --max-parallelism=1000 --memory=1GB --cluster=us-central2 \
    lib/marin/src/marin/transform/legal/transform_edgar.py \
    --input_path gs://marin-us-central2/raw/legal-edgar/f7d3ba7/ \
    --output_path gs://marin-data/processed/law/edgar-v1.0/txt/documents
"""

from dataclasses import dataclass

import draccus
from zephyr import Dataset, flow_backend, load_parquet


def convert_to_dolma(row: dict):
    """Transform EDGAR record to Dolma format.

    Args:
        row: Record from parquet file

    Returns:
        Transformed record in Dolma format
    """
    source = "edgar"
    sections = [
        "section_1",
        "section_1A",
        "section_1B",
        "section_2",
        "section_3",
        "section_4",
        "section_5",
        "section_6",
        "section_7",
        "section_7A",
        "section_8",
        "section_9",
        "section_9A",
        "section_9B",
        "section_10",
        "section_11",
        "section_12",
        "section_13",
        "section_14",
        "section_15",
    ]
    text = "\n\n".join([row[section] for section in sections]).strip()

    return {
        "id": row["cik"],
        "text": text,
        "source": source,
        "metadata": {
            "year": row["year"],
            "filename": row["filename"],
        },
    }


@dataclass
class Config:
    """Configuration for EDGAR transformation."""

    input_path: str
    """Path to the edgar raw directory"""
    output_path: str
    """Path to store edgar dolma files"""


@draccus.wrap()
def main(cfg: Config) -> None:
    """Transform EDGAR data to Dolma format."""
    backend = flow_backend()
    pipeline = (
        Dataset.from_files(f"{cfg.input_path}/**/*.parquet")
        .flat_map(load_parquet)
        .map(convert_to_dolma)
        .write_jsonl(f"{cfg.output_path}/data-{{shard:05d}}-of-{{total:05d}}.jsonl.gz")
    )
    list(backend.execute(pipeline))
