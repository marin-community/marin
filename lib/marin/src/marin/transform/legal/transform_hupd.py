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
Inputs: one tar.gz file per year containing multiple json files, Output: one jsonl.gz file in dolma format per year

Example Usage:
uv run zephyr --backend=ray --max-parallelism=1000 --memory=10GB --cluster=us-central2 \
    lib/marin/src/marin/transform/legal/transform_hupd.py \
    --input_path gs://marin-us-central2/raw/legal-hupd/f570a84/ \
    --output_path gs://marin-data/processed/law/hupd-v1.0/txt/documents
"""

import json
import tarfile
from dataclasses import dataclass

import draccus
import fsspec
from zephyr import Dataset, flow_backend


def convert_to_dolma(input_file_path):
    """Process HUPD tar.gz file and yield transformed records in Dolma format.

    Args:
        input_file_path: Path to input tar.gz file

    Yields:
        Transformed records in Dolma format
    """
    source = "hupd"

    with fsspec.open(
        input_file_path, mode="rb", compression="gzip", block_size=16_000_000, cache_type="background"
    ) as f:
        with tarfile.open(fileobj=f) as tar:
            for member in tar:
                if member.name.endswith(".json"):
                    f = tar.extractfile(member)
                    if f is not None:
                        content = f.read()
                        row = json.loads(content)

                        # The background and summary should be part of the full description, so we leave them out
                        text = (
                            f"Title:\n{row['title']}\n\n"
                            f"Abstract:\n{row['abstract']}\n\n"
                            f"Claims:\n{row['claims']}\n\n"
                            f"Full Description:\n{row['full_description']}"
                        )

                        yield {
                            "id": row["application_number"],
                            "text": text,
                            "created": row["date_published"],
                            "source": source,
                            "metadata": {
                                "publication_number": row["publication_number"],
                                "decision": row["decision"],
                                "date_produced": row["date_produced"],
                                "main_cpc_label": row["main_cpc_label"],
                                "cpc_labels": row["cpc_labels"],
                                "main_ipcr_label": row["main_ipcr_label"],
                                "ipcr_labels": row["ipcr_labels"],
                                "patent_number": row["patent_number"],
                                "filing_date": row["filing_date"],
                                "patent_issue_date": row["patent_issue_date"],
                                "abandon_date": row["abandon_date"],
                                "uspc_class": row["uspc_class"],
                                "uspc_subclass": row["uspc_subclass"],
                                "examiner_id": row["examiner_id"],
                                "examiner_name_last": row["examiner_name_last"],
                                "examiner_name_first": row["examiner_name_first"],
                                "examiner_name_middle": row["examiner_name_middle"],
                                # Leave out inventor_list for simplicity
                            },
                        }


@dataclass
class Config:
    """Configuration for HUPD transformation."""

    input_path: str
    """Path to the hupd raw directory"""
    output_path: str
    """Path to store hupd dolma files"""


@draccus.wrap()
def main(cfg: Config) -> None:
    """Transform HUPD data to Dolma format."""
    backend = flow_backend()
    pipeline = (
        Dataset.from_files(cfg.input_path, "**/*.tar.gz")
        .flat_map(convert_to_dolma)
        .write_jsonl(f"{cfg.output_path}/data-{{shard:05d}}-of-{{total:05d}}.jsonl.gz")
    )
    list(backend.execute(pipeline))
