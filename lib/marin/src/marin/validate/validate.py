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
validate.py

Script for validating Dolma-formatted datasets using Zephyr pipelines.

Given an input path to a top-level `documents/` directory on GCS, runs a Zephyr job that
validates all documents, computes dataset statistics, and writes a `metadata.json` file
to the same directory.

Run with:
    uv run zephyr --backend=ray --max-parallelism=1000 --memory=4GB --cluster=us-central2 \
        lib/marin/src/marin/validate/validate.py \
        --input_path gs://marin-us-central2/documents/hello_world_fw/v1.0/quickstart \
        --num_examples_to_sample 1024
"""

import json
from collections.abc import Iterator
from dataclasses import dataclass

import draccus
import fsspec
import numpy as np
from marin.utilities.validation_utils import compute_global_mean_std, summarize_document
from zephyr import Dataset, load_jsonl
from zephyr.backends import Backend


@dataclass
class ValidationConfig:
    input_path: str
    """Path to top-level documents directory to validate"""

    num_examples_to_sample: int = 1024
    """Number of documents to sample on each shard"""


def validate_shard(documents: Iterator[dict], num_samples: int) -> Iterator[dict]:
    """Validate documents in a shard.

    This function processes all documents in a shard, performing:
    - Validation and byte counting for each document
    - Reservoir sampling of examples
    - Computation of per-shard statistics

    Args:
        documents: Iterator of documents in this shard
        num_samples: Number of examples to reservoir sample from this shard

    Yields:
        Single dict with shard metadata including statistics and examples
    """
    num_documents = 0
    all_document_bytes = []
    all_text_bytes = []

    rng = np.random.default_rng()
    example_reservoir = []

    for doc_idx, doc in enumerate(documents):
        footprint = summarize_document(doc)
        all_document_bytes.append(footprint.document_bytes)
        all_text_bytes.append(footprint.text_bytes)
        num_documents += 1

        # Reservoir sampling
        if doc_idx < num_samples:
            example_reservoir.append(doc)
        else:
            j = rng.integers(0, doc_idx + 1)
            if j < num_samples:
                example_reservoir[j] = doc

    if num_documents == 0:
        return

    all_document_bytes_arr = np.asarray(all_document_bytes)
    all_text_bytes_arr = np.asarray(all_text_bytes)

    yield {
        "num_documents": num_documents,
        "document_bytes_mean": float(np.mean(all_document_bytes_arr)),
        "document_bytes_std": float(np.std(all_document_bytes_arr)),
        "text_bytes_mean": float(np.mean(all_text_bytes_arr)),
        "text_bytes_std": float(np.std(all_text_bytes_arr)),
        "examples": example_reservoir,
    }


def aggregate_and_write_metadata(shard_metadata_iter: Iterator[list[dict]], output_path: str) -> dict:
    """Aggregate per-shard metadata and write global metadata.json.

    Args:
        shard_metadata_iter: Iterator of lists of per-shard metadata dicts
        output_path: Path to write metadata.json

    Returns:
        Dict with summary of what was written
    """
    # Collect all shard metadata - flatten the lists
    all_shard_metadata = [item for shard_list in shard_metadata_iter for item in shard_list]

    if not all_shard_metadata:
        raise ValueError("No shards were validated")

    # Extract per-shard statistics
    shard_num_documents = []
    shard_document_bytes_means = []
    shard_document_bytes_stds = []
    shard_text_bytes_means = []
    shard_text_bytes_stds = []
    examples = []

    for shard in all_shard_metadata:
        shard_num_documents.append(shard["num_documents"])
        shard_document_bytes_means.append(shard["document_bytes_mean"])
        shard_document_bytes_stds.append(shard["document_bytes_std"])
        shard_text_bytes_means.append(shard["text_bytes_mean"])
        shard_text_bytes_stds.append(shard["text_bytes_std"])
        examples.extend(shard["examples"])

    doc_bytes_stats = compute_global_mean_std(shard_num_documents, shard_document_bytes_means, shard_document_bytes_stds)
    text_bytes_stats = compute_global_mean_std(shard_num_documents, shard_text_bytes_means, shard_text_bytes_stds)

    # Write metadata.json
    metadata = {
        "num_documents": sum(shard_num_documents),
        "document_bytes_mean": doc_bytes_stats.mean,
        "document_bytes_std": doc_bytes_stats.std,
        "text_bytes_mean": text_bytes_stats.mean,
        "text_bytes_std": text_bytes_stats.std,
        "num_shards": len(all_shard_metadata),
        "shard_num_documents": shard_num_documents,
        "examples": examples,
    }

    with fsspec.open(output_path, "wt") as f:
        json.dump(metadata, f, indent=2)

    return {
        "path": output_path,
        "num_shards": len(all_shard_metadata),
        "num_documents": sum(shard_num_documents),
    }


@draccus.wrap()
def main(cfg: ValidationConfig) -> None:
    """Validate Dolma-formatted documents using Zephyr pipeline."""
    pipeline = (
        Dataset.from_files(f"{cfg.input_path}/**/*.jsonl.gz")
        .flat_map(load_jsonl)
        .map_shard(lambda docs: validate_shard(docs, cfg.num_examples_to_sample))
        .reduce(
            local_reducer=list,
            global_reducer=lambda shards: aggregate_and_write_metadata(shards, f"{cfg.input_path}/metadata.json"),
        )
    )

    result = list(Backend.execute(pipeline))
    print(f"Validation complete: {result[0]}")


if __name__ == "__main__":
    main()
