"""
validate.py

Script for launching validation jobs; given an input path to a top-level `documents/` or (TODO: `tokenized/`)
directory on GCS, runs a Ray job (resume-able) that iteratively validates each Dolma-formatted `.jsonl.gz` shard,
dumping a top-level `metadata.json` file with summary statistics.

Specifically, this script:
    1) [Global]  Creates a list of `.jsonl.gz` files to validate / iterate over (glob on **/*.jsonl.gz).
    2) [Map]     Checks that each shard file is a valid `.jsonl.gz` file; logs file size on disk.
    3) [Map]     Checks that each document (line of file) is properly formatted; logs summary statistics.
    4) [Map]     Samples `n_documents` randomly from the file.
    5) [Global]  Compiles global statistics for all shards / all documents and writes `metadata.json`

Run with:
    - [Ray] ray job submit --address=http://127.0.0.1:8265 --working-dir . --no-wait -- \
            python operations/validate/validate.py \
            --input_path="gs://marin-us-central2/documents/hello_world_fw/v1.0/quickstart"
"""

import json
import logging
import os
from dataclasses import dataclass

import draccus
import fsspec
import numpy as np
import ray

from marin.utilities.validation_utils import compute_global_mean_std, summarize_document_from_json
from marin.utils import fsspec_exists, fsspec_glob

# Initialize Logger
logger = logging.getLogger(__name__)


@dataclass
class ValidationConfig:
    # fmt: off
    input_path: str                             # Path to top-level `documents` directory to validate
    n_examples: int = 1024                      # Number of Documents to Sample for Example Server (across all shards)

    def __post_init__(self) -> None:
        if not self.input_path.startswith("gs://"):
            raise ValueError(
                f"Invalid `{self.input_path = }`; expected URI of form `gs://BUCKET/path/to/resource`"
            )

    # fmt: on


@ray.remote(memory=4 * 1024 * 1024 * 1024)  # 4 GB of RAM by Default
def validate_jsonl_gz(input_file_path: str, n_examples_per_shard: int) -> dict:
    fs = fsspec.filesystem("gcs")

    # Short-Circuit on Metadata File =>> this is basically an extended version of @cached_or_construct_output
    success_file_path = input_file_path + ".METADATA"
    if fsspec_exists(success_file_path):
        logger.info(f"Metadata for `{input_file_path = }` already exists; skipping validation!")
        with fsspec.open(success_file_path, "rt") as f:
            return json.load(f)

    # Create Per-Document Metadata Trackers, Example Reservoir for Sampling
    n_documents, all_document_bytes, all_text_bytes = 0, [], []
    example_reservoir, rng = [], np.random.default_rng(abs(hash(input_file_path)))  # Controlled Chaos

    # Lazily iterate + validate individual lines (documents) in the `.jsonl.gz` file
    with fsspec.open(input_file_path, "rt", compression="gzip") as input_jsonl_gz:
        for doc_idx, line in enumerate(input_jsonl_gz):
            document_bytes, text_bytes = summarize_document_from_json(line)

            all_document_bytes.append(document_bytes)
            all_text_bytes.append(text_bytes)
            n_documents += 1

            # Reservoir Sampling
            if doc_idx < n_examples_per_shard:
                example_reservoir.append(json.loads(line))
            else:
                j = rng.integers(0, doc_idx)
                if j < n_examples_per_shard:
                    example_reservoir[j] = json.loads(line)

    # Compute Document Size Summary Statistics (Mean, Standard Deviation)
    all_document_bytes, all_text_bytes = np.asarray(all_document_bytes), np.asarray(all_text_bytes)
    doc_bytes_mean, doc_bytes_std = np.mean(all_document_bytes), np.std(all_document_bytes)
    text_bytes_mean, text_bytes_std = np.mean(all_text_bytes), np.std(all_text_bytes)
    file_metadata = {
        "file_path": input_file_path,
        "file_size_bytes": fs.size(input_file_path),
        "n_documents": n_documents,
        "document_bytes_mean": doc_bytes_mean,
        "document_bytes_std": doc_bytes_std,
        "text_bytes_mean": text_bytes_mean,
        "text_bytes_std": text_bytes_std,
        "examples": example_reservoir,
    }

    # Write Per-File (Shard) Metadata to GCS
    with fsspec.open(success_file_path, "wt") as f:
        json.dump(file_metadata, f, indent=2)

    return file_metadata


def write_global_metadata(metadata_output_path: str, all_shard_metadata: list[dict]) -> bool:
    if fsspec_exists(metadata_output_path):
        logger.info(f"Global Metadata File `{metadata_output_path = }` already exists; skipping aggregation")
        return True

    # Aggregate Summary Statistics & Examples across Shards
    shard_file_paths, shard_file_size_bytes, shard_n_documents = [], [], []
    shard_document_bytes_means, shard_document_bytes_stds = [], []
    shard_text_bytes_means, shard_text_bytes_stds = [], []
    examples = []

    for shard_metadata in all_shard_metadata:
        shard_file_paths.append(shard_metadata["file_path"])
        shard_file_size_bytes.append(shard_metadata["file_size_bytes"])
        shard_n_documents.append(shard_metadata["n_documents"])

        shard_document_bytes_means.append(shard_metadata["document_bytes_mean"])
        shard_document_bytes_stds.append(shard_metadata["document_bytes_std"])

        shard_text_bytes_means.append(shard_metadata["text_bytes_mean"])
        shard_text_bytes_stds.append(shard_metadata["text_bytes_std"])

        examples.extend(shard_metadata["examples"])

    # Reconstitute Global Mean/Standard Deviations
    global_document_bytes_mean, global_document_bytes_std = compute_global_mean_std(
        shard_n_documents, shard_document_bytes_means, shard_document_bytes_stds
    )

    global_text_bytes_mean, global_text_bytes_std = compute_global_mean_std(
        shard_n_documents, shard_text_bytes_means, shard_text_bytes_stds
    )

    # Write `metadata.json`
    with fsspec.open(metadata_output_path, "wt") as f:
        json.dump(
            {
                "n_documents": sum(shard_n_documents),
                "file_size_bytes": sum(shard_file_size_bytes),
                "document_bytes_mean": global_document_bytes_mean,
                "document_bytes_std": global_document_bytes_std,
                "text_bytes_mean": global_text_bytes_mean,
                "text_bytes_std": global_text_bytes_std,
                "n_shards": len(shard_n_documents),
                "shard_file_paths": shard_file_paths,
                "shard_file_size_bites": shard_file_size_bytes,
                "shard_n_documents": shard_n_documents,
                "examples": examples,
            },
            f,
            indent=2,
        )

    return True


@draccus.wrap()
def validate(cfg: ValidationConfig) -> None:
    logger.info(f"Validating Dolma-Formatted Documents at `{cfg.input_path}`")

    # Initialize Connection to Cluster
    ray.init()

    # Identify the complete set of files to validate; note that this doesn't explicitly check subdirectory structure!
    shard_files = fsspec_glob(os.path.join(cfg.input_path, "**/*.jsonl.gz"))
    n_examples_per_shard = (cfg.n_examples // len(shard_files)) + 1

    # Invoke Ray Functions --> we're going to aggregate the function outputs
    responses: list[ray.ObjectRef] = []
    for shard_file_path in shard_files:
        responses.append(validate_jsonl_gz.remote(shard_file_path, n_examples_per_shard))

    # Wait on Success --> compute global mean/std, write top-level `metadata.json`
    try:
        all_shard_metadata = ray.get(responses)
        write_global_metadata(os.path.join(cfg.input_path, "metadata.json"), all_shard_metadata)
    except Exception as e:
        logger.error(f"Error Aggregating Validation Metadata: {e}")


if __name__ == "__main__":
    validate()
