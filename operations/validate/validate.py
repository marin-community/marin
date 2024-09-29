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
from marin.utils import fsspec_exists, fsspec_glob, fsspec_rm, fsspec_size

# Initialize Logger
logger = logging.getLogger(__name__)


@dataclass
class ValidationConfig:
    # fmt: off
    input_path: str                             # Path to top-level `documents` directory to validate
    num_examples_to_sample: int = 1024          # Number of Documents to Sample for Example Server (across all shards)

    overwrite_validation_cache: bool = False    # Whether to invalidate existing ledgers/metadata files (re-validate)

    # fmt: on


@ray.remote(memory=4 * 1024 * 1024 * 1024)  # 4 GB of RAM by Default
def validate_jsonl_gz(input_file_path: str, num_samples_per_shard: int, overwrite_validation_cache: bool) -> dict:
    # Short-Circuit on Metadata File =>> this is basically an extended version of @cached_or_construct_output with
    #   extra (custom) logic for "invalidating" ledgers
    success_file_path = input_file_path + ".METADATA"
    if fsspec_exists(success_file_path) and overwrite_validation_cache:
        logger.info(f"Metadata for `{input_file_path = }` exists but `{overwrite_validation_cache = }`; unlink")
        fsspec_rm(success_file_path)

    elif fsspec_exists(success_file_path):
        logger.info(f"Metadata for `{input_file_path = }` exists; skipping validation!")
        with fsspec.open(success_file_path, "rt") as f:
            return json.load(f)

    # Create Per-Document Metadata Trackers
    num_documents, all_document_bytes, all_text_bytes = 0, [], []

    # Create Reservoir for Sampling Example Documents & Seed Random Number Generator (for determinism)
    #
    # We're going to try our best to uniformly sample `num_example_per_shard_documents` from each `.jsonl.gz` file
    # Rather than precompute total number of documents/sample up front, we're going to reservoir sample; for each new
    # document we process, decide whether to include it with decreasing probability over time.
    #
    # Note (@siddk): This is very overkill for the current implementation, but I originally wrote this to smoothly
    #   handle preemption *during* the course of reading an individual shard file (some shard files are *huge*);
    #   leaving it as is in case we want to try getting that working later on...
    rng = np.random.default_rng(abs(hash(input_file_path)))
    example_reservoir = []

    # Lazily iterate + validate individual lines (documents) in the `.jsonl.gz` file
    with fsspec.open(input_file_path, "rt", compression="gzip") as input_jsonl_gz:
        for doc_idx, line in enumerate(input_jsonl_gz):
            document_footprint = summarize_document_from_json(line)
            document_bytes, text_bytes = document_footprint.document_bytes, document_footprint.text_bytes

            all_document_bytes.append(document_bytes)
            all_text_bytes.append(text_bytes)
            num_documents += 1

            # Reservoir Sampling
            if doc_idx < num_samples_per_shard:
                example_reservoir.append(json.loads(line))
            else:
                j = rng.integers(0, doc_idx)
                if j < num_samples_per_shard:
                    example_reservoir[j] = json.loads(line)

    # Compute Document Size Summary Statistics (Mean, Standard Deviation)
    all_document_bytes, all_text_bytes = np.asarray(all_document_bytes), np.asarray(all_text_bytes)
    doc_bytes_mean, doc_bytes_std = np.mean(all_document_bytes), np.std(all_document_bytes)
    text_bytes_mean, text_bytes_std = np.mean(all_text_bytes), np.std(all_text_bytes)
    file_metadata = {
        "file_path": input_file_path,
        "file_size_bytes": fsspec_size(input_file_path),
        "num_documents": num_documents,
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


def write_global_metadata(
    metadata_output_path: str, all_shard_metadata: list[dict], overwrite_validation_cache: bool
) -> bool:
    # Short-Circuit on Metadata File (respecting `overwrite_validation_cache`)
    if fsspec_exists(metadata_output_path) and overwrite_validation_cache:
        logger.info(f"Global Metadata `{metadata_output_path = }` exists but `{overwrite_validation_cache}`; unlink")
        fsspec_rm(metadata_output_path)

    elif fsspec_exists(metadata_output_path):
        logger.info(f"Global Metadata `{metadata_output_path = }` exists; skipping aggregation")
        return True

    # Aggregate Summary Statistics & Examples across Shards
    shard_file_paths, shard_file_size_bytes, shard_num_documents = [], [], []
    shard_document_bytes_means, shard_document_bytes_stds = [], []
    shard_text_bytes_means, shard_text_bytes_stds = [], []
    examples = []

    for shard_metadata in all_shard_metadata:
        shard_file_paths.append(shard_metadata["file_path"])
        shard_file_size_bytes.append(shard_metadata["file_size_bytes"])
        shard_num_documents.append(shard_metadata["num_documents"])

        shard_document_bytes_means.append(shard_metadata["document_bytes_mean"])
        shard_document_bytes_stds.append(shard_metadata["document_bytes_std"])

        shard_text_bytes_means.append(shard_metadata["text_bytes_mean"])
        shard_text_bytes_stds.append(shard_metadata["text_bytes_std"])

        examples.extend(shard_metadata["examples"])

    # Reconstitute Global Mean/Standard Deviations
    document_bytes_stats = compute_global_mean_std(
        shard_num_documents, shard_document_bytes_means, shard_document_bytes_stds
    )
    document_bytes_mean, document_bytes_std = document_bytes_stats.mean, document_bytes_stats.std

    text_bytes_stats = compute_global_mean_std(shard_num_documents, shard_text_bytes_means, shard_text_bytes_stds)
    text_bytes_mean, text_bytes_std = text_bytes_stats.mean, text_bytes_stats.std

    # Write `metadata.json`
    with fsspec.open(metadata_output_path, "wt") as f:
        json.dump(
            {
                "num_documents": sum(shard_num_documents),
                "file_size_bytes": sum(shard_file_size_bytes),
                "document_bytes_mean": document_bytes_mean,
                "document_bytes_std": document_bytes_std,
                "text_bytes_mean": text_bytes_mean,
                "text_bytes_std": text_bytes_std,
                "num_shards": len(shard_num_documents),
                "shard_file_paths": shard_file_paths,
                "shard_file_size_bites": shard_file_size_bytes,
                "shard_num_documents": shard_num_documents,
                "examples": examples,
            },
            f,
            indent=2,
        )

    return True


@draccus.wrap()
def validate(cfg: ValidationConfig) -> None:
    logger.info(f"Validating Dolma-Formatted Documents at `{cfg.input_path}`")

    # Identify the complete set of files to validate
    #
    # Note (@siddk): Right now this will match on *any* `.jsonl.gz` file in the directory, rather than verifying all
    #   files live at the same level of the directory tree. This could be a potential source of bugs later!
    shard_files = fsspec_glob(os.path.join(cfg.input_path, "**/*.jsonl.gz"))
    num_samples_per_shard = (cfg.num_examples_to_sample // len(shard_files)) + 1

    # Invoke Ray Functions --> we're going to aggregate the function outputs
    responses: list[ray.ObjectRef] = []
    for shard_file_path in shard_files:
        responses.append(
            validate_jsonl_gz.remote(shard_file_path, num_samples_per_shard, cfg.overwrite_validation_cache)
        )

    # Wait on Success --> compute global mean/std, write top-level `metadata.json`
    try:
        all_shard_metadata = ray.get(responses)
        write_global_metadata(
            os.path.join(cfg.input_path, "metadata.json"), all_shard_metadata, cfg.overwrite_validation_cache
        )
    except Exception as e:
        logger.error(f"Error Aggregating Validation Metadata: {e}")


if __name__ == "__main__":

    # Launch Validation Jobs
    validate()
