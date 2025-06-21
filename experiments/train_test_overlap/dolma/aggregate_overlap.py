#!/usr/bin/env python3
"""
aggregate_overlap.py

Aggregate n-gram overlap results across multiple evaluation datasets,
using the Ray backpressure architecture. Updated to handle the new
directory structure where each n-gram size has its own subdirectory.
"""
import csv
import json
import os
from collections import defaultdict
from dataclasses import dataclass

import fsspec

from experiments.train_test_overlap.dolma.debug_sharded_parquet import BASE_DEDUPE_CONFIG, dedupe_step
from marin.execution.executor import ExecutorStep, executor_main, this_output_path
from marin.utils import fsspec_glob, fsspec_mkdirs


@dataclass(frozen=True)
class AggregateOverlapConfig:
    # Input path to the dedupe step outputs
    input_path: str
    # Output path for aggregated results
    output_path: str = ""


def get_ngram_sizes():
    """Extract n-gram sizes from the base dedupe config."""
    ngram_length = BASE_DEDUPE_CONFIG.ngram.ngram_length
    if isinstance(ngram_length, list):
        return tuple(ngram_length)
    else:
        return (ngram_length,)


def get_attribute_name():
    """Get the base attribute name from the dedupe config."""
    return BASE_DEDUPE_CONFIG.attribute_name


def discover_datasets_from_outputs(input_path: str, ngram_size: int) -> dict[str, list[str]]:
    """
    Discover evaluation datasets from dedupe output directory structure.
    Updated to handle the new structure with n-gram-specific subdirectories.

    Directory structure:
    input_path/debug/train-00000-of-00128/{ngram_size}/{attribute_name}/mmlu-9fbdd5/

    Args:
        input_path: Base path to dedupe outputs
        ngram_size: N-gram size to look for

    Returns:
        Dict mapping dataset_name -> list of dataset_dirs across all shards
    """
    dataset_to_dirs = defaultdict(list)

    print(f"[DEBUG] Discovering datasets for {ngram_size}-gram in {input_path}", flush=True)

    # Get attribute name to look for
    base_attr_name = get_attribute_name()
    ngram_sizes = get_ngram_sizes()

    # Look for debug/ subdirectories
    debug_pattern = f"{input_path.rstrip('/')}/debug/*"
    shard_dirs = fsspec_glob(debug_pattern)
    print(f"[DEBUG] Found {len(shard_dirs)} shard directories under debug/", flush=True)

    if not shard_dirs:
        print("[DEBUG] No debug/ structure found", flush=True)
        return dict(dataset_to_dirs)

    for shard_dir in shard_dirs:
        print(f"[DEBUG] Scanning shard directory: {shard_dir}", flush=True)
        # Look for the n-gram size specific subdirectory
        ngram_dir = f"{shard_dir.rstrip('/')}/{ngram_size}"
        print(f"[DEBUG] Looking for n-gram directory: {ngram_dir}", flush=True)
        # Check if this ngram directory exists
        try:
            fs = fsspec.get_filesystem_class("gs")()
            if not fs.exists(ngram_dir):
                print(f"[DEBUG] n-gram directory {ngram_dir} does not exist", flush=True)
                continue
        except Exception as e:
            print(f"[DEBUG] Error checking n-gram directory {ngram_dir}: {e}", flush=True)
            continue
        # Find all dataset directories directly under this ngram directory
        dataset_pattern = f"{ngram_dir.rstrip('/')}/*"
        print(f"[DEBUG] Looking for datasets with pattern: {dataset_pattern}", flush=True)
        all_items = fsspec_glob(dataset_pattern)
        print(f"[DEBUG] Found {len(all_items)} items under {ngram_dir}", flush=True)
        if not all_items:
            continue
        # Filter to only directories
        dataset_dirs = []
        try:
            fs = fsspec.get_filesystem_class("gs")()
            for item in all_items:
                try:
                    if fs.isdir(item):
                        dataset_dirs.append(item)
                        print(f"[DEBUG] Confirmed dataset directory: {os.path.basename(item)}", flush=True)
                except Exception as e:
                    print(f"[DEBUG] Error checking {item}: {e}", flush=True)
        except Exception as e:
            print(f"[DEBUG] Falling back to name-based detection due to: {e}", flush=True)
            dataset_dirs = [item for item in all_items if "." not in os.path.basename(item)]

        # Process each dataset directory
        for dataset_dir in dataset_dirs:
            dir_name = os.path.basename(dataset_dir.rstrip("/"))
            print(f"[DEBUG] Processing dataset directory: {dir_name}", flush=True)

            # Skip executor metadata
            if dir_name.startswith("."):
                print(f"[DEBUG] Skipping metadata directory: {dir_name}", flush=True)
                continue

            # Extract dataset name from directory (e.g., "mmlu-9fbdd5" -> "mmlu")
            dataset_name = dir_name.split("-")[0]
            dataset_to_dirs[dataset_name].append(dataset_dir)
            print(f"[DEBUG] Added {dataset_name} -> {dataset_dir}", flush=True)

    # Log discovered datasets
    for dataset_name, dirs in dataset_to_dirs.items():
        print(f"[DEBUG] Dataset '{dataset_name}': {len(dirs)} directories for {ngram_size}-gram", flush=True)

    return dict(dataset_to_dirs)


def aggregate_single_dataset(dataset_name: str, dataset_dirs: list[str], ngram_size: int, base_attr_name: str) -> dict:
    """
    Aggregate overlap results for a single dataset across all training shards.

    Args:
        dataset_name: Name of the dataset (e.g., "mmlu")
        dataset_dirs: List of directories containing this dataset's results
        ngram_size: N-gram size
        base_attr_name: Base attribute name (e.g., "ngram_overlap")

    Returns:
        Dict with aggregation results
    """
    # Use n-gram size specific attribute name for multiple n-gram sizes
    ngram_sizes = get_ngram_sizes()
    if len(ngram_sizes) > 1:
        attr_key = f"{base_attr_name}_{ngram_size}"
    else:
        attr_key = base_attr_name

    print(f"[DEBUG] Aggregating {dataset_name} for {ngram_size}-gram, attribute key={attr_key}", flush=True)

    id_to_shards = defaultdict(list)
    total_test = 0
    overlapped_ids = set()

    for dataset_dir in dataset_dirs:
        print(f"[DEBUG] Processing dataset directory: {dataset_dir}", flush=True)

        # Find all JSONL files recursively under this dataset directory
        rec_paths = fsspec_glob(f"{dataset_dir.rstrip('/')}/**/*.jsonl*")
        print(f"[DEBUG] Found {len(rec_paths)} JSONL files recursively in {dataset_dir}", flush=True)

        if not rec_paths:
            print(f"[DEBUG] No JSONL files found in {dataset_dir}", flush=True)
            continue

        overlapped_in_shard = set()

        # Process all JSONL files in this dataset directory
        for rec_path in rec_paths:
            print(f"[DEBUG] Processing file: {rec_path}", flush=True)
            try:
                with fsspec.open(rec_path, "rt", compression="infer") as f:
                    line_count = 0
                    overlaps_in_file = 0
                    for line in f:
                        line_count += 1
                        total_test += 1
                        rec = json.loads(line)

                        # Check if the overlap attribute list is non-empty
                        overlap_val = rec.get("attributes", {}).get(attr_key, [])

                        if line_count <= 5:  # Debug first few records
                            print(
                                f"[DEBUG] Record {line_count}: id={rec.get('id', 'NO_ID')}, {attr_key}={overlap_val}",
                                flush=True,
                            )

                        # Non-empty list means overlap
                        if overlap_val:  # This checks if list is non-empty
                            overlapped_ids.add(rec["id"])
                            overlapped_in_shard.add(rec["id"])
                            id_to_shards[rec["id"]].append(dataset_dir)
                            overlaps_in_file += 1
                            if overlaps_in_file <= 3:  # Debug first few overlaps
                                print(
                                    f"[DEBUG] Found overlap! id={rec['id']}, {attr_key}={overlap_val} in {rec_path}",
                                    flush=True,
                                )

                    print(
                        f"[DEBUG] Processed {line_count} lines from {rec_path}, found {overlaps_in_file} overlaps",
                        flush=True,
                    )

            except Exception as e:
                print(f"[DEBUG] Error processing {rec_path}: {e}", flush=True)
                continue

        shard_name = os.path.basename(dataset_dir.rstrip("/"))
        print(f"[DEBUG] Shard {shard_name}: processed, overlapped_count={len(overlapped_in_shard)}", flush=True)

    # Calculate metrics
    overlap_count = len(overlapped_ids)
    overlap_fraction = overlap_count / total_test if total_test > 0 else 0.0

    print(
        f"[DEBUG] Dataset {dataset_name} ({ngram_size}-gram): {overlap_count}/{total_test} = {overlap_fraction:.4f}",
        flush=True,
    )

    return {
        "dataset_name": dataset_name,
        "ngram_size": ngram_size,
        "total_test": total_test,
        "overlap_count": overlap_count,
        "overlap_fraction": overlap_fraction,
        "overlapped_ids": overlapped_ids,
        "id_to_shards": dict(id_to_shards),
    }


def aggregate_overlap(config: AggregateOverlapConfig):
    """Main aggregation function with path-based dataset discovery."""

    print("[DEBUG] Starting aggregate_overlap", flush=True)
    print(f"[DEBUG] Input path: {config.input_path}", flush=True)

    # Read configuration from base dedupe config
    ngram_sizes = get_ngram_sizes()
    base_attr_name = get_attribute_name()

    print(f"[DEBUG] Using ngram sizes: {ngram_sizes}", flush=True)
    print(f"[DEBUG] Using base attribute name: {base_attr_name}", flush=True)

    # Process each n-gram size
    for ngram_size in ngram_sizes:
        print(f"[INFO] Processing {ngram_size}-gram overlaps", flush=True)

        # Discover datasets for this n-gram size
        dataset_to_dirs = discover_datasets_from_outputs(config.input_path, ngram_size)

        if not dataset_to_dirs:
            print(f"[WARNING] No datasets discovered for {ngram_size}-gram!", flush=True)
            continue

        print(
            f"[INFO] Discovered {len(dataset_to_dirs)} datasets for {ngram_size}-gram: {list(dataset_to_dirs.keys())}",
            flush=True,
        )

        # Aggregate each dataset separately
        dataset_results = {}
        combined_overlapped_ids = set()
        combined_total_test = 0

        for dataset_name, dataset_dirs in dataset_to_dirs.items():
            print(f"[DEBUG] Starting aggregation for dataset: {dataset_name}", flush=True)
            result = aggregate_single_dataset(dataset_name, dataset_dirs, ngram_size, base_attr_name)
            dataset_results[dataset_name] = result

            # Add to combined statistics
            combined_overlapped_ids.update(result["overlapped_ids"])
            combined_total_test += result["total_test"]
            print(
                f"[DEBUG] Completed {dataset_name}: {result['overlap_count']} overlaps from {result['total_test']} examples",
                flush=True,
            )

        # Create hierarchical output structure
        base_out = config.output_path.rstrip("/")

        # Per-dataset results
        for dataset_name, result in dataset_results.items():
            dataset_out_dir = os.path.join(base_out, "by_dataset", dataset_name, str(ngram_size))
            fsspec_mkdirs(dataset_out_dir)

            # CSV: dataset-specific overlap fraction
            with fsspec.open(os.path.join(dataset_out_dir, "fractions.csv"), "wt") as cf:
                writer = csv.writer(cf)
                writer.writerow(["total_examples", "overlap_count", "overlap_fraction"])
                writer.writerow([result["total_test"], result["overlap_count"], result["overlap_fraction"]])

            # JSONL: dataset-specific ID to shards mapping
            with fsspec.open(os.path.join(dataset_out_dir, "id_to_shards.jsonl"), "wt") as jf:
                for _id, shards_list in result["id_to_shards"].items():
                    jf.write(json.dumps({"id": _id, "shards": shards_list}) + "\n")

            print(f"[INFO] Wrote {dataset_name} {ngram_size}-gram results to {dataset_out_dir}", flush=True)

        # Combined results across all datasets
        combined_out_dir = os.path.join(base_out, "combined", str(ngram_size))
        fsspec_mkdirs(combined_out_dir)

        combined_overlap_fraction = (
            len(combined_overlapped_ids) / combined_total_test if combined_total_test > 0 else 0.0
        )

        # Combined CSV
        with fsspec.open(os.path.join(combined_out_dir, "fractions.csv"), "wt") as cf:
            writer = csv.writer(cf)
            writer.writerow(["total_examples", "overlap_count", "combined_overlap_fraction"])
            writer.writerow([combined_total_test, len(combined_overlapped_ids), combined_overlap_fraction])

        print(
            f"[INFO] Combined {ngram_size}-gram results: {len(combined_overlapped_ids)}/{combined_total_test} = {combined_overlap_fraction:.4f}",
            flush=True,
        )

        # Summary table with per-dataset breakdown
        summary_out_dir = os.path.join(base_out, "summary")
        fsspec_mkdirs(summary_out_dir)

        with fsspec.open(os.path.join(summary_out_dir, f"per_dataset_{ngram_size}gram.csv"), "wt") as sf:
            writer = csv.writer(sf)
            writer.writerow(["dataset", "total_examples", "overlap_count", "overlap_fraction"])

            # Sort by overlap fraction descending
            sorted_results = sorted(dataset_results.items(), key=lambda x: x[1]["overlap_fraction"], reverse=True)

            for dataset_name, result in sorted_results:
                writer.writerow(
                    [dataset_name, result["total_test"], result["overlap_count"], result["overlap_fraction"]]
                )

        print(f"[INFO] Wrote summary table to {summary_out_dir}/per_dataset_{ngram_size}gram.csv", flush=True)


# Create the configuration and executor step
cfg = AggregateOverlapConfig(
    input_path=dedupe_step,  # Upstream step: executor will replace with its output_path
    output_path=this_output_path(),
)

aggregate_step = ExecutorStep(
    name="train_test_overlap/dolma/aggregate_overlap",
    fn=aggregate_overlap,
    config=cfg,
)

if __name__ == "__main__":
    executor_main(
        # run sharded dedupe first, then aggregation
        steps=[dedupe_step, aggregate_step],
        description="Aggregate n-gram overlap across all discovered evaluation datasets",
    )
