#!/usr/bin/env python3
"""
aggregate_total_overlap.py

Aggregate n-gram overlap results across multiple evaluation datasets,
using path-based dataset discovery and hierarchical output structure.
"""
import csv
import json
import os
from collections import defaultdict
from dataclasses import dataclass

import fsspec

from experiments.train_test_overlap.dolma.debug_sharded_parquet import steps as dedupe_steps
from marin.execution.executor import ExecutorStep, executor_main, this_output_path
from marin.utils import fsspec_glob, fsspec_mkdirs


@dataclass(frozen=True)
class AggregateTotalOverlapConfig:
    # List of InputName references to dedupe step output paths to be scanned
    dedupe_steps: list[str]
    # n-gram sizes to aggregate (will be read from dedupe config)
    ngrams: tuple[int, ...] = (15,)  # Default, but will read from config
    # where to write aggregated outputs (Executor will fill this)
    output_path: str = ""


def get_ngram_config_from_dedupe_steps():
    """Extract ngram configuration from dedupe steps."""
    print(f"[DEBUG] Extracting ngram config from {len(dedupe_steps)} dedupe steps", flush=True)

    if dedupe_steps:
        # Get the first step's config to read ngram settings
        first_step = dedupe_steps[0]
        if hasattr(first_step, "config") and hasattr(first_step.config, "ngram"):
            ngram_length = first_step.config.ngram.ngram_length
            print(f"[DEBUG] Found ngram_length = {ngram_length} from dedupe config", flush=True)
            return (ngram_length,)

    print("[DEBUG] Could not extract ngram config, using default (15,)", flush=True)
    return (15,)


def discover_datasets_from_outputs(shard_outputs: list[str]) -> dict[str, list[str]]:
    """
    Discover evaluation datasets from dedupe output directory structure.

    Args:
        shard_outputs: List of output paths from dedupe steps

    Returns:
        Dict mapping dataset_name -> list of dataset_dirs across all shards
    """
    dataset_to_dirs = defaultdict(list)

    print(f"[DEBUG] Starting dataset discovery from {len(shard_outputs)} shard outputs", flush=True)

    for shard_output in shard_outputs:
        print(f"[DEBUG] Scanning shard output: {shard_output}", flush=True)

        # Find all dataset directories under this shard output
        # Try both with and without trailing slash for fsspec compatibility
        pattern1 = f"{shard_output.rstrip('/')}/*/"
        pattern2 = f"{shard_output.rstrip('/')}/*"
        print(f"[DEBUG] Trying glob patterns: {pattern1} and {pattern2}", flush=True)

        dataset_dirs = fsspec_glob(pattern1)
        if not dataset_dirs:
            print("[DEBUG] First pattern failed, trying second pattern", flush=True)
            all_items = fsspec_glob(pattern2)
            if not all_items:
                print("[DEBUG] Second pattern also failed", flush=True)
                dataset_dirs = []
            else:
                print(f"[DEBUG] Second pattern found {len(all_items)} items", flush=True)
                # Use fsspec to check which items are directories
                try:
                    fs = fsspec.get_filesystem_class("gs")()
                    dataset_dirs = []
                    for item in all_items:
                        try:
                            if fs.isdir(item):
                                dataset_dirs.append(item)
                                print(f"[DEBUG] Confirmed directory: {os.path.basename(item)}", flush=True)
                            else:
                                print(f"[DEBUG] Not a directory: {os.path.basename(item)}", flush=True)
                        except Exception as e:
                            print(f"[DEBUG] Error checking {item}: {e}", flush=True)
                except Exception as e:
                    print(f"[DEBUG] Falling back to name-based detection due to: {e}", flush=True)
                    # Fallback: assume items without extensions are directories
                    dataset_dirs = [item for item in all_items if "." not in os.path.basename(item)]

                print(f"[DEBUG] Final directory count: {len(dataset_dirs)}", flush=True)

        print(f"[DEBUG] Final glob results: {dataset_dirs[:5] if len(dataset_dirs) > 5 else dataset_dirs}", flush=True)

        for dataset_dir in dataset_dirs:
            dir_name = os.path.basename(dataset_dir.rstrip("/"))
            print(f"[DEBUG] Processing directory: {dir_name}", flush=True)

            # Skip executor metadata
            if dir_name.startswith("."):
                print(f"[DEBUG] Skipping metadata directory: {dir_name}", flush=True)
                continue

            # Extract dataset name from directory (e.g., "mmlu-9fbdd5" -> "mmlu")
            dataset_name = dir_name.split("-")[0]
            dataset_to_dirs[dataset_name].append(dataset_dir)
            print(f"[DEBUG] Added {dataset_name} -> {dataset_dir}", flush=True)

        print(
            f"[DEBUG] Found {len(dataset_dirs)} total dirs, processed {len([d for d in dataset_dirs if not os.path.basename(d.rstrip('/')).startswith('.')])} non-metadata dirs in {shard_output}",
            flush=True,
        )

    # Log discovered datasets
    for dataset_name, dirs in dataset_to_dirs.items():
        print(f"[DEBUG] Dataset '{dataset_name}': {len(dirs)} directories across shards", flush=True)

    return dict(dataset_to_dirs)


def aggregate_single_dataset(dataset_name: str, dataset_dirs: list[str], n: int) -> dict:
    """
    Aggregate overlap results for a single dataset across all training shards.

    Args:
        dataset_name: Name of the dataset (e.g., "mmlu")
        dataset_dirs: List of directories containing this dataset's results
        n: N-gram size (for reference, actual attribute is just "mmlu_overlap")

    Returns:
        Dict with aggregation results
    """
    # FIXED: Use just "mmlu_overlap" without the n-gram suffix
    attr_key = "mmlu_overlap"
    print(f"[DEBUG] Aggregating {dataset_name} for {n}-gram, attribute key={attr_key}", flush=True)

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

                        # FIXED: Check if mmlu_overlap list is non-empty
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
        f"[DEBUG] Dataset {dataset_name} ({n}-gram): {overlap_count}/{total_test} = {overlap_fraction:.4f}", flush=True
    )

    return {
        "dataset_name": dataset_name,
        "ngram_size": n,
        "total_test": total_test,
        "overlap_count": overlap_count,
        "overlap_fraction": overlap_fraction,
        "overlapped_ids": overlapped_ids,
        "id_to_shards": dict(id_to_shards),
    }


def aggregate_total_overlap(config: AggregateTotalOverlapConfig):
    """Main aggregation function with path-based dataset discovery."""

    print("[DEBUG] Starting aggregate_total_overlap", flush=True)

    # FIXED: Read ngram config from dedupe steps
    actual_ngrams = get_ngram_config_from_dedupe_steps()
    print(f"[DEBUG] Using ngram sizes: {actual_ngrams}", flush=True)

    # The config.dedupe_steps should be InputName objects pointing to the actual output paths
    # We'll get the resolved paths from the config
    shard_outputs = config.dedupe_steps

    print(f"[DEBUG] Processing {len(shard_outputs)} shard outputs", flush=True)
    print(f"[DEBUG] First few shard outputs: {shard_outputs[:3]}", flush=True)

    # Test the specific shard we know has data
    test_shard = (
        "gs://marin-us-central2/train_test_overlap/dolma/parquet_finemath3plus_dedupe/train-00127-of-00128-9f2666"
    )
    if test_shard in shard_outputs:
        print("[DEBUG] Found the test shard in our list!", flush=True)
    else:
        print("[DEBUG] Test shard not in our list. Checking if any shard ends with train-00127...", flush=True)
        matching = [s for s in shard_outputs if "train-00127" in s]
        print(f"[DEBUG] Found matching shards: {matching}", flush=True)

    # Discover datasets from output directory structure
    dataset_to_dirs = discover_datasets_from_outputs(shard_outputs)

    if not dataset_to_dirs:
        print("[WARNING] No datasets discovered from output directories!", flush=True)
        return

    print(f"[INFO] Discovered {len(dataset_to_dirs)} datasets: {list(dataset_to_dirs.keys())}", flush=True)

    # Process each n-gram size
    for n in actual_ngrams:  # FIXED: Use actual ngrams from config
        print(f"[INFO] Processing {n}-gram overlaps", flush=True)

        # Aggregate each dataset separately
        dataset_results = {}
        combined_overlapped_ids = set()
        combined_total_test = 0

        for dataset_name, dataset_dirs in dataset_to_dirs.items():
            print(f"[DEBUG] Starting aggregation for dataset: {dataset_name}", flush=True)
            result = aggregate_single_dataset(dataset_name, dataset_dirs, n)
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
            dataset_out_dir = os.path.join(base_out, "by_dataset", dataset_name, str(n))
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

            print(f"[INFO] Wrote {dataset_name} {n}-gram results to {dataset_out_dir}", flush=True)

        # Combined results across all datasets
        combined_out_dir = os.path.join(base_out, "combined", str(n))
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
            f"[INFO] Combined {n}-gram results: {len(combined_overlapped_ids)}/{combined_total_test} = {combined_overlap_fraction:.4f}",
            flush=True,
        )

        # Summary table with per-dataset breakdown
        summary_out_dir = os.path.join(base_out, "summary")
        fsspec_mkdirs(summary_out_dir)

        with fsspec.open(os.path.join(summary_out_dir, f"per_dataset_{n}gram.csv"), "wt") as sf:
            writer = csv.writer(sf)
            writer.writerow(["dataset", "total_examples", "overlap_count", "overlap_fraction"])

            # Sort by overlap fraction descending
            sorted_results = sorted(dataset_results.items(), key=lambda x: x[1]["overlap_fraction"], reverse=True)

            for dataset_name, result in sorted_results:
                writer.writerow(
                    [dataset_name, result["total_test"], result["overlap_count"], result["overlap_fraction"]]
                )

        print(f"[INFO] Wrote summary table to {summary_out_dir}/per_dataset_{n}gram.csv", flush=True)


# Configuration - convert ExecutorSteps to InputName references
dedupe_input_paths = [step.cd("") for step in dedupe_steps]  # Get InputName for each step's output

cfg = AggregateTotalOverlapConfig(
    dedupe_steps=dedupe_input_paths,
    ngrams=(15,),  # Will be overridden by config reading
    output_path=this_output_path(),
)

aggregate_step = ExecutorStep(
    name="train_test_overlap/dolma/aggregate_total_overlap",
    fn=aggregate_total_overlap,
    config=cfg,
)

if __name__ == "__main__":
    executor_main(
        steps=[aggregate_step],
        description="Aggregate n-gram overlap across all discovered evaluation datasets",
    )
