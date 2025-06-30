#!/usr/bin/env python3
"""
aggregate_total_overlap.py

Aggregate n-gram overlap results across multiple training datasets and evaluation datasets,
creating matrices that show contamination percentages per training dataset and overall.
"""
import csv
import json
import os
from collections import defaultdict
from dataclasses import dataclass

import fsspec

from experiments.train_test_overlap.dolma.dedupe_total import (
    finemath_dedupe_step,
    proofpile_dedupe_step,
)
from experiments.train_test_overlap.utils import find_dataset_shards, get_relative_path_no_extension
from marin.execution.executor import ExecutorStep, executor_main, this_output_path
from marin.utils import fsspec_mkdirs


@dataclass(frozen=True)
class AggregateTotalOverlapConfig:
    # Input paths to the dedupe step outputs (list of paths, one per training dataset)
    input_paths: list[str] = None
    # Training dataset names corresponding to input_paths (optional, will be inferred if not provided)
    training_dataset_names: list[str] = None
    # Output path for aggregated results
    output_path: str = ""
    # N-gram sizes to process (defaults will be discovered from directory structure)
    ngram_sizes: list[int] = None
    # Base attribute name (defaults will be discovered from directory structure)
    attribute_name: str = "ngram_overlap"


def discover_training_datasets_from_paths(input_paths: list[str], training_dataset_names: list[str] = None) -> list[str]:
    """
    Get training dataset names from input paths.

    Args:
        input_paths: List of paths to dedupe outputs
        training_dataset_names: Optional list of explicit names, otherwise inferred from paths
    """
    if training_dataset_names:
        if len(training_dataset_names) != len(input_paths):
            raise ValueError("training_dataset_names length must match input_paths length")
        return training_dataset_names

    # Infer from path names
    training_datasets = []
    for path in input_paths:
        # Extract name from path like "train_test_overlap/dolma/total_finemath" -> "finemath"
        path_parts = path.rstrip("/").split("/")
        if path_parts and path_parts[-1].startswith("total_"):
            name = path_parts[-1][6:]  # Remove "total_" prefix
        else:
            name = path_parts[-1] if path_parts else "unknown"
        training_datasets.append(name)

    print(f"[DEBUG] Inferred training datasets: {training_datasets}")
    return training_datasets


def discover_ngram_sizes_from_paths(input_paths: list[str]) -> list[int]:
    """Discover available n-gram sizes from multiple input paths."""
    print(f"[DEBUG] Discovering n-gram sizes from {len(input_paths)} input paths")

    # Gather all n-gram sizes from shard file paths under each training-output root
    ngram_sizes = set()
    for input_path in input_paths:
        for shard_path in find_dataset_shards(input_path):
            rel = get_relative_path_no_extension(shard_path, input_path)
            for seg in rel.split(os.sep):
                if seg.isdigit():
                    ngram_sizes.add(int(seg))

    ngram_sizes = sorted(list(ngram_sizes))
    print(f"[DEBUG] Discovered n-gram sizes: {ngram_sizes}")
    return ngram_sizes


def discover_test_datasets_from_paths(input_paths: list[str], ngram_size: int) -> set[str]:
    """Discover all test datasets for a given n-gram size across all input paths."""
    print(f"[DEBUG] Discovering test datasets for {ngram_size}-gram across {len(input_paths)} paths")

    # Discover test datasets by scanning all attribute shard files under each training-output root
    test_datasets = set()
    for input_path in input_paths:
        for shard_path in find_dataset_shards(input_path):
            rel = get_relative_path_no_extension(shard_path, input_path)
            parts = rel.split(os.sep)
            if str(ngram_size) not in parts:
                continue
            # basename without extension encodes the test dataset
            file_base = parts[-1]
            test_dataset = file_base.split("-")[0]
            test_datasets.add(test_dataset)

    test_datasets = sorted(list(test_datasets))
    print(f"[DEBUG] Discovered test datasets for {ngram_size}-gram: {test_datasets}")
    return test_datasets


def aggregate_training_dataset_overlap_from_path(
    input_path: str, training_dataset: str, ngram_size: int, attribute_name: str
) -> dict[str, dict]:
    """
    Aggregate overlap results for a single training dataset from a specific input path.

    Returns:
        Dict mapping test_dataset_name -> {total_test, overlap_count, overlap_fraction, overlapped_ids}
    """
    print(f"[DEBUG] Aggregating {training_dataset} for {ngram_size}-gram from {input_path}")

    # Build attribute key and initialize results
    attr_key = f"{attribute_name}_{ngram_size}"
    test_dataset_results = defaultdict(
        lambda: {"total_test": 0, "overlapped_ids": set(), "id_to_shards": defaultdict(list)}
    )
    # Iterate through all attribute shard files under this training-output root
    for shard_path in find_dataset_shards(input_path):
        rel = get_relative_path_no_extension(shard_path, input_path)
        parts = rel.split(os.sep)
        if str(ngram_size) not in parts:
            continue
        shard_name = parts[0]
        file_base = parts[-1]
        test_dataset_name = file_base.split("-")[0]
        try:
            with fsspec.open(shard_path, "rt", compression="infer") as f:
                for line in f:
                    rec = json.loads(line)
                    test_dataset_results[test_dataset_name]["total_test"] += 1
                    overlap_val = rec.get("attributes", {}).get(attr_key, [])
                    if overlap_val:
                        test_dataset_results[test_dataset_name]["overlapped_ids"].add(rec["id"])
                        test_dataset_results[test_dataset_name]["id_to_shards"][rec["id"]].append(shard_name)
        except Exception as e:
            print(f"[DEBUG] Error processing {shard_path}: {e}")
            continue

    # Calculate final results
    final_results = {}
    for test_dataset_name, data in test_dataset_results.items():
        overlap_count = len(data["overlapped_ids"])
        total_test = data["total_test"]
        overlap_fraction = overlap_count / total_test if total_test > 0 else 0.0

        final_results[test_dataset_name] = {
            "total_test": total_test,
            "overlap_count": overlap_count,
            "overlap_fraction": overlap_fraction,
            "overlapped_ids": data["overlapped_ids"],
            "id_to_shards": dict(data["id_to_shards"]),
        }

        print(
            f"[DEBUG] {training_dataset} -> {test_dataset_name}: {overlap_count}/{total_test} = {overlap_fraction:.4f}"
        )

    return final_results


def aggregate_total_overlap(config: AggregateTotalOverlapConfig):
    """Main aggregation function that creates matrices for training vs test dataset contamination."""

    print("[DEBUG] Starting aggregate_total_overlap")
    print(f"[DEBUG] Input paths: {config.input_paths}")

    if not config.input_paths:
        print("[ERROR] No input paths provided!")
        return

    # Get training dataset names
    training_datasets = discover_training_datasets_from_paths(config.input_paths, config.training_dataset_names)

    # Discover n-gram sizes from all paths
    ngram_sizes = config.ngram_sizes or discover_ngram_sizes_from_paths(config.input_paths)
    if not ngram_sizes:
        print("[ERROR] No n-gram sizes discovered!")
        return

    print(f"[INFO] Processing {len(training_datasets)} training datasets and {len(ngram_sizes)} n-gram sizes")

    # Process each n-gram size
    for ngram_size in ngram_sizes:
        print(f"[INFO] Processing {ngram_size}-gram overlaps")

        # Discover test datasets for this n-gram size
        test_datasets = discover_test_datasets_from_paths(config.input_paths, ngram_size)
        if not test_datasets:
            print(f"[WARNING] No test datasets found for {ngram_size}-gram")
            continue

        print(f"[INFO] Found {len(test_datasets)} test datasets for {ngram_size}-gram: {test_datasets}")

        # Aggregate results for each training dataset
        training_results = {}
        for i, training_dataset in enumerate(training_datasets):
            input_path = config.input_paths[i]
            training_results[training_dataset] = aggregate_training_dataset_overlap_from_path(
                input_path, training_dataset, ngram_size, config.attribute_name
            )

        # Create output directory for this n-gram size
        ngram_output_dir = os.path.join(config.output_path, str(ngram_size))
        fsspec_mkdirs(ngram_output_dir)

        # Create contamination matrix (training dataset vs test dataset)
        contamination_matrix_path = os.path.join(ngram_output_dir, "contamination_matrix.csv")
        with fsspec.open(contamination_matrix_path, "wt") as f:
            writer = csv.writer(f)

            # Header: training datasets as columns
            header = ["test_dataset", *training_datasets]
            writer.writerow(header)

            # Rows: test datasets
            for test_dataset in test_datasets:
                row = [test_dataset]
                for training_dataset in training_datasets:
                    if test_dataset in training_results[training_dataset]:
                        fraction = training_results[training_dataset][test_dataset]["overlap_fraction"]
                        row.append(f"{fraction:.6f}")
                    else:
                        row.append("0.000000")
                writer.writerow(row)

        print(f"[INFO] Created contamination matrix: {contamination_matrix_path}")

        # Create overall contamination summary (what % of each test dataset is contaminated across all training datasets)
        overall_contamination = {}
        for test_dataset in test_datasets:
            all_overlapped_ids = set()
            total_test_examples = 0

            for training_dataset in training_datasets:
                if test_dataset in training_results[training_dataset]:
                    result = training_results[training_dataset][test_dataset]
                    all_overlapped_ids.update(result["overlapped_ids"])
                    total_test_examples = max(total_test_examples, result["total_test"])

            overall_fraction = len(all_overlapped_ids) / total_test_examples if total_test_examples > 0 else 0.0
            overall_contamination[test_dataset] = {
                "total_test": total_test_examples,
                "overall_overlap_count": len(all_overlapped_ids),
                "overall_overlap_fraction": overall_fraction,
            }

        # Write overall contamination summary
        overall_summary_path = os.path.join(ngram_output_dir, "overall_contamination.csv")
        with fsspec.open(overall_summary_path, "wt") as f:
            writer = csv.writer(f)
            writer.writerow(["test_dataset", "total_examples", "contaminated_examples", "contamination_fraction"])

            # Sort by contamination fraction descending
            sorted_datasets = sorted(
                overall_contamination.items(), key=lambda x: x[1]["overall_overlap_fraction"], reverse=True
            )

            for test_dataset, data in sorted_datasets:
                writer.writerow(
                    [
                        test_dataset,
                        data["total_test"],
                        data["overall_overlap_count"],
                        f"{data['overall_overlap_fraction']:.6f}",
                    ]
                )

        print(f"[INFO] Created overall contamination summary: {overall_summary_path}")

        # Create detailed per-training-dataset results
        detailed_dir = os.path.join(ngram_output_dir, "detailed")
        fsspec_mkdirs(detailed_dir)

        for training_dataset in training_datasets:
            training_dir = os.path.join(detailed_dir, training_dataset)
            fsspec_mkdirs(training_dir)

            # Summary CSV for this training dataset
            summary_path = os.path.join(training_dir, "summary.csv")
            with fsspec.open(summary_path, "wt") as f:
                writer = csv.writer(f)
                writer.writerow(["test_dataset", "total_examples", "overlap_count", "overlap_fraction"])

                for test_dataset, result in training_results[training_dataset].items():
                    writer.writerow(
                        [
                            test_dataset,
                            result["total_test"],
                            result["overlap_count"],
                            f"{result['overlap_fraction']:.6f}",
                        ]
                    )

            # Detailed ID mappings for this training dataset
            for test_dataset, result in training_results[training_dataset].items():
                if result["overlap_count"] > 0:
                    id_mapping_path = os.path.join(training_dir, f"{test_dataset}_overlaps.jsonl")
                    with fsspec.open(id_mapping_path, "wt") as f:
                        for _id, shards in result["id_to_shards"].items():
                            f.write(
                                json.dumps(
                                    {
                                        "id": _id,
                                        "test_dataset": test_dataset,
                                        "training_dataset": training_dataset,
                                        "shards": shards,
                                    }
                                )
                                + "\n"
                            )

        print(f"[INFO] Created detailed results in {detailed_dir}")

        # Print summary to console
        print(f"\n[SUMMARY] {ngram_size}-gram Results:")
        print(f"Training datasets: {len(training_datasets)}")
        print(f"Test datasets: {len(test_datasets)}")
        print("\nOverall contamination rates:")
        for test_dataset, data in sorted_datasets:
            print(
                f"{test_dataset}: {data['overall_overlap_count']}/{data['total_test']} = {data['overall_overlap_fraction']:.4f}"
            )


cfg = AggregateTotalOverlapConfig(
    input_paths=[
        finemath_dedupe_step,
        # dclm_dedupe_step,
        # starcoder_dedupe_step,
        proofpile_dedupe_step,
        # nemotron_dedupe_step
    ],
    training_dataset_names=["finemath", "proofpile"],
    output_path=this_output_path(),
)

aggregate_total_step = ExecutorStep(
    name="train_test_overlap/dolma/aggregate_total_overlap",
    fn=aggregate_total_overlap,
    config=cfg,
)

if __name__ == "__main__":
    executor_main(
        steps=[
            finemath_dedupe_step,
            # #dclm_dedupe_step,
            # starcoder_dedupe_step,
            proofpile_dedupe_step,
            # nemotron_dedupe_step,
            aggregate_total_step,
        ],
        description="Aggregate n-gram overlap across all training and test datasets, creating contamination matrices",
    )
