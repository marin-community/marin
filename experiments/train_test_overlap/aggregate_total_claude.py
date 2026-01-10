#!/usr/bin/env python3
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

"""aggregate_total_claude.py

Refactored version of aggregate_total.py with improved readability,
bug fixes, and performance optimizations.

Aggregate overlap across all discovered training datasets with two levels:
1. Per-training-dataset aggregation (using ALL shards)
2. Union aggregation across all training datasets
3. Overlap matrix showing training_datasets X evaluation_datasets

Auto-discovers training datasets from attribute directory structure.

Example Usage:
    python experiments/train_test_overlap/aggregate_total_claude.py --prefix gs://my-bucket
"""

import csv
import logging
import os
from collections.abc import Iterator
from dataclasses import dataclass, field

import fsspec
from marin.execution.executor import ExecutorStep, executor_main, this_output_path
from marin.utils import fsspec_glob
from zephyr import Backend, Dataset, load_file, load_jsonl

from experiments.train_test_overlap.eval_datasets_overlap import EVAL_DATASET_STEPS

logger = logging.getLogger(__name__)


###############################################################################
# Result Dataclasses
###############################################################################


@dataclass(frozen=True)
class ShardMetadata:
    """Metadata extracted from a shard path."""

    test_dataset: str
    shard_path: str


@dataclass(frozen=True)
class DatasetOverlapStats:
    """Overlap statistics for a dataset.

    Uses frozenset for immutability after aggregation is complete.
    """

    unique_ids: frozenset[str]
    overlap_ids: frozenset[str]

    @property
    def total(self) -> int:
        return len(self.unique_ids)

    @property
    def contaminated(self) -> int:
        return len(self.overlap_ids)

    @property
    def fraction(self) -> float:
        return self.contaminated / self.total if self.total else 0.0


@dataclass(frozen=True)
class AggregationResult:
    """Complete aggregation result for a training dataset."""

    training_name: str
    shard_count: int
    overall_stats: DatasetOverlapStats
    per_test_stats: dict[str, DatasetOverlapStats]


###############################################################################
# Configuration
###############################################################################


@dataclass
class AggregateConfig:
    """Aggregate overlap across all training datasets.

    The attributes_base_path should point to a directory containing attribute files
    organized as: {attributes_base_path}/{training_dataset}/{ngram}/{eval_dataset}/{split}/*.jsonl*

    For example: gs://bucket/total/proofpile-346f8a/15/algebraic-stack/train/*.jsonl.zst
    """

    attributes_base_path: str
    """Base path containing attribute files (e.g., gs://bucket/total/)"""

    output_path: str
    """Where to write the aggregated results"""

    ngram_size: int = 15
    """N-gram size to process"""

    attribute_name: str = "ngram_overlap"
    """Attribute name in the JSON files"""

    eval_dataset_steps: list[ExecutorStep] = field(default_factory=lambda: EVAL_DATASET_STEPS)
    """Evaluation dataset steps to compute sizes for"""


###############################################################################
# Helper Functions - Discovery
###############################################################################


def extract_shard_metadata(shard_path: str, training_root: str, ngram_size: int) -> ShardMetadata:
    """Extract test dataset name from shard path structure.

    Args:
        shard_path: Full path to the attribute shard
        training_root: Root directory of the training dataset
        ngram_size: N-gram size (used to parse path structure)

    Returns:
        ShardMetadata with test_dataset and shard_path
    """
    rel = os.path.relpath(shard_path, training_root)
    parts = rel.split(os.sep)
    try:
        idx_n = parts.index(str(ngram_size))
        test_ds_segment = parts[idx_n + 1] if idx_n + 1 < len(parts) else "unknown"
    except ValueError:
        test_ds_segment = parts[0]
    test_ds = test_ds_segment.split("-")[0]

    return ShardMetadata(test_dataset=test_ds, shard_path=shard_path)


def discover_training_datasets(base_path: str, ngram_size: int = 15) -> list[str]:
    """Auto-discover training dataset root directories (e.g., proofpile-xxxx).

    We treat the *first* path component after `base_path` as the training dataset
    name. For example, if `base_path` is:

        gs://bucket/total/

    and an attribute file lives at:

        gs://bucket/total/proofpile-6b5995/15/algebraic-stack/train/c0000.jsonl.zst

    then the training dataset root is:

        gs://bucket/total/proofpile-6b5995
    """
    base_prefix = base_path.rstrip("/")
    prefix_parts = base_prefix.split("/")
    prefix_len = len(prefix_parts)

    pattern = os.path.join(base_prefix, "*", str(ngram_size), "**", "*.jsonl*")
    attribute_files = fsspec_glob(pattern)

    training_datasets = set()
    for filepath in attribute_files:
        parts = filepath.split("/")
        if len(parts) <= prefix_len:
            continue

        dataset_root = "/".join(parts[: prefix_len + 1])
        training_datasets.add(dataset_root)

    result = sorted(training_datasets)
    logger.info("Discovered %d training datasets:", len(result))
    for ds in result:
        logger.info("    %s", os.path.basename(ds))
    return result


def _resolve_step_path(step: ExecutorStep) -> str:
    """Get the output path from an ExecutorStep.

    NOTE: This assumes override_output_path is set, which is typical for
    EVAL_DATASET_STEPS. In a full executor context, paths are resolved dynamically.
    """
    if step.override_output_path:
        return step.override_output_path
    # Fallback: derive from step name (may not be accurate without executor)
    logger.warning("Step %s has no override_output_path; using name as fallback", step.name)
    return step.name


###############################################################################
# Helper Functions - Dataset Sizes
###############################################################################


def _count_examples_in_path(path: str) -> int:
    """Count total examples in a directory using Zephyr."""
    pattern = os.path.join(path.rstrip("/"), "**", "*.jsonl*")
    pipeline = Dataset.from_files(pattern, empty_glob_ok=True).flat_map(load_file).map(lambda _: 1).reduce(sum)
    results = Backend.execute(pipeline)
    return results[0] if results else 0


def _compute_dataset_sizes(dataset_steps: list[ExecutorStep]) -> dict[str, int]:
    """Return mapping dataset_name -> total example count using Zephyr.

    Fixed: Properly handles ExecutorStep objects instead of assuming string paths.
    """
    size_map: dict[str, int] = {}
    for step in dataset_steps:
        step_path = _resolve_step_path(step)
        ds_name = os.path.basename(step_path.rstrip("/"))
        size_map[ds_name.split("-")[0]] = _count_examples_in_path(step_path)

    logger.info("Pre-computed dataset sizes:")
    for k, v in sorted(size_map.items()):
        logger.info("    %s: %s", k, v)
    return size_map


###############################################################################
# Core Processing - Phase 1: Zephyr Processing
###############################################################################


def _process_shards_to_intermediate(
    shard_paths: list[str],
    path_to_test_ds: dict[str, str],
    training_name: str,
    attr_key: str,
    output_dir: str,
) -> list[str]:
    """Phase 1: Use Zephyr to process shards and write intermediate files.

    Args:
        shard_paths: List of shard file paths to process
        path_to_test_ds: Mapping from shard path to test dataset name
        training_name: Name of the training dataset being processed
        attr_key: Attribute key to look for overlap (e.g., "ngram_overlap_15")
        output_dir: Directory to write intermediate files

    Returns:
        List of intermediate file paths written
    """

    def extract_overlap_records(shard_path: str) -> Iterator[dict]:
        """Read attribute shard and extract overlap information."""
        test_dataset = path_to_test_ds.get(shard_path, "unknown")

        logger.info("Loading from: %s", shard_path)
        for rec in load_file(shard_path):
            doc_id = rec.get("id")
            if doc_id is None:
                continue

            attrs = rec.get("attributes", {})
            has_overlap = bool(attrs.get(attr_key))

            yield {
                "id": doc_id,
                "test_dataset": test_dataset,
                "training_dataset": training_name,
                "has_overlap": has_overlap,
            }

    intermediate_paths = Backend.execute(
        Dataset.from_list(shard_paths)
        .flat_map(extract_overlap_records)
        .write_jsonl(f"{output_dir}/overlap-{{shard:05d}}.jsonl.gz", skip_existing=True)
    )

    logger.info("Wrote %d intermediate files to %s", len(intermediate_paths), output_dir)
    return intermediate_paths


###############################################################################
# Core Processing - Phase 2: In-Memory Aggregation
###############################################################################


def _aggregate_intermediate_files(
    intermediate_paths: list[str],
    known_test_datasets: set[str],
) -> tuple[DatasetOverlapStats, dict[str, DatasetOverlapStats]]:
    """Phase 2: Read intermediate files and build aggregation stats.

    Args:
        intermediate_paths: List of intermediate file paths to read
        known_test_datasets: Set of known test dataset names for pre-allocation

    Returns:
        Tuple of (overall_stats, per_test_stats)
    """
    # Mutable accumulators during aggregation
    overall_unique: set[str] = set()
    overall_overlap: set[str] = set()
    per_test: dict[str, dict[str, set[str]]] = {ds: {"unique": set(), "overlap": set()} for ds in known_test_datasets}

    for intermediate_path in intermediate_paths:
        for rec in load_jsonl(intermediate_path):
            doc_id = rec["id"]
            test_ds = rec["test_dataset"]
            has_overlap = rec["has_overlap"]

            overall_unique.add(doc_id)

            # Handle unknown test datasets dynamically
            if test_ds not in per_test:
                per_test[test_ds] = {"unique": set(), "overlap": set()}
            per_test[test_ds]["unique"].add(doc_id)

            if has_overlap:
                overall_overlap.add(doc_id)
                per_test[test_ds]["overlap"].add(doc_id)

    # Convert to immutable dataclasses
    overall_stats = DatasetOverlapStats(
        unique_ids=frozenset(overall_unique),
        overlap_ids=frozenset(overall_overlap),
    )

    per_test_stats = {
        ds: DatasetOverlapStats(
            unique_ids=frozenset(data["unique"]),
            overlap_ids=frozenset(data["overlap"]),
        )
        for ds, data in per_test.items()
    }

    return overall_stats, per_test_stats


def _merge_into_union(
    union_unique: set[str],
    union_overlap: set[str],
    union_per_test: dict[str, dict[str, set[str]]],
    result: AggregationResult,
) -> None:
    """Merge a single dataset result into union statistics (in-place).

    Args:
        union_unique: Mutable set of all unique doc IDs (modified in-place)
        union_overlap: Mutable set of all overlapping doc IDs (modified in-place)
        union_per_test: Mutable per-test tracking dict (modified in-place)
        result: Aggregation result to merge
    """
    union_unique.update(result.overall_stats.unique_ids)
    union_overlap.update(result.overall_stats.overlap_ids)

    for test_ds, stats in result.per_test_stats.items():
        if test_ds not in union_per_test:
            union_per_test[test_ds] = {"unique": set(), "overlap": set()}
        union_per_test[test_ds]["unique"].update(stats.unique_ids)
        union_per_test[test_ds]["overlap"].update(stats.overlap_ids)


###############################################################################
# Core Processing - Main Aggregation
###############################################################################


def aggregate_single_dataset(
    training_root: str,
    cfg: AggregateConfig,
    known_test_datasets: set[str],
) -> AggregationResult | None:
    """Aggregate overlap for a single training dataset.

    Args:
        training_root: Root path of the training dataset
        cfg: Aggregation configuration
        known_test_datasets: Set of known test dataset names

    Returns:
        AggregationResult or None if no shards found
    """
    attr_key = f"{cfg.attribute_name}_{cfg.ngram_size}"
    training_name = os.path.basename(training_root)

    # Discover all shards
    pattern = os.path.join(training_root, "**", str(cfg.ngram_size), "**", "*.jsonl*.zst")
    shard_paths = sorted(fsspec_glob(pattern))

    if not shard_paths:
        logger.warning("No attribute shards found for %s", training_name)
        return None

    logger.info("Processing %s with %d shards", training_name, len(shard_paths))

    # Build shard metadata mapping
    shard_metadata = [extract_shard_metadata(path, training_root, cfg.ngram_size) for path in shard_paths]
    path_to_test_ds = {meta.shard_path: meta.test_dataset for meta in shard_metadata}

    # Phase 1: Zephyr processing
    intermediate_dir = os.path.join(cfg.output_path, ".intermediate", training_name)
    intermediate_paths = _process_shards_to_intermediate(
        shard_paths, path_to_test_ds, training_name, attr_key, intermediate_dir
    )

    # Phase 2: In-memory aggregation
    overall_stats, per_test_stats = _aggregate_intermediate_files(intermediate_paths, known_test_datasets)

    logger.info(
        "%s - %d-gram - shards=%d => %d/%d (fraction %.4f)",
        training_name,
        cfg.ngram_size,
        len(shard_paths),
        overall_stats.contaminated,
        overall_stats.total,
        overall_stats.fraction,
    )

    return AggregationResult(
        training_name=training_name,
        shard_count=len(shard_paths),
        overall_stats=overall_stats,
        per_test_stats=per_test_stats,
    )


def aggregate_total(cfg: AggregateConfig) -> None:
    """Main function to aggregate overlap across all training datasets.

    Outputs consolidated results in two files:
    1. summary.csv - All training datasets + union summary
    2. overlap_matrix.csv - Evaluation x Training datasets overlap fractions
    """
    # Auto-discover training datasets first (no dataset size computation yet)
    training_datasets = discover_training_datasets(cfg.attributes_base_path, cfg.ngram_size)

    if not training_datasets:
        raise ValueError(f"No training datasets found in {cfg.attributes_base_path}")

    # Get known test dataset names for pre-allocation
    known_test_datasets = {
        os.path.basename(_resolve_step_path(step).rstrip("/")).split("-")[0] for step in cfg.eval_dataset_steps
    }

    # Track results
    all_results: dict[str, AggregationResult] = {}

    # Union accumulators (mutable during processing)
    union_unique: set[str] = set()
    union_overlap: set[str] = set()
    union_per_test: dict[str, dict[str, set[str]]] = {
        ds: {"unique": set(), "overlap": set()} for ds in known_test_datasets
    }

    # Process each training dataset
    for training_root in training_datasets:
        result = aggregate_single_dataset(training_root, cfg, known_test_datasets)

        if result is None:
            continue

        all_results[result.training_name] = result
        _merge_into_union(union_unique, union_overlap, union_per_test, result)

    # Convert union to immutable stats
    union_overall = DatasetOverlapStats(
        unique_ids=frozenset(union_unique),
        overlap_ids=frozenset(union_overlap),
    )
    union_per_test_stats = {
        ds: DatasetOverlapStats(
            unique_ids=frozenset(data["unique"]),
            overlap_ids=frozenset(data["overlap"]),
        )
        for ds, data in union_per_test.items()
    }

    logger.info(
        "All datasets - %d-gram => %d/%d (fraction %.4f)",
        cfg.ngram_size,
        union_overall.contaminated,
        union_overall.total,
        union_overall.fraction,
    )

    # Compute dataset sizes lazily (only needed for CSV output)
    dataset_sizes = _compute_dataset_sizes(cfg.eval_dataset_steps)

    # Write outputs
    _write_summary_csv(cfg.output_path, all_results, union_overall, cfg.ngram_size)
    _write_overlap_matrix_csv(cfg.output_path, all_results, union_per_test_stats, dataset_sizes)


###############################################################################
# Output Writing
###############################################################################


def _write_summary_csv(
    output_path: str,
    results: dict[str, AggregationResult],
    union_stats: DatasetOverlapStats,
    ngram_size: int,
) -> None:
    """Write consolidated summary CSV."""
    summary_path = os.path.join(output_path, "summary.csv")

    with fsspec.open(summary_path, "wt") as f:
        writer = csv.writer(f)
        writer.writerow(["training_dataset", "ngram", "total_examples", "contaminated", "fraction"])

        for training_name in sorted(results.keys()):
            result = results[training_name]
            writer.writerow(
                [
                    training_name,
                    ngram_size,
                    result.overall_stats.total,
                    result.overall_stats.contaminated,
                    f"{result.overall_stats.fraction:.6f}",
                ]
            )

        writer.writerow(
            [
                "union",
                ngram_size,
                union_stats.total,
                union_stats.contaminated,
                f"{union_stats.fraction:.6f}",
            ]
        )

    logger.info("Wrote consolidated summary: %s", summary_path)


def _write_overlap_matrix_csv(
    output_path: str,
    results: dict[str, AggregationResult],
    union_per_test: dict[str, DatasetOverlapStats],
    dataset_sizes: dict[str, int],
) -> None:
    """Write overlap matrix CSV (evaluation x training datasets)."""
    matrix_path = os.path.join(output_path, "overlap_matrix.csv")

    with fsspec.open(matrix_path, "wt") as f:
        writer = csv.writer(f)

        # Header: training datasets as columns + union
        training_names = sorted(results.keys())
        header = ["evaluation_dataset", *training_names, "union"]
        writer.writerow(header)

        # Rows: one per evaluation dataset
        for eval_ds in sorted(dataset_sizes.keys()):
            row = [eval_ds]
            total = dataset_sizes.get(eval_ds, 0)

            # Individual training datasets
            for train_ds in training_names:
                result = results.get(train_ds)
                if result and eval_ds in result.per_test_stats:
                    cont = result.per_test_stats[eval_ds].contaminated
                    frac = cont / total if total else 0.0
                    row.append(f"{frac:.6f}")
                else:
                    row.append("0.000000")

            # Union column
            if eval_ds in union_per_test:
                union_cont = union_per_test[eval_ds].contaminated
                union_frac = union_cont / total if total else 0.0
                row.append(f"{union_frac:.6f}")
            else:
                row.append("0.000000")

            writer.writerow(row)

    logger.info("Wrote overlap matrix: %s", matrix_path)
    logger.info(
        "Matrix dimensions: %d evaluation datasets x %d training datasets (+ union)",
        len(dataset_sizes),
        len(training_names),
    )


###############################################################################
# Public API (preserved from original)
###############################################################################


def run_aggregate_total(config: AggregateConfig) -> str:
    """Run aggregation and return output path."""
    logger.info("Starting train-test overlap aggregation with config: %s", config)
    aggregate_total(config)
    logger.info("Aggregation completed! Results written to %s", config.output_path)
    return config.output_path


def build_aggregate_total_step(
    attributes_base_path: str = "gs://marin-us-central2/train_test_overlap/",
    output_path: str | None = None,
    ngram_size: int = 15,
    eval_dataset_steps: list[ExecutorStep] | None = None,
) -> ExecutorStep:
    """Create an ExecutorStep for aggregating train-test overlap.

    Args:
        attributes_base_path: Base path containing attribute files
        output_path: Where to write results. If None, uses this_output_path()
        ngram_size: N-gram size to process
        eval_dataset_steps: Evaluation dataset steps. If None, uses EVAL_DATASET_STEPS

    Returns:
        ExecutorStep that can be integrated into a pipeline
    """
    cfg = AggregateConfig(
        attributes_base_path=attributes_base_path,
        output_path=output_path or this_output_path(),
        ngram_size=ngram_size,
        eval_dataset_steps=eval_dataset_steps or EVAL_DATASET_STEPS,
    )

    return ExecutorStep(
        name="train_test_overlap/aggregate_total",
        fn=run_aggregate_total,
        config=cfg,
        description="Aggregate overlap across all training datasets with individual and union views",
    )


STEPS = [build_aggregate_total_step()]

if __name__ == "__main__":
    executor_main(
        steps=STEPS,
        description="Aggregate train-test overlap across all training datasets",
    )
