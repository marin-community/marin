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

"""aggregate_total.py

Aggregate overlap across all discovered training datasets with two levels:
1. Per-training-dataset aggregation (like debug scripts but using ALL shards)
2. Union aggregation across all training datasets
3. Overlap matrix showing training_datasets X evaluation_datasets

Auto-discovers training datasets from attribute directory structure.

Example Usage:
    python experiments/train_test_overlap/aggregate_total.py --prefix gs://my-bucket
"""

import csv
import json
import logging
import os
from collections.abc import Iterator
from dataclasses import dataclass, field

import fsspec
import ray
from marin.execution.executor import ExecutorStep, executor_main, this_output_path
from marin.utils import fsspec_glob
from zephyr import Dataset, flow_backend, load_file

from experiments.train_test_overlap.eval_datasets_overlap import EVAL_DATASET_STEPS

logger = logging.getLogger(__name__)


def extract_shard_metadata(shard_path: str, training_root: str, ngram_size: int) -> dict:
    """Extract test dataset name from shard path structure.

    Args:
        shard_path: Full path to the attribute shard
        training_root: Root directory of the training dataset
        ngram_size: N-gram size (used to parse path structure)

    Returns:
        Dict with test_dataset and shard_path
    """
    rel = os.path.relpath(shard_path, training_root)
    parts = rel.split(os.sep)
    try:
        idx_n = parts.index(str(ngram_size))
        test_ds_segment = parts[idx_n + 1] if idx_n + 1 < len(parts) else "unknown"
    except ValueError:
        test_ds_segment = parts[0]
    test_ds = test_ds_segment.split("-")[0]

    return {
        "test_dataset": test_ds,
        "shard_path": shard_path,
    }


def _compute_dataset_sizes(dataset_steps: list[ExecutorStep]) -> dict[str, int]:
    """Return mapping dataset_name -> total example count using Zephyr."""

    def count_dir(path: str) -> int:
        pattern = os.path.join(path.rstrip("/"), "**", "*.jsonl*")
        pipeline = Dataset.from_files(pattern, empty_glob_ok=True).flat_map(load_file).map(lambda _: 1).reduce(sum)
        results = list(flow_backend().execute(pipeline))
        return results[0]

    size_map: dict[str, int] = {}
    for step in dataset_steps:
        ds_name = os.path.basename(step.rstrip("/"))
        size_map[ds_name.split("-")[0]] = count_dir(step)

    logger.info("Pre-computed dataset sizes:")
    for k, v in sorted(size_map.items()):
        logger.info("    %s: %s", k, v)
    return size_map


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

    # Find any attribute file under the base path with the pattern:
    # {base_path}/{training_dataset}/{ngram}/{eval_dataset}/**/*.jsonl*
    pattern = os.path.join(base_prefix, "*", str(ngram_size), "**", "*.jsonl*")
    attribute_files = fsspec_glob(pattern)

    training_datasets = set()
    for filepath in attribute_files:
        parts = filepath.split("/")
        if len(parts) <= prefix_len:
            continue  # malformed

        # The first component after the base path is the training dataset name
        dataset_root = "/".join(parts[: prefix_len + 1])
        training_datasets.add(dataset_root)

    result = sorted(training_datasets)
    logger.info("Discovered %d training datasets:", len(result))
    for ds in result:
        logger.info("    %s", os.path.basename(ds))
    return result


###############################################################################
# Config dataclass
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
# Core aggregation functions
###############################################################################


def aggregate_single_dataset(
    training_root: str, cfg: AggregateConfig, dataset_sizes: dict[str, int]
) -> tuple[dict, dict]:
    """Aggregate overlap for a single training dataset using Zephyr.

    Phase 1: Use Zephyr to read all attribute shards and write intermediate overlap data
    Phase 2: Read intermediate data and build aggregations in controller
    """
    attr_key = f"{cfg.attribute_name}_{cfg.ngram_size}"
    training_name = os.path.basename(training_root)

    # 1. Discover ALL shards for this training dataset
    pattern = os.path.join(training_root, "**", str(cfg.ngram_size), "**", "*.jsonl*")
    shard_paths: list[str] = sorted(fsspec_glob(pattern))
    if not shard_paths:
        logger.warning("No attribute shards found for %s", training_name)
        return {}, {}

    logger.info("Processing %s with %d shards", training_name, len(shard_paths))

    # 2. Create metadata for each shard (maps shard path -> test_dataset)
    shard_metadata = [extract_shard_metadata(path, training_root, cfg.ngram_size) for path in shard_paths]

    # Build lookup: shard_path -> test_dataset
    path_to_test_ds = {meta["shard_path"]: meta["test_dataset"] for meta in shard_metadata}

    def extract_overlap_records(shard_path: str) -> Iterator[dict]:
        """Read attribute shard and extract overlap information."""
        test_dataset = path_to_test_ds.get(shard_path, "unknown")

        with fsspec.open(shard_path, "rt", compression="infer") as f:
            for line in f:
                rec = json.loads(line)
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

    # 3. Run Zephyr pipeline to extract overlap data
    intermediate_dir = os.path.join(cfg.output_path, ".intermediate", training_name)
    intermediate_paths = list(
        flow_backend().execute(
            Dataset.from_list(shard_paths)
            .flat_map(load_file)
            .flat_map(extract_overlap_records)
            .write_jsonl(f"{intermediate_dir}/overlap-{{shard:05d}}.jsonl.gz", skip_existing=True)
        )
    )

    logger.info(f"Wrote {len(intermediate_paths)} intermediate files to {intermediate_dir}")

    # 4. Read intermediate files and build aggregations
    overall_unique: set[str] = set()
    overall_overlap: set[str] = set()
    per_test: dict[str, dict[str, set[str]]] = {ds: {"unique": set(), "overlap": set()} for ds in dataset_sizes.keys()}

    for intermediate_path in intermediate_paths:
        with fsspec.open(intermediate_path, "rt", compression="gzip") as f:
            for line in f:
                rec = json.loads(line)
                doc_id = rec["id"]
                test_ds = rec["test_dataset"]
                has_overlap = rec["has_overlap"]

                overall_unique.add(doc_id)
                if test_ds not in per_test:
                    per_test[test_ds] = {"unique": set(), "overlap": set()}
                per_test[test_ds]["unique"].add(doc_id)

                if has_overlap:
                    overall_overlap.add(doc_id)
                    per_test[test_ds]["overlap"].add(doc_id)

    total = len(overall_unique)
    contaminated = len(overall_overlap)
    frac = contaminated / total if total else 0.0

    logger.info(
        "%s • %d-gram • shards=%d ⇒ %d/%d (fraction %.4f)",
        training_name,
        cfg.ngram_size,
        len(shard_paths),
        contaminated,
        total,
        frac,
    )

    return {
        "training_name": training_name,
        "total_examples": total,
        "contaminated": contaminated,
        "fraction": frac,
        "per_test": per_test,
        "shard_count": len(shard_paths),
    }, {"overall_unique": overall_unique, "overall_overlap": overall_overlap, "per_test": per_test}


###############################################################################
# Main aggregation function
###############################################################################


def aggregate_total(cfg: AggregateConfig):
    """Main function to aggregate overlap across all training datasets.

    Outputs consolidated results in two files:
    1. summary.csv - All training datasets + union summary
    2. overlap_matrix.csv - Evaluation x Training datasets overlap fractions
    """

    # Pre-compute test dataset sizes
    dataset_sizes = _compute_dataset_sizes(cfg.eval_dataset_steps)

    # Auto-discover training datasets
    training_datasets = discover_training_datasets(cfg.attributes_base_path, cfg.ngram_size)

    if not training_datasets:
        raise ValueError(f"No training datasets found in {cfg.attributes_base_path}")

    # Track results for matrix generation
    all_results = {}
    union_unique = set()
    union_overlap = set()
    union_per_test = {ds: {"unique": set(), "overlap": set()} for ds in dataset_sizes.keys()}

    # Process each training dataset
    for training_root in training_datasets:
        training_name = os.path.basename(training_root)

        # Aggregate this dataset
        detailed_results, summary_results = aggregate_single_dataset(training_root, cfg, dataset_sizes)

        if not detailed_results:  # Skip if no results
            continue

        # Store for matrix generation
        all_results[training_name] = detailed_results

        # Update union statistics
        union_unique.update(summary_results["overall_unique"])
        union_overlap.update(summary_results["overall_overlap"])

        for tds in dataset_sizes.keys():
            if tds in summary_results["per_test"]:
                union_per_test[tds]["unique"].update(summary_results["per_test"][tds]["unique"])
                union_per_test[tds]["overlap"].update(summary_results["per_test"][tds]["overlap"])

    # Compute union statistics
    union_total = len(union_unique)
    union_contaminated = len(union_overlap)
    union_frac = union_contaminated / union_total if union_total else 0.0

    logger.info(
        "All datasets • %d-gram ⇒ %d/%d (fraction %.4f)",
        cfg.ngram_size,
        union_contaminated,
        union_total,
        union_frac,
    )

    # Write consolidated summary CSV (all datasets + union)
    summary_path = os.path.join(cfg.output_path, "summary.csv")
    with fsspec.open(summary_path, "wt") as f:
        writer = csv.writer(f)
        writer.writerow(["training_dataset", "ngram", "total_examples", "contaminated", "fraction"])

        # Write each training dataset
        for training_name in sorted(all_results.keys()):
            result = all_results[training_name]
            writer.writerow(
                [
                    training_name,
                    cfg.ngram_size,
                    result["total_examples"],
                    result["contaminated"],
                    f"{result['fraction']:.6f}",
                ]
            )

        # Write union row
        writer.writerow(
            [
                "union",
                cfg.ngram_size,
                union_total,
                union_contaminated,
                f"{union_frac:.6f}",
            ]
        )

    logger.info("Wrote consolidated summary: %s", summary_path)

    # Write overlap matrix CSV (evaluation x training datasets)
    matrix_path = os.path.join(cfg.output_path, "overlap_matrix.csv")
    with fsspec.open(matrix_path, "wt") as f:
        writer = csv.writer(f)

        # Header: training datasets as columns + union
        training_names = sorted(all_results.keys())
        header = ["evaluation_dataset", *training_names, "union"]
        writer.writerow(header)

        # Rows: one per evaluation dataset
        for eval_ds in sorted(dataset_sizes.keys()):
            row = [eval_ds]
            tot = dataset_sizes.get(eval_ds, 0)

            # Individual training datasets
            for train_ds in training_names:
                if train_ds in all_results and eval_ds in all_results[train_ds]["per_test"]:
                    cont = len(all_results[train_ds]["per_test"][eval_ds]["overlap"])
                    frac = cont / tot if tot else 0.0
                    row.append(f"{frac:.6f}")
                else:
                    row.append("0.000000")

            # Union column
            union_cont = len(union_per_test[eval_ds]["overlap"])
            union_frac_eval = union_cont / tot if tot else 0.0
            row.append(f"{union_frac_eval:.6f}")

            writer.writerow(row)

    logger.info("Wrote overlap matrix: %s", matrix_path)
    logger.info(
        "Matrix dimensions: %d evaluation datasets x %d training datasets (+ union)",
        len(dataset_sizes),
        len(training_names),
    )


@ray.remote(runtime_env={"env_vars": {"JAX_PLATFORMS": "cpu", "PJRT_DEVICE": "cpu"}})
def run_aggregate_total(config: AggregateConfig) -> str:
    logger.info(f"Starting train-test overlap aggregation with config: {config}")
    aggregate_total(config)
    logger.info(f"Aggregation completed! Results written to {config.output_path}")
    return config.output_path


def build_aggregate_total_step(
    attributes_base_path: str = "gs://marin-us-central2/tmp/power/train_test_overlap/dolma/total",
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
        name="tmp/train_test_overlap/aggregate_total",
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
