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

Auto-discovers training datasets from a given GCP path.
"""

from __future__ import annotations

import csv
import json
import logging
import os
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass

import fsspec
import ray

from experiments.train_test_overlap.utils import EVAL_DATASET_STEPS
from marin.core.runtime import simple_backpressure
from marin.execution.executor import ExecutorStep, executor_main, this_output_path
from marin.utils import fsspec_glob

logger = logging.getLogger(__name__)


@ray.remote
def summarize_shard(shard_path: str, test_dataset: str, training_dataset: str, attr_key: str) -> dict:
    """Return basic overlap statistics for a single attribute shard.

    Args:
        shard_path: Path to the attribute shard file containing overlap data
        test_dataset: Name of the evaluation dataset being checked for overlap
        training_dataset: Name of the training dataset that may contain overlapping content
        attr_key: The attribute key to check for overlap (e.g., "ngram_overlap_15")

    Returns:
        Dictionary containing:
        - shard_path: The input shard path
        - test_dataset: The test dataset name
        - training_dataset: The training dataset name
        - ids_seen: List of all document IDs found in this shard
        - overlap_ids: List of document IDs that have overlap with the test dataset
    """

    ids_seen: set[str] = set()
    overlap_ids: set[str] = set()
    with fsspec.open(shard_path, "rt", compression="infer") as f:
        for line in f:
            rec = json.loads(line)
            doc_id = rec.get("id")
            if doc_id is None:
                continue
            ids_seen.add(doc_id)
            attrs = rec.get("attributes", {})
            if attrs.get(attr_key):
                overlap_ids.add(doc_id)

    return {
        "shard_path": shard_path,
        "test_dataset": test_dataset,
        "training_dataset": training_dataset,
        "ids_seen": list(ids_seen),
        "overlap_ids": list(overlap_ids),
    }


# Helper: cache for dataset sizes so we compute them only once per run
_DATASET_SIZE_CACHE: dict[str, int] | None = None
_TEST_LOOKUP_CACHE: dict[str, list[str]] | None = None


def _compute_dataset_sizes(dataset_steps: list[ExecutorStep]) -> dict[str, int]:
    """Return mapping dataset_name -> total example count."""
    global _DATASET_SIZE_CACHE
    if _DATASET_SIZE_CACHE is not None:
        return _DATASET_SIZE_CACHE

    # Local helper to count a single dataset directory (non-parallel: datasets are small)
    def count_dir(path: str) -> int:
        pattern = os.path.join(path.rstrip("/"), "**", "*.jsonl*")
        total = 0
        for fp in fsspec_glob(pattern):
            with fsspec.open(fp, "rt", compression="infer") as f:
                for _ in f:
                    total += 1
        return total

    size_map: dict[str, int] = {}
    for step in dataset_steps:
        ds_name = os.path.basename(step.rstrip("/"))
        size_map[ds_name.split("-")[0]] = count_dir(step)

    _DATASET_SIZE_CACHE = size_map
    logger.info("Pre-computed dataset sizes:")
    for k, v in sorted(size_map.items()):
        logger.info("    %s: %s", k, v)
    return size_map


def _build_test_lookup(dataset_steps: list[ExecutorStep]) -> dict[str, list[str]]:
    global _TEST_LOOKUP_CACHE
    if _TEST_LOOKUP_CACHE is not None:
        return _TEST_LOOKUP_CACHE

    lookup: dict[str, list[str]] = defaultdict(list)
    for step in dataset_steps:
        root = step.rstrip("/")
        for fp in fsspec_glob(os.path.join(root, "**", "*.jsonl*")):
            fname = os.path.basename(fp)
            lookup[fname].append(fp)
    _TEST_LOOKUP_CACHE = lookup
    return lookup


def discover_training_datasets(base_path: str, ngram_size: int = 15) -> list[str]:
    """Auto-discover training dataset root directories (e.g., proofpile-xxxx).

    We treat the *first* path component after `base_path` as the training dataset
    name.  For example, if `base_path` is:

        gs://bucket/train_test_overlap/dolma/total/

    and an attribute file lives at:

        gs://bucket/train_test_overlap/dolma/total/proofpile-6b5995/algebraic-stack/train/c0000/15/...

    then the training dataset root is:

        gs://bucket/train_test_overlap/dolma/total/proofpile-6b5995
    """

    base_prefix = base_path.rstrip("/")
    prefix_parts = base_prefix.split("/")
    prefix_len = len(prefix_parts)

    # Find any attribute file under the base path
    pattern = os.path.join(base_prefix, "*", "**", str(ngram_size), "**", "*.jsonl*")
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


@dataclass(frozen=True)
class AggregateConfig:
    """Aggregate overlap across all training datasets."""

    training_datasets_base_path: str  # base path to discover training datasets
    output_path: str  # where to write the results
    ngram_size: int = 15
    attribute_name: str = "ngram_overlap"
    max_in_flight: int = 64  # ray back-pressure
    dataset_steps: list[ExecutorStep] = None  # steps for eval datasets


###############################################################################
# Core aggregation functions
###############################################################################


def aggregate_single_dataset(
    training_root: str, cfg: AggregateConfig, dataset_sizes: dict[str, int], test_lookup: dict[str, list[str]]
) -> tuple[dict, dict]:
    """Aggregate overlap for a single training dataset."""
    attr_key = f"{cfg.attribute_name}_{cfg.ngram_size}"
    training_name = os.path.basename(training_root)

    # 1. Discover ALL shards for this training dataset
    pattern = os.path.join(training_root, "**", str(cfg.ngram_size), "**", "*.jsonl*")
    shard_paths: list[str] = sorted(fsspec_glob(pattern))
    if not shard_paths:
        logger.warning("No attribute shards found for %s", training_name)
        return {}, {}

    logger.info("Processing %s with %d shards", training_name, len(shard_paths))

    # 2. Submit Ray tasks
    task_args: list[tuple[str, str, str, str]] = []
    for shard in shard_paths:
        rel = os.path.relpath(shard, training_root)
        parts = rel.split(os.sep)
        try:
            idx_n = parts.index(str(cfg.ngram_size))
            test_ds_segment = parts[idx_n + 1] if idx_n + 1 < len(parts) else "unknown"
        except ValueError:
            test_ds_segment = parts[0]
        test_ds = test_ds_segment.split("-")[0]
        task_args.append((shard, test_ds, training_name, attr_key))

    refs: Iterator = simple_backpressure(summarize_shard, iter(task_args), cfg.max_in_flight, fetch_local=True)

    # 3. Aggregate results
    overall_unique: set[str] = set()
    overall_overlap: set[str] = set()
    per_test: dict[str, dict[str, set[str]]] = {ds: {"unique": set(), "overlap": set()} for ds in dataset_sizes.keys()}

    # detailed mapping: test_ds -> id -> {training_dataset: set(shard_paths)}
    contam_map: dict[str, dict[str, dict[str, dict[str, set[str]]]]] = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(lambda: {"dedupe_shards": set(), "training_shards": set(), "test_shards": set()})
        )
    )

    for ref in refs:
        res = ray.get(ref)
        overall_unique.update(res["ids_seen"])
        overall_overlap.update(res["overlap_ids"])

        tds = res["test_dataset"]
        if tds not in per_test:
            per_test[tds] = {"unique": set(), "overlap": set()}
        per_test[tds]["unique"].update(res["ids_seen"])
        per_test[tds]["overlap"].update(res["overlap_ids"])

        # detailed shard mapping
        if res["overlap_ids"]:
            train_name = res["training_dataset"]
            dedupe_shard_path = res["shard_path"]

            for _id in res["overlap_ids"]:
                entry = contam_map[tds][_id][train_name]
                entry["dedupe_shards"].add(dedupe_shard_path)

                # add test shard paths
                test_fname = os.path.basename(dedupe_shard_path)
                for tp in test_lookup.get(test_fname, []):
                    entry["test_shards"].add(tp)

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
        "contam_map": contam_map,
        "shard_count": len(shard_paths),
    }, {"overall_unique": overall_unique, "overall_overlap": overall_overlap, "per_test": per_test}


def write_dataset_results(results: dict, output_dir: str, cfg: AggregateConfig, dataset_sizes: dict[str, int]):
    """Write results for a single dataset to its output directory."""
    os.makedirs(output_dir, exist_ok=True)

    # 1. Write summary CSV
    csv_path = os.path.join(output_dir, "summary.csv")
    with fsspec.open(csv_path, "wt") as f:
        writer = csv.writer(f)
        writer.writerow(["training_dataset", "ngram", "shards", "total_examples", "contaminated", "fraction"])
        writer.writerow(
            [
                results["training_name"],
                cfg.ngram_size,
                results["shard_count"],
                results["total_examples"],
                results["contaminated"],
                f"{results['fraction']:.6f}",
            ]
        )
    logger.info("Wrote %s", csv_path)

    # 2. Write per-test breakdown CSV
    per_test_csv = os.path.join(output_dir, "per_test_breakdown.csv")
    with fsspec.open(per_test_csv, "wt") as f:
        writer = csv.writer(f)
        writer.writerow(["test_dataset", "total_examples", "contaminated", "fraction"])
        for tds in sorted(dataset_sizes.keys()):
            tot = dataset_sizes.get(tds, 0)
            cont = len(results["per_test"][tds]["overlap"])
            frac_t = cont / tot if tot else 0.0
            writer.writerow([tds, tot, cont, f"{frac_t:.6f}"])
    logger.info("Wrote %s", per_test_csv)

    # 3. Write detailed contamination JSONL files
    for tds, id_map in results["contam_map"].items():
        file_path = os.path.join(output_dir, f"{tds}_overlap_map.jsonl.gz")
        with fsspec.open(file_path, "wt", compression="gzip") as f:
            for _id, train_dict in id_map.items():
                for train_name, maps in train_dict.items():
                    f.write(
                        json.dumps(
                            {
                                "test_dataset": tds,
                                "id": _id,
                                "training_dataset": train_name,
                                "dedupe_shards": sorted(maps["dedupe_shards"]),
                                "training_shards": sorted(maps["training_shards"]),
                                "test_shards": sorted(maps["test_shards"]),
                            }
                        )
                        + "\n"
                    )
        logger.info("Wrote %s", file_path)


###############################################################################
# Main aggregation function
###############################################################################


def aggregate_total(cfg: AggregateConfig):
    """Main function to aggregate overlap across all training datasets."""

    # Pre-compute test dataset sizes
    dataset_sizes = _compute_dataset_sizes(cfg.dataset_steps)
    test_lookup = _build_test_lookup(cfg.dataset_steps)

    # Auto-discover training datasets
    training_datasets = discover_training_datasets(cfg.training_datasets_base_path, cfg.ngram_size)

    if not training_datasets:
        raise ValueError(f"No training datasets found in {cfg.training_datasets_base_path}")

    # Track results for matrix generation
    all_results = {}
    union_unique = set()
    union_overlap = set()
    union_per_test = {ds: {"unique": set(), "overlap": set()} for ds in dataset_sizes.keys()}

    # Process each training dataset
    for training_root in training_datasets:
        training_name = os.path.basename(training_root)

        # Aggregate this dataset
        detailed_results, summary_results = aggregate_single_dataset(training_root, cfg, dataset_sizes, test_lookup)

        if not detailed_results:  # Skip if no results
            continue

        # Store for matrix generation
        all_results[training_name] = detailed_results

        # Write individual results
        out_dir = os.path.join(cfg.output_path, training_name, str(cfg.ngram_size))
        write_dataset_results(detailed_results, out_dir, cfg, dataset_sizes)

        # Update union statistics
        union_unique.update(summary_results["overall_unique"])
        union_overlap.update(summary_results["overall_overlap"])

        for tds in dataset_sizes.keys():
            if tds in summary_results["per_test"]:
                union_per_test[tds]["unique"].update(summary_results["per_test"][tds]["unique"])
                union_per_test[tds]["overlap"].update(summary_results["per_test"][tds]["overlap"])

    # Write union results
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

    union_dir = os.path.join(cfg.output_path, "union", str(cfg.ngram_size))
    os.makedirs(union_dir, exist_ok=True)

    # Union summary CSV
    union_csv = os.path.join(union_dir, "summary.csv")
    with fsspec.open(union_csv, "wt") as f:
        writer = csv.writer(f)
        writer.writerow(["training_dataset", "ngram", "shards", "total_examples", "contaminated", "fraction"])
        writer.writerow(["union", cfg.ngram_size, "all", union_total, union_contaminated, f"{union_frac:.6f}"])
    logger.info("Wrote %s", union_csv)

    # Union per-test breakdown CSV
    union_per_test_csv = os.path.join(union_dir, "per_test_breakdown.csv")
    with fsspec.open(union_per_test_csv, "wt") as f:
        writer = csv.writer(f)
        writer.writerow(["test_dataset", "total_examples", "contaminated", "fraction"])
        for tds in sorted(dataset_sizes.keys()):
            tot = dataset_sizes.get(tds, 0)
            cont = len(union_per_test[tds]["overlap"])
            frac_t = cont / tot if tot else 0.0
            writer.writerow([tds, tot, cont, f"{frac_t:.6f}"])
    logger.info("Wrote %s", union_per_test_csv)

    # Generate contamination matrix CSV
    matrix_path = os.path.join(cfg.output_path, "overlap_matrix.csv")
    with fsspec.open(matrix_path, "wt") as f:
        writer = csv.writer(f)

        # Header: evaluation datasets as columns
        training_names = sorted(all_results.keys())
        header = ["evaluation_dataset", *training_names]
        writer.writerow(header)

        # Rows: one per evaluation dataset
        for eval_ds in sorted(dataset_sizes.keys()):
            row = [eval_ds]
            tot = dataset_sizes.get(eval_ds, 0)

            for train_ds in training_names:
                if train_ds in all_results and eval_ds in all_results[train_ds]["per_test"]:
                    cont = len(all_results[train_ds]["per_test"][eval_ds]["overlap"])
                    frac = cont / tot if tot else 0.0
                    row.append(f"{frac:.6f}")
                else:
                    row.append("0.000000")

            writer.writerow(row)

    logger.info("Wrote overlap matrix: %s", matrix_path)
    logger.info(
        "Matrix dimensions: %d evaluation datasets x %d training datasets",
        len(dataset_sizes),
        len(training_names),
    )


###############################################################################
# Executor step
###############################################################################

cfg = AggregateConfig(
    training_datasets_base_path="gs://marin-us-central2/train_test_overlap/dolma/total/",
    output_path=this_output_path(),
    ngram_size=15,
    dataset_steps=EVAL_DATASET_STEPS,
)

aggregate_total_step = ExecutorStep(
    name="train_test_overlap/dolma/aggregate_total_final",
    fn=aggregate_total,
    config=cfg,
    description="Aggregate overlap across all training datasets with individual and union views",
)

###############################################################################
# Main entry-point
###############################################################################

if __name__ == "__main__":
    executor_main(
        steps=[aggregate_total_step],
        description="Aggregate train-test overlap across all training datasets",
    )
