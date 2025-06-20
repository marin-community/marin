#!/usr/bin/env python3
"""
aggregate_parallel_test.py

Parallel version of the overlap aggregator that unions contamination across *all* training
 datasets for each test dataset.

Strategy (approach "B" from design notes):
• Each shard file (produced by the dedupe pipeline) is processed by an independent
  Ray remote task that summarises, for one n-gram size, the set of overlapped IDs and
  the total number of test examples in the shard.
• Back-pressure (`simple_backpressure`) limits concurrency, so we can launch thousands of
  shards without overwhelming the cluster.
• The driver merges shard-level summaries into a final union per test-dataset which is
  then written to CSV (one `overall_contamination.csv` per n-gram).

The module is designed to be used inside Marin's Executor framework, so it does **not** call
`ray.init()` – that's handled by the surrounding experiment.
"""
from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterator

import fsspec
import ray

from marin.execution.executor import ExecutorStep, executor_main, this_output_path
from marin.core.runtime import simple_backpressure
from marin.utils import fsspec_mkdirs, fsspec_glob

from experiments.train_test_overlap.utils import (
    find_dataset_shards,
    get_relative_path_no_extension,
)
from experiments.train_test_overlap.dolma.dedupe_total import (
    finemath_dedupe_step,
    proofpile_dedupe_step,
)

#####################################################################
# Configuration dataclass
#####################################################################


@dataclass(frozen=True)
class ParallelAggregateConfig:
    """Configuration for the parallel overlap aggregator."""

    input_paths: list[str]
    """Roots of dedupe outputs – one per training dataset."""

    output_path: str
    """Base directory where aggregated CSVs will be written."""

    ngram_sizes: list[int] | None = None  # If None we introspect
    attribute_name: str = "ngram_overlap"
    max_in_flight: int = 128  # controls back-pressure when launching shard tasks


#####################################################################
# Ray remote task – processes one shard file
#####################################################################


@ray.remote(num_cpus=1)
def summarise_shard(
    shard_path: str,
    test_dataset: str,
    training_dataset: str,
    attr_key: str,
) -> dict:
    """Summarise one attribute shard.

    Returns dict with keys:
      test_dataset, training_dataset, shard_name, total, overlap_count, ids_seen, overlap_ids
    All sets are converted to lists for Ray serialisation.
    """
    ids_seen_set: set[str] = set()
    overlap_ids_set: set[str] = set()

    # Shard identifier = path tail without attr part
    shard_name = os.path.dirname(shard_path).split(os.sep)[-1]

    try:
        with fsspec.open(shard_path, "rt", compression="infer") as f:
            for line in f:
                rec = json.loads(line)
                _id = rec["id"]
                ids_seen_set.add(_id)
                if rec.get("attributes", {}).get(attr_key, []):
                    overlap_ids_set.add(_id)
    except Exception as e:
        print(f"[WARN] Error processing shard {shard_path}: {e}")

    return {
        "test_dataset": test_dataset,
        "training_dataset": training_dataset,
        "shard_name": shard_name,
        "shard_path": shard_path,
        "total": len(ids_seen_set),
        "overlap_count": len(overlap_ids_set),
        "ids_seen": list(ids_seen_set),
        "overlap_ids": list(overlap_ids_set),
    }


#####################################################################
# Driver function – executed as an ExecutorStep
#####################################################################


def aggregate_parallel(config: ParallelAggregateConfig):
    print("[INFO] Starting parallel aggregation")
    print(f"[INFO] Input paths: {config.input_paths}")

    # -----------------------------------------------------------------
    # 1. Discover n-gram sizes if not given
    # -----------------------------------------------------------------
    if config.ngram_sizes is None:
        detected: set[int] = set()
        for ip in config.input_paths:
            for shard in find_dataset_shards(ip):
                rel = get_relative_path_no_extension(shard, ip)
                for seg in rel.split(os.sep):
                    if seg.isdigit():
                        detected.add(int(seg))
        ngram_sizes = sorted(detected)
    else:
        ngram_sizes = config.ngram_sizes

    print(f"[INFO] N-gram sizes: {ngram_sizes}")

    # -----------------------------------------------------------------
    # 2. Aggregate for each n-gram size
    # -----------------------------------------------------------------
    for n in ngram_sizes:
        attr_key = f"{config.attribute_name}_{n}"
        print(f"[INFO] Processing {n}-gram… (attr_key={attr_key})")

        # Build list of shard tasks
        task_args: list[tuple[str, str, str, str]] = []  # (shard_path, test_ds, training_name, attr_key)

        for idx, train_root in enumerate(config.input_paths):
            if isinstance(train_root, str):
                training_name = os.path.basename(train_root).replace("total_", "")
            else:
                training_name = f"train_{idx}"

            pattern = os.path.join(train_root, "**", str(n), "**", "*.jsonl*")
            shard_paths = fsspec_glob(pattern)
            print(f"[DEBUG] ({training_name}) found {len(shard_paths)} shards for {n}-gram", flush=True)

            # DEBUG: list a few jsonl files under train_root to ensure globbing works
            sample_paths = fsspec_glob(os.path.join(train_root, "**", "*.jsonl*"))[:5]
            print(f"[DEBUG] sample paths under {training_name}: {sample_paths}", flush=True)

            for shard in shard_paths:
                # Determine test dataset name from path segment immediately inside the n-gram folder
                rel = os.path.relpath(shard, train_root)
                parts = rel.split(os.sep)

                # find the index of the n-gram segment
                try:
                    idx_ngram = parts.index(str(n))
                    test_ds_segment = parts[idx_ngram + 1] if idx_ngram + 1 < len(parts) else "unknown"
                except ValueError:
                    # should not happen; fallback to first segment
                    test_ds_segment = parts[0]

                test_ds = test_ds_segment.split("-")[0]

                task_args.append((shard, test_ds, training_name, attr_key))
        print(f"[INFO]  – Found {len(task_args)} shard files for {n}-gram", flush=True)
        if not task_args:
            print("[WARN] No shard files found, skipping this n-gram", flush=True)
            continue

        # Submit Ray tasks with back-pressure
        refs: Iterator = simple_backpressure(
            summarise_shard, iter(task_args), config.max_in_flight, fetch_local=True
        )

        # -----------------------------------------------------------------
        # 3. Reduce – union overlaps across all shards/training datasets
        # -----------------------------------------------------------------
        # overall union across all training datasets
        overall_unique_ids: dict[str, set[str]] = defaultdict(set)
        overall_overlap_ids: dict[str, set[str]] = defaultdict(set)

        # per-training-dataset stats: keep only counts to save memory
        training_stats: dict[str, dict[str, dict]] = defaultdict(
            lambda: defaultdict(
                lambda: {
                    "overlap_ids": set(),
                    "id_to_shards": defaultdict(list),
                }
            )
        )

        for ref in refs:
            shard_res = ray.get(ref)
            test_ds = shard_res["test_dataset"]
            train_name = shard_res["training_dataset"]
            overall_unique_ids[test_ds].update(shard_res["ids_seen"])
            overall_overlap_ids[test_ds].update(shard_res["overlap_ids"])

            # per-training
            ts_entry = training_stats[train_name][test_ds]
            ts_entry["overlap_ids"].update(shard_res["overlap_ids"])
            for _id in shard_res["overlap_ids"]:
                ts_entry["id_to_shards"][_id].append(shard_res["shard_path"])

        # -----------------------------------------------------------------
        # 4. Write overall contamination CSV
        # -----------------------------------------------------------------
        ngram_dir = os.path.join(config.output_path, str(n))
        fsspec_mkdirs(ngram_dir)
        csv_path = os.path.join(ngram_dir, "overall_contamination.csv")
        with fsspec.open(csv_path, "wt") as f:
            writer = csv.writer(f)
            writer.writerow([
                "test_dataset",
                "total_examples",
                "contaminated_examples",
                "contamination_fraction",
            ])

            for test_ds, ids in sorted(overall_unique_ids.items(), key=lambda x: len(overall_overlap_ids[x[0]]), reverse=True):
                total = len(ids)
                contaminated = len(overall_overlap_ids[test_ds])
                frac = contaminated / total if total else 0.0
                writer.writerow([test_ds, total, contaminated, f"{frac:.6f}"])
        print(f"[INFO]  – Wrote {csv_path}", flush=True)

        # -----------------------------------------------------------------
        # 5. Per-training summaries & detailed mappings
        # -----------------------------------------------------------------

        detailed_dir = os.path.join(ngram_dir, "detailed")
        fsspec_mkdirs(detailed_dir)

        for train_name, per_test in training_stats.items():
            train_dir = os.path.join(detailed_dir, train_name)
            fsspec_mkdirs(train_dir)

            # summary csv
            summary_path = os.path.join(train_dir, "summary.csv")
            with fsspec.open(summary_path, "wt") as f:
                writer = csv.writer(f)
                writer.writerow(["test_dataset", "total_examples", "overlap_count", "overlap_fraction"])

                for test_ds, stats in per_test.items():
                    total = len(overall_unique_ids[test_ds])
                    overlap_cnt = len(stats["overlap_ids"])
                    frac = overlap_cnt / total if total else 0.0
                    writer.writerow([test_ds, total, overlap_cnt, f"{frac:.6f}"])

            # detailed mapping per ID
            for test_ds, stats in per_test.items():
                if stats["overlap_ids"]:
                    detail_path = os.path.join(train_dir, f"{test_ds}_overlaps.jsonl")
                    with fsspec.open(detail_path, "wt") as f:
                        for _id in stats["overlap_ids"]:
                            f.write(json.dumps({
                                "id": _id,
                                "test_dataset": test_ds,
                                "training_dataset": train_name,
                                "shards": stats["id_to_shards"][_id],
                            }) + "\n")

        print(f"[INFO]  – Wrote detailed stats into {detailed_dir}", flush=True)

    print("[INFO] Parallel aggregation finished", flush=True)


#####################################################################
# Hook into the Executor framework
#####################################################################

parallel_cfg = ParallelAggregateConfig(
    input_paths=[
        finemath_dedupe_step,
        #proofpile_dedupe_step,
    ],
    ngram_sizes=[15],
    output_path=this_output_path(),
)

aggregate_parallel_step = ExecutorStep(
    name="train_test_overlap/dolma/aggregate_parallel",
    fn=aggregate_parallel,
    config=parallel_cfg,
    description="Union contamination across ALL training datasets (parallel version)",
)

if __name__ == "__main__":
    executor_main(
        steps=[
            finemath_dedupe_step,
            #proofpile_dedupe_step,
            aggregate_parallel_step,
        ],
        description="Parallel aggregation of train-test overlap across datasets",
    ) 