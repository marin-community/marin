#!/usr/bin/env python3
"""aggregate_total.py

Aggregate contamination across all discovered training datasets with two levels:
1. Per-training-dataset aggregation (like debug scripts but using ALL shards)
2. Union aggregation across all training datasets
3. Contamination matrix showing training_datasets X evaluation_datasets

Auto-discovers training datasets from a given GCP path.
"""
from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass

import fsspec
import ray

# ---------------------------------------------------------------------------
# Import evaluation dataset conversion steps so executor can resolve paths and
# we can count their sizes on-the-fly.
# ---------------------------------------------------------------------------
from experiments.train_test_overlap.eval_datasets_overlap import (
    ai2_arc_convert_dolma,
    bbh_convert_dolma,
    boolq_convert_dolma,
    commonsense_qa_convert_dolma,
    gpqa_convert_dolma,
    gsm8k_convert_dolma,
    hellaswag_convert_dolma,
    humaneval_convert_dolma,
    instruction_following_convert_dolma,
    lambada_openai_convert_dolma,
    math_convert_dolma,
    mmlu_convert_dolma,
    mmlu_pro_convert_dolma,
    musr_convert_dolma,
    openbookqa_convert_dolma,
    piqa_convert_dolma,
    truthful_qa_convert_dolma,
    winograd_wsc_convert_dolma,
)

# Re-use helper functions + remote task from the existing aggregator
from marin.core.runtime import simple_backpressure
from marin.execution.executor import ExecutorStep, executor_main, this_output_path
from marin.utils import fsspec_glob

# List of evaluator conversion steps
EVAL_DATASET_STEPS: list[ExecutorStep] = [
    gsm8k_convert_dolma,
    math_convert_dolma,
    truthful_qa_convert_dolma,
    bbh_convert_dolma,
    mmlu_convert_dolma,
    humaneval_convert_dolma,
    instruction_following_convert_dolma,
    gpqa_convert_dolma,
    musr_convert_dolma,
    mmlu_pro_convert_dolma,
    hellaswag_convert_dolma,
    ai2_arc_convert_dolma,
    boolq_convert_dolma,
    commonsense_qa_convert_dolma,
    lambada_openai_convert_dolma,
    openbookqa_convert_dolma,
    piqa_convert_dolma,
    winograd_wsc_convert_dolma,
]


@ray.remote
def summarise_shard(shard_path: str, test_dataset: str, training_dataset: str, attr_key: str) -> dict:
    """Return basic overlap statistics for a single attribute shard."""

    ids_seen: set[str] = set()
    overlap_ids: set[str] = set()
    with fsspec.open(shard_path, "rt", compression="infer") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
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
    print("[INFO] Pre-computed dataset sizes:")
    for k, v in sorted(size_map.items()):
        print(f"    {k}: {v}")
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
    print(f"[INFO] Discovered {len(result)} training datasets:")
    for ds in result:
        print(f"    {os.path.basename(ds)}")
    return result


###############################################################################
# Config dataclass
###############################################################################


@dataclass(frozen=True)
class AggregateConfig:
    """Aggregate contamination across all training datasets."""

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
    """Aggregate contamination for a single training dataset."""
    attr_key = f"{cfg.attribute_name}_{cfg.ngram_size}"
    training_name = os.path.basename(training_root)

    # 1. Discover ALL shards for this training dataset
    pattern = os.path.join(training_root, "**", str(cfg.ngram_size), "**", "*.jsonl*")
    shard_paths: list[str] = sorted(fsspec_glob(pattern))
    if not shard_paths:
        print(f"[WARN] No attribute shards found for {training_name}")
        return {}, {}

    print(f"[INFO] Processing {training_name} with {len(shard_paths)} shards")

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

    refs: Iterator = simple_backpressure(summarise_shard, iter(task_args), cfg.max_in_flight, fetch_local=True)

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

    print(
        f"[RESULT] {training_name} • {cfg.ngram_size}-gram "
        f"• shards={len(shard_paths)} ⇒ {contaminated}/{total} "
        f"(fraction {frac:.4f})"
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
    print(f"[INFO] Wrote {csv_path}")

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
    print(f"[INFO] Wrote {per_test_csv}")

    # 3. Write detailed contamination JSONL files
    for tds, id_map in results["contam_map"].items():
        file_path = os.path.join(output_dir, f"{tds}_contamination_map.jsonl.gz")
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
        print(f"[INFO] Wrote {file_path}")


###############################################################################
# Main aggregation function
###############################################################################


def aggregate_total(cfg: AggregateConfig):
    """Main function to aggregate contamination across all training datasets."""

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

    print(f"All datasets • {cfg.ngram_size}-gram ⇒ {union_contaminated}/{union_total} (fraction {union_frac:.4f})")

    union_dir = os.path.join(cfg.output_path, "union", str(cfg.ngram_size))
    os.makedirs(union_dir, exist_ok=True)

    # Union summary CSV
    union_csv = os.path.join(union_dir, "summary.csv")
    with fsspec.open(union_csv, "wt") as f:
        writer = csv.writer(f)
        writer.writerow(["training_dataset", "ngram", "shards", "total_examples", "contaminated", "fraction"])
        writer.writerow(["union", cfg.ngram_size, "all", union_total, union_contaminated, f"{union_frac:.6f}"])
    print(f"[INFO] Wrote {union_csv}")

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
    print(f"[INFO] Wrote {union_per_test_csv}")

    # Generate contamination matrix CSV
    matrix_path = os.path.join(cfg.output_path, "contamination_matrix.csv")
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

    print(f"[INFO] Wrote contamination matrix: {matrix_path}")
    print(
        f"[INFO] Matrix dimensions: {len(dataset_sizes)} evaluation datasets x {len(training_names)} training datasets"
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
    description="Aggregate contamination across all training datasets with individual and union views",
)

###############################################################################
# Main entry-point
###############################################################################

if __name__ == "__main__":
    executor_main(
        steps=[aggregate_total_step],
        description="Aggregate train-test contamination across all training datasets",
    )
