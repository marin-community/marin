#!/usr/bin/env python3
"""debug_parallel_test.py

Run the parallel-overlap aggregator on *only the first N shards* of a
training-dataset output directory.  Intended for interactively debugging how
the union-contamination curve grows as we add more shards.

Usage inside Marin/Executor: the script defines several `ExecutorStep`s – one
per value of `shard_limit` – that depend on the upstream `finemath_dedupe_step`
(so they run *after* the attribute files have been produced).

The logic is identical to `aggregate_parallel_test.py` but we:
  * slice the discovered shard list to the first `shard_limit` elements
    (after sorting for determinism);
  * write the summary into `<output_path>/<ngram>/<shard_limit>/...` so you can
    compare contamination as more shards are included.
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

from experiments.train_test_overlap.dolma.aggregate_parallel_test import summarise_shard
from experiments.train_test_overlap.dolma.dedupe_total import proofpile_dedupe_step

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
from experiments.train_test_overlap.utils import (
    find_dataset_shards,
)
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


###############################################################################
# Config dataclass
###############################################################################


@dataclass(frozen=True)
class DebugAggregateConfig:
    """Aggregate contamination using at most `shard_limit` shards."""

    training_root: str | ExecutorStep  # root directory that holds attribute files
    output_path: str  # where to write the CSVs
    ngram_size: int = 15
    shard_limit: int | None = None  # if None ⇒ use *all* shards
    attribute_name: str = "ngram_overlap"
    max_in_flight: int = 64  # ray back-pressure
    dataset_steps: list[ExecutorStep] = None  # steps for eval datasets
    training_dataset_dir: str | None = None  # path where raw training shards live


###############################################################################
# Driver function
###############################################################################


def aggregate_debug(cfg: DebugAggregateConfig):
    attr_key = f"{cfg.attribute_name}_{cfg.ngram_size}"
    training_name = os.path.basename(cfg.training_root)

    # Pre-compute test dataset sizes
    dataset_sizes = _compute_dataset_sizes(cfg.dataset_steps)
    test_lookup = _build_test_lookup(cfg.dataset_steps)

    # 1. Discover shards and slice
    pattern = os.path.join(cfg.training_root, "**", str(cfg.ngram_size), "**", "*.jsonl*")
    shard_paths: list[str] = sorted(fsspec_glob(pattern))
    if cfg.shard_limit is not None:
        shard_paths = shard_paths[: cfg.shard_limit]
    if not shard_paths:
        raise FileNotFoundError(f"No attribute shards found under {cfg.training_root} for {cfg.ngram_size}-gram")

    print(f"[INFO] Using {len(shard_paths)} shard files")

    # ------------------------------------------------------------------
    # 2. Submit Ray tasks
    # ------------------------------------------------------------------
    task_args: list[tuple[str, str, str, str]] = []
    for shard in shard_paths:
        rel = os.path.relpath(shard, cfg.training_root)
        parts = rel.split(os.sep)
        try:
            idx_n = parts.index(str(cfg.ngram_size))
            test_ds_segment = parts[idx_n + 1] if idx_n + 1 < len(parts) else "unknown"
        except ValueError:
            test_ds_segment = parts[0]
        test_ds = test_ds_segment.split("-")[0]
        task_args.append((shard, test_ds, training_name, attr_key))

    refs: Iterator = simple_backpressure(summarise_shard, iter(task_args), cfg.max_in_flight, fetch_local=True)

    # 3. union contamination rate (plus per-test breakdown)
    overall_unique: set[str] = set()
    overall_overlap: set[str] = set()
    per_test: dict[str, dict[str, set[str]]] = {ds: {"unique": set(), "overlap": set()} for ds in dataset_sizes.keys()}

    # detailed mapping: test_ds -> id -> {training_dataset: set(shard_paths)}
    contam_map: dict[str, dict[str, dict[str, dict[str, set[str]]]]] = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(lambda: {"dedupe_shards": set(), "training_shards": set(), "test_shards": set()})
        )
    )

    # ------------------------------------------------------------------
    # Build lookup from dedupe-shard directory name -> original training shard
    # ------------------------------------------------------------------
    training_shard_lookup: dict[str, str] = {}

    def _strip_multi_ext(path: str) -> str:
        base = path
        while True:
            base, ext = os.path.splitext(base)
            if ext in (".zst", ".gz", ".jsonl", ".json"):
                continue
            else:
                return base

    def _rel_no_ext(p: str, root: str) -> str:
        rel = os.path.relpath(p, root)
        return _strip_multi_ext(rel)

    if cfg.training_dataset_dir:
        print("[DEBUG] Building training_shard_lookup", flush=True)
        for idx, original_path in enumerate(find_dataset_shards(cfg.training_dataset_dir)):
            key = _rel_no_ext(original_path, cfg.training_dataset_dir)
            training_shard_lookup[key] = original_path
            if idx < 10:
                print(f"[DEBUG] lookup key[{idx}] = {key} -> {original_path}", flush=True)
    else:
        print("[WARN] No training dataset directory provided", flush=True)
        print("[WARN] No training dataset directory provided\n\n", flush=True)

    for ref in refs:
        res = ray.get(ref)
        overall_unique.update(res["ids_seen"])
        overall_overlap.update(res["overlap_ids"])

        tds = res["test_dataset"]
        if tds not in per_test:
            # defensive: dataset not pre-registered (rare)
            per_test[tds] = {"unique": set(), "overlap": set()}
        per_test[tds]["unique"].update(res["ids_seen"])
        per_test[tds]["overlap"].update(res["overlap_ids"])

        # detailed shard mapping
        if res["overlap_ids"]:
            train_name = res["training_dataset"]
            dedupe_shard_path = res["shard_path"]
            print(f"[DEBUG] Processing dedupe shard: {dedupe_shard_path}", flush=True)

            # infer training shard dir component
            dedupe_parts = dedupe_shard_path.split("/")
            try:
                idx_train = dedupe_parts.index(train_name) + 1
                idx_ngram = dedupe_parts.index(str(cfg.ngram_size))
                rel_key = "/".join(dedupe_parts[idx_train:idx_ngram])
                print(f"[DEBUG] Derived rel_key: {rel_key}", flush=True)
                original_shard_path = training_shard_lookup.get(rel_key)
                if original_shard_path is None:
                    print("[WARN] rel_key not found in lookup", flush=True)
                else:
                    print(f"[DEBUG] Found original shard: {original_shard_path}", flush=True)
            except ValueError:
                original_shard_path = None

            for _id in res["overlap_ids"]:
                entry = contam_map[tds][_id][train_name]
                entry["dedupe_shards"].add(dedupe_shard_path)
                if original_shard_path:
                    entry["training_shards"].add(original_shard_path)

                test_fname = os.path.basename(dedupe_shard_path)
                for tp in test_lookup.get(test_fname, []):
                    entry["test_shards"].add(tp)

    total = len(overall_unique)
    contaminated = len(overall_overlap)
    frac = contaminated / total if total else 0.0

    print(
        f"[RESULT] {training_name} • {cfg.ngram_size}-gram • shards={len(shard_paths)} ⇒ {contaminated}/{total} (fraction {frac:.4f})"
    )

    # ------------------------------------------------------------------
    # 4. Write simple CSV
    # ------------------------------------------------------------------
    out_dir = os.path.join(cfg.output_path, str(cfg.ngram_size))
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "summary.csv")
    with fsspec.open(csv_path, "wt") as f:
        writer = csv.writer(f)
        writer.writerow(["training_dataset", "ngram", "shards", "total_examples", "contaminated", "fraction"])
        writer.writerow([training_name, cfg.ngram_size, len(shard_paths), total, contaminated, f"{frac:.6f}"])
    print(f"[INFO] Wrote {csv_path}")

    # ------------------------------------------------------------------
    # 5. Per-test-dataset CSV
    # ------------------------------------------------------------------
    per_test_csv = os.path.join(out_dir, "per_test_breakdown.csv")
    with fsspec.open(per_test_csv, "wt") as f:
        writer = csv.writer(f)
        writer.writerow(["test_dataset", "total_examples", "contaminated", "fraction"])
        for tds in sorted(dataset_sizes.keys()):
            tot = dataset_sizes.get(tds, 0)
            cont = len(per_test[tds]["overlap"])
            frac_t = cont / tot if tot else 0.0
            print(f"    {tds}: {cont}/{tot} -> {frac_t:.4%}")
            writer.writerow([tds, tot, cont, f"{frac_t:.6f}"])
    print(f"[INFO] Wrote {per_test_csv}")

    # ------------------------------------------------------------------
    # 6. Detailed contamination JSONL (id → shard paths)
    # ------------------------------------------------------------------
    # Write per-test dataset gzipped JSONL files
    for tds, id_map in contam_map.items():
        file_path = os.path.join(out_dir, f"{tds}_contamination_map.jsonl.gz")
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
# Build Executor steps (Finemath example)
###############################################################################

# Define which shard counts you'd like to inspect
SHARD_COUNTS = [1, 50, 100, 1000, 9999999]  # adjust as needed

steps = []
for n_shards in SHARD_COUNTS:
    cfg = DebugAggregateConfig(
        training_root=proofpile_dedupe_step,
        output_path=this_output_path(),
        ngram_size=15,
        shard_limit=n_shards,
        dataset_steps=EVAL_DATASET_STEPS,
        training_dataset_dir=proofpile_dedupe_step.config.dataset_dir,
    )
    step = ExecutorStep(
        name=f"debug/train_test_overlap/proofpile_15gram_{n_shards}shards",
        fn=aggregate_debug,
        config=cfg,
        description=f"Union contamination for Proofpile, 15-gram, first {n_shards} shards",
    )
    steps.append(step)

###############################################################################
# Main entry-point
###############################################################################

if __name__ == "__main__":
    executor_main(
        steps=steps,
        description="Debug contamination curve for Proofpile 15-gram (true denominators)",
    )
