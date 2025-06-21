#!/usr/bin/env python3
"""testset_size_summary.py

Utility to compute the *true* number of JSONL examples in every evaluation
 dataset we run de-contamination against.  It reads the *converted* Dolma
 files produced by `eval_datasets_overlap.py` (the `*_convert_dolma` executor
 steps) and counts records across **all** `.jsonl` / `.jsonl.gz` shards.

We rely on Marin's executor: pass the list of executor steps as config; at
runtime they are replaced by their resolved GCS output paths.  The script then
spawns one Ray task per dataset, counts the lines, and writes a CSV summary.
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from typing import Iterator, Tuple, List

import fsspec
import ray

from marin.core.runtime import simple_backpressure
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, output_path_of
from marin.utils import fsspec_glob

# Import every *_convert_dolma executor step
from experiments.train_test_overlap.eval_datasets_overlap import (
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
)

###############################################################################
# Config dataclass
###############################################################################


@dataclass(frozen=True)
class SizeSummaryConfig:
    datasets: List[str | ExecutorStep]  # list of executor steps or raw paths
    output_path: str
    max_in_flight: int = 32


###############################################################################
# Ray task – count records in one dataset directory
###############################################################################


@ray.remote(num_cpus=1)
def count_dataset(dataset_path: str) -> Tuple[str, int]:
    """Return (dataset_name, record_count)."""
    pattern = os.path.join(dataset_path.rstrip("/"), "**", "*.jsonl*")
    files = fsspec_glob(pattern)
    if not files:
        print(f"[WARN] No JSONL files found under {dataset_path}")
        return os.path.basename(dataset_path.rstrip("/")), 0

    total = 0
    for fp in files:
        try:
            with fsspec.open(fp, "rt", compression="infer") as f:
                for _ in f:
                    total += 1
        except Exception as e:
            print(f"[WARN] Error counting {fp}: {e}")
    return os.path.basename(dataset_path.rstrip("/")), total


###############################################################################
# Driver function
###############################################################################


def summarise_sizes(cfg: SizeSummaryConfig):
    # Resolve any ExecutorStep → path
    resolved_paths: List[str] = []
    for item in cfg.datasets:
        if isinstance(item, str):
            resolved_paths.append(item)
        else:
            resolved_paths.append(output_path_of(item))

    # Submit Ray tasks
    task_args = [(p,) for p in resolved_paths]
    refs: Iterator = simple_backpressure(count_dataset, iter(task_args), cfg.max_in_flight, fetch_local=True)

    results: List[Tuple[str, int]] = []
    for ref in refs:
        results.append(ray.get(ref))

    # Write CSV
    csv_path = os.path.join(cfg.output_path, "eval_testset_sizes.csv")
    os.makedirs(cfg.output_path, exist_ok=True)
    with fsspec.open(csv_path, "wt") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "examples"])
        for name, cnt in sorted(results, key=lambda x: x[0]):
            writer.writerow([name, cnt])
    print(f"[INFO] Wrote {csv_path}")

###############################################################################
# Build Executor step list and main
###############################################################################

EVAL_DATASET_STEPS: List[ExecutorStep] = [
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

size_cfg = SizeSummaryConfig(datasets=EVAL_DATASET_STEPS, output_path=this_output_path())

testset_size_step = ExecutorStep(
    name="train_test_overlap/eval_testset_sizes",
    fn=summarise_sizes,
    config=size_cfg,
    description="Count total examples in each evaluation dataset (Dolma format)",
)

if __name__ == "__main__":
    executor_main(
        steps=EVAL_DATASET_STEPS + [testset_size_step],
        description="Compute and log test-set sizes for all evaluation datasets",
    ) 