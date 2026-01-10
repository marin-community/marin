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

"""aggregate_proofpile.py

Aggregate train-test overlap results for Proofpile.

This is a Proofpile-focused variant of aggregate_total.py that derives the
training output path from the train_test_proofpile executor step and aggregates
directly from that step output.

Example Usage:
    python experiments/train_test_overlap/aggregate_proofpile.py --prefix gs://my-bucket
"""

import csv
import logging
import os
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import cast

import fsspec
from marin.execution.executor import ExecutorStep, executor_main, this_output_path
from marin.utils import fsspec_glob
from zephyr import Backend, Dataset, load_jsonl

from experiments.train_test_overlap.train_test_proofpile import STEPS as PROOFPILE_STEPS

logger = logging.getLogger(__name__)

PROOFPILE_STEP = PROOFPILE_STEPS[0]


def _default_ngram_size(step: ExecutorStep, fallback: int = 15) -> int:
    ngram = getattr(step.config, "ngram", None)
    if ngram is None:
        return fallback
    lengths = getattr(ngram, "ngram_length", None)
    if isinstance(lengths, int):
        return lengths
    if lengths:
        return max(lengths)
    return fallback


def _default_attribute_name(step: ExecutorStep, fallback: str = "ngram_overlap") -> str:
    return cast(str, getattr(step.config, "attribute_name", fallback))


def _default_eval_steps(step: ExecutorStep) -> list[ExecutorStep]:
    source = getattr(step.config, "decontaminate_source", None)
    if isinstance(source, list):
        return source
    return []


def extract_shard_metadata(shard_path: str, training_root: str, ngram_size: int) -> dict:
    """Extract test dataset name from shard path structure."""
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
        pipeline = Dataset.from_files(pattern, empty_glob_ok=True).flat_map(load_jsonl).map(lambda _: 1).reduce(sum)
        results = Backend.execute(pipeline)
        return results[0]

    size_map: dict[str, int] = {}
    for step in dataset_steps:
        ds_name = os.path.basename(cast(str, step).rstrip("/"))
        size_map[ds_name.split("-")[0]] = count_dir(cast(str, step))

    logger.info("Pre-computed dataset sizes:")
    for k, v in sorted(size_map.items()):
        logger.info("    %s: %s", k, v)
    return size_map


def _discover_attribute_shards(training_root: str, ngram_size: int) -> list[str]:
    patterns = [
        os.path.join(training_root, "**", str(ngram_size), "**", "*.jsonl*"),
        os.path.join(training_root, "**", str(ngram_size), "**", "*.parquet"),
        os.path.join(training_root, "**", str(ngram_size), "**", "*.vortex"),
    ]
    shard_paths: set[str] = set()
    for pattern in patterns:
        shard_paths.update(fsspec_glob(pattern))
    return sorted(shard_paths)


@dataclass
class AggregateConfig:
    """Aggregate overlap for Proofpile based on a training overlap step output."""

    training_output_path: str | ExecutorStep
    """Output path of the train_test_proofpile step."""

    output_path: str
    """Where to write the aggregated results."""

    ngram_size: int
    """N-gram size to process."""

    attribute_name: str = "ngram_overlap"
    """Attribute name in the JSON files."""

    eval_dataset_steps: list[ExecutorStep] = field(default_factory=list)
    """Evaluation dataset steps to compute sizes for."""


def aggregate_single_dataset(
    training_root: str, cfg: AggregateConfig, dataset_sizes: dict[str, int]
) -> tuple[dict, dict]:
    """Aggregate overlap for a single training dataset using Zephyr."""
    attr_key = f"{cfg.attribute_name}_{cfg.ngram_size}"
    training_name = os.path.basename(training_root.rstrip("/"))

    shard_paths = _discover_attribute_shards(training_root, cfg.ngram_size)
    if not shard_paths:
        logger.warning("No attribute shards found for %s", training_name)
        return {}, {}

    logger.info("Processing %s with %d shards", training_name, len(shard_paths))

    shard_metadata = [extract_shard_metadata(path, training_root, cfg.ngram_size) for path in shard_paths]
    path_to_test_ds = {meta["shard_path"]: meta["test_dataset"] for meta in shard_metadata}

    def extract_overlap_records(shard_path: str) -> Iterator[dict]:
        test_dataset = path_to_test_ds.get(shard_path, "unknown")

        logger.info("Loading from: %s", shard_path)
        for rec in load_jsonl(shard_path):
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

    intermediate_dir = os.path.join(cfg.output_path, ".intermediate", training_name)
    intermediate_paths = Backend.execute(
        Dataset.from_list(shard_paths)
        .flat_map(extract_overlap_records)
        .write_jsonl(f"{intermediate_dir}/overlap-{{shard:05d}}.jsonl.gz", skip_existing=True)
    )

    logger.info("Wrote %d intermediate files to %s", len(intermediate_paths), intermediate_dir)

    overall_unique: set[str] = set()
    overall_overlap: set[str] = set()
    per_test: dict[str, dict[str, set[str]]] = {ds: {"unique": set(), "overlap": set()} for ds in dataset_sizes.keys()}

    for intermediate_path in intermediate_paths:
        for rec in load_jsonl(intermediate_path):
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


def aggregate_proofpile(cfg: AggregateConfig) -> None:
    """Aggregate overlap results for Proofpile."""
    dataset_sizes = _compute_dataset_sizes(cfg.eval_dataset_steps)

    training_root = cast(str, cfg.training_output_path).rstrip("/")
    detailed_results, summary_results = aggregate_single_dataset(training_root, cfg, dataset_sizes)
    if not detailed_results:
        raise ValueError(f"No attribute shards found under {training_root}")

    training_name = detailed_results["training_name"]
    all_results = {training_name: detailed_results}

    union_unique = summary_results["overall_unique"]
    union_overlap = summary_results["overall_overlap"]
    union_per_test = summary_results["per_test"]

    union_total = len(union_unique)
    union_contaminated = len(union_overlap)
    union_frac = union_contaminated / union_total if union_total else 0.0

    logger.info(
        "Proofpile • %d-gram ⇒ %d/%d (fraction %.4f)",
        cfg.ngram_size,
        union_contaminated,
        union_total,
        union_frac,
    )

    summary_path = os.path.join(cfg.output_path, "summary.csv")
    with fsspec.open(summary_path, "wt") as f:
        writer = csv.writer(f)
        writer.writerow(["training_dataset", "ngram", "total_examples", "contaminated", "fraction"])

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

        writer.writerow(["union", cfg.ngram_size, union_total, union_contaminated, f"{union_frac:.6f}"])

    logger.info("Wrote consolidated summary: %s", summary_path)

    matrix_path = os.path.join(cfg.output_path, "overlap_matrix.csv")
    with fsspec.open(matrix_path, "wt") as f:
        writer = csv.writer(f)

        writer.writerow(["evaluation_dataset", training_name, "union"])

        for eval_ds in sorted(dataset_sizes.keys()):
            row = [eval_ds]
            tot = dataset_sizes.get(eval_ds, 0)

            if eval_ds in all_results[training_name]["per_test"]:
                cont = len(all_results[training_name]["per_test"][eval_ds]["overlap"])
                frac = cont / tot if tot else 0.0
                row.append(f"{frac:.6f}")
            else:
                row.append("0.000000")

            union_cont = len(union_per_test.get(eval_ds, {"overlap": set()})["overlap"])
            union_frac_eval = union_cont / tot if tot else 0.0
            row.append(f"{union_frac_eval:.6f}")

            writer.writerow(row)

    logger.info("Wrote overlap matrix: %s", matrix_path)
    logger.info("Matrix dimensions: %d evaluation datasets x 1 training dataset (+ union)", len(dataset_sizes))


def run_aggregate_proofpile(config: AggregateConfig) -> str:
    logger.info("Starting Proofpile overlap aggregation with config: %s", config)
    aggregate_proofpile(config)
    logger.info("Aggregation completed! Results written to %s", config.output_path)
    return config.output_path


def build_aggregate_proofpile_step(
    *,
    training_step: ExecutorStep = PROOFPILE_STEP,
    output_path: str | None = None,
    ngram_size: int | None = None,
    attribute_name: str | None = None,
    eval_dataset_steps: list[ExecutorStep] | None = None,
) -> ExecutorStep:
    """Create an ExecutorStep for aggregating Proofpile overlap results."""
    resolved_ngram_size = ngram_size or _default_ngram_size(training_step)
    resolved_attribute_name = attribute_name or _default_attribute_name(training_step)
    resolved_eval_steps = eval_dataset_steps or _default_eval_steps(training_step)

    cfg = AggregateConfig(
        training_output_path=training_step,
        output_path=output_path or this_output_path(),
        ngram_size=resolved_ngram_size,
        attribute_name=resolved_attribute_name,
        eval_dataset_steps=resolved_eval_steps,
    )

    return ExecutorStep(
        name="train_test_overlap/aggregate_proofpile",
        fn=run_aggregate_proofpile,
        config=cfg,
        description="Aggregate train-test overlap results for Proofpile",
    )


STEPS = [build_aggregate_proofpile_step()]

if __name__ == "__main__":
    executor_main(
        steps=STEPS,
        description="Aggregate train-test overlap results for Proofpile",
    )
