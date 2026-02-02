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

"""
Aggregate eval-side overlap outputs from per-file Bloom filters.

This produces ngram_new-style overlap stats for eval instances and reports
per-training-file provenance using the per-file eval-side outputs.
"""

import json
import logging
import os
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field

import fsspec
from marin.execution.executor import ExecutorStep, InputName, executor_main, this_output_path
from marin.processing.classification import decon as decon_mod
from marin.utils import fsspec_glob, fsspec_isdir
from zephyr import load_file

from experiments.train_test_overlap.eval_datasets_overlap import EVAL_DATASET_STEPS
from experiments.train_test_overlap.ngram_new.overlap_reduce import _attach_instance_links
from experiments.train_test_overlap.ngram_new.utils import parse_eval_dataset_name
from experiments.train_test_overlap.train_test_eval_side_finemath import TRAIN_METADATA_FILENAME

logger = logging.getLogger(__name__)

DEFAULT_NGRAM_LENGTHS = [15]


@dataclass(frozen=True)
class DeconEvalReduceConfig:
    """Configuration for eval-side overlap reduction."""

    eval_side_base_path: str | None
    eval_side_output_paths: list[str] | list[ExecutorStep] | None
    output_path: str
    eval_dataset_paths: str | list[str] | None = None
    eval_text_field: str = "text"
    ngram_lengths: list[int] = field(default_factory=lambda: DEFAULT_NGRAM_LENGTHS)
    attribute_name: str = "ngram_overlap"
    skip_existing: bool = True
    write_details: bool = False
    training_name: str = "finemath"
    train_metadata_filename: str = TRAIN_METADATA_FILENAME


def _normalize_paths(paths: str | Sequence[str]) -> list[str]:
    if isinstance(paths, str):
        return [paths]
    return [str(path) for path in paths]


def _collect_eval_file_specs(eval_paths: str | Sequence[str]) -> list[dict]:
    specs: list[dict] = []
    for eval_path in _normalize_paths(eval_paths):
        eval_dataset = parse_eval_dataset_name(eval_path)
        for file_path in decon_mod._collect_input_files(eval_path):
            specs.append({"path": file_path, "eval_dataset": eval_dataset})
    return specs


def _compute_eval_counts_and_links(
    eval_dataset_paths: str | Sequence[str], text_field: str
) -> tuple[dict[str, int], dict[tuple[str, str], str]]:
    counts: dict[str, set[str]] = defaultdict(set)
    links: dict[tuple[str, str], str] = {}
    for spec in _collect_eval_file_specs(eval_dataset_paths):
        eval_dataset = spec["eval_dataset"]
        for record in load_file(spec["path"]):
            text = record.get(text_field)
            if text is None:
                continue
            instance_id = str(decon_mod._record_id(record))
            counts[eval_dataset].add(instance_id)
            links[(eval_dataset, instance_id)] = spec["path"]
    return {dataset: len(instance_ids) for dataset, instance_ids in counts.items()}, links


def _discover_training_outputs(eval_side_base_path: str) -> list[str]:
    base_path = eval_side_base_path.rstrip("/")
    candidates = fsspec_glob(f"{base_path}/*")
    outputs = [path for path in candidates if fsspec_isdir(path)]
    return sorted(outputs)


def _resolve_training_output_paths(config: DeconEvalReduceConfig) -> list[str]:
    if config.eval_side_output_paths:
        outputs: list[str] = []
        for base_path in config.eval_side_output_paths:
            outputs.extend(_discover_training_outputs(str(base_path)))
        return sorted(outputs)
    if config.eval_side_base_path:
        return _discover_training_outputs(config.eval_side_base_path)
    raise ValueError("eval_side_base_path or eval_side_output_paths must be provided")


def _load_train_metadata(output_path: str, metadata_filename: str) -> dict:
    metadata_path = os.path.join(output_path, metadata_filename)
    fs, fs_path = fsspec.core.url_to_fs(metadata_path)
    if not fs.exists(fs_path):
        return {}
    with fs.open(fs_path, "r") as f:
        return json.loads(f.read())


def run_decon_eval_reduce(config: DeconEvalReduceConfig) -> str:
    """Aggregate eval-side overlap outputs into ngram_new-style stats."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    stats_dir = os.path.join(config.output_path, "stats")
    stats_path = os.path.join(stats_dir, "overlap_stats.jsonl")
    train_path_stats_path = os.path.join(stats_dir, "overlap_stats_by_train_path.jsonl")
    details_path = os.path.join(stats_dir, "overlap_details.jsonl.gz")
    stats_fs, stats_fs_path = fsspec.core.url_to_fs(stats_path)
    train_fs, train_fs_path = fsspec.core.url_to_fs(train_path_stats_path)
    details_fs, details_fs_path = fsspec.core.url_to_fs(details_path)
    stats_exists = stats_fs.exists(stats_fs_path)
    train_path_exists = train_fs.exists(train_fs_path)
    details_exists = details_fs.exists(details_fs_path) if config.write_details else True
    if config.skip_existing and stats_exists and train_path_exists and details_exists:
        logger.info(
            "Skipping reduce; outputs exist at %s, %s, and %s",
            stats_path,
            train_path_stats_path,
            details_path,
        )
        return config.output_path

    n_values = sorted(set(config.ngram_lengths))
    if not n_values:
        raise ValueError("ngram_lengths must be non-empty")

    if config.eval_dataset_paths is None:
        raise ValueError("eval_dataset_paths is required for eval-side reduction")
    counts, link_map = _compute_eval_counts_and_links(config.eval_dataset_paths, config.eval_text_field)

    training_outputs = _resolve_training_output_paths(config)
    if not training_outputs:
        raise ValueError("No training-file outputs found for reduction")

    attr_key_template = f"{config.attribute_name}_{{n}}"
    overlap_map: dict[tuple[str, int], set[str]] = defaultdict(set)
    overlap_by_train: dict[tuple[str, int, str], set[str]] = defaultdict(set)

    details_handle = None
    if config.write_details and (not config.skip_existing or not details_exists):
        details_fs.makedirs(os.path.dirname(details_fs_path), exist_ok=True)
        details_handle = fsspec.open(details_path, "wt", compression="infer")

    try:
        for training_output in training_outputs:
            metadata = _load_train_metadata(training_output, config.train_metadata_filename)
            train_path = metadata.get("train_path", training_output)
            for n_val in n_values:
                eval_root = os.path.join(training_output, str(n_val))
                eval_dirs = [path for path in fsspec_glob(f"{eval_root}/*") if fsspec_isdir(path)]
                for eval_dir in eval_dirs:
                    eval_dataset = parse_eval_dataset_name(eval_dir)
                    try:
                        attr_files = decon_mod._collect_input_files(eval_dir)
                    except FileNotFoundError:
                        continue
                    attr_key = attr_key_template.format(n=n_val)
                    for file_path in attr_files:
                        for record in load_file(file_path):
                            attrs = record.get("attributes", {})
                            if not attrs.get(attr_key):
                                continue
                            instance_id = record.get("id")
                            if instance_id is None:
                                instance_id = decon_mod._record_id(record)
                            instance_id = str(instance_id)
                            overlap_map[(eval_dataset, n_val)].add(instance_id)
                            overlap_by_train[(eval_dataset, n_val, train_path)].add(instance_id)
                            if details_handle is not None:
                                details_handle.write(
                                    json.dumps(
                                        {
                                            "eval_dataset": eval_dataset,
                                            "n": n_val,
                                            "eval_instance_id": instance_id,
                                            "train_path": train_path,
                                        }
                                    )
                                    + "\n"
                                )
    finally:
        if details_handle is not None:
            details_handle.close()

    stats_fs.makedirs(os.path.dirname(stats_fs_path), exist_ok=True)
    with stats_fs.open(stats_fs_path, "w") as f:
        for eval_dataset, num_instances in sorted(counts.items()):
            for n_val in n_values:
                instance_ids = sorted(overlap_map.get((eval_dataset, n_val), set()))
                f.write(
                    json.dumps(
                        {
                            "eval_dataset": eval_dataset,
                            "n": n_val,
                            "num_instances": num_instances,
                            "instance_ids": instance_ids,
                            "instance_links": _attach_instance_links(eval_dataset, instance_ids, link_map),
                        }
                    )
                    + "\n"
                )

    train_fs.makedirs(os.path.dirname(train_fs_path), exist_ok=True)
    with train_fs.open(train_fs_path, "w") as f:
        for eval_dataset, n_val, train_path in sorted(overlap_by_train.keys()):
            instance_ids = sorted(overlap_by_train[(eval_dataset, n_val, train_path)])
            f.write(
                json.dumps(
                    {
                        "eval_dataset": eval_dataset,
                        "n": n_val,
                        "train_path": train_path,
                        "train_doc_ids": [],
                        "instance_ids": instance_ids,
                        "instance_links": _attach_instance_links(eval_dataset, instance_ids, link_map),
                        "overlap_count": len(instance_ids),
                    }
                )
                + "\n"
            )

    success_path = os.path.join(config.output_path, ".SUCCESS")
    success_fs, success_fs_path = fsspec.core.url_to_fs(success_path)
    success_fs.makedirs(os.path.dirname(success_fs_path), exist_ok=True)
    with success_fs.open(success_fs_path, "w") as f:
        f.write("")

    logger.info("Wrote eval-side overlap stats to %s", stats_path)
    if config.write_details:
        logger.info("Wrote eval-side overlap details to %s", details_path)
    return config.output_path


def build_decon_eval_reduce_step(
    eval_side_base_path: str | None = None,
    eval_side_output_paths: list[ExecutorStep] | None = None,
    output_path: str | None = None,
    ngram_lengths: list[int] | None = None,
    eval_dataset_paths: list[ExecutorStep] | None = None,
    training_name: str = "finemath",
) -> ExecutorStep:
    cfg = DeconEvalReduceConfig(
        eval_side_base_path=eval_side_base_path
        or InputName.hardcoded("train_test_overlap/decon/eval_side_per_file/finemath"),
        eval_side_output_paths=eval_side_output_paths,
        output_path=output_path or this_output_path(),
        eval_dataset_paths=eval_dataset_paths or EVAL_DATASET_STEPS,
        ngram_lengths=ngram_lengths or DEFAULT_NGRAM_LENGTHS,
        training_name=training_name,
    )

    return ExecutorStep(
        name=f"train_test_overlap/decon/decon_eval_reduce/{training_name}",
        fn=run_decon_eval_reduce,
        config=cfg,
        description=f"Reduce eval-side decon overlap for {training_name}",
    )


STEPS = [build_decon_eval_reduce_step()]

if __name__ == "__main__":
    executor_main(
        steps=STEPS,
        description="Aggregate eval-side decon overlap outputs into ngram_new-style stats",
    )
