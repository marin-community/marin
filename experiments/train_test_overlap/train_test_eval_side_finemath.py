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
Run eval-side train/test overlap detection for FineMath, one step per training file.

This builds one Bloom filter per training file and applies it to each eval dataset
so we can report overlaps with training-file provenance and run steps in parallel.
takes 27 min on central2
"""

import hashlib
import json
import logging
import os
from dataclasses import dataclass

import draccus
import fsspec

from marin.execution.executor import Executor, ExecutorMainConfig, ExecutorStep, this_output_path
from marin.processing.classification import decon as decon_mod
from marin.processing.classification.decon import DeconConfig, NGramConfig

from experiments.midtraining_datasets import finemath_3_plus
from experiments.train_test_overlap.eval_datasets_overlap import EVAL_DATASET_STEPS
from experiments.train_test_overlap.ngram_new.utils import parse_eval_dataset_name
from experiments.train_test_overlap.train_test_total import DatasetConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

TRAIN_METADATA_FILENAME = "train_metadata.json"


@dataclass(frozen=True)
class PerFileEvalOverlapConfig:
    """Configuration for per-file eval-side overlap detection."""

    train_path: str
    output_path: str
    eval_dataset_paths: list[str]
    ngram: NGramConfig
    attribute_name: str = "ngram_overlap"
    false_positive_rate: float = 1e-20
    processes: int = 1024
    train_text_field: str = "text"
    eval_text_field: str = "text"
    estimated_doc_count: int = 1_000_000
    write_train_metadata: bool = True
    training_root: str | None = None


FINEMATH_CONFIG = DatasetConfig(name="finemath", path=finemath_3_plus, text_field="text")

DEFAULT_NGRAM_CONFIG = NGramConfig(
    ngram_length=[20],
    overlap_threshold=1e-6,
    stride=0,
)


def _train_label(path: str) -> str:
    basename = os.path.basename(path)
    digest = hashlib.md5(path.encode()).hexdigest()[:6]
    return f"{basename}-{digest}"


def _write_train_metadata(output_path: str, train_path: str, training_root: str | None, label: str) -> None:
    metadata = {"train_path": train_path, "train_label": label}
    if training_root:
        root = training_root.rstrip("/")
        if train_path.startswith(root + "/"):
            metadata["train_relpath"] = train_path[len(root) + 1 :]
    metadata_path = os.path.join(output_path, TRAIN_METADATA_FILENAME)
    fs, fs_path = fsspec.core.url_to_fs(metadata_path)
    fs.makedirs(os.path.dirname(fs_path), exist_ok=True)
    with fs.open(fs_path, "w") as f:
        f.write(json.dumps(metadata))


def run_eval_side_per_file(config: PerFileEvalOverlapConfig) -> str:
    """Run eval-side overlap detection for a single training file."""
    ngram_lengths = (
        config.ngram.ngram_length if isinstance(config.ngram.ngram_length, list) else [config.ngram.ngram_length]
    )
    if config.write_train_metadata:
        _write_train_metadata(
            config.output_path, config.train_path, config.training_root, _train_label(config.train_path)
        )

    for ngram_len in ngram_lengths:
        current_ngram = NGramConfig(
            ngram_length=ngram_len,
            stride=config.ngram.stride,
            overlap_threshold=config.ngram.overlap_threshold,
        )

        train_config = DeconConfig(
            input_path=config.train_path,
            output_path=config.output_path,
            ngram=current_ngram,
            text_field=config.train_text_field,
            estimated_doc_count=config.estimated_doc_count,
            false_positive_rate=config.false_positive_rate,
            processes=config.processes,
            attribute_name=config.attribute_name,
        )

        bloom_path = os.path.join(config.output_path, "bloom", f"{ngram_len}.bin")
        bloom_path = decon_mod.build_filter(config.train_path, bloom_path, train_config)

        for eval_path in config.eval_dataset_paths:
            eval_path_str = str(eval_path)
            eval_name = parse_eval_dataset_name(eval_path_str)
            eval_output = os.path.join(config.output_path, str(ngram_len), eval_name)
            test_config = DeconConfig(
                input_path=eval_path_str,
                output_path=eval_output,
                attribute_name=f"{config.attribute_name}_{ngram_len}",
                ngram=current_ngram,
                text_field=config.eval_text_field,
                estimated_doc_count=config.estimated_doc_count,
                false_positive_rate=config.false_positive_rate,
                processes=config.processes,
            )
            decon_mod.mark_duplicates_bloom(eval_path_str, bloom_path, eval_output, test_config)

    return config.output_path


def _resolve_training_root(prefix: str, training_config: DatasetConfig) -> str:
    executor = Executor(prefix=prefix, executor_info_base_path=os.path.join(prefix, "experiments"))
    if not hasattr(training_config.path, "step"):
        return str(training_config.path)
    executor.compute_version(training_config.path.step, is_pseudo_dep=False)
    output_path = executor.output_paths[training_config.path.step]
    if training_config.path.name:
        return os.path.join(output_path, training_config.path.name)
    return output_path


def build_eval_side_steps(training_config: DatasetConfig, prefix: str) -> list[ExecutorStep]:
    training_root = _resolve_training_root(prefix, training_config)
    training_files = sorted(decon_mod._collect_input_files(training_root))
    steps: list[ExecutorStep] = []
    for train_path in training_files:
        label = _train_label(train_path)
        config = PerFileEvalOverlapConfig(
            train_path=train_path,
            output_path=this_output_path(),
            eval_dataset_paths=EVAL_DATASET_STEPS,
            ngram=DEFAULT_NGRAM_CONFIG,
            attribute_name="ngram_overlap",
            false_positive_rate=1e-20,
            processes=1024,
            train_text_field=training_config.text_field,
            eval_text_field="text",
            training_root=training_root,
        )
        steps.append(
            ExecutorStep(
                name=f"train_test_overlap/decon/eval_side_per_file/{training_config.name}/{label}",
                fn=run_eval_side_per_file,
                config=config,
                description=f"Eval-side overlap per file: {training_config.name}/{os.path.basename(train_path)}",
            )
        )
    return steps


@draccus.wrap()
def main(config: ExecutorMainConfig):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    prefix = config.prefix or os.environ.get("MARIN_PREFIX")
    if prefix is None:
        raise ValueError("Must specify --prefix or set MARIN_PREFIX")

    executor_info_base_path = config.executor_info_base_path
    if executor_info_base_path is None:
        executor_info_base_path = os.path.join(prefix, "experiments")

    steps = build_eval_side_steps(FINEMATH_CONFIG, prefix)
    executor = Executor(
        prefix=prefix,
        executor_info_base_path=executor_info_base_path,
        description="Run eval-side train/test overlap per FineMath training file",
    )
    executor.run(steps=steps, dry_run=config.dry_run, run_only=config.run_only, force_run_failed=config.force_run_failed)


if __name__ == "__main__":
    main()
