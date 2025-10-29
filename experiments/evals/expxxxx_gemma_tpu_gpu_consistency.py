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
#1603: Gemma TPU vs GPU Logprob Consistency

This experiment computes log probabilities for Gemma checkpoints on both TPU and GPU
backends and compares the resulting evaluation summaries to ensure hardware parity.
"""

from __future__ import annotations

import json
import logging
import os
import posixpath
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import fsspec
from fsspec.core import url_to_fs

from levanter.compat.hf_checkpoints import HFCheckpointConverter

from experiments.evals.exp1600_uncheatable_evals import (
    get_directory_friendly_name,
    truncate_model_name,
    uncheatable_eval_tokenized,
)
from experiments.evals.resource_configs import SINGLE_TPU_V5p_8_FULL
from experiments.models import ModelConfig as HFModelConfig, download_model_step
from marin.evaluation.log_probs import default_lm_log_probs
from marin.execution.executor import ExecutorStep, executor_main, output_path_of, this_output_path
from marin.processing.tokenize.data_configs import mixture_for_evaluation
from marin.resources import GpuConfig

logger = logging.getLogger(__name__)

TPU_RESOURCE = SINGLE_TPU_V5p_8_FULL
GPU_RESOURCE = GpuConfig(gpu_count=1, accelerator_type="A100-80G")

MODEL_NAME = "google/gemma-2-27b"
MODEL_REVISION = "main"
MAX_SAMPLES_PER_DATASET = 128


@dataclass(frozen=True)
class HardwareEvalSpec:
    name: str
    resource_config: Any
    wandb_tag: str
    per_device_batch_size: int = 1


HARDWARE_SPECS = (
    HardwareEvalSpec(
        name="tpu",
        resource_config=TPU_RESOURCE,
        wandb_tag="hw=tpu-v5p-8",
        per_device_batch_size=1,
    ),
    HardwareEvalSpec(
        name="gpu",
        resource_config=GPU_RESOURCE,
        wandb_tag="hw=gpu-a100-80g",
        per_device_batch_size=1,
    ),
)


@dataclass
class ConsistencyCheckConfig:
    tpu_output_path: str
    gpu_output_path: str
    tolerance: float = 5e-5
    output_path: str = field(default_factory=this_output_path)  # type: ignore[misc]


def compare_logprob_runs(config: ConsistencyCheckConfig) -> None:
    """Compare wandb summaries produced by TPU and GPU logprob runs."""
    tpu_summary = _load_wandb_summary(config.tpu_output_path)
    gpu_summary = _load_wandb_summary(config.gpu_output_path)

    numeric_keys = sorted(
        key
        for key in tpu_summary
        if key in gpu_summary and isinstance(tpu_summary[key], (int, float)) and isinstance(gpu_summary[key], (int, float))
    )
    if not numeric_keys:
        raise RuntimeError(
            "No numeric metrics found in wandb summaries. "
            "Ensure the evaluation runs completed successfully with wandb enabled."
        )

    differences: dict[str, float] = {
        key: abs(tpu_summary[key] - gpu_summary[key]) for key in numeric_keys
    }
    violations = {k: v for k, v in differences.items() if v > config.tolerance}

    fs, base_path = url_to_fs(config.output_path)
    output_file = posixpath.join(base_path, "comparison.json")
    fs.makedirs(base_path, exist_ok=True)
    with fs.open(output_file, "w") as f:
        json.dump(
            {
                "tolerance": config.tolerance,
                "differences": differences,
                "violations": violations,
                "tpu_output_path": config.tpu_output_path,
                "gpu_output_path": config.gpu_output_path,
            },
            f,
            indent=2,
            sort_keys=True,
        )
    logger.info("Wrote hardware comparison report to %s", output_file)

    if violations:
        raise ValueError(
            f"Found {len(violations)} metrics exceeding tolerance {config.tolerance}: {violations}"
        )


def _load_wandb_summary(output_path: str) -> dict[str, Any]:
    """Load wandb summary JSON from a logprob run output path."""
    fs, base_path = url_to_fs(output_path)
    candidate_paths = [
        posixpath.join(base_path, "wandb/latest-run/files/wandb-summary.json"),
    ]

    for candidate in candidate_paths:
        if fs.exists(candidate):
            with fs.open(candidate) as f:
                return json.load(f)

    wandb_root = posixpath.join(base_path, "wandb")
    if fs.exists(wandb_root):
        for entry in fs.ls(wandb_root):
            summary_path = posixpath.join(entry, "files", "wandb-summary.json")
            if fs.exists(summary_path):
                with fs.open(summary_path) as f:
                    return json.load(f)

    raise FileNotFoundError(
        f"Unable to locate wandb-summary.json under {output_path}. "
        "Confirm that the logprob run finished and that wandb logging is enabled."
    )


@lru_cache(maxsize=1)
def build_steps() -> list[ExecutorStep[Any]]:
    steps: list[ExecutorStep[Any]] = []

    tokenizer_name = MODEL_NAME
    eval_data = mixture_for_evaluation(uncheatable_eval_tokenized(tokenizer=tokenizer_name))

    model_identifier = f"{MODEL_NAME}@{MODEL_REVISION}"
    model_download_step = download_model_step(
        HFModelConfig(hf_repo_id=MODEL_NAME, hf_revision=MODEL_REVISION)
    )
    hf_model_config = HFCheckpointConverter.from_hf(model_identifier).config_from_hf_checkpoint(model_identifier)
    directory_name = get_directory_friendly_name(MODEL_NAME)

    eval_steps: dict[str, ExecutorStep[Any]] = {}
    for spec in HARDWARE_SPECS:
        eval_step = default_lm_log_probs(
            checkpoint=output_path_of(model_download_step),
            model=hf_model_config,
            data=eval_data,
            resource_config=spec.resource_config,
            checkpoint_is_hf=True,
            per_device_batch_size=spec.per_device_batch_size,
            max_samples_per_dataset=MAX_SAMPLES_PER_DATASET,
            name=f"{directory_name}-logprobs-{spec.name}",
            wandb_tags=[
                f"M={truncate_model_name(MODEL_NAME)}",
                "eval=gemma-logprob-consistency",
                spec.wandb_tag,
            ],
        )
        steps.append(eval_step)
        eval_steps[spec.name] = eval_step

    # Consistency check step compares the wandb summaries from both hardware runs.
    consistency_step = ExecutorStep(
        name=f"analysis/log_probs/{directory_name}-tpu-gpu-consistency-check",
        fn=compare_logprob_runs,
        config=ConsistencyCheckConfig(
            tpu_output_path=output_path_of(eval_steps["tpu"]),
            gpu_output_path=output_path_of(eval_steps["gpu"]),
        ),
    )
    steps.append(consistency_step)

    return steps


def main():
    if os.getenv("CI"):
        logger.info("Skipping experiment execution on CI environment, needs HF access.")
        return

    for step in build_steps():
        executor_main(steps=[step])


if __name__ == "__main__":
    main()
