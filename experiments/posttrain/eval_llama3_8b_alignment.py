# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run Bloom-format alignment eval inference for Marin seed-0 targets.

This entry point intentionally runs inference only. GPT-4.1 judging happens in a
separate follow-up job via `experiments/posttrain/run_bloom_judge.py` after
manual review of the inference artifacts.

By default, this script submits inference steps for all seed-0 targets in the
current Bloom reproduction sweep. Use `--target` to restrict the run to a
subset.

Usage:

    uv run iris --config lib/iris/examples/marin.yaml job run \
        --no-wait \
        --job-name eval-marin-seed0-alignment-us-east1 \
        --cpu 4 --memory 16GB --disk 10GB \
        --region us-east1 \
        -- python experiments/posttrain/eval_llama3_8b_alignment.py \
            --region us-east1 \
            --target sft

    uv run iris --config lib/iris/examples/marin.yaml job run \
        --no-wait \
        --job-name eval-marin-seed0-alignment-us-east5 \
        --cpu 4 --memory 16GB --disk 10GB \
        --region us-east5 \
        -- python experiments/posttrain/eval_llama3_8b_alignment.py \
            --region us-east5 \
            --target beta001_lr5e7_seed0
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from marin.alignment.align import EvalConfig, evaluate
from marin.alignment.evaluate import PromptFormat
from marin.alignment.inference_config import VLLMConfig
from marin.execution.executor import ExecutorStep, executor_main, mirrored
from rigging.filesystem import REGION_TO_DATA_BUCKET

SPEC_PATH = str(Path(__file__).parent / "specs" / "openai_model_spec.jsonl")

SUPPORTED_REGIONS = ("us-central1", "us-east5", "us-east1", "europe-west4")
REGION_ALIASES = {
    "eu-west4": "europe-west4",
}

BLOOM_PROMPTS_RELATIVE_PATH = "alignment/gpt-4.1-eval-split"

DEFAULT_TPU_TYPE = "v6e-4"
TENSOR_PARALLEL_SIZE = 4
MAX_MODEL_LEN = 4096
GPU_MEMORY_UTILIZATION = 0.9
TPU_CPU = 16
TPU_DISK = "100g"
DEFAULT_TPU_RAM = "256g"
TPU_RAM_BY_PREFIX = {
    "v5p": "128g",
}
JUDGE_MODEL = "gpt-4.1"
MIRRORED_MODEL_BUDGET_GB = 80

SFT_MODEL_RELATIVE_PATH = "models/marin-community--marin-8b-instruct--0378f9c"


def _normalize_region(region: str) -> str:
    normalized = REGION_ALIASES.get(region, region)
    if normalized not in SUPPORTED_REGIONS:
        supported = ", ".join(sorted((*SUPPORTED_REGIONS, *REGION_ALIASES)))
        raise ValueError(f"Unsupported region {region!r}. Expected one of: {supported}")
    return normalized


def _region_prefix(region: str) -> str:
    bucket = REGION_TO_DATA_BUCKET[_normalize_region(region)]
    return f"gs://{bucket}"


def _regional_path(region: str, relative_path: str) -> str:
    return f"{_region_prefix(region)}/{relative_path.strip('/')}"


def _tpu_ram_for_type(tpu_type: str) -> str:
    for prefix, ram in TPU_RAM_BY_PREFIX.items():
        if tpu_type.startswith(prefix):
            return ram
    return DEFAULT_TPU_RAM


@dataclass(frozen=True)
class EvalTarget:
    key: str
    name: str
    description: str
    model_relative_path: str | None = None
    model_path: str | None = None
    mirror_model: bool = False
    mirror_budget_gb: float = MIRRORED_MODEL_BUDGET_GB


TARGETS = {
    "sft": EvalTarget(
        key="sft",
        name="marin_8b_instruct_bloom_speceval",
        description="Marin 8B Instruct SFT on 2,576 Bloom eval prompts (46 statements)",
        model_relative_path=SFT_MODEL_RELATIVE_PATH,
    ),
    "beta001_lr5e7_seed0": EvalTarget(
        key="beta001_lr5e7_seed0",
        name="marin_dpo_beta001_lr5e7_seed0_bloom_speceval",
        description="Marin DPO beta0.01 lr5e-7 seed0 on 2,576 Bloom eval prompts (46 statements)",
        model_relative_path=("checkpoints/dpo/" "bloom_speceval_v2_marin_instruct_beta0.01_seed0-e2b733/hf/step-849"),
    ),
    "beta001_lr75e7_seed0": EvalTarget(
        key="beta001_lr75e7_seed0",
        name="marin_dpo_beta001_lr75e7_seed0_bloom_speceval",
        description="Marin DPO beta0.01 lr7.5e-7 seed0 on 2,576 Bloom eval prompts (46 statements)",
        model_relative_path=(
            "checkpoints/dpo/" "bloom_speceval_v2_marin_instruct_beta0.01_lr7.5e-7_seed0-872f2e/hf/step-849"
        ),
    ),
    "beta01_lr5e7_seed0": EvalTarget(
        key="beta01_lr5e7_seed0",
        name="marin_dpo_beta01_lr5e7_seed0_bloom_speceval",
        description="Marin DPO beta0.1 lr5e-7 seed0 on 2,576 Bloom eval prompts (46 statements)",
        model_relative_path=("checkpoints/dpo/" "bloom_speceval_v2_marin_instruct_beta0.1_seed0-4f9703/hf/step-849"),
    ),
    "beta01_lr75e7_seed0": EvalTarget(
        key="beta01_lr75e7_seed0",
        name="marin_dpo_beta01_lr75e7_seed0_bloom_speceval",
        description="Marin DPO beta0.1 lr7.5e-7 seed0 on 2,576 Bloom eval prompts (46 statements)",
        model_relative_path=(
            "checkpoints/dpo/" "bloom_speceval_v2_marin_instruct_beta0.1_lr7.5e-7_seed0-cc50ad/hf/step-849"
        ),
    ),
    "compare_lora_beta01_seed0_b64_step1699": EvalTarget(
        key="compare_lora_beta01_seed0_b64_step1699",
        name="marin_dpo_compare_lora_beta01_seed0_b64_step1699_bloom_speceval",
        description="Marin full DPO beta0.1 batch64 seed0 step1699 on 2,576 Bloom eval prompts (46 statements)",
        model_relative_path=(
            "checkpoints/dpo/compare_lora/" "bloom_speceval_v2_beta0.1_seed0_b64_v5p16-68f963/hf/step-1699"
        ),
    ),
    "tune_lora_lr1e5_seed0_step1699": EvalTarget(
        key="tune_lora_lr1e5_seed0_step1699",
        name="marin_dpo_tune_lora_lr1e5_seed0_step1699_bloom_speceval",
        description=(
            "Marin tune_lora DPO lr1e5 seed0 step1699 clean re-export on 2,576 Bloom eval prompts (46 statements)"
        ),
        model_relative_path=(
            "checkpoints/dpo/tune_lora/" "bloom_speceval_v2_marin_lr1e5_seed0_b64_v5p8-41586d/hf-reexport-r3/step-1699"
        ),
    ),
    "tune_lora_lr5e6_seed0_step1699": EvalTarget(
        key="tune_lora_lr5e6_seed0_step1699",
        name="marin_dpo_tune_lora_lr5e6_seed0_step1699_bloom_speceval",
        description=(
            "Marin tune_lora DPO lr5e6 seed0 step1699 clean re-export on 2,576 Bloom eval prompts (46 statements)"
        ),
        model_relative_path=(
            "checkpoints/dpo/tune_lora/" "bloom_speceval_v2_marin_lr5e6_seed0_b64_v5p8-274540/hf-fixed-r1/step-1699"
        ),
    ),
    "azero_lr1e6_seed0_step1699": EvalTarget(
        key="azero_lr1e6_seed0_step1699",
        name="marin_dpo_lora_azero_lr1e6_seed0_step1699_bloom_speceval",
        description="A=0 LoRA DPO lr1e-6 seed0 step1699 on 2,576 Bloom eval prompts (46 statements)",
        model_relative_path=(
            "checkpoints/dpo/tune_lora/" "lora_bloom_speceval_v2_lr1e6_seed0_b64_v5p8_azero-8e1101/hf/step-1699"
        ),
        mirror_model=True,
    ),
    "azero_lr1e5_seed0_step1699": EvalTarget(
        key="azero_lr1e5_seed0_step1699",
        name="marin_dpo_lora_azero_lr1e5_seed0_step1699_bloom_speceval",
        description="A=0 LoRA DPO lr1e-5 seed0 step1699 on 2,576 Bloom eval prompts (46 statements)",
        model_relative_path=(
            "checkpoints/dpo/tune_lora/" "lora_bloom_speceval_v2_lr1e5_seed0_b64_v5p8_azero-d93e61/hf/step-1699"
        ),
        mirror_model=True,
    ),
    "azero_lr8p75e6_seed0_step1699": EvalTarget(
        key="azero_lr8p75e6_seed0_step1699",
        name="marin_dpo_lora_azero_lr8p75e6_seed0_step1699_bloom_speceval",
        description="A=0 LoRA DPO lr8.75e-6 seed0 step1699 on 2,576 Bloom eval prompts (46 statements)",
        model_relative_path=(
            "checkpoints/dpo/tune_lora/" "lora_bloom_speceval_lr8p75e6_seed0_b64_v5p8_azero-4a1bf7/hf/step-1699"
        ),
        mirror_model=True,
    ),
    "azero_lr5e6_seed0_step1699": EvalTarget(
        key="azero_lr5e6_seed0_step1699",
        name="marin_dpo_lora_azero_lr5e6_seed0_step1699_bloom_speceval",
        description="A=0 LoRA DPO lr5e-6 seed0 step1699 on 2,576 Bloom eval prompts (46 statements)",
        model_relative_path=(
            "checkpoints/dpo/tune_lora/" "lora_bloom_speceval_v2_lr5e6_seed0_b64_v5p8_azero-a9e388/hf/step-1699"
        ),
        mirror_model=True,
    ),
}

DEFAULT_TARGET_KEYS = tuple(key for key in TARGETS if not key.startswith("azero_"))

EVAL_CONFIG = EvalConfig(
    prompt_format=PromptFormat.BLOOM,
    temperature=0.7,
    max_tokens=1500,
    n=3,
    inference_batch_size=256,
    judge_workers=64,
    judge_batch_size=8,
    judge_max_tokens=1000,
)


def _target_vllm_config(target: EvalTarget, region: str, tpu_type: str) -> VLLMConfig:
    if target.mirror_model:
        if target.model_relative_path is None:
            raise ValueError(f"Mirrored target {target.key} must define model_relative_path")
        model = cast(str, mirrored(target.model_relative_path, budget_gb=target.mirror_budget_gb))
    elif target.model_path is not None:
        model = target.model_path
    elif target.model_relative_path is not None:
        model = _regional_path(region, target.model_relative_path)
    else:
        raise ValueError(f"Target {target.key} does not define a model path")
    return VLLMConfig(
        model=model,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        tpu_type=tpu_type,
        cpu=TPU_CPU,
        disk=TPU_DISK,
        ram=_tpu_ram_for_type(tpu_type),
    )


def _step_name(target: EvalTarget, run_label: str | None) -> str:
    return target.name if run_label is None else f"{target.name}_{run_label}"


def build_inference_steps(
    target_keys: list[str], region: str, tpu_type: str, run_label: str | None
) -> list[ExecutorStep]:
    prompts_path = _regional_path(region, BLOOM_PROMPTS_RELATIVE_PATH)
    steps: list[ExecutorStep] = []
    for key in target_keys:
        target = TARGETS[key]
        eval_steps = evaluate(
            name=_step_name(target, run_label),
            target_model=_target_vllm_config(target, region, tpu_type),
            prompts=prompts_path,
            spec=SPEC_PATH,
            eval_config=EVAL_CONFIG,
            judge_model=JUDGE_MODEL,
        )
        # Inference only — judge step runs separately after manual review.
        steps.extend(eval_steps[:1])
    return steps


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Run Marin Bloom-format eval inference for one or more seed-0 targets.")
    parser.add_argument(
        "--region",
        required=True,
        choices=sorted((*SUPPORTED_REGIONS, *REGION_ALIASES)),
        help="GCP region to use for prompts, checkpoints, and TPU placement.",
    )
    parser.add_argument(
        "--target",
        dest="targets",
        action="append",
        choices=sorted(TARGETS),
        help="Target key to run. Repeat the flag to run a subset. Defaults to all targets.",
    )
    parser.add_argument(
        "--tpu-type",
        default=DEFAULT_TPU_TYPE,
        help="TPU type for inference workers. Defaults to v6e-4.",
    )
    parser.add_argument(
        "--run-label",
        help="Optional suffix added to step names so reruns land in a distinct output prefix.",
    )
    return parser.parse_known_args()


def _run_description(target_keys: list[str], tpu_type: str, run_label: str | None) -> str:
    if len(target_keys) == 1:
        label_suffix = "" if run_label is None else f" [{run_label}]"
        return f"Inference only: {TARGETS[target_keys[0]].description} on {tpu_type}{label_suffix}"
    target_labels = ", ".join(target_keys)
    label_suffix = "" if run_label is None else f" [{run_label}]"
    return f"Inference only: Marin seed-0 Bloom eval sweep for {target_labels} on {tpu_type}{label_suffix}"


if __name__ == "__main__":
    args, executor_args = parse_args()
    sys.argv = [sys.argv[0], *executor_args]
    region = _normalize_region(args.region)
    target_keys = args.targets or list(DEFAULT_TARGET_KEYS)
    executor_main(
        steps=build_inference_steps(target_keys, region, args.tpu_type, args.run_label),
        description=_run_description(target_keys, args.tpu_type, args.run_label),
    )
