#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""BCG probe inference driver — runs local-vLLM inference via Iris for one
(target, region, tpu_type) combination on the 50-prompt BCG probe eval set.

Only the inference step runs; paired-rubric scoring happens separately via
`stage4_bcg_eval.py score-submit` against the Iris output.

Usage (typically wrapped by `uv run iris job run`):

    python experiments/posttrain/bcg_probe_infer.py \
        --target sft \
        --region us-east1 \
        --tpu-type v6e-4
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

from marin.alignment.align import EvalConfig, evaluate
from marin.alignment.evaluate import PromptFormat
from marin.alignment.inference_config import VLLMConfig
from marin.execution.executor import ExecutorStep, executor_main
from rigging.filesystem import REGION_TO_DATA_BUCKET


SPEC_PATH = str(Path(__file__).parent / "specs" / "openai_model_spec.jsonl")

SUPPORTED_REGIONS = ("us-central1", "us-east5", "us-east1", "europe-west4")
REGION_ALIASES = {"eu-west4": "europe-west4"}

PROMPTS_RELATIVE_PATH_DEFAULT = "alignment/bcg_probe_50_prompts"
# Full 2573-prompt variant lives at "alignment/bcg_full_2573_prompts" — set
# via --prompts-relative-path.

# TPU resource per type.
TPU_TYPE_CONFIG = {
    "v6e-4": {"tensor_parallel_size": 4, "ram": "256g"},
    "v6e-8": {"tensor_parallel_size": 8, "ram": "256g"},
    "v5p-8": {"tensor_parallel_size": 4, "ram": "128g"},
}

MAX_MODEL_LEN = 4096
GPU_MEMORY_UTILIZATION = 0.9
TPU_CPU = 16
TPU_DISK = "100g"


@dataclass(frozen=True)
class EvalTarget:
    key: str
    name: str
    description: str
    model_relative_path: str


TARGETS: dict[str, EvalTarget] = {
    "sft": EvalTarget(
        key="sft",
        name="bcg_probe_sft",
        description="Marin 8B Instruct SFT on 50 BCG-probe tension-corner prompts",
        model_relative_path="models/marin-community--marin-8b-instruct--0378f9c",
    ),
    "tune_lora_lr1e5_seed0_step1699": EvalTarget(
        key="tune_lora_lr1e5_seed0_step1699",
        name="bcg_probe_tune_lora_lr1e5_seed0_step1699",
        description="Marin tune_lora DPO lr=1e-5 seed0 step-1699 on 50 BCG-probe prompts",
        model_relative_path=(
            "checkpoints/dpo/tune_lora/"
            "bloom_speceval_v2_marin_lr1e5_seed0_b64_v5p8-41586d/hf-reexport-r3/step-1699"
        ),
    ),
}


EVAL_CONFIG = EvalConfig(
    prompt_format=PromptFormat.MARIN,
    temperature=0.7,
    max_tokens=1500,
    n=4,  # 4 samples per prompt for BCG marginal/joint estimation
    inference_batch_size=64,
    judge_workers=64,  # not used (no judge step)
    judge_batch_size=8,  # not used
    judge_max_tokens=1000,  # not used
)


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


def _target_vllm_config(target: EvalTarget, region: str, tpu_type: str) -> VLLMConfig:
    tpu_cfg = TPU_TYPE_CONFIG[tpu_type]
    return VLLMConfig(
        model=_regional_path(region, target.model_relative_path),
        tensor_parallel_size=tpu_cfg["tensor_parallel_size"],
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        tpu_type=tpu_type,
        cpu=TPU_CPU,
        disk=TPU_DISK,
        ram=tpu_cfg["ram"],
    )


def _step_name(target: EvalTarget, region: str, tpu_type: str) -> str:
    # Include region + tpu_type in step name so each (region, tpu_type) combo
    # writes to a distinct executor output prefix. First successful job wins.
    return f"{target.name}_{region.replace('-', '')}_{tpu_type.replace('-', '')}"


def build_inference_steps(
    target_key: str,
    region: str,
    tpu_type: str,
    prompts_relative_path: str,
    step_suffix: str = "",
) -> list[ExecutorStep]:
    target = TARGETS[target_key]
    prompts_path = _regional_path(region, prompts_relative_path)
    step_name = _step_name(target, region, tpu_type) + step_suffix
    eval_steps = evaluate(
        name=step_name,
        target_model=_target_vllm_config(target, region, tpu_type),
        prompts=prompts_path,
        spec=SPEC_PATH,
        eval_config=EVAL_CONFIG,
        judge_model="gpt-4.1",  # unused — we only include eval_steps[:1]
    )
    # Inference only — paired-rubric scoring happens in stage4_bcg_eval.py.
    return [eval_steps[0]]


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True, choices=sorted(TARGETS))
    parser.add_argument(
        "--region",
        required=True,
        choices=sorted((*SUPPORTED_REGIONS, *REGION_ALIASES)),
    )
    parser.add_argument(
        "--tpu-type",
        required=True,
        choices=sorted(TPU_TYPE_CONFIG),
    )
    parser.add_argument(
        "--prompts-relative-path",
        default=PROMPTS_RELATIVE_PATH_DEFAULT,
        help="Region-relative prompts path. Default: 50-point probe set. "
             "Use 'alignment/bcg_full_2573_prompts' for the full atlas.",
    )
    parser.add_argument(
        "--step-suffix",
        default="",
        help="Suffix appended to the executor step name so full-atlas runs do not collide with probe outputs.",
    )
    return parser.parse_known_args()


if __name__ == "__main__":
    args, executor_args = parse_args()
    sys.argv = [sys.argv[0], *executor_args]
    region = _normalize_region(args.region)
    steps = build_inference_steps(args.target, region, args.tpu_type, args.prompts_relative_path, args.step_suffix)
    target = TARGETS[args.target]
    executor_main(
        steps=steps,
        description=(
            f"BCG inference: {target.description} on {args.tpu_type} @ {region} "
            f"prompts={args.prompts_relative_path}"
        ),
    )
