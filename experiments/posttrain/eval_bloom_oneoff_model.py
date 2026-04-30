#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run Bloom-format eval inference for a one-off HF model path."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from marin.alignment.align import EvalConfig, evaluate
from marin.alignment.evaluate import PromptFormat
from marin.alignment.inference_config import VLLMConfig
from marin.execution.executor import executor_main
from rigging.filesystem import REGION_TO_DATA_BUCKET

SPEC_PATH = str(Path(__file__).parent / "specs" / "openai_model_spec.jsonl")
BLOOM_PROMPTS_RELATIVE_PATH = "alignment/gpt-4.1-eval-split"
SUPPORTED_REGIONS = ("us-central1", "us-east5", "us-east1", "europe-west4")
REGION_ALIASES = {"eu-west4": "europe-west4"}
DEFAULT_TPU_TYPE = "v6e-4"
TENSOR_PARALLEL_SIZE = 4
MAX_MODEL_LEN = 4096
GPU_MEMORY_UTILIZATION = 0.9
TPU_CPU = 16
TPU_DISK = "100g"
DEFAULT_TPU_RAM = "256g"
TPU_RAM_BY_PREFIX = {"v5p": "128g"}
JUDGE_MODEL = "gpt-4.1"

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


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Run one-off Bloom-format eval inference for an HF model path.")
    parser.add_argument("--region", required=True, choices=sorted((*SUPPORTED_REGIONS, *REGION_ALIASES)))
    parser.add_argument("--model-path", required=True, help="Absolute gs:// HF model path to serve.")
    parser.add_argument("--name", required=True, help="Logical run name used for the output prefix.")
    parser.add_argument("--description", required=True, help="Human-readable description for the executor run.")
    parser.add_argument("--tpu-type", default=DEFAULT_TPU_TYPE, help="TPU type for inference workers.")
    parser.add_argument(
        "--non-preemptible",
        action="store_true",
        help="Request non-preemptible inference workers for this one-off run.",
    )
    parser.add_argument("--run-label", help="Optional suffix added to the output name.")
    return parser.parse_known_args()


def main() -> int:
    args, executor_args = parse_args()
    sys.argv = [sys.argv[0], *executor_args]
    region = _normalize_region(args.region)
    prompts_path = _regional_path(region, BLOOM_PROMPTS_RELATIVE_PATH)
    step_name = args.name if args.run_label is None else f"{args.name}_{args.run_label}"
    target_model = VLLMConfig(
        model=args.model_path,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        tpu_type=args.tpu_type,
        cpu=TPU_CPU,
        disk=TPU_DISK,
        ram=_tpu_ram_for_type(args.tpu_type),
        preemptible=not args.non_preemptible,
    )
    eval_steps = evaluate(
        name=step_name,
        target_model=target_model,
        prompts=prompts_path,
        spec=SPEC_PATH,
        eval_config=EVAL_CONFIG,
        judge_model=JUDGE_MODEL,
    )
    executor_main(
        steps=eval_steps[:1],
        description=f"Inference only: {args.description} on {args.tpu_type}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
