#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Scratch-only pressure-test runner for the Iris vLLM design.

from __future__ import annotations

import argparse
import importlib
import logging
import os
import socket
import time
from dataclasses import dataclass

DEFAULT_TASK = "standard_humaneval_smoke"
DEFAULT_MODEL_NAME = "qwen3-0.6b-standard-smoke"
DEFAULT_MODEL_PATH = "Qwen/Qwen3-0.6B"
DEFAULT_TOKENIZER = "Qwen/Qwen3-0.6B"
DEFAULT_OUTPUT_PATH = "tmp/iris_pressure_test_20260504/results/standard_humaneval_smoke"
DEFAULT_MAX_NUM_BATCHED_TOKENS = 1024
DEFAULT_MAX_GEN_TOKS = 128

VLLM_ENV_DEFAULTS = {
    "MARIN_VLLM_MODE": "native",
    "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
    "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1",
    "VLLM_TPU_DISABLE_TOPK_TOPP_OPTIMIZATION": "1",
    "VLLM_TPU_SKIP_PRECOMPILE": "1",
    "HF_ALLOW_CODE_EVAL": "1",
    "WANDB_MODE": "offline",
}


def install_lm_eval_vllm_import_shim() -> None:
    """Keep lm-eval importable with the vLLM package currently installed on Iris."""
    try:
        vllm_utils = importlib.import_module("vllm.utils")
    except Exception:
        return

    if hasattr(vllm_utils, "get_open_port"):
        return

    def get_open_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("", 0))
            return int(sock.getsockname()[1])

    vllm_utils.get_open_port = get_open_port


install_lm_eval_vllm_import_shim()

# Imports are intentionally below the shim: lm-eval (pulled in by marin.evaluation.run) imports
# vllm.utils.get_open_port at module load, which is missing on the installed vLLM build.
from fray.cluster import ResourceConfig  # noqa: E402
from marin.evaluation.evaluation_config import EvalTaskConfig, EvaluationConfig  # noqa: E402
from marin.evaluation.run import evaluate  # noqa: E402


@dataclass(frozen=True)
class TaskSpec:
    task: EvalTaskConfig
    apply_chat_template: bool
    note: str


TASKS = {
    "standard_humaneval_smoke": TaskSpec(
        task=EvalTaskConfig("humaneval", 0, task_alias="humaneval_0shot"),
        apply_chat_template=True,
        note="MVP smoke: stock HumanEval generation through local-chat-completions.",
    ),
    "humaneval_0shot": TaskSpec(
        task=EvalTaskConfig("humaneval", 0, task_alias="humaneval_0shot"),
        apply_chat_template=True,
        note="Stock lm-eval HumanEval generation. This is the honest few-shot setting lm-eval records.",
    ),
    "mmlu_sl_verb_5shot": TaskSpec(
        task=EvalTaskConfig("mmlu_sl_verb", 5, task_alias="mmlu_sl_verb_5shot"),
        apply_chat_template=False,
        note="Canonical scoring follow-up; currently gated on TPU vLLM prompt-logprob support.",
    ),
    "humaneval_5shot": TaskSpec(
        task=EvalTaskConfig("humaneval", 5, task_alias="humaneval_5shot"),
        apply_chat_template=True,
        note="Legacy pressure-test alias. Stock lm-eval may still record HumanEval as 0-shot.",
    ),
}


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for key, value in VLLM_ENV_DEFAULTS.items():
        os.environ.setdefault(key, value)

    task_spec = TASKS[args.task]
    engine_kwargs = {
        "tokenizer": args.tokenizer,
        "max_num_batched_tokens": args.max_num_batched_tokens,
    }
    if args.max_model_len is not None:
        engine_kwargs["max_model_len"] = args.max_model_len
    if args.lm_eval_max_length is not None:
        engine_kwargs["max_length"] = args.lm_eval_max_length
    generation_params = {}
    if args.max_gen_toks is not None:
        generation_params["max_gen_toks"] = args.max_gen_toks
    config = EvaluationConfig(
        evaluator="lm_evaluation_harness",
        resource_config=ResourceConfig.with_tpu(args.tpu_type),
        model_name=args.model_name,
        model_path=args.model_path,
        evaluation_path=args.output_path,
        evals=[task_spec.task],
        max_eval_instances=args.limit,
        engine_kwargs=engine_kwargs,
        generation_params=generation_params or None,
        discover_latest_checkpoint=False,
        apply_chat_template=task_spec.apply_chat_template,
        wandb_tags=["iris-pressure-test", args.task],
    )

    print("scratch_runner_config_start")
    print(f"task={args.task}")
    print(f"model_name={args.model_name}")
    print(f"model_path={args.model_path}")
    print(f"tokenizer={args.tokenizer}")
    print(f"output_path={args.output_path}")
    print(f"tpu_type={args.tpu_type}")
    print(f"limit={args.limit}")
    print(f"apply_chat_template={task_spec.apply_chat_template}")
    print(f"task_note={task_spec.note}")
    print(f"eval_task_name={task_spec.task.name}")
    print(f"eval_num_fewshot={task_spec.task.num_fewshot}")
    print(f"eval_task_alias={task_spec.task.task_alias}")
    print(f"engine_kwargs={engine_kwargs}")
    print(f"generation_params={generation_params}")
    print("scratch_runner_config_end")

    if args.dry_run:
        return

    start = time.monotonic()
    evaluate(config)
    elapsed = time.monotonic() - start
    print(f"scratch_runner_elapsed_seconds={elapsed:.3f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=sorted(TASKS), default=DEFAULT_TASK)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--tpu-type", default="v5p-8")
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--lm-eval-max-length", type=int, default=None)
    parser.add_argument("--max-gen-toks", type=int, default=DEFAULT_MAX_GEN_TOKS)
    parser.add_argument("--max-num-batched-tokens", type=int, default=DEFAULT_MAX_NUM_BATCHED_TOKENS)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
