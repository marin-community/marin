# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tokenize OT Agent traces and evaluate logprobs with a configurable model.

Reuses the trace download/convert steps from download_ot_traces.py, tokenizes
each model's traces as a separate component, then evaluates logprobs over all
traces in a single mixture_for_evaluation call.

Usage:
    uv run lib/marin/src/marin/run/ray_run.py -- python experiments/agent_scaling/ot_trace_logprobs.py
"""

import argparse
import os
import sys

from levanter.data.text import ChatLmDatasetFormat
from levanter.models.qwen import Qwen3Config

from experiments.chat_templates.qwen3_chat_template import QWEN_3_CHAT_TEMPLATE

from experiments.agent_scaling.download_ot_traces import (
    BASE_MODEL,
    K,
    build_steps as build_trace_steps,
)
from iris.marin_fs import marin_prefix
from experiments.defaults import default_tokenize
from experiments.models import ModelConfig, download_model_step
from fray.cluster import ResourceConfig
from marin.evaluation.save_logprobs import default_save_logprobs
from marin.execution.executor import ExecutorStep, executor_main, output_path_of, unwrap_versioned_value, versioned
from marin.processing.tokenize.data_configs import mixture_for_evaluation
from marin.utils import get_directory_friendly_name

DEFAULT_TPU_TYPE = "v4-8"


def build_tokenize_steps_for_model(
    eval_model_step: ExecutorStep,
    trace_steps: dict,
):
    tokenizer = os.path.join(marin_prefix(), eval_model_step.override_output_path or eval_model_step.name)

    tokenized_traces = {}
    for name, step_dict in trace_steps.items():
        convert_step = step_dict["traces"]
        tokenized = default_tokenize(
            name=f"ot_traces/{name}",
            dataset=convert_step,
            tokenizer=tokenizer,
            format=ChatLmDatasetFormat(
                messages_field="conversations",
                chat_template=versioned(QWEN_3_CHAT_TEMPLATE),
                mask_user_turns=versioned(False),
            ),
            is_validation=True,
        )
        tokenized_traces[name] = tokenized

    eval_data = mixture_for_evaluation(tokenized_traces)
    return eval_data, tokenized_traces


def build_logprobs_step_for_model(
    eval_model_step: ExecutorStep,
    eval_data,
    tpu_type: str = DEFAULT_TPU_TYPE,
    logprobs_top_k: int | None = 10,
    max_eval_length: int = 4096,
):
    eval_model = unwrap_versioned_value(eval_model_step.config.hf_dataset_id)
    gcs_path = os.path.join(marin_prefix(), eval_model_step.override_output_path or eval_model_step.name)
    hf_model_config = Qwen3Config().hf_checkpoint_converter(ref_checkpoint=gcs_path).config_from_hf_checkpoint(gcs_path)
    name = get_directory_friendly_name(eval_model)

    return default_save_logprobs(
        checkpoint=output_path_of(eval_model_step),
        model=hf_model_config,
        data=eval_data,
        resource_config=ResourceConfig.with_tpu(tpu_type),
        checkpoint_is_hf=True,
        top_k=versioned(logprobs_top_k),
        max_eval_length=versioned(max_eval_length),
        name=f"ot_traces/{name}",
    )


def build_steps(
    top_k_traces: int = K,
    base_model: str = BASE_MODEL,
    tpu_type: str = DEFAULT_TPU_TYPE,
    logprobs_top_k: int | None = 10,
    max_eval_length: int = 4096,
) -> dict[str, dict]:
    trace_steps = build_trace_steps(k=top_k_traces, base_model=base_model)

    base_model_step = download_model_step(ModelConfig(hf_repo_id=base_model, hf_revision="main"))

    eval_model_steps = [base_model_step]
    for step_dict in trace_steps.values():
        eval_model_steps.append(step_dict["model"])

    # Tokenize once with the base model tokenizer (all finetunes share it)
    eval_data, tokenized_traces = build_tokenize_steps_for_model(base_model_step, trace_steps)

    steps = {}
    for model_step in eval_model_steps:
        eval_model = unwrap_versioned_value(model_step.config.hf_dataset_id)
        name = get_directory_friendly_name(eval_model)
        logprobs_step = build_logprobs_step_for_model(
            model_step,
            eval_data,
            tpu_type,
            logprobs_top_k,
            max_eval_length,
        )
        steps[name] = {"logprobs": logprobs_step, "tokenized": tokenized_traces}

    return steps


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate logprobs on OT Agent traces.")
    parser.add_argument("--top-k-traces", type=int, default=K)
    parser.add_argument("--base-model", type=str, default=BASE_MODEL)
    parser.add_argument("--tpu-type", type=str, default=DEFAULT_TPU_TYPE)
    parser.add_argument("--logprobs-top-k", type=int, default=10)
    parser.add_argument("--max-eval-length", type=int, default=4096)
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]

    steps_dict = build_steps(
        top_k_traces=args.top_k_traces,
        base_model=args.base_model,
        tpu_type=args.tpu_type,
        logprobs_top_k=args.logprobs_top_k,
        max_eval_length=args.max_eval_length,
    )

    all_steps = [s["logprobs"] for s in steps_dict.values()]
    executor_main(steps=all_steps, description="Evaluate logprobs on OT Agent traces.")
