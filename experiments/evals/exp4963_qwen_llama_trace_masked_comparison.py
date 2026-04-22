# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""#4963: Qwen3/Llama3.1 8B trace-masked loss comparison in us-central1."""

from typing import Literal

from experiments.chat_templates.llama3pt1_chat_template import LLAMA_3_1_CHAT_TEMPLATE
from experiments.chat_templates.qwen3_chat_template import QWEN_3_CHAT_TEMPLATE
from experiments.llama import (
    llama_3_1_8b,
    llama_3_1_8b_instruct,
    llama_3_1_8b_instruct_tokenizer,
    llama_3_1_8b_tokenizer,
)
from experiments.models import ModelConfig, download_model_step, llama_3_1_8b as llama_3_1_8b_hf
from experiments.models import llama_3_1_8b_instruct as llama_3_1_8b_instruct_hf
from experiments.qwen3 import qwen3_8b, qwen3_8b_tokenizer
from fray.cluster import ResourceConfig
from levanter.data.text import HfDatasetSourceConfig, TraceChatEvaluationFormat
from marin.evaluation.trace_masked_eval import (
    TraceMaskedEvalDatasetConfig,
    TraceRowAdapterConfig,
    default_trace_masked_eval,
)
from marin.execution.executor import ExecutorMainConfig, ExecutorStep, executor_main, output_path_of
from marin.rl.placement import marin_prefix_for_region, singleton_region_list

REGION = "us-central1"
MAX_EXAMPLES_PER_DATASET = 32
MAX_EVAL_LENGTH = 8192
WANDB_GROUP = "exp4963-qwen-llama-trace-masked"

LOCAL_LOSS_TAGS = ("assistant", "tool_call", "tool", "observation", "final_assistant")
OUTCOME_LOSS_TAGS = ("patch", "outcome")


def trace_format(
    chat_template: str,
    *,
    loss_tags: tuple[str, ...],
    slice_strategy: Literal["left", "right", "raise"] = "left",
) -> TraceChatEvaluationFormat:
    return TraceChatEvaluationFormat(
        messages_field="messages",
        chat_template=chat_template,
        chat_template_kwargs="chat_template_kwargs",
        loss_tags=loss_tags,
        pack=4,
        slice_strategy=slice_strategy,
    )


def trace_datasets(chat_template: str) -> dict[str, TraceMaskedEvalDatasetConfig]:
    local_fmt = trace_format(chat_template, loss_tags=LOCAL_LOSS_TAGS)
    outcome_fmt = trace_format(chat_template, loss_tags=OUTCOME_LOSS_TAGS, slice_strategy="right")
    return {
        "nemotron_v1_tool_calling": TraceMaskedEvalDatasetConfig(
            source=HfDatasetSourceConfig(
                id="nvidia/Nemotron-Post-Training-Dataset-v1",
                splits=["tool_calling"],
            ),
            split="tool_calling",
            trace_format=local_fmt,
            max_examples=MAX_EXAMPLES_PER_DATASET,
        ),
        "smoltalk2_smolagents_toolcalling_traces_think": TraceMaskedEvalDatasetConfig(
            source=HfDatasetSourceConfig(
                id="HuggingFaceTB/smoltalk2",
                name="SFT",
                splits=["smolagents_toolcalling_traces_think"],
            ),
            split="smolagents_toolcalling_traces_think",
            trace_format=local_fmt,
            max_examples=MAX_EXAMPLES_PER_DATASET,
        ),
        "smoltalk2_xlam_traces_no_think": TraceMaskedEvalDatasetConfig(
            source=HfDatasetSourceConfig(
                id="HuggingFaceTB/smoltalk2",
                name="SFT",
                splits=["xlam_traces_no_think"],
            ),
            split="xlam_traces_no_think",
            trace_format=local_fmt,
            max_examples=MAX_EXAMPLES_PER_DATASET,
        ),
        "openhands_swe_rebench_outcome": TraceMaskedEvalDatasetConfig(
            source=HfDatasetSourceConfig(
                id="nebius/SWE-rebench-openhands-trajectories",
                splits=["train"],
            ),
            split="train",
            trace_format=outcome_fmt,
            max_examples=MAX_EXAMPLES_PER_DATASET,
            row_adapter=TraceRowAdapterConfig(
                input_messages_field="trajectory",
                patch_field="model_patch",
                outcome_field="resolved",
            ),
        ),
        "swe_agent_nebius_outcome": TraceMaskedEvalDatasetConfig(
            source=HfDatasetSourceConfig(
                id="nebius/SWE-agent-trajectories",
                splits=["train"],
            ),
            split="train",
            trace_format=outcome_fmt,
            max_examples=MAX_EXAMPLES_PER_DATASET,
            row_adapter=TraceRowAdapterConfig(
                input_messages_field="trajectory",
                patch_field="generated_patch",
                outcome_field="target",
            ),
        ),
        "swe_smith_tool_outcome": TraceMaskedEvalDatasetConfig(
            source=HfDatasetSourceConfig(
                id="SWE-bench/SWE-smith-trajectories",
                splits=["tool"],
            ),
            split="tool",
            trace_format=outcome_fmt,
            max_examples=MAX_EXAMPLES_PER_DATASET,
            row_adapter=TraceRowAdapterConfig(
                input_messages_field="messages",
                patch_field="patch",
                outcome_field="resolved",
            ),
        ),
        "swe_gym_openhands_sampled_outcome": TraceMaskedEvalDatasetConfig(
            source=HfDatasetSourceConfig(
                id="SWE-Gym/OpenHands-Sampled-Trajectories",
                splits=["train.raw"],
            ),
            split="train.raw",
            trace_format=outcome_fmt,
            max_examples=MAX_EXAMPLES_PER_DATASET,
            row_adapter=TraceRowAdapterConfig(
                input_messages_field="messages",
                patch_field="test_result.git_patch",
                outcome_field="resolved",
            ),
        ),
        "coderforge_swebench_verified_outcome": TraceMaskedEvalDatasetConfig(
            source=HfDatasetSourceConfig(
                id="togethercomputer/CoderForge-Preview-32B-SWE-Bench-Verified-Evaluation-trajectories",
                splits=["train"],
            ),
            split="train",
            trace_format=outcome_fmt,
            max_examples=MAX_EXAMPLES_PER_DATASET,
            row_adapter=TraceRowAdapterConfig(
                input_messages_field="messages",
                patch_field="output_patch",
                outcome_field="reward",
            ),
        ),
    }


def trace_eval_step(
    *,
    name: str,
    checkpoint: ExecutorStep,
    model,
    tokenizer: str,
    chat_template: str,
    tags: tuple[str, ...],
) -> ExecutorStep:
    return default_trace_masked_eval(
        name=name,
        checkpoint=output_path_of(checkpoint),
        checkpoint_is_hf=True,
        model=model,
        tokenizer=tokenizer,
        datasets=trace_datasets(chat_template),
        resource_config=ResourceConfig.with_tpu(
            "v5p-8",
            regions=singleton_region_list(REGION),
            ram="256g",
            disk="100g",
        ),
        per_device_batch_size=1,
        max_eval_length=MAX_EVAL_LENGTH,
        wandb_tags=("trace_masked_eval", "exp4963", *tags),
        wandb_group=WANDB_GROUP,
    )


qwen3_8b_hf = download_model_step(
    ModelConfig(
        hf_repo_id="Qwen/Qwen3-8B",
        hf_revision="b968826",
    )
).with_output_path("models/Qwen--Qwen3-8B--b968826-complete")

qwen3_8b_base_hf = download_model_step(
    ModelConfig(
        hf_repo_id="Qwen/Qwen3-8B-Base",
        hf_revision="49e3418",
    )
).with_output_path("models/Qwen--Qwen3-8B-Base--49e3418-complete")

qwen3_8b_eval = trace_eval_step(
    name="qwen3-8b-posttrained-trace-masked-eval-us-central1",
    checkpoint=qwen3_8b_hf,
    model=qwen3_8b,
    tokenizer=qwen3_8b_tokenizer,
    chat_template=QWEN_3_CHAT_TEMPLATE,
    tags=("qwen3", "8b", "posttrained"),
)

qwen3_8b_base_eval = trace_eval_step(
    name="qwen3-8b-base-trace-masked-eval-us-central1",
    checkpoint=qwen3_8b_base_hf,
    model=qwen3_8b,
    tokenizer="Qwen/Qwen3-8B-Base",
    chat_template=QWEN_3_CHAT_TEMPLATE,
    tags=("qwen3", "8b", "base"),
)

llama_3_1_8b_eval = trace_eval_step(
    name="llama-3-1-8b-trace-masked-eval-us-central1-templatefix",
    checkpoint=llama_3_1_8b_hf,
    model=llama_3_1_8b,
    tokenizer=llama_3_1_8b_tokenizer,
    chat_template=LLAMA_3_1_CHAT_TEMPLATE,
    tags=("llama3.1", "8b", "base"),
)

llama_3_1_8b_instruct_eval = trace_eval_step(
    name="llama-3-1-8b-instruct-trace-masked-eval-us-central1-templatefix",
    checkpoint=llama_3_1_8b_instruct_hf,
    model=llama_3_1_8b_instruct,
    tokenizer=llama_3_1_8b_instruct_tokenizer,
    chat_template=LLAMA_3_1_CHAT_TEMPLATE,
    tags=("llama3.1", "8b", "instruct"),
)


if __name__ == "__main__":
    executor_main(
        ExecutorMainConfig(prefix=marin_prefix_for_region(REGION)),
        steps=[
            qwen3_8b_hf,
            qwen3_8b_base_hf,
            llama_3_1_8b_hf,
            llama_3_1_8b_instruct_hf,
            qwen3_8b_eval,
            qwen3_8b_base_eval,
            llama_3_1_8b_eval,
            llama_3_1_8b_instruct_eval,
        ],
        description="Qwen3/Llama3.1 8B trace-masked loss comparison on Nemotron and SmolTalk2 traces.",
    )
