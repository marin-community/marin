# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""#5724: Repeatable 8B agent trace labeled-loss head-to-head."""

from typing import Literal

from fray.cluster import ResourceConfig
from levanter.data.text import HfDatasetSourceConfig, TraceChatEvaluationFormat
from marin.evaluation.trace_labeled_eval import (
    TraceLabeledEvalDatasetConfig,
    TraceRowAdapterConfig,
    trace_labeled_eval_step,
)
from marin.execution.executor import ExecutorStep, executor_main, output_path_of

from experiments.chat_templates.llama3pt1_chat_template import LLAMA_3_1_CHAT_TEMPLATE
from experiments.chat_templates.qwen2pt5_instruct_chat_template import QWEN_2_5_INSTRUCT_CHAT_TEMPLATE
from experiments.chat_templates.qwen3_chat_template import QWEN_3_CHAT_TEMPLATE
from experiments.llama import (
    llama_3_1_8b,
    llama_3_1_8b_instruct,
    llama_3_1_8b_instruct_tokenizer,
    llama_3_1_8b_tokenizer,
)
from experiments.marin_models import MARIN_CHAT_TEMPLATE, marin_tokenizer
from experiments.models import ModelConfig, download_model_step
from experiments.models import llama_3_1_8b as llama_3_1_8b_hf
from experiments.models import llama_3_1_8b_instruct as llama_3_1_8b_instruct_hf
from experiments.models import marin_8b_base as marin_8b_base_hf
from experiments.models import qwen2_5_7b as qwen2_5_7b_hf_raw
from experiments.models import qwen2_5_7b_instruct as qwen2_5_7b_instruct_hf_raw
from experiments.models import qwen3_8b as qwen3_8b_hf_raw
from experiments.models import qwen3_8b_base as qwen3_8b_base_hf
from experiments.qwen3 import (
    qwen2_5_7b,
    qwen2_5_7b_instruct,
    qwen2_5_7b_instruct_tokenizer,
    qwen2_5_7b_tokenizer,
    qwen3_8b,
    qwen3_8b_tokenizer,
)

TRACE_EVAL_TPU_TYPE = "v5p-8"
MAX_EXAMPLES_PER_DATASET = 128
MAX_EVAL_LENGTH = 8192
PATCH_ABLATION_PREFIX_FRACTIONS = (0.0, 0.5)
WANDB_GROUP = "exp5724-trace-labeled-head-to-head"
RUN_SUFFIX = "trace-labeled-prefix-patch-ablation-128ex"

LOCAL_LOSS_TAGS = ("assistant", "assistant_text", "tool_call", "tool", "observation", "final_assistant")
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


def outcome_adapter(
    *,
    input_messages_field: str,
    patch_field: str,
    outcome_field: str,
    task_id_field: str | None = "instance_id",
    record_id_field: str | None = None,
) -> TraceRowAdapterConfig:
    return TraceRowAdapterConfig(
        input_messages_field=input_messages_field,
        patch_field=patch_field,
        outcome_field=outcome_field,
        task_id_field=task_id_field,
        record_id_field=record_id_field,
    )


def trace_datasets(chat_template: str) -> dict[str, TraceLabeledEvalDatasetConfig]:
    local_format = trace_format(chat_template, loss_tags=LOCAL_LOSS_TAGS)
    outcome_format = trace_format(chat_template, loss_tags=OUTCOME_LOSS_TAGS, slice_strategy="right")
    datasets = {
        "nemotron_v1_tool_calling": TraceLabeledEvalDatasetConfig(
            source=HfDatasetSourceConfig(
                id="nvidia/Nemotron-Post-Training-Dataset-v1",
                splits=["tool_calling"],
            ),
            split="tool_calling",
            trace_format=local_format,
            max_examples=MAX_EXAMPLES_PER_DATASET,
        ),
        "smoltalk2_smolagents_toolcalling_traces_think": TraceLabeledEvalDatasetConfig(
            source=HfDatasetSourceConfig(
                id="HuggingFaceTB/smoltalk2",
                name="SFT",
                splits=["smolagents_toolcalling_traces_think"],
            ),
            split="smolagents_toolcalling_traces_think",
            trace_format=local_format,
            max_examples=MAX_EXAMPLES_PER_DATASET,
        ),
        "smoltalk2_xlam_traces_no_think": TraceLabeledEvalDatasetConfig(
            source=HfDatasetSourceConfig(
                id="HuggingFaceTB/smoltalk2",
                name="SFT",
                splits=["xlam_traces_no_think"],
            ),
            split="xlam_traces_no_think",
            trace_format=local_format,
            max_examples=MAX_EXAMPLES_PER_DATASET,
        ),
        "openhands_swe_rebench_outcome": TraceLabeledEvalDatasetConfig(
            source=HfDatasetSourceConfig(
                id="nebius/SWE-rebench-openhands-trajectories",
                splits=["train"],
            ),
            split="train",
            trace_format=outcome_format,
            max_examples=MAX_EXAMPLES_PER_DATASET,
            row_adapter=outcome_adapter(
                input_messages_field="trajectory",
                patch_field="model_patch",
                outcome_field="resolved",
                record_id_field="trajectory_id",
            ),
        ),
        "swe_agent_nebius_outcome": TraceLabeledEvalDatasetConfig(
            source=HfDatasetSourceConfig(
                id="nebius/SWE-agent-trajectories",
                splits=["train"],
            ),
            split="train",
            trace_format=outcome_format,
            max_examples=MAX_EXAMPLES_PER_DATASET,
            row_adapter=outcome_adapter(
                input_messages_field="trajectory",
                patch_field="generated_patch",
                outcome_field="target",
            ),
        ),
        "swe_smith_tool_outcome": TraceLabeledEvalDatasetConfig(
            source=HfDatasetSourceConfig(
                id="SWE-bench/SWE-smith-trajectories",
                splits=["tool"],
            ),
            split="tool",
            trace_format=outcome_format,
            max_examples=MAX_EXAMPLES_PER_DATASET,
            row_adapter=outcome_adapter(
                input_messages_field="messages",
                patch_field="patch",
                outcome_field="resolved",
            ),
        ),
        "swe_gym_openhands_sampled_outcome": TraceLabeledEvalDatasetConfig(
            source=HfDatasetSourceConfig(
                id="SWE-Gym/OpenHands-Sampled-Trajectories",
                splits=["train.raw"],
            ),
            split="train.raw",
            trace_format=outcome_format,
            max_examples=MAX_EXAMPLES_PER_DATASET,
            row_adapter=outcome_adapter(
                input_messages_field="messages",
                patch_field="test_result.git_patch",
                outcome_field="resolved",
            ),
        ),
        "coderforge_swebench_verified_outcome": TraceLabeledEvalDatasetConfig(
            source=HfDatasetSourceConfig(
                id="togethercomputer/CoderForge-Preview-32B-SWE-Bench-Verified-Evaluation-trajectories",
                splits=["train"],
            ),
            split="train",
            trace_format=outcome_format,
            max_examples=MAX_EXAMPLES_PER_DATASET,
            row_adapter=outcome_adapter(
                input_messages_field="messages",
                patch_field="output_patch",
                outcome_field="reward",
            ),
        ),
    }

    for outcome_name, dataset_config in list(datasets.items()):
        if dataset_config.row_adapter is None:
            continue
        for prefix_fraction in PATCH_ABLATION_PREFIX_FRACTIONS:
            prefix_name = int(prefix_fraction * 100)
            datasets[f"{outcome_name}_patch_prefix_{prefix_name:02d}"] = TraceLabeledEvalDatasetConfig(
                source=dataset_config.source,
                split=dataset_config.split,
                trace_format=dataset_config.trace_format,
                max_examples=dataset_config.max_examples,
                row_adapter=dataset_config.row_adapter,
                row_prefix_fraction=prefix_fraction,
            )

    return datasets


def trace_eval_step(
    *,
    name: str,
    checkpoint: ExecutorStep,
    model,
    tokenizer: str,
    chat_template: str,
    tags: tuple[str, ...],
) -> ExecutorStep:
    return trace_labeled_eval_step(
        name=f"{name}-{RUN_SUFFIX}",
        checkpoint=output_path_of(checkpoint),
        checkpoint_is_hf=True,
        model=model,
        tokenizer=tokenizer,
        datasets=trace_datasets(chat_template),
        resource_config=ResourceConfig.with_tpu(
            TRACE_EVAL_TPU_TYPE,
            ram="256g",
            disk="100g",
        ),
        per_device_batch_size=1,
        max_eval_length=MAX_EVAL_LENGTH,
        wandb_tags=("trace_labeled_eval", "exp5724", *tags),
        wandb_group=WANDB_GROUP,
    )


marin_8b_instruct_hf = download_model_step(
    ModelConfig(
        hf_repo_id="marin-community/marin-8b-instruct",
        hf_revision="0378f9c932bd64ca189d2f2ca5bd2fb076cba61d",
    )
)

qwen2_5_7b_hf = qwen2_5_7b_hf_raw.with_output_path(
    "models/Qwen--Qwen2.5-7B--d149729398750b98c0af14eb82c78cfe92750796-complete"
)
qwen2_5_7b_instruct_hf = qwen2_5_7b_instruct_hf_raw.with_output_path(
    "models/Qwen--Qwen2.5-7B-Instruct--a09a354-complete"
)
qwen3_8b_hf = qwen3_8b_hf_raw.with_output_path("models/Qwen--Qwen3-8B--b968826-complete")

qwen2_5_7b_base_eval = trace_eval_step(
    name="qwen2-5-7b-base",
    checkpoint=qwen2_5_7b_hf,
    model=qwen2_5_7b,
    tokenizer=qwen2_5_7b_tokenizer,
    chat_template=QWEN_2_5_INSTRUCT_CHAT_TEMPLATE,
    tags=("qwen2.5", "7b", "base"),
)

qwen2_5_7b_instruct_eval = trace_eval_step(
    name="qwen2-5-7b-instruct",
    checkpoint=qwen2_5_7b_instruct_hf,
    model=qwen2_5_7b_instruct,
    tokenizer=qwen2_5_7b_instruct_tokenizer,
    chat_template=QWEN_2_5_INSTRUCT_CHAT_TEMPLATE,
    tags=("qwen2.5", "7b", "instruct"),
)

qwen3_8b_eval = trace_eval_step(
    name="qwen3-8b-posttrained",
    checkpoint=qwen3_8b_hf,
    model=qwen3_8b,
    tokenizer=qwen3_8b_tokenizer,
    chat_template=QWEN_3_CHAT_TEMPLATE,
    tags=("qwen3", "8b", "posttrained"),
)

qwen3_8b_base_eval = trace_eval_step(
    name="qwen3-8b-base",
    checkpoint=qwen3_8b_base_hf,
    model=qwen3_8b,
    tokenizer="Qwen/Qwen3-8B-Base",
    chat_template=QWEN_3_CHAT_TEMPLATE,
    tags=("qwen3", "8b", "base"),
)

llama_3_1_8b_eval = trace_eval_step(
    name="llama-3-1-8b-base",
    checkpoint=llama_3_1_8b_hf,
    model=llama_3_1_8b,
    tokenizer=llama_3_1_8b_tokenizer,
    chat_template=LLAMA_3_1_CHAT_TEMPLATE,
    tags=("llama3.1", "8b", "base"),
)

llama_3_1_8b_instruct_eval = trace_eval_step(
    name="llama-3-1-8b-instruct",
    checkpoint=llama_3_1_8b_instruct_hf,
    model=llama_3_1_8b_instruct,
    tokenizer=llama_3_1_8b_instruct_tokenizer,
    chat_template=LLAMA_3_1_CHAT_TEMPLATE,
    tags=("llama3.1", "8b", "instruct"),
)

marin_8b_base_eval = trace_eval_step(
    name="marin-8b-base",
    checkpoint=marin_8b_base_hf,
    model=llama_3_1_8b,
    tokenizer=marin_tokenizer,
    chat_template=MARIN_CHAT_TEMPLATE,
    tags=("marin", "8b", "base"),
)

marin_8b_instruct_eval = trace_eval_step(
    name="marin-8b-instruct",
    checkpoint=marin_8b_instruct_hf,
    model=llama_3_1_8b,
    tokenizer=marin_tokenizer,
    chat_template=MARIN_CHAT_TEMPLATE,
    tags=("marin", "8b", "instruct"),
)

eval_steps = [
    qwen2_5_7b_base_eval,
    qwen2_5_7b_instruct_eval,
    qwen3_8b_eval,
    qwen3_8b_base_eval,
    llama_3_1_8b_eval,
    llama_3_1_8b_instruct_eval,
    marin_8b_base_eval,
    marin_8b_instruct_eval,
]


if __name__ == "__main__":
    executor_main(
        steps=[
            qwen2_5_7b_hf,
            qwen2_5_7b_instruct_hf,
            qwen3_8b_hf,
            qwen3_8b_base_hf,
            llama_3_1_8b_hf,
            llama_3_1_8b_instruct_hf,
            marin_8b_base_hf,
            marin_8b_instruct_hf,
            *eval_steps,
        ],
        description="8B trace-labeled head-to-head on Qwen2.5, Qwen3, Llama3.1, and Marin.",
    )
