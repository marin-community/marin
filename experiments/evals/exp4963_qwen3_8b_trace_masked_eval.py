# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""#4963: Qwen3-8B trace-masked loss smoke test in us-central1."""

from experiments.chat_templates.qwen3_chat_template import QWEN_3_CHAT_TEMPLATE
from experiments.models import ModelConfig, download_model_step
from experiments.qwen3 import qwen3_8b, qwen3_8b_tokenizer
from fray.cluster import ResourceConfig
from levanter.data.text import HfDatasetSourceConfig, TraceChatEvaluationFormat
from marin.evaluation.trace_masked_eval import TraceMaskedEvalDatasetConfig, default_trace_masked_eval
from marin.execution.executor import ExecutorMainConfig, executor_main, output_path_of
from marin.rl.placement import marin_prefix_for_region, singleton_region_list

REGION = "us-central1"
MAX_EXAMPLES_PER_DATASET = 32
MAX_EVAL_LENGTH = 8192

TRACE_FORMAT = TraceChatEvaluationFormat(
    messages_field="messages",
    chat_template=QWEN_3_CHAT_TEMPLATE,
    chat_template_kwargs="chat_template_kwargs",
    loss_tags=("assistant", "assistant_text", "tool_call", "tool", "observation", "final_assistant"),
    pack=4,
)

TRACE_DATASETS = {
    "nemotron_v1_tool_calling": TraceMaskedEvalDatasetConfig(
        source=HfDatasetSourceConfig(
            id="nvidia/Nemotron-Post-Training-Dataset-v1",
            splits=["tool_calling"],
        ),
        split="tool_calling",
        trace_format=TRACE_FORMAT,
        max_examples=MAX_EXAMPLES_PER_DATASET,
    ),
    "smoltalk2_smolagents_toolcalling_traces_think": TraceMaskedEvalDatasetConfig(
        source=HfDatasetSourceConfig(
            id="HuggingFaceTB/smoltalk2",
            name="SFT",
            splits=["smolagents_toolcalling_traces_think"],
        ),
        split="smolagents_toolcalling_traces_think",
        trace_format=TRACE_FORMAT,
        max_examples=MAX_EXAMPLES_PER_DATASET,
    ),
    "smoltalk2_xlam_traces_no_think": TraceMaskedEvalDatasetConfig(
        source=HfDatasetSourceConfig(
            id="HuggingFaceTB/smoltalk2",
            name="SFT",
            splits=["xlam_traces_no_think"],
        ),
        split="xlam_traces_no_think",
        trace_format=TRACE_FORMAT,
        max_examples=MAX_EXAMPLES_PER_DATASET,
    ),
}

qwen3_8b_hf = download_model_step(
    ModelConfig(
        hf_repo_id="Qwen/Qwen3-8B",
        hf_revision="b968826",
    )
).with_output_path("models/Qwen--Qwen3-8B--b968826-complete")

qwen3_8b_trace_masked_eval = default_trace_masked_eval(
    name="qwen3-8b-trace-masked-eval-us-central1",
    checkpoint=output_path_of(qwen3_8b_hf),
    checkpoint_is_hf=True,
    model=qwen3_8b,
    tokenizer=qwen3_8b_tokenizer,
    datasets=TRACE_DATASETS,
    resource_config=ResourceConfig.with_tpu(
        "v5p-8",
        regions=singleton_region_list(REGION),
        ram="256g",
        disk="100g",
    ),
    per_device_batch_size=1,
    max_eval_length=MAX_EVAL_LENGTH,
)


if __name__ == "__main__":
    executor_main(
        ExecutorMainConfig(prefix=marin_prefix_for_region(REGION)),
        steps=[qwen3_8b_hf, qwen3_8b_trace_masked_eval],
        description="Qwen3-8B trace-masked loss smoke test on Nemotron and SmolTalk2 traces.",
    )
