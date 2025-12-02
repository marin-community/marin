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

"""Scaling law comparison between Stack v2 datasets and StarCoderData with simulated epoching."""

import dataclasses
import logging
from collections.abc import Sequence

from levanter.data.text import LMMixtureDatasetConfig
from levanter.models.llama import LlamaConfig

from experiments.common_pile.tokenize_common_pile import stackv2, stackv2_edu_filtered
from experiments.pretraining_datasets.dclm import dclm_components_llama3
from experiments.defaults import default_tokenize, simulated_epoching_train
from experiments.llama import llama3_tokenizer, llama_1_4b
from experiments.simple_train_config import SimpleTrainConfig
from marin.execution.executor import ExecutorStep, InputName, executor_main
from marin.resources import TpuPodConfig
from experiments.evals.task_configs import CORE_TASKS

TPU_TYPE = "v5p-8"
TAG = ["exp1752", "simulated_epoching"]

STACK_V2_SWEEP_NAME = "exp1752-stack-v2-sim"
STACK_V2_EDU_SWEEP_NAME = "exp1752-stack-v2-edu-sim"
STARCODER_SWEEP_NAME = "exp1752-starcoderdata-sim"

SIMULATED_TARGET_BUDGET_TOKENS = 15_000_000_000_000  # 15T tokens to mimic full-budget epoching behaviour

training_config = SimpleTrainConfig(
    resources=TpuPodConfig(tpu_type=TPU_TYPE, slice_count=1),
    train_batch_size=256,
    learning_rate=1e-3,
    weight_decay=0.1,
    num_train_steps=200000,
    warmup=1000,
    decay=0.0,
    lr_schedule="constant",
    ema_beta=0.995,
    steps_per_eval=500,
    steps_per_task_eval=500,
)


def simulated_scaling_law_suite(
    sweep_name: str,
    tokenized: InputName | ExecutorStep | LMMixtureDatasetConfig,
    *,
    widths: Sequence[int] = (512, 768, 1024, 1536, 2048),
    base_model_config: LlamaConfig = llama_1_4b,
    tags: Sequence[str] = (),
    intermediate_scale: float = 4,
    training_config: SimpleTrainConfig = training_config,
    base_lr: float = 3e-4 * 4096,
    max_lr: float = 5e-3,
    target_budget: int = SIMULATED_TARGET_BUDGET_TOKENS,
) -> Sequence[ExecutorStep]:
    """Mirror scaling_law_suite but replace training with simulated epoching."""

    steps: list[ExecutorStep] = []
    for width in widths:
        intermediate_dim = _round_to_multiple(intermediate_scale * width, 128)
        head_size = 128  # keeping this 128 means we can use splash attention
        num_heads = width // head_size
        num_kv_heads = min(num_heads, 8)
        assert num_heads * head_size == width, f"Number of heads must divide width: {width} % {head_size} != 0"

        if num_heads % num_kv_heads != 0:
            num_kv_heads = num_heads

        model_config = dataclasses.replace(
            base_model_config,
            hidden_dim=width,
            intermediate_dim=intermediate_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
        )

        learning_rate = min(base_lr / width, max_lr)
        lr_training_config = dataclasses.replace(training_config, learning_rate=learning_rate)

        logging.info(f"Creating simulated epoching step for {sweep_name}-{width} with lr {learning_rate}")

        steps.append(
            simulated_epoching_train(
                name=f"{sweep_name}-{width}",
                tokenized=tokenized,
                model_config=model_config,
                train_config=lr_training_config,
                target_budget=target_budget,
                tags=tags,
                eval_harness_tasks=CORE_TASKS,
            )
        )
    return steps


def _round_to_multiple(x: float, multiple: int) -> int:
    return int(multiple * round(x / multiple))


stackv2_tokenized = default_tokenize(
    name="common_pile_stackv2",
    dataset=stackv2 / "documents",
    tokenizer=llama3_tokenizer,
)

stackv2_edu_tokenized = default_tokenize(
    name="common_pile_stackv2_edu",
    dataset=stackv2_edu_filtered,
    tokenizer=llama3_tokenizer,
)

stackv2_suite = simulated_scaling_law_suite(
    sweep_name=STACK_V2_SWEEP_NAME,
    tokenized=stackv2_tokenized,
    tags=[*TAG, "stackv2"],
    intermediate_scale=4,
    training_config=training_config,
)

stackv2_edu_suite = simulated_scaling_law_suite(
    sweep_name=STACK_V2_EDU_SWEEP_NAME,
    tokenized=stackv2_edu_tokenized,
    tags=[*TAG, "stackv2_edu"],
    intermediate_scale=4,
    training_config=training_config,
)

starcoder_suite = simulated_scaling_law_suite(
    sweep_name=STARCODER_SWEEP_NAME,
    tokenized=dclm_components_llama3["starcoderdata"],
    tags=[*TAG, "starcoderdata"],
    intermediate_scale=4,
    training_config=training_config,
)

if __name__ == "__main__":
    executor_main(
        steps=[
            *stackv2_suite,
            *stackv2_edu_suite,
            *starcoder_suite,
        ],
        description="Scaling law sweeps comparing Stack v2 with StarCoderData using simulated epoching.",
    )
