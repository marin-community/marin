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

"""Scaling law comparison between Stack v2 datasets and StarCoderData."""

from experiments.common_pile.tokenize_common_pile import stackv2, stackv2_edu_filtered
from experiments.defaults import default_tokenize
from experiments.llama import llama3_tokenizer
from experiments.pretraining_datasets.dclm import dclm_components_llama3
from experiments.simple_train_config import SimpleTrainConfig
from fray.cluster import ResourceConfig
from marin.execution.executor import executor_main, step
from marin.scaling_laws.create_ladder_suite import scaling_law_suite

TPU_TYPE = "v5p-8"
TAG = ["exp1752_stackv2_vs_starcoder"]

STACK_V2_SWEEP_NAME = "exp1752-stack-v2"
STACK_V2_EDU_SWEEP_NAME = "exp1752-stack-v2-edu"
STARCODER_SWEEP_NAME = "exp1752-starcoderdata"

training_config = SimpleTrainConfig(
    resources=ResourceConfig.with_tpu(TPU_TYPE, slice_count=1),
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


def tokenize_stackv2():
    return default_tokenize(
        name="common_pile_stackv2",
        dataset=stackv2 / "documents",
        tokenizer=llama3_tokenizer,
    )


def tokenize_stackv2_edu():
    return default_tokenize(
        name="common_pile_stackv2_edu",
        dataset=stackv2_edu_filtered,
        tokenizer=llama3_tokenizer,
    )


@step(name="exp1752-stackv2-vs-starcoder/all")
def run_scaling_law_comparison():
    """Entry point for Stack v2 vs StarCoderData scaling law comparison."""
    stackv2_tokenized = tokenize_stackv2()
    stackv2_edu_tokenized = tokenize_stackv2_edu()

    stackv2_suite = scaling_law_suite(
        sweep_name=STACK_V2_SWEEP_NAME,
        tokenized=stackv2_tokenized,
        tags=[*TAG, "stackv2"],
        intermediate_scale=4,
        training_config=training_config,
    )

    stackv2_edu_suite = scaling_law_suite(
        sweep_name=STACK_V2_EDU_SWEEP_NAME,
        tokenized=stackv2_edu_tokenized,
        tags=[*TAG, "stackv2_edu"],
        intermediate_scale=4,
        training_config=training_config,
    )

    starcoder_suite = scaling_law_suite(
        sweep_name=STARCODER_SWEEP_NAME,
        tokenized=dclm_components_llama3["starcoderdata"],
        tags=[*TAG, "starcoderdata"],
        intermediate_scale=4,
        training_config=training_config,
    )

    return [*stackv2_suite, *stackv2_edu_suite, *starcoder_suite]


if __name__ == "__main__":
    executor_main(
        steps=run_scaling_law_comparison(),
        description="Scaling law sweeps comparing Stack v2 with StarCoderData.",
    )
