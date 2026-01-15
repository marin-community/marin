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

"""
Switch over to Qwen3 architecture, warmstarting from the llama-32b-tootsie
"""

import dataclasses

import haliax
from levanter.optim import AdamConfig
from levanter.optim.clip_update_norm import ClipUpdateNormConfig

from experiments.defaults import default_train
from experiments.qwen3 import qwen3_32b
from experiments.tootsie.exp1295_32b import llama_32b_tootsie, llama_32b_train_config, nemotron_mix
from marin.execution import StepRef, executor_main, step
from fray.cluster import ResourceConfig

# We have doctored the opt state to include update history from
# gs://marin-us-central2/checkpoints/llama-32b-tootsie-2/checkpoints/step-77096 for clipping
warmstart_checkpoint = llama_32b_tootsie.cd("checkpoints/step-80000/").nonblocking()

qwen3_32b_remat = dataclasses.replace(
    qwen3_32b, gradient_checkpointing=haliax.ScanCheckpointPolicy(save_carries="offload")
)


qwen_32b_warmstart_train = dataclasses.replace(
    llama_32b_train_config,
    initialize_from_checkpoint_path=warmstart_checkpoint,
    resources=ResourceConfig.with_tpu("v4-2048", 1),
    reset_data_loader_on_init=False,
    allow_partial_checkpoint=True,
    optimizer_config=AdamConfig(
        beta1=0.9,
        beta2=0.95,
        epsilon=1e-8,
        max_grad_norm=0.2,  # we're almost always < .2 except during spikes
        # width is a little smaller than the 24B and we're using a much larger batch size
        # 4.2e-4 * sqrt(8192/3072) ≈ 7e-4
        learning_rate=7e-4,
        weight_decay=0.05,
        skip_bad_steps=True,
        # update_rms_clipping=1.0,  # added at 67522, removed at 72233
        lr_schedule="linear",
        warmup=0.01,
        rewarmup=1000,
        decay=0.4,
        # using WSD-S to rewarmup given that we're adding new weights
        cycles=[80000, 1_000_000_000],
        # this was inadvertently off from about 74k to 80k
        clip_update_norm=ClipUpdateNormConfig(rolling_interval_length=128, sigma_factor=2.0),
    ),
)

MARIN_32B_QWEN_OUTPUT_PATH = "checkpoints/marin-32b-qwen"
MARIN_32B_QWEN_V5P_OUTPUT_PATH = "checkpoints/marin-32b-v5p-qwen-2"

marin_32b_qwen = StepRef(MARIN_32B_QWEN_OUTPUT_PATH)
marin_32b_qwen_v5p = StepRef(MARIN_32B_QWEN_V5P_OUTPUT_PATH)


# we got some v5p spot capacity. gonna try it over there

# TODO: should probably tune the FSDP axis size a bit.

qwen_32b_warmstart_train_v5p = dataclasses.replace(
    qwen_32b_warmstart_train,
    initialize_from_checkpoint_path=None,
    resources=ResourceConfig.with_tpu("v5p-2048", 1),
    reset_data_loader_on_init=False,
    allow_partial_checkpoint=False,
)


@step(name="tootsie/exp1395_qwen3_32b/all")
def run_experiment():
    """Entry point for Qwen3 32B training experiment."""
    default_train(
        name="marin-32b-qwen",
        tokenized=nemotron_mix,
        model_config=qwen3_32b_remat,
        train_config=qwen_32b_warmstart_train,
        tags=["qwen", "32b", "ema", "exp859", "exp1395", "tootsie"],
        eval_harness_tasks=[],
    ).with_output_path(MARIN_32B_QWEN_OUTPUT_PATH)

    default_train(
        name="marin-32b-v5p-qwen-2",
        tokenized=nemotron_mix,
        model_config=qwen3_32b,  # no remat. tons of hbm
        train_config=qwen_32b_warmstart_train_v5p,
        tags=["qwen", "32b", "ema", "exp859", "exp1395", "tootsie"],
        eval_harness_tasks=[],
    ).with_output_path(MARIN_32B_QWEN_V5P_OUTPUT_PATH)


if __name__ == "__main__":
    executor_main(
        steps=[run_experiment()],
        description="Warmstart 32B Qwen3 from Llama 32B Tootsie checkpoint and train on Nemotron etc",
    )
