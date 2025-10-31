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

import math

from levanter.optim import MuonHConfig

from experiments.defaults import SimpleTrainConfig, default_train
from marin.processing.tokenize.data_configs import lm_varying_mixture_data_config
from marin.execution.executor import executor_main
from marin.resources import TpuPodConfig
from experiments.qwen3 import qwen3_1_7b, qwen3_8b

from experiments.ferries.initial_ferry import (
    BATCH_SIZE_1B,
    BATCH_SIZE_8B,
    NUM_1B_TRAIN_STEPS,
    NUM_8B_TRAIN_STEPS,
    NEMOTRON_PT_MIX_WEIGHTS,
    nemotron_steps,
    proofpile_2_step,
    starcoder_step,
    phase_3_tokenized,
    pt_vs_hq_components,
    megamath_tokenized,
    starling_components,
    STACKV2_EDU_PYTHON_KEY,
    stackv2_edu_filtered_python_tokenized,
    mantis_cooldown_weights_with_stackv2_python,
)


muonh_cfg_1b = MuonHConfig(
    learning_rate=0.01,
    adam_lr=0.0015,
    min_lr_ratio=0.0,
    momentum=0.98,
    beta1=0.9,
    beta2=0.98,
    epsilon=1e-15,
    muon_epsilon=1e-5,
    max_grad_norm=2.0,
    warmup=1000,
)

muonh_cfg_8b = MuonHConfig(
    learning_rate=0.01,
    adam_lr=0.0015,
    min_lr_ratio=0.0,
    momentum=0.98,
    beta1=0.9,
    beta2=0.98,
    epsilon=1e-15,
    muon_epsilon=1e-5,
    max_grad_norm=2.0,
    warmup=1000,
)


train_config_1b = SimpleTrainConfig(
    resources=TpuPodConfig(tpu_type="v5p-32"),
    train_batch_size=BATCH_SIZE_1B,
    num_train_steps=NUM_1B_TRAIN_STEPS,
    learning_rate=muonh_cfg_1b.learning_rate,
    z_loss_weight=5e-6,
    optimizer_config=muonh_cfg_1b,
)

train_config_8b = SimpleTrainConfig(
    resources=TpuPodConfig(tpu_type="v5p-64"),
    train_batch_size=BATCH_SIZE_8B,
    num_train_steps=NUM_8B_TRAIN_STEPS,
    learning_rate=muonh_cfg_8b.learning_rate,
    z_loss_weight=5e-6,
    optimizer_config=muonh_cfg_8b,
)


def _build_varying_mixture_for_steps(
    num_train_steps: int,
    *,
    train_batch_size: int,
    mixture_block_size: int = 2048,
):
    """
    Build the varying data mixture with phase cutovers proportional to the
    Tootsie schedules (160k/192k and 174k/192k) for a given training horizon.

    Uses Feistel shuffle for sequence sampling to avoid linear-permutation artifacts.
    """
    # Allocate 20% of steps to the cooldown (midtraining) phase
    requested_cooldown_start_step = max(1, int(num_train_steps * 0.8))

    step_multiple = mixture_block_size // math.gcd(mixture_block_size, train_batch_size)
    cooldown_start_step = (requested_cooldown_start_step // step_multiple) * step_multiple
    if cooldown_start_step == 0:
        cooldown_start_step = step_multiple

    return lm_varying_mixture_data_config(
        components={
            **nemotron_steps,
            "starcoderdata": starcoder_step,
            "proofpile_2": proofpile_2_step,
            **phase_3_tokenized,
            **{k: v for k, v in pt_vs_hq_components.items() if k != "all_math"},
            **megamath_tokenized,
            **starling_components,
            STACKV2_EDU_PYTHON_KEY: stackv2_edu_filtered_python_tokenized,
        },
        weights_list=[
            (0, NEMOTRON_PT_MIX_WEIGHTS),  # Pretraining
            (cooldown_start_step, mantis_cooldown_weights_with_stackv2_python),  # Midtraining
        ],
        permutation_type="feistel",
        mixture_block_size=mixture_block_size,
    )


varying_mixture_1b = _build_varying_mixture_for_steps(
    NUM_1B_TRAIN_STEPS, train_batch_size=BATCH_SIZE_1B, mixture_block_size=2048
)
varying_mixture_8b = _build_varying_mixture_for_steps(
    NUM_8B_TRAIN_STEPS, train_batch_size=BATCH_SIZE_8B, mixture_block_size=2048
)

ferry_model_1b = default_train(
    name="ferry_muonh_qwen3_1_7b_feistel",
    tokenized=varying_mixture_1b,
    model_config=qwen3_1_7b,
    train_config=train_config_1b,
)

ferry_model_8b = default_train(
    name="ferry_muonh_qwen3_8b_feistel",
    tokenized=varying_mixture_8b,
    model_config=qwen3_8b,
    train_config=train_config_8b,
)


if __name__ == "__main__":
    executor_main(
        steps=[ferry_model_1b, ferry_model_8b],
        description=(
            "Ferry (Feistel + MuonH): PT on Nemotron+Code then cooldown to HQ mix, scaled from Tootsie schedules"
        ),
    )
