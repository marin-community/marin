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
Codename: hypnotic-spoonbill

This experiment is a cooldown run for the tootsie-8b model. It is trained on the
same mix as monument-jellyfish, except that we are adding some flan and tulu.

We are also only lowering the LR to 2.75e-5 instead of 1.7e-5. After around
that point the loss started to increase again. I still don't know why.
"""

import dataclasses

from levanter.callbacks.watch import WatchConfig

from experiments.pretraining_datasets.dclm import DCLM_MIXTURE_WEIGHTS
from experiments.defaults import default_sft, default_train
from experiments.pretraining_datasets import tokenize_dolmino_subset
from experiments.exp606_sft import tulu3_llama_data_old, tulu_sft_config
from experiments.llama import llama_8b
from experiments.posttrain.instruction_datasets import (
    tulu3_flat_llama_tokenized_as_train,
    tulu3_flat_llama_tokenized_as_validation,
)
from experiments.tootsie.exp600_tootsie import (
    PHASE3_CHECKPOINT_PATH,
    PHASE_3_END,
    PHASE_3_START,
    cooldown_mixture_weights_v1,
    llama_8b_train_config_phase3,
    phase_3_tokenized,
)
from fray.cluster import ResourceConfig
from marin.execution import StepRef, executor_main, step
from marin.processing.tokenize.data_configs import lm_varying_mixture_data_config

# 3072 * 4096 * 10000 is 125B tokens
COOLDOWN_LEN = 10000
COOLDOWN_END = PHASE_3_END + COOLDOWN_LEN


tootsie_8b_hypnotic_spoonbill_train = dataclasses.replace(
    llama_8b_train_config_phase3,
    resources=ResourceConfig.with_tpu("v4-128", slice_count=4),
    learning_rate=1.7e-4,  # only does what we want b/c we're warmstarting from the ckpt below
    num_train_steps=COOLDOWN_END,
    min_lr_ratio=2.75e-5 / 1.7e-4,
    decay=COOLDOWN_LEN,
    initialize_from_checkpoint_path=f"{PHASE3_CHECKPOINT_PATH}/checkpoints/step-819924",
    reset_data_loader_on_init=False,
    per_device_eval_parallelism=16,
    allow_partial_checkpoint=False,
)

flan = tokenize_dolmino_subset("flan")


def _normalize_weights(weights, scale=1.0):
    total = sum(weights.values())
    return {k: v / total * scale for k, v in weights.items()}


spoonbill_weights = {
    **_normalize_weights(cooldown_mixture_weights_v1, 0.97),
    "tulu_sft_train": 0.001,
    "flan": 0.029,
}

spoonbill_mixture = lm_varying_mixture_data_config(
    components={
        **phase_3_tokenized,
        "flan": flan,
        "tulu_sft": tulu3_flat_llama_tokenized_as_validation,
        "tulu_sft_train": tulu3_flat_llama_tokenized_as_train,
    },
    weights_list=[
        (0, DCLM_MIXTURE_WEIGHTS),
        (PHASE_3_START, cooldown_mixture_weights_v1),
        (PHASE_3_END, spoonbill_weights),
    ],
    permutation_type="linear",
)


HYPNOTIC_SPOONBILL_PATH = "checkpoints/tootsie-8b-hypnotic-spoonbill-2"
tootsie_8b_hypnotic_spoonbill = StepRef(HYPNOTIC_SPOONBILL_PATH)

norm_tracking_spoonbill_train = dataclasses.replace(
    tootsie_8b_hypnotic_spoonbill_train,
    watch=WatchConfig(
        interval=10,
        include_histograms=True,
        watch_targets=["grads", "params", "opt_state", "updates"],
    ),
)


HYPNOTIC_SPOONBILL_NORMS_PATH = "checkpoints/tootsie-8b-hypnotic-spoonbill-norms-2"
norm_tootsie_8b_hypnotic_spoonbill = StepRef(HYPNOTIC_SPOONBILL_NORMS_PATH)


llama_8b_fp32_attn = dataclasses.replace(llama_8b, upcast_attn=True)

tootsie_8b_focused_spoonbill_train = dataclasses.replace(
    norm_tracking_spoonbill_train,
    # I saved this from spoonbill-norms-2 so it wouldn't get deleted
    initialize_from_checkpoint_path="gs://marin-us-central2/checkpoints/scratch-dlwh/spoonbill-norms/step-824524/",
    watch=WatchConfig(
        interval=10,
        # fp32 pushes the ram too high here
        include_histograms=False,
        watch_targets=["grads", "params", "opt_state", "updates"],
    ),
)

FOCUSED_SPOONBILL_FP32_PATH = "checkpoints/tootsie-8b-focused-spoonbill_fp32"
norm_tootsie_8b_focused_spoonbill_fp32_attention = StepRef(FOCUSED_SPOONBILL_FP32_PATH)


tootsie_8b_focused_spoonbill_train_zloss = dataclasses.replace(
    norm_tracking_spoonbill_train,
    # I saved this from spoonbill-norms-2 so it wouldn't get deleted
    initialize_from_checkpoint_path="gs://marin-us-central2/checkpoints/scratch-dlwh/spoonbill-norms/step-824524/",
    watch=WatchConfig(
        interval=10,
        # fp32 pushes the ram too high here
        include_histograms=False,
        watch_targets=["grads", "params", "opt_state", "updates"],
    ),
    z_loss_weight=1e-4,  # same as olmo
    # let's try to get a good checkpoint before everything goes to hell
    steps_per_hf_export=5000,
)

FOCUSED_SPOONBILL_ZLOSS_PATH = "checkpoints/tootsie-8b-focused-spoonbill-zloss"
norm_tootsie_8b_focused_spoonbill_zloss = StepRef(FOCUSED_SPOONBILL_ZLOSS_PATH)

EXTRA_STEPS = 10000
DEEPER_END = COOLDOWN_END + EXTRA_STEPS


tootsie_8b_deeper_spoonbill_train = dataclasses.replace(
    norm_tracking_spoonbill_train,
    initialize_from_checkpoint_path=f"{FOCUSED_SPOONBILL_ZLOSS_PATH}/checkpoints/step-829947",
    watch=WatchConfig(
        interval=10,
        # fp32 pushes the ram too high here
        include_histograms=False,
        watch_targets=["grads", "params", "opt_state", "updates"],
    ),
    num_train_steps=DEEPER_END,
    decay=EXTRA_STEPS,
    z_loss_weight=1e-4,  # same as olmo
    # let's try to get a good checkpoint before everything goes to hell
    steps_per_hf_export=5000,
    # set to final LR of focused spoonbill:
    learning_rate=2.75e-5,
    min_lr_ratio=2.75e-6 / 2.75e-5,  # 1e-6
)


DEEPER_SPOONBILL_PATH = "checkpoints/tootsie-8b-deeper-spoonbill-2"
tootsie_8b_deeper_spoonbill = StepRef(DEEPER_SPOONBILL_PATH)

# do some sfts

spoonbill_zloss_tulu3_sft_config = dataclasses.replace(
    tulu_sft_config,
    model_name_or_path=f"{FOCUSED_SPOONBILL_ZLOSS_PATH}/hf/step-829999/",
)


SFT_TULU3_SPOONBILL_ZLOSS_PATH = "checkpoints/sft/tulu3_tootsie_sft_spoonbill_zloss"
sft_tulu3_spoonbill_zloss = StepRef(SFT_TULU3_SPOONBILL_ZLOSS_PATH)

SFT_TULU3_DEEPER_SPOONBILL_PATH = "checkpoints/sft/tulu3_tootsie_sft_deeper_spoonbill_zloss"
sft_tulu3_deeper_spoonbill = StepRef(SFT_TULU3_DEEPER_SPOONBILL_PATH)


@step(name="tootsie/exp916_spoonbill_cooldown/all")
def run_spoonbill_cooldown():
    """Entry point for Spoonbill cooldown experiments."""
    dataclasses.replace(
        default_train(
            name="tootsie-8b-hypnotic-spoonbill",
            tokenized=spoonbill_mixture,
            model_config=llama_8b,
            train_config=tootsie_8b_hypnotic_spoonbill_train,
            use_default_validation=True,
            tags=["llama", "8b", "ema", "exp916", "tootsie"],
            eval_harness_tasks=[],
        ),
        override_output_path=HYPNOTIC_SPOONBILL_PATH,
    )

    dataclasses.replace(
        default_train(
            name="tootsie-8b-hypnotic-spoonbill-norms-2",
            tokenized=spoonbill_mixture,
            model_config=llama_8b,
            train_config=norm_tracking_spoonbill_train,
            use_default_validation=True,
            tags=["llama", "8b", "ema", "exp916", "tootsie"],
            eval_harness_tasks=[],
        ),
        override_output_path=HYPNOTIC_SPOONBILL_NORMS_PATH,
    )

    dataclasses.replace(
        default_train(
            name="tootsie-8b-focused-spoonbill-fp32",
            tokenized=spoonbill_mixture,
            model_config=llama_8b_fp32_attn,
            train_config=tootsie_8b_focused_spoonbill_train,
            use_default_validation=True,
            tags=["llama", "8b", "ema", "exp916", "tootsie"],
            eval_harness_tasks=[],
        ),
        override_output_path=FOCUSED_SPOONBILL_FP32_PATH,
    )

    dataclasses.replace(
        default_train(
            name="tootsie-8b-focused-spoonbill-zloss",
            tokenized=spoonbill_mixture,
            model_config=llama_8b_fp32_attn,
            train_config=tootsie_8b_focused_spoonbill_train_zloss,
            use_default_validation=True,
            tags=["llama", "8b", "ema", "exp916", "tootsie"],
            eval_harness_tasks=[],
        ),
        override_output_path=FOCUSED_SPOONBILL_ZLOSS_PATH,
    )

    dataclasses.replace(
        default_train(
            name="tootsie-8b-deeper-spoonbill-2",
            tokenized=spoonbill_mixture,
            model_config=llama_8b_fp32_attn,
            train_config=tootsie_8b_deeper_spoonbill_train,
            use_default_validation=True,
            tags=["llama", "8b", "ema", "exp916", "tootsie"],
            eval_harness_tasks=[],
        ),
        override_output_path=DEEPER_SPOONBILL_PATH,
    )

    default_sft(
        name="sft/tulu3_tootsie_sft_spoonbill_zloss",
        tokenized=tulu3_llama_data_old,
        model_config=llama_8b_fp32_attn,
        sft_config=spoonbill_zloss_tulu3_sft_config,
        tags=["llama", "8b", "exp916", "tootsie", "sft", "spoonbill"],
    ).with_output_path(SFT_TULU3_SPOONBILL_ZLOSS_PATH)

    default_sft(
        name="sft/tulu3_tootsie_deeper_spoonbill",
        tokenized=tulu3_llama_data_old,
        model_config=llama_8b_fp32_attn,
        sft_config=dataclasses.replace(
            spoonbill_zloss_tulu3_sft_config,
            model_name_or_path=f"{DEEPER_SPOONBILL_PATH}/hf/step-839999/",
        ),
        tags=["llama", "8b", "exp916", "tootsie", "sft", "spoonbill"],
    ).with_output_path(SFT_TULU3_DEEPER_SPOONBILL_PATH)


if __name__ == "__main__":
    executor_main(
        steps=[run_spoonbill_cooldown()],
        description="Cooldown run for tootsie-8b model with some flan and tulu",
    )
