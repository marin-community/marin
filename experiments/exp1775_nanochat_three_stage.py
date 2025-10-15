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

"""NanoChat-style pretraining, midtraining, and chat SFT in a single pipeline.

We collapse the previous multi-run setup into a single executor graph that warms up on the
DCLM mixture, transitions into the reasoning-focused midtraining mix with a 0.2/0.6/0.2 WSD
schedule, and finally runs a SmolTalk chat SFT stage on top of the resulting checkpoint.
"""

from experiments.dclm.tokenize_dclm import DCLM_MIXTURE_WEIGHTS, dclm_components_llama3
from experiments.defaults import default_sft, default_train
from experiments.llama import llama3_tokenizer, llama_300m
from experiments.midtraining_datasets import finemath_3_plus_tokenized, megamath_tokenized
from experiments.simple_sft_config import SimpleSFTConfig
from experiments.simple_train_config import SimpleTrainConfig
from experiments.exp808_sft_mixture import mixture_config as sft_mixture_llama3
from marin.execution.executor import executor_main
from marin.processing.tokenize.data_configs import lm_varying_mixture_data_config
from marin.resources import TpuPodConfig

# ------------------------------------------------------------------------------------
# Shared hyperparameters and helpers
MODEL_CONFIG = llama_300m
TOKENIZER_NAME = llama3_tokenizer

# Warmup/stable/decay proportions requested for the combined run.
WARMUP_FRACTION = 0.2
STABLE_FRACTION = 0.6
DECAY_FRACTION = 0.2

# Blend fractions for the pretraining and midtraining mixtures once the run transitions.
PRETRAIN_FRACTION = 0.7
MIDTRAIN_FRACTION = 0.3

# Keep the total step budget consistent with the previous pretrain + midtrain stages.
BASE_NUM_STEPS = 2_000
MID_NUM_STEPS = 600
TOTAL_NUM_STEPS = BASE_NUM_STEPS + MID_NUM_STEPS

BATCH_SIZE = 512
PEAK_LEARNING_RATE = 3e-3
WEIGHT_DECAY = 0.1

# Stage 3 SFT configuration.
SFT_NUM_STEPS = 1_000
SFT_LEARNING_RATE = 5e-5
SFT_WARMUP_FRACTION = 0.05
SFT_COOLDOWN_FRACTION = 0.2

# Transition point where we swap from the base mixture into the midtraining mixture.
MID_START_STEP = int(TOTAL_NUM_STEPS * WARMUP_FRACTION)


def _scale_weights(weights: dict[str, float], fraction: float) -> dict[str, float]:
    """Normalize `weights` and rescale them so they sum to `fraction`."""
    total_weight = sum(weights.values())
    if total_weight <= 0:
        raise ValueError("Mixture weights must sum to a positive value.")
    return {name: (value * fraction) / total_weight for name, value in weights.items()}


# ------------------------------------------------------------------------------------
# Data mixture schedule across the run
mid_components = {
    "finemath_3_plus": finemath_3_plus_tokenized,
    "megamath/qa": megamath_tokenized["megamath/qa"],
    "megamath/web": megamath_tokenized["megamath/web"],
    "megamath/web_pro": megamath_tokenized["megamath/web_pro"],
}
mid_weights = {
    "finemath_3_plus": 0.6,
    "megamath/qa": 0.15,
    "megamath/web": 0.15,
    "megamath/web_pro": 0.1,
}

mid_blended_weights = _scale_weights(DCLM_MIXTURE_WEIGHTS, PRETRAIN_FRACTION)
for component, weight in _scale_weights(mid_weights, MIDTRAIN_FRACTION).items():
    mid_blended_weights[component] = mid_blended_weights.get(component, 0.0) + weight

nanochat_pretrain_mid_mixture = lm_varying_mixture_data_config(
    components={**dclm_components_llama3, **mid_components},
    weights_list=[
        (0, DCLM_MIXTURE_WEIGHTS),
        (MID_START_STEP, mid_blended_weights),
    ],
)

nanochat_train_config = SimpleTrainConfig(
    resources=TpuPodConfig(tpu_type="v5p-8"),
    train_batch_size=BATCH_SIZE,
    num_train_steps=TOTAL_NUM_STEPS,
    learning_rate=PEAK_LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    warmup=WARMUP_FRACTION,
    decay=DECAY_FRACTION,
    lr_schedule="linear",
    steps_per_eval=250,
    steps_per_export=250,
    steps_per_task_eval=TOTAL_NUM_STEPS,
)

nanochat_pre_mid_step = default_train(
    name="nanochat-style-pre-mid",
    tokenized=nanochat_pretrain_mid_mixture,
    model_config=MODEL_CONFIG,
    train_config=nanochat_train_config,
    tags=["nanochat-style", "pretrain-midtrain"],
).with_output_path("checkpoints/nanochat-style-pre-mid")


sft_config = SimpleSFTConfig(
    resources=TpuPodConfig(tpu_type="v5p-8"),
    train_batch_size=BATCH_SIZE,
    num_train_steps=SFT_NUM_STEPS,
    learning_rate=SFT_LEARNING_RATE,
    tokenizer=TOKENIZER_NAME,
    model_name_or_path=nanochat_pre_mid_step,
    steps_per_eval=100,
    steps_per_checkpoint=200,
    steps_per_hf_export=200,
    warmup=SFT_WARMUP_FRACTION,
    cooldown=SFT_COOLDOWN_FRACTION,
)

nanochat_sft_step = default_sft(
    name="nanochat-style-sft",
    tokenized=sft_mixture_llama3,
    model_config=MODEL_CONFIG,
    sft_config=sft_config,
    tags=["nanochat-style", "sft"],
).with_output_path("checkpoints/nanochat-style-sft")

# ------------------------------------------------------------------------------------
# Pipeline entry point
if __name__ == "__main__":
    executor_main(
        steps=[nanochat_sft_step],
        description=(
            "Single-run NanoChat-style training that warms up on DCLM before shifting to the "
            "reasoning-focused midtraining mixture with a 0.2/0.6/0.2 WSD schedule, followed by SmolTalk SFT."
        ),
    )
