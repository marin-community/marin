"""
Codename: hypnotic-spoonbill

This experiment is a cooldown run for the tootsie-8b model. It is trained on the
same mix as monument-jellyfish, except that we are adding some flan and tulu.

We are also only lowering the LR to 2.75e-5 instead of 1.7e-5. After around
that point the loss started to increase again. I still don't know why.
"""

import dataclasses

from experiments.dclm.tokenize_dclm import DCLM_MIXTURE_WEIGHTS
from experiments.defaults import default_train
from experiments.dolmino.tokenize_dolmino import get_dolmino_step
from experiments.tootsie.exp600_tootsie import (
    PHASE_3_END,
    PHASE_3_START,
    cooldown_mixture_weights_v1,
    llama_8b_tootsie_phase3,
    llama_8b_train_config_phase3,
    phase_3_tokenized,
)
from experiments.instruction_datasets import (
    tulu3_flat_llama_tokenized_as_train,
    tulu3_flat_llama_tokenized_as_validation,
)
from experiments.llama import llama_8b
from marin.execution.executor import executor_main, output_path_of
from marin.processing.tokenize.data_configs import lm_varying_mixture_data_config

# 3072 * 4096 * 10000 is 125B tokens
COOLDOWN_LEN = 10000
COOLDOWN_END = PHASE_3_END + COOLDOWN_LEN


tootsie_8b_hypnotic_spoonbill_train = dataclasses.replace(
    llama_8b_train_config_phase3,
    tpu_type="v4-128",
    node_count=4,
    learning_rate=1.7e-4,  # only does what we want b/c we're warmstarting from the ckpt below
    num_train_steps=COOLDOWN_END,
    min_lr_ratio=2.75e-5 / 1.7e-4,
    decay=COOLDOWN_LEN,
    initialize_from_checkpoint_path=output_path_of(llama_8b_tootsie_phase3, "checkpoints/step-819924"),
    reset_data_loader_on_init=False,
    per_device_eval_parallelism=16,
    allow_partial_checkpoint=False,
)

flan = get_dolmino_step("flan")


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
)


tootsie_8b_hypnotic_spoonbill = dataclasses.replace(
    default_train(
        name="tootsie-8b-hypnotic-spoonbill",
        tokenized=spoonbill_mixture,
        model_config=llama_8b,
        train_config=tootsie_8b_hypnotic_spoonbill_train,
        use_default_validation=True,
        tags=["llama", "8b", "ema", "exp916", "tootsie"],
        # HF is having trouble today so skipping this.
        eval_harness_tasks=[],
    ),
    override_output_path="checkpoints/tootsie-8b-hypnotic-spoonbill-2",
)


if __name__ == "__main__":
    executor_main(
        [tootsie_8b_hypnotic_spoonbill], description="Cooldown run for tootsie-8b model with some flan and tulu"
    )
