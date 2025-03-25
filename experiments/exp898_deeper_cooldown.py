"""
https://github.com/stanford-crfm/marin/issues/898

Codename: soft-raccoon

The tootsie 8b model (#600) used a much higher peak LR than Olmo2 (1.7e-3 vs 3e-4) and our cooled down model therefore
has a much higher LR than theirs (1.7e-4 vs 3e-5). Indeed, our cooldown LR is closer to their peak LR...

Starting from cooldown v1 (monumental-jellyfish ) we're going to just keep the same cooldown mix from 1.7e-4 down to
1.7e-5 over 125B tokens
"""

import dataclasses

from experiments.defaults import default_train
from experiments.exp600_tootsie import (
    PHASE_3_END,
    llama_8b_tootsie_phase3,
    llama_8b_train_config_phase3,
    phase_3_data_mixture,
)
from experiments.instruction_datasets import tulu3_flat_llama_tokenized
from experiments.llama import llama_8b
from marin.execution.executor import executor_main, output_path_of
from marin.processing.tokenize import add_validation_sets_to_mixture

# 3072 * 4096 * 10000 is 125B tokens
COOLDOWN_LEN = 10000
COOLDOWN_END = PHASE_3_END + COOLDOWN_LEN

tootsie_8b_soft_raccoon_train = dataclasses.replace(
    llama_8b_train_config_phase3,
    tpu_type="v4-128",
    node_count=4,
    learning_rate=1.7e-4,  # only does what we want b/c we're warmstarting from the ckpt below
    num_train_steps=COOLDOWN_END,
    min_lr_ratio=0.1,  # 1.7e-5
    decay=COOLDOWN_LEN,
    initialize_from_checkpoint_path=output_path_of(llama_8b_tootsie_phase3, "checkpoints/step-819924"),
    reset_data_loader_on_init=False,
    per_device_eval_parallelism=16,
)

tulu_3_sft_data_as_validation = tulu3_flat_llama_tokenized

raccoon_mixture = add_validation_sets_to_mixture(
    phase_3_data_mixture,
    {"tulu_sft": tulu_3_sft_data_as_validation},
)

# -3 because we had a little snafu in the original.
tootsie_8b_soft_raccoon = dataclasses.replace(
    default_train(
        name="tootsie-8b-soft-raccoon-3",
        tokenized=raccoon_mixture,
        model_config=llama_8b,
        train_config=tootsie_8b_soft_raccoon_train,
        use_default_validation=True,
        tags=["llama", "8b", "ema", "exp898", "tootsie"],
        # HF is having trouble today so skipping this.
        eval_harness_tasks=[],
    ),
    override_output_path="checkpoints/tootsie-8b-soft-raccoon-3",
)

# we have to go deeper

COOLER_DOWN_LEN = 10000
COOLER_DOWN_END = COOLDOWN_END + COOLER_DOWN_LEN

tootsie_8b_softer_raccoon_train = dataclasses.replace(
    llama_8b_train_config_phase3,
    tpu_type="v4-128",
    node_count=4,
    learning_rate=1.7e-5,  # only does what we want b/c we're warmstarting from soft-raccoon
    num_train_steps=COOLER_DOWN_END,
    min_lr_ratio=1e-6 / 1.7e-5,  # 1e-6
    decay=COOLER_DOWN_LEN,
    initialize_from_checkpoint_path=output_path_of(tootsie_8b_soft_raccoon, "checkpoints/step-829992"),
    reset_data_loader_on_init=False,
    per_device_eval_parallelism=16,
)

tootsie_8b_softer_raccoon = dataclasses.replace(
    default_train(
        name="tootsie-8b-softer-raccoon",
        tokenized=raccoon_mixture,
        model_config=llama_8b,
        train_config=tootsie_8b_softer_raccoon_train,
        use_default_validation=True,
        tags=["llama", "8b", "ema", "exp898", "tootsie"],
        # HF is having trouble today so skipping this.
        eval_harness_tasks=[],
    ),
    override_output_path="checkpoints/tootsie-8b-softer-raccoon",
)


# We found that loss starts increasing when the LR is too low, so we're going to turn off weight decay and see if that
# helps.

tootsie_8b_softer_raccoon_train_no_decay = dataclasses.replace(
    llama_8b_train_config_phase3,
    tpu_type="v4-128",
    node_count=4,
    learning_rate=1.7e-5,  # only does what we want b/c we're warmstarting from soft-raccoon
    num_train_steps=COOLER_DOWN_END,
    min_lr_ratio=1e-6 / 1.7e-5,  # 1e-6
    decay=COOLER_DOWN_LEN,
    initialize_from_checkpoint_path=output_path_of(tootsie_8b_soft_raccoon, "checkpoints/step-829992"),
    reset_data_loader_on_init=False,
    per_device_eval_parallelism=16,
    weight_decay=0.0,
)

tootsie_8b_softer_raccoon_no_decay = dataclasses.replace(
    default_train(
        name="tootsie-8b-softer-raccoon-no-decay",
        tokenized=raccoon_mixture,
        model_config=llama_8b,
        train_config=tootsie_8b_softer_raccoon_train_no_decay,
        use_default_validation=True,
        tags=["llama", "8b", "ema", "exp898", "tootsie"],
        # HF is having trouble today so skipping this.
        eval_harness_tasks=[],
    ),
    override_output_path="checkpoints/tootsie-8b-softer-raccoon-no-decay",
)


# That made no difference. We're going to try *resetting the AdamW optimizer state*
# We don't have a mechanism for this yet in code. So we're going to manually delete it outside the framework
# and tell Levanter to allow partial checkpoint loading.

# Specifically we're going to delete the first and second moments of the optimizer state.
# $CHKPT/opt_state/inner_state/1/mu/
# $CHKPT/opt_state/inner_state/1/nu/


tootsie_8b_softer_raccoon_reset_train = dataclasses.replace(
    # NB: starting from the one WITH weight decay
    tootsie_8b_softer_raccoon_train,
    #initialize_from_checkpoint_path=output_path_of(tootsie_8b_softer_raccoon_no_decay, "checkpoints/step-839992"),
    allow_partial_checkpoint=True
)

tootsie_8b_softer_raccoon_reset_adamw = dataclasses.replace(
    default_train(
        name="tootsie-8b-softer-raccoon-reset-adamw",
        tokenized=raccoon_mixture,
        model_config=llama_8b,
        train_config=tootsie_8b_softer_raccoon_reset_train,
        use_default_validation=True,
        tags=["llama", "8b", "ema", "exp898", "tootsie"],
        # HF is having trouble today so skipping this.
        eval_harness_tasks=[],
    ),
    override_output_path="checkpoints/tootsie-8b-softer-raccoon-reset-adamw",
)


if __name__ == "__main__":
    executor_main(
        [tootsie_8b_soft_raccoon, tootsie_8b_softer_raccoon_reset_adamw],
        description="Train Tootsie 8b with cooldown from 1.7e-4 to 1.7e-5 over 125B tokens",
    )
