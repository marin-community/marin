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
from experiments.evals.task_configs import CORE_TASKS_PLUS_MMLU
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
    learning_rate=1.7e-4,  # only does what we want b/c we're warmstarting from the ckpt below
    num_train_steps=COOLDOWN_END,
    min_lr_ratio=0.1,  # 1.7e-5
    decay=COOLDOWN_LEN,
    initialize_from_checkpoint_path=output_path_of(llama_8b_tootsie_phase3, "checkpoints/step-819924"),
    tpu_type="v4-128",
    node_count=2,
)

tulu_3_sft_data_as_validation = tulu3_flat_llama_tokenized

raccoon_mixture = add_validation_sets_to_mixture(
    phase_3_data_mixture,
    {"tulu_sft": tulu_3_sft_data_as_validation},
)

tootsie_8b_soft_raccoon = dataclasses.replace(
    default_train(
        name="llama-8b-tootsie-phase3",
        tokenized=raccoon_mixture,
        model_config=llama_8b,
        train_config=tootsie_8b_soft_raccoon_train,
        use_default_validation=True,
        tags=["llama", "8b", "ema", "exp898", "tootsie"],
        eval_harness_tasks=CORE_TASKS_PLUS_MMLU,
    ),
    override_output_path="checkpoints/tootsie-8b-soft-raccoon",
)

if __name__ == "__main__":
    executor_main(
        [tootsie_8b_soft_raccoon], description="Train Tootsie 8b with cooldown from 1.7e-4 to 1.7e-5 over 125B tokens"
    )
