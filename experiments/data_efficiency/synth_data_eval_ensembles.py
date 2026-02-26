"""Evaluate ensembles for all configs in synth_data_regularized_ensembles.py
and synth_data_synth_ensembles.py.

For each hyperparameter configuration, we gather the 5-seed training steps
and evaluate ensembles of sizes 1 through 5.

Usage:
    uv run python experiments/data_efficiency/synth_data_eval_ensembles.py \
        --prefix gs://marin-us-central2 --dry_run true
"""

from experiments.data_efficiency.synth_data_regularized_ensembles import (
    ensemble_members_train_steps_dict as regularized_dict,
)
from experiments.data_efficiency.synth_data_synth_ensembles import (
    ensemble_members_train_steps_dict as synth_dict,
)
from experiments.data_efficiency.train import data_efficiency_eval_ensemble
from marin.execution.executor import ExecutorStep, executor_main

TOTAL_SEEDS = 5


def _collect_ensemble_eval_steps(
    train_steps_dict: dict,
    seed_index: int,
) -> list[ExecutorStep]:
    """Build ensemble eval steps for every config in a train-steps dict.

    Args:
        train_steps_dict: mapping from hparam-tuple (including seed) to ExecutorStep.
        seed_index: position of the seed field within the key tuple.
    """
    eval_steps: list[ExecutorStep] = []
    for key in train_steps_dict:
        if key[seed_index] != 0:
            continue
        members = []
        for seed in range(TOTAL_SEEDS):
            seed_key = key[:seed_index] + (seed,) + key[seed_index + 1 :]
            members.append(train_steps_dict[seed_key])
        for seed_count in range(1, TOTAL_SEEDS + 1):
            eval_steps.append(data_efficiency_eval_ensemble(members[:seed_count]))
    return eval_steps


# Regularized ensembles: key is
#   (base_train_steps, epochs, lr, weight_decay, model_name, block_cda, seed)
#   seed is at index 6
# eval_steps = _collect_ensemble_eval_steps(regularized_dict, seed_index=6)

# Synth ensembles: key is
#   (synthetic_data_name, synthetic_data_weight, base_train_steps, epochs,
#    lr, weight_decay, model_name, seed)
#   seed is at index 7
eval_steps = _collect_ensemble_eval_steps(synth_dict, seed_index=7)


if __name__ == "__main__":
    executor_main(
        steps=eval_steps,
        description="Evaluate all synth data ensemble configs",
    )
