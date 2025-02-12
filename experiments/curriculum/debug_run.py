"""Debug script for curriculum training experiments."""

from itertools import chain

from marin.execution.executor import executor_main

# Import required functions from exp702_targeted_curriculum.py
from experiments.curriculum.exp702_targeted_curriculum import (
    full_training_varying_mixture,
)

if __name__ == "__main__":
    stage_pairs = [
        full_training_varying_mixture(
            data1_name="flan",
            data2_name="c4",
            total_data1_portion=0.99,
            duration_frac_stage2=0.5,
            data1_frac_alloc_stage2=0.5,
            schedule_type="linear",
            cooldown_frac=0.25,
            model_size="150m",
            num_train_steps=1000,
            learning_rate=0.003,
            num_eval=8,
            num_rare_epochs=16,
            additional_tags=["debug-repetition"],
            version_tag="-v2"
        )
    ]

    steps = list(chain(*stage_pairs))

    executor_main(
        steps=steps,
        description="Test training with varying mixtures",
    )
