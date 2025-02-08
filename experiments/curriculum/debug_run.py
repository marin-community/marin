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
            total_data1_portion=0.005,
            duration_frac_stage2=duration_frac_stage2,
            data1_frac_alloc_stage2=1.0,
            schedule_type="linear",
            cooldown_frac=cooldown_frac,
            model_size="600m",
            num_train_steps=num_train_steps,
            learning_rate=0.003,
            additional_tags=["debug-modules-not-found"],
            version_tag="-v4"
        )
        for num_train_steps in [1200]
        for cooldown_frac in [0.02]
        for duration_frac_stage2 in [0.02]
    ]

    steps = list(chain(*stage_pairs))

    executor_main(
        steps=steps,
        description="Test training with varying mixtures",
    )
