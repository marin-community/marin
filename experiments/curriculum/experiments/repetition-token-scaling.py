from itertools import chain

from marin.execution.executor import executor_main

# Import required functions from exp702_targeted_curriculum.py
from experiments.curriculum.exp702_targeted_curriculum import (
    full_training_varying_mixture,
)

if __name__ == "__main__":
    stage_pairs = [
        full_training_varying_mixture(
            data1_name="c4",
            data2_name="flan",
            total_data1_portion=1.0 / num_data1_repetitions,
            duration_frac_stage2=1.0,
            data1_frac_alloc_stage2=1.0,
            schedule_type="linear",
            cooldown_frac=0.30,
            model_size="600m",
            num_train_steps=base_num_steps * num_data1_repetitions,
            learning_rate=0.003,
            num_eval=20,
            num_data1_repetitions=num_data1_repetitions,
            additional_tags=["flan-c4-repetition-token-scaling-c4"],
        )
        for base_num_steps in [1000, 2000, 4000, 8000]
        for num_data1_repetitions in [1, 2, 4, 8, 16]
    ]

    steps = list(chain(*stage_pairs))

    executor_main(
        steps=steps,
        description="Test training with varying mixtures",
    )
