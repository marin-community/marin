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
            cooldown_frac=0.05,
            model_size="150m",
            num_train_steps=3000,
            learning_rate=0.003,
            num_eval=4,
            num_data1_repetitions=num_data1_repetitions,
            version_tag="-v2",
            additional_tags=["flan-c4-repetition-all-stage2"],
        )
        for num_data1_repetitions in [6]
        for duration_frac_stage2 in [0.05, 0.1, 0.2, 0.4, 0.8, 1.0]
    ]


    # stage_pairs = [
    #     full_training_varying_mixture(
    #         data1_name="flan",
    #         data2_name="c4",
    #         total_data1_portion=0.005,
    #         duration_frac_stage2=duration_frac_stage2,
    #         data1_frac_alloc_stage2=1.0,
    #         schedule_type="linear",
    #         cooldown_frac=cooldown_frac,
    #         model_size="150m",
    #         num_train_steps=3000,
    #         learning_rate=0.003,
    #         num_eval=20,
    #         num_data1_repetitions=num_data1_repetitions,
    #         additional_tags=["flan-c4-4-repetition-lr-sweep"],
    #     )
    #     for num_data1_repetitions in [4]
    #     for duration_frac_stage2 in [0.05, 0.1, 0.2, 0.4]
    #     for cooldown_frac in [0.01, 0.05, 0.1, 0.25]
    # ]

    steps = list(chain(*stage_pairs))

    executor_main(
        steps=steps,
        description="Test training with varying mixtures",
    )
