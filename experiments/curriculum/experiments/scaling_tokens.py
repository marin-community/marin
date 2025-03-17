from itertools import chain

from marin.execution.executor import executor_main

from experiments.curriculum.exp702_targeted_curriculum import (
    full_training_varying_mixture,
)

from experiments.curriculum.experiments.scaling_configs import (
    learning_rate_dict,
    chinchilla_steps,
    version_tag,
    correct_model_size,
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
            model_size=correct_model_size(model_size),
            num_train_steps=num_train_steps,
            learning_rate=learning_rate_dict[model_size],
            additional_tags=["flan-c4-eu-token-scaling"],
            version_tag=version_tag(learning_rate_dict[model_size])
        )
        for model_size in ["600m_0.001"]
        for num_train_steps in [48000]
        for duration_frac_stage2 in [0.01, 0.02, 0.05, 0.1, 0.2, 0.4]
    ]

    steps = list(chain(*stage_pairs))

    executor_main(
        steps=steps,
        description=f"Test training with varying mixtures",
    )