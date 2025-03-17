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
            data1_name="wiki",
            data2_name="c4",
            total_data1_portion=total_data1_portion,
            duration_frac_stage2=duration_frac_stage2,
            data1_frac_alloc_stage2=1.0,
            schedule_type="linear",
            cooldown_frac=0.05,
            model_size="150m_4096",
            num_train_steps=200,
            learning_rate=0.003,
            num_eval=3,
            num_data1_repetitions=1,
            additional_tags=["debug-internal-eval"],
            num_lm_eval_harness=1,
            version_tag="",
        )
        for duration_frac_stage2 in [1.0] 
        for total_data1_portion in [0.05]
    ]

    steps = list(chain(*stage_pairs))

    executor_main(
        steps=steps,
        description="Test training with varying mixtures",
    )
