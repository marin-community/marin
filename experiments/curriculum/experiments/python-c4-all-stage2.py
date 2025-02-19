from itertools import chain

from marin.execution.executor import executor_main

# Import required functions from exp702_targeted_curriculum.py
from experiments.curriculum.exp702_targeted_curriculum import (
    full_training_varying_mixture,
)

if __name__ == "__main__":
    stage_pairs = [
        full_training_varying_mixture(
            data1_name="stack_dedup",
            data2_name="c4",
            total_data1_portion=0.005,
            duration_frac_stage2=duration_frac_stage2,
            data1_frac_alloc_stage2=1.0,
            schedule_type="linear",
            cooldown_frac=0.05,
            model_size="150m_4096",
            num_train_steps=1000,
            learning_rate=0.003,
            num_eval=20,
            num_data1_repetitions=1,
            additional_tags=["eval-python-c4-allstage2"],
            num_lm_eval_harness=4,
            version_tag="-debug",
        )
        for duration_frac_stage2 in [0.5] 
    ]

    steps = list(chain(*stage_pairs))

    executor_main(
        steps=steps,
        description="Test training with varying mixtures",
    )