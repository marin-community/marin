from itertools import chain

from marin.execution.executor import executor_main

# Import required functions from exp702_targeted_curriculum.py
from experiments.curriculum.exp702_targeted_curriculum import (
    full_training_varying_mixture,
)

if __name__ == "__main__":
    num_train_steps = 1000
    rare_data_portion = 0.04
    stage_pairs = [
        full_training_varying_mixture(
            data1_name="flan",
            data2_name="c4",
            total_data1_portion=rare_data_portion,
            duration_frac_stage2=duration_frac_stage2,
            data1_frac_alloc_stage2=1.0,
            schedule_type=schedule_type,
            cooldown_frac=0.05,
            sft_learning_rate=sft_learning_rate if schedule_type == "sft" else None,
            sft_steps=num_train_steps * duration_frac_stage2 if schedule_type == "sft" else None,
            model_size="150m_4096",
            num_train_steps=num_train_steps,
            learning_rate=0.003,
            num_eval=4,
            num_data1_repetitions=1,
            additional_tags=[f"flan-c4-sft-schedule-lrsweep"],
            num_lm_eval_harness=None,
            version_tag="-v2",
            tpu_type="v4-128",
            min_lr_ratio=0.0,
        )
        # for schedule_type in ["sft"]
        # for sft_learning_rate in [3e-3, 3e-4, 3e-5, 3e-6, 3e-7]
        for sft_learning_rate in [None]
        for duration_frac_stage2 in [0.04, 0.08, 0.16]
        for schedule_type in ["linear"]
    ]

    steps = list(chain(*stage_pairs))

    executor_main(
        steps=steps,
        description="Test training with varying mixtures",
    )