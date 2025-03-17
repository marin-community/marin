from itertools import chain

from marin.execution.executor import executor_main

# Import required functions from exp702_targeted_curriculum.py
from experiments.curriculum.exp702_targeted_curriculum import (
    full_training_varying_mixture,
)

if __name__ == "__main__":
    # stage_pairs = [
    #     full_training_varying_mixture(
    #         data1_name="wiki",
    #         data2_name=data2_name,
    #         total_data1_portion=0.02,
    #         duration_frac_stage2=duration_frac_stage2,
    #         data1_frac_alloc_stage2=data1_frac_alloc_stage2,
    #         schedule_type="linear",
    #         cooldown_frac=0.05,
    #         model_size="600m_4096",
    #         num_train_steps=3000,
    #         learning_rate=0.001,
    #         num_eval=20,
    #         num_data1_repetitions=1,
    #         additional_tags=[f"eval-wiki-{data2_name}-allstage2"],
    #         num_lm_eval_harness=4,
    #         version_tag="",
    #         tpu_type="v4-128",
    #     )
    #     for duration_frac_stage2 in [0.02, 0.05, 0.1, 0.2, 0.5, 1.0] 
    #     for data1_frac_alloc_stage2 in [1.0, 0.75, 0.5, 0.25]
    #     # for duration_frac_stage2 in [0.02]
    #     # for data1_frac_alloc_stage2 in [1.0]
    #     for data2_name in ["dclm", "c4"]
    # ]

    model_size = "150m_4096"
    total_data1_portion = 0.05
    stage_pairs = [
        full_training_varying_mixture(
            data1_name="finemath",
            data2_name="c4",
            total_data1_portion=total_data1_portion,
            duration_frac_stage2=duration_frac_stage2,
            data1_frac_alloc_stage2=data1_frac_alloc_stage2,
            schedule_type="linear",
            cooldown_frac=0.2,
            model_size=model_size,
            num_train_steps=1000,
            learning_rate=0.001 if model_size == "600m_4096" else 0.003,
            num_eval=20,
            num_data1_repetitions=1,
            additional_tags=[f"eval-finemath-c4-allstage2-{model_size[:4]}-{total_data1_portion}"],
            num_lm_eval_harness=4,
            version_tag="",
            tpu_type="v4-128",
        )
        for duration_frac_stage2 in (list(filter(lambda x: x >= total_data1_portion, [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0])))
        for data1_frac_alloc_stage2 in ([0.25, 0.5, 0.75, 1.0] if model_size == "600m_4096" else [1.0])
    ]

    # stage_pairs = [
    #     full_training_varying_mixture(
    #         data1_name=data1_name,
    #         data2_name="wiki",
    #         total_data1_portion=1.0,
    #         duration_frac_stage2=1.0,
    #         data1_frac_alloc_stage2=1.0,
    #         schedule_type="linear",
    #         cooldown_frac=0.05,
    #         model_size="150m_4096",
    #         num_train_steps=1000,
    #         learning_rate=0.003,
    #         num_eval=20,
    #         num_data1_repetitions=1,
    #         additional_tags=[f"eval-debug-alldata"],
    #         num_lm_eval_harness=4,
    #         version_tag="-debug",
    #         tpu_type="v4-128",
    #     )
    #     for data1_name in ["c4", "wiki"]
    # ]

    steps = list(chain(*stage_pairs))

    executor_main(
        steps=steps,
        description="Test training with varying mixtures",
    )